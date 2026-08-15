import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const compiledRoot = process.env.LIMCODE_COMPILED_ROOT
  ? path.resolve(root, process.env.LIMCODE_COMPILED_ROOT)
  : path.join(root, 'dist/extension');
const kernel = await import(pathToFileURL(
  path.join(compiledRoot, 'backend/reliableKernel/index.js')
).href);

function dependencies() {
  return {
    authorityCompiler: {
      async compile(request) {
        return {
          turnId: request.turnId,
          executorAgentId: request.executorAgentId,
          executionPreset: {
            content: JSON.stringify({ providerConfigId: 'provider-watchdog', modelId: 'model-watchdog' })
          },
          authoritySnapshot: {
            content: JSON.stringify({
              kind: 'effective-turn-authority',
              turnId: request.turnId,
              conversationId: request.conversationId,
              executorAgentId: request.executorAgentId,
              model: {
                providerConfigId: 'provider-watchdog',
                provider: 'openai-responses',
                modelId: 'model-watchdog',
                retryPolicy: { enabled: true, maxRetries: 3 }
              },
              modelProfile: {
                compressionThresholdTokens: 100000,
                contextWindowTokens: 128000,
                tokenEstimator: { kind: 'utf8-bytes-ceil', bytesPerToken: 4 }
              },
              toolPolicy: { id: 'tools-default', allowedTools: [], preset: 'custom', toolConfigs: {}, sourceConfigs: {} },
              systemPrompt: { id: 'prompt-default', text: '' },
              runtimeContext: { id: null, name: '', template: '' },
              workEnvironmentPolicy: { id: null, enabled: false, allowedWorkEnvironmentIds: [], defaultWorkEnvironmentId: null }
            })
          }
        };
      }
    },
    resolveWorkEnvironment: async () => undefined,
    mcpConnections: {
      async toolAnnotations() { return {}; },
      async callTool() { return null; }
    },
    mcpPolicyGate: {
      async authorize() { return { toolPolicyAllowed: true, planReviewAllowed: true }; }
    },
    attachmentSettings: {
      async loadGlobalSettings() {
        return {
          section: 'attachments',
          settings: { maxStoredInlineFileMb: 25 },
          filePath: 'settings/attachments.json'
        };
      }
    },
    providers: {
      resolve(providerId) {
        return {
          providerId,
          async sendFullRequest() { throw new Error('fixture provider must be supplied explicitly'); }
        };
      }
    },
    toolDispatcher: {
      definitions() { return []; },
      async dispatch() { throw new Error('fixture has no tools'); }
    }
  };
}

async function withApp(name, run) {
  const parent = await fs.mkdtemp(path.join(os.tmpdir(), `${name}-`));
  const authority = new kernel.RootAuthority(() => path.join(parent, 'runtime'));
  await kernel.initializeEmptyRuntimeRoot(authority);
  const app = await kernel.ReliableKernelApplication.open(authority, dependencies());
  try {
    const now = new Date().toISOString();
    await app.database.transaction([
      kernel.DOMAIN_REPOSITORIES.domain('Conversation').insert({
        id: name, title: name, status: 'active', created_at: now, updated_at: now
      }),
      kernel.DOMAIN_REPOSITORIES.domain('AgentConversationLink').insert({
        id: `${name}-agent-link`, conversation_id: name, agent_id: 'agent-main',
        role: 'default', created_at: now, updated_at: now
      })
    ]);
    const started = await app.turns.input({
      source: { kind: 'command', key: `${name}-input` },
      conversationId: name,
      leaseOwnerId: `${name}-owner`,
      hostBootId: app.database.hostBootId,
      leaseExpiresAt: new Date(Date.now() + 120_000).toISOString(),
      content: 'submit_plan approved, then update_task_list completed'
    });
    await run(app, name, started.turnId);
  } finally {
    await app.close();
    await fs.rm(parent, { recursive: true, force: true });
  }
}

async function list(app, domain, where = {}) {
  return (await app.database.snapshotAll(kernel.DOMAIN_REPOSITORIES.domain(domain).list({
    where,
    orderBy: { column: 'id', direction: 'asc' },
    limit: 100
  }))).snapshot;
}

async function get(app, domain, id) {
  return (await app.database.snapshot([kernel.DOMAIN_REPOSITORIES.domain(domain).get(id)])).snapshot[0];
}

async function createRequest(app, conversationId, turnId, key) {
  const head = (await list(app, 'ConversationContextHeadLink', { conversation_id: conversationId }))[0];
  const authority = (await list(app, 'AuthoritySnapshot', { turn_id: turnId }))[0];
  return app.modelProvider.createModelRequest({
    turnId,
    contextRootId: head.root_id,
    authoritySnapshotId: authority.id,
    recipe: {
      kind: 'reliable-agent-turn',
      round: '3',
      previousTool: 'update_task_list',
      tools: []
    },
    idempotencyKey: key
  });
}

function controlPlane(app, overrides = {}) {
  return new kernel.ModelProviderControlPlane(app.database, app.contentStore, {
    semanticTimeouts: {
      firstSemanticMs: 40,
      semanticIdleMs: 30,
      compressionCompletionMs: 80,
      ...(overrides.semanticTimeouts ?? {})
    },
    retryDelaysMs: overrides.retryDelaysMs ?? [0],
    adapterDrainTimeoutMs: 20
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForRequestStatus(app, requestId, status, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const request = await get(app, 'ModelRequest', requestId);
    if (request.status === status) return request;
    await sleep(5);
  }
  throw new Error(`ModelRequest ${requestId} did not reach ${status}.`);
}

test('可靠 Provider 请求携带冻结 Context root 所属的 conversationId', async () => {
  await withApp('provider-conversation-scope', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'conversation-scope');
    let capturedConversationId;
    await controlPlane(app).dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(fullRequest, controls) {
        capturedConversationId = fullRequest.conversationId;
        await controls.onEvent({
          kind: 'completed', streamSeq: '1',
          content: { text: 'done', thought: '', toolCalls: [] }
        });
      }
    });

    assert.equal(capturedConversationId, conversationId);
  });
});

test('plan→update_task_list 后只有伪 thought progress 不会续命，semantic stall 自动创建新 Attempt 并完成', async () => {
  await withApp('provider-semantic-stall', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'task-list-stall');
    const provider = controlPlane(app);
    let pseudoProgressStats;
    const originalEpochNow = provider.epochNow;
    let epoch = 10_000;
    provider.epochNow = () => epoch;
    const transient = [];
    let calls = 0;
    const startedAt = Date.now();
    const result = await provider.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(fullRequest, controls) {
        calls += 1;
        if (fullRequest.attemptSeq === '1') {
          await controls.onEvent({
            kind: 'output_delta', streamSeq: '1', semanticProgress: false,
            content: { type: 'thought_delta', text: '正在分析' }
          });
          for (let seq = 2; seq <= 5; seq += 1) {
            await sleep(5);
            epoch += 5_000;
            await controls.onEvent({
              kind: 'output_delta',
              streamSeq: String(seq),
              semanticProgress: false,
              content: { type: 'thought_progress', thoughtElapsedMs: seq * 500 }
            });
          }
          pseudoProgressStats = (await get(app, 'ModelRequest', request.modelRequestId)).stream_stats_json;
          await new Promise((resolve) => controls.signal.addEventListener('abort', resolve, { once: true }));
          const aborted = new Error('watchdog aborted stalled socket');
          aborted.name = 'AbortError';
          throw aborted;
        }
        await controls.onEvent({
          kind: 'completed', streamSeq: '1',
          content: { text: '无需用户继续即可完成', thought: '', toolCalls: [] }
        });
      }
    }, { onTransientTerminal: (event) => transient.push(event) });

    assert.equal(result.terminalState, 'completed');
    provider.epochNow = originalEpochNow;
    assert.equal(pseudoProgressStats.lastStreamEventAt, undefined,
      '本地 thought_progress 不能伪装成 durable Provider 活动心跳');
    assert.equal(pseudoProgressStats.lastStreamSeq, undefined);
    assert.equal(calls, 2);
    assert.ok(Date.now() - startedAt < 500, 'fixture must detect the stall promptly');
    assert.ok(transient.some((entry) =>
      entry.event.content.retrying === true
      && entry.event.content.terminalState === 'provider_transient_first_semantic_timeout'
    ));
    const operation = (await list(app, 'Operation', {
      owner_kind: 'model_request', owner_id: request.modelRequestId
    }))[0];
    assert.equal((await list(app, 'Attempt', { operation_id: operation.id })).length, 2);
    assert.equal((await list(app, 'MessageRevision')).filter((revision) => revision.role === 'user').length, 1);
  });
});

test('已收到文本、思考或工具输出后发生 semantic idle stall 会切换 Attempt，只采纳恢复输出', async () => {
  for (const fixture of [
    {
      kind: 'text',
      content: { type: 'text_delta', text: 'discarded stall text' }
    },
    {
      kind: 'thought',
      content: { type: 'thought_delta', text: 'discarded stall thought' }
    },
    {
      kind: 'tool',
      content: {
        type: 'tool_call_delta',
        calls: [{ id: 'discarded-stall-call', name: 'echo', argumentsDelta: '{"partial":', streamIndex: '0' }]
      }
    }
  ]) {
    await withApp(`provider-stall-after-${fixture.kind}`, async (app, conversationId, turnId) => {
      const request = await createRequest(app, conversationId, turnId, `stall-after-${fixture.kind}`);
      const terminals = [];
      let calls = 0;
      await controlPlane(app, {
        semanticTimeouts: { firstSemanticMs: 1_000, semanticIdleMs: 500 }
      }).dispatch(request.modelRequestId, {
        providerId: 'provider-watchdog',
        async sendFullRequest(fullRequest, controls) {
          calls += 1;
          if (fullRequest.attemptSeq === '1') {
            await controls.onEvent({
              kind: 'output_delta', streamSeq: '1', content: fixture.content
            });
            await new Promise((resolve) => controls.signal.addEventListener('abort', resolve, { once: true }));
            const aborted = new Error('watchdog replaced stalled partial stream');
            aborted.name = 'AbortError';
            throw aborted;
          }
          await controls.onEvent({
            kind: 'completed', streamSeq: '1',
            content: { text: `recovered-${fixture.kind}`, thought: '', toolCalls: [] }
          });
        }
      }, { onTransientTerminal: (event) => terminals.push(event) });

      assert.equal(calls, 2);
      const durableRequest = await get(app, 'ModelRequest', request.modelRequestId);
      assert.equal(durableRequest.terminal_state, 'completed');
      assert.equal(durableRequest.stream_stats_json.attemptSeq, '2');
      const operation = (await list(app, 'Operation', {
        owner_kind: 'model_request', owner_id: request.modelRequestId
      }))[0];
      const attempts = (await list(app, 'Attempt', { operation_id: operation.id }))
        .slice()
        .sort((left, right) => Number(left.attempt_seq) - Number(right.attempt_seq));
      assert.deepEqual(attempts.map((entry) => entry.status), ['transient_failed', 'completed']);
      assert.ok(terminals.some((terminal) =>
        terminal.attemptSeq === '1'
        && terminal.event.content.terminalState === 'provider_transient_stream_stalled'
        && terminal.event.content.retrying === true
        && terminal.event.content.discardOutput === true
      ));
      const completed = await app.modelProvider.completedEvent(request.modelRequestId);
      assert.equal(completed.content.text, `recovered-${fixture.kind}`);
      assert.equal(completed.content.thought, '');
      assert.deepEqual(completed.content.toolCalls, []);
      const checkpoints = await list(app, 'ModelStreamCheckpoint', { model_request_id: request.modelRequestId });
      assert.ok(checkpoints.every((checkpoint) => checkpoint.attempt_seq === 2n),
        'terminal prune must discard the stalled Attempt checkpoint');
    });
  }
});

test('普通 transient error 在已有语义输出后仍不盲目重放', async () => {
  await withApp('provider-generic-no-replay-after-output', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'generic-no-replay-after-output');
    let calls = 0;
    await assert.rejects(
      controlPlane(app, {
        semanticTimeouts: { firstSemanticMs: 500, semanticIdleMs: 500 }
      }).dispatch(request.modelRequestId, {
        providerId: 'provider-watchdog',
        async sendFullRequest(_fullRequest, controls) {
          calls += 1;
          await controls.onEvent({
            kind: 'output_delta', streamSeq: '1',
            content: { type: 'text_delta', text: 'generic partial output' }
          });
          throw new kernel.ProviderTransientError('temporary_service_error', 'generic transient after output');
        }
      }),
      /不自动重放请求/
    );
    assert.equal(calls, 1);
    assert.equal((await get(app, 'ModelRequest', request.modelRequestId)).terminal_state, 'provider_failed');
  });
});

test('连续 semantic idle stall 会自动重试到冻结预算上限后才终止', async () => {
  await withApp('provider-stall-retry-exhaustion', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'stall-retry-exhaustion');
    let calls = 0;
    await assert.rejects(
      controlPlane(app, {
        semanticTimeouts: { firstSemanticMs: 80, semanticIdleMs: 20 }
      }).dispatch(request.modelRequestId, {
        providerId: 'provider-watchdog',
        async sendFullRequest(_fullRequest, controls) {
          calls += 1;
          await controls.onEvent({
            kind: 'output_delta', streamSeq: '1',
            content: { type: 'text_delta', text: `stalled-${calls}` }
          });
          await new Promise((resolve) => controls.signal.addEventListener('abort', resolve, { once: true }));
          const aborted = new Error(`watchdog aborted stalled Attempt ${calls}`);
          aborted.name = 'AbortError';
          throw aborted;
        }
      }),
      (error) => /no semantic progress for 20ms/.test(error.message)
        && !/不自动重放请求/.test(error.message)
    );
    assert.equal(calls, 4);
    const durableRequest = await get(app, 'ModelRequest', request.modelRequestId);
    assert.equal(durableRequest.terminal_state, 'provider_transient_stream_stalled');
    const operation = (await list(app, 'Operation', {
      owner_kind: 'model_request', owner_id: request.modelRequestId
    }))[0];
    const attempts = (await list(app, 'Attempt', { operation_id: operation.id }))
      .slice()
      .sort((left, right) => Number(left.attempt_seq) - Number(right.attempt_seq));
    assert.deepEqual(attempts.map((entry) => entry.status), [
      'transient_failed', 'transient_failed', 'transient_failed', 'failed'
    ]);
  });
});

test('冻结 retryMaxAttempts=3 允许连续 transient failures 后第四个 Attempt 恢复', async () => {
  await withApp('provider-bounded-retry', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'bounded-retry');
    const provider = controlPlane(app);
    let calls = 0;
    await provider.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(_fullRequest, controls) {
        calls += 1;
        if (calls <= 3) throw new kernel.ProviderTransientError('temporary_service_error', `temporary-${calls}`);
        await controls.onEvent({
          kind: 'completed', streamSeq: '1', content: { text: 'recovered', thought: '', toolCalls: [] }
        });
      }
    });
    assert.equal(calls, 4);
    const operation = (await list(app, 'Operation', {
      owner_kind: 'model_request', owner_id: request.modelRequestId
    }))[0];
    const attempts = await list(app, 'Attempt', { operation_id: operation.id });
    assert.equal(attempts.length, 4);
    assert.equal(attempts.filter((entry) => entry.status === 'transient_failed').length, 3);
  });
});

test('已提交 retrying/not-before 在 Host handoff 后由新 ControlPlane 恢复，且永久错误不重试', async () => {
  await withApp('provider-retry-recovery', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'retry-recovery');
    const firstHost = controlPlane(app, { retryDelaysMs: [150] });
    const firstDispatch = firstHost.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest() {
        throw new kernel.ProviderTransientError('connection_interrupted', 'temporary handoff fixture');
      }
    });
    const retrying = await waitForRequestStatus(app, request.modelRequestId, 'retrying');
    assert.equal(retrying.stream_stats_json.attemptSeq, '2');
    assert.ok(retrying.stream_stats_json.retryNotBeforeAt > Date.now());
    await firstHost.quiesceAllActiveDispatches(new kernel.ExecutionHandoffError('fixture handoff'));
    await assert.rejects(firstDispatch, /handoff/i);

    const secondHost = controlPlane(app, { retryDelaysMs: [150] });
    await secondHost.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(fullRequest, controls) {
        assert.equal(fullRequest.attemptSeq, '2');
        await controls.onEvent({
          kind: 'completed', streamSeq: '1', content: { text: 'resumed', thought: '', toolCalls: [] }
        });
      }
    }, { reconnect: true });
    assert.equal((await get(app, 'ModelRequest', request.modelRequestId)).terminal_state, 'completed');

    const permanent = await createRequest(app, conversationId, turnId, 'permanent-no-retry');
    let permanentCalls = 0;
    await assert.rejects(secondHost.dispatch(permanent.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest() {
        permanentCalls += 1;
        throw new Error('invalid request schema');
      }
    }), /invalid request schema/);
    assert.equal(permanentCalls, 1);
  });
});

test('生产 deadline 固定为普通首语义300秒/idle600秒与压缩终态270秒', () => {
  assert.deepEqual(kernel.RELIABLE_PROVIDER_SEMANTIC_DEADLINES_MS, {
    ordinaryFirst: 300_000,
    ordinaryIdle: 600_000,
    compressionCompletion: 270_000
  });
});

test('Phase 0 Provider/Process/Client Feed 里程碑基准报告真实原始计数', { timeout: 120_000 }, () => {
  const run = childProcess.spawnSync(process.execPath, [
    'scripts/reliable-kernel/benchmark-phase0-milestones.mjs',
    '--samples=1'
  ], {
    cwd: root,
    encoding: 'utf8',
    timeout: 110_000,
    maxBuffer: 16 * 1024 * 1024
  });
  assert.equal(run.status, 0, [run.stdout, run.stderr].filter(Boolean).join('\n'));
  const report = JSON.parse(run.stdout);
  assert.deepEqual(report.existingAggregation, {
    intervalMs: 32,
    maxBatchEvents: 24,
    maxBufferedChars: 1024
  });
  assert.equal(report.checkpointCapacity, 33);

  const expected = new Map([
    [1, { transactions: 2, rows: 2, drops: 0 }],
    [10, { transactions: 2, rows: 2, drops: 0 }],
    [33, { transactions: 2, rows: 2, drops: 0 }],
    [100, { transactions: 2, rows: 2, drops: 0 }]
  ]);
  for (const providerCase of report.provider) {
    const measurement = providerCase.measurements[0];
    const target = expected.get(providerCase.eventCount);
    assert.equal(measurement.durableStreamTransactions, target.transactions);
    assert.equal(measurement.modelStreamWorkerTransactions, target.transactions);
    assert.equal(measurement.retainedCheckpointRows, target.rows);
    assert.equal(measurement.capacityDrops, target.drops);
    assert.equal(measurement.contextMaterializeCalls, 1);
  }
  for (const boundary of report.providerCapacityBoundary.measurements) {
    assert.equal(boundary.checkpointed, true);
    assert.equal(boundary.ignoredReason, null);
    assert.equal(boundary.transactionCount, 1);
  }

  if (!report.process.skipped) {
    assert.deepEqual(report.process.map((entry) => entry.command), ['true', 'printf_x', 'rg']);
    for (const processCase of report.process) {
      const measurement = processCase.measurements[0];
      assert.equal(measurement.terminalStatus, 'succeeded');
      for (const phase of ['spawn', 'identity_ready', 'terminal_receipt']) {
        assert.ok(measurement.phases[phase].count >= 1, `${processCase.command} lacks ${phase}`);
      }
      if (processCase.command === 'true') {
        assert.equal(measurement.phases.output_import.count, 0);
      } else {
        assert.ok(measurement.phases.output_import.count >= 1, `${processCase.command} lacks output_import`);
      }
    }
  }

  const feed = report.clientFeed.measurements[0];
  assert.equal(feed.withoutWebview.feedListenerEvents, 0);
  assert.equal(feed.withWebview.feedListenerEvents, 1);
  assert.ok(feed.withWebview.databaseListenerCount > feed.withoutWebview.databaseListenerCount);
});

test('Provider persists only the first delta while item_done and terminal remain durable', async () => {
  await withApp('provider-recovery-checkpoint-policy', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'recovery-checkpoint-policy');
    const results = [];
    let liveHeartbeat;
    const provider = controlPlane(app, {
      semanticTimeouts: { firstSemanticMs: 1_000, semanticIdleMs: 1_000 }
    });
    const originalEpochNow = provider.epochNow;
    let epoch = 10_000;
    provider.epochNow = () => epoch;
    try {
      await provider.dispatch(request.modelRequestId, {
        providerId: 'provider-watchdog',
        async sendFullRequest(_request, controls) {
          results.push(await controls.onEvent({
            kind: 'output_delta', streamSeq: '1', content: { type: 'text_delta', text: 'first' }
          }));
          results.push(await controls.onEvent({
            kind: 'output_delta', streamSeq: '2', content: { type: 'text_delta', text: 'x'.repeat(9_000) }
          }));
          results.push(await controls.onEvent({
            kind: 'output_delta', streamSeq: '3', content: { type: 'text_delta', text: 'y'.repeat(9_000) }
          }));
          await sleep(270);
          epoch += 5_000;
          results.push(await controls.onEvent({
            kind: 'output_delta', streamSeq: '4', content: { type: 'text_delta', text: 'after-time-window' }
          }));
          liveHeartbeat = (await get(app, 'ModelRequest', request.modelRequestId)).stream_stats_json;
          results.push(await controls.onEvent({
            kind: 'output_item_done', streamSeq: '5', content: { type: 'thought_done' }
          }));
          results.push(await controls.onEvent({
            kind: 'completed', streamSeq: '6', content: { text: 'done', thought: '', toolCalls: [] }
          }));
        }
      });
    } finally {
      provider.epochNow = originalEpochNow;
    }
    assert.deepEqual(results.map((result) => [result.checkpointed, result.ignoredReason ?? null]), [
      [true, null],
      [false, 'coalesced'],
      [false, 'coalesced'],
      [false, 'coalesced'],
      [true, null],
      [true, null]
    ]);
    assert.equal(liveHeartbeat.lastStreamSeq, '4');
    assert.equal(liveHeartbeat.lastStreamEventAt, 15_000);
    const durableRequest = await get(app, 'ModelRequest', request.modelRequestId);
    assert.equal(durableRequest.stream_stats_json.lastStreamSeq, undefined,
      'terminal summary replaces live heartbeat fields with authoritative terminal timing');
    const checkpoints = await list(app, 'ModelStreamCheckpoint', { model_request_id: request.modelRequestId });
    assert.equal(checkpoints.filter((row) => row.checkpoint_kind === 'output_item_done').length, 1);
  });
});

test('Provider semantic checkpoint overflow 有界合并且 terminal summary 仍可提交', async () => {
  await withApp('provider-semantic-checkpoint-overflow', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'semantic-checkpoint-overflow');
    const results = [];
    await controlPlane(app, {
      // This fixture intentionally performs 41 sequential SQLite transactions. Keep the semantic
      // deadline above test-runner/worker scheduling jitter; the dedicated commit-stall case below
      // owns the short watchdog boundary.
      semanticTimeouts: { firstSemanticMs: 500, semanticIdleMs: 500 }
    }).dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(_request, controls) {
        await controls.onEvent({
          kind: 'output_delta', streamSeq: '1', content: { type: 'text_delta', text: 'first' }
        });
        for (let item = 1; item <= 40; item += 1) {
          results.push(await controls.onEvent({
            kind: 'output_item_done',
            streamSeq: String(item + 1),
            content: { type: 'synthetic_item', item }
          }));
        }
        results.push(await controls.onEvent({
          kind: 'completed', streamSeq: '42',
          content: { text: 'terminal survives capacity', thought: '', toolCalls: [] }
        }));
      }
    });
    const durableRequest = await get(app, 'ModelRequest', request.modelRequestId);
    assert.equal(durableRequest.status, 'terminal');
    assert.equal(durableRequest.terminal_state, 'completed');
    assert.ok(results.some((result) => result.ignoredReason === 'checkpoint-capacity'));
    assert.equal(results.at(-1).terminal, true);
    const checkpoints = await list(app, 'ModelStreamCheckpoint', {
      model_request_id: request.modelRequestId
    });
    assert.equal(checkpoints.length, 33);
    assert.equal(checkpoints.filter((row) => row.checkpoint_kind === 'terminal_summary').length, 1);
  });
});

test('单条 thought 后的合法静默在 idle 边界内完成且不创建重试 Attempt', async () => {
  await withApp('provider-legitimate-thought-silence', async (app, conversationId, turnId) => {
    const provider = controlPlane(app, {
      // 合法静默只需小于 idle 边界；并行测试负载下 SQLite/CAS 调度抖动可达数百毫秒，
      // 边界留足余量，避免把测试环境抖动误判成看门狗误触发。
      semanticTimeouts: {
        firstSemanticMs: 1_000,
        semanticIdleMs: 1_000,
        compressionCompletionMs: 2_000
      }
    });
    const request = await createRequest(app, conversationId, turnId, 'legitimate-thought-silence');
    let calls = 0;
    await provider.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(_request, controls) {
        calls += 1;
        await controls.onEvent({
          kind: 'output_delta', streamSeq: '1',
          content: { type: 'thought_delta', text: 'planning one long step' }
        });
        await sleep(80);
        await controls.onEvent({
          kind: 'completed', streamSeq: '2',
          content: { text: 'completed without reconnect', thought: '', toolCalls: [] }
        });
      }
    });
    assert.equal(calls, 1);
    const operation = (await list(app, 'Operation', {
      owner_kind: 'model_request', owner_id: request.modelRequestId
    }))[0];
    assert.equal((await list(app, 'Attempt', { operation_id: operation.id })).length, 1);
  });
});

test('正常 semantic progress 会刷新 idle watchdog，首语义 black-hole 会自动 retry', async () => {
  await withApp('provider-semantic-progress', async (app, conversationId, turnId) => {
    const provider = controlPlane(app, {
      // Repeated progress must outlive the first-event deadline, while the idle deadline allows
      // ordinary SQLite scheduling jitter now that it intentionally stays armed during each commit.
      semanticTimeouts: { firstSemanticMs: 50, semanticIdleMs: 150 }
    });
    const progressing = await createRequest(app, conversationId, turnId, 'normal-semantic-progress');
    let progressCalls = 0;
    await provider.dispatch(progressing.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(_request, controls) {
        progressCalls += 1;
        for (let seq = 1; seq <= 4; seq += 1) {
          await sleep(20);
          await controls.onEvent({
            kind: 'output_delta', streamSeq: String(seq),
            content: { type: 'thought_delta', text: `semantic-${seq}` }
          });
        }
        await controls.onEvent({
          kind: 'completed', streamSeq: '5', content: { text: 'normal', thought: '', toolCalls: [] }
        });
      }
    });
    assert.equal(progressCalls, 1);

    const firstEventBlackHole = await createRequest(app, conversationId, turnId, 'first-semantic-timeout');
    const terminals = [];
    let blackHoleCalls = 0;
    await provider.dispatch(firstEventBlackHole.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(request, controls) {
        blackHoleCalls += 1;
        if (request.attemptSeq === '1') {
          await new Promise((resolve) => controls.signal.addEventListener('abort', resolve, { once: true }));
          const aborted = new Error('first semantic watchdog abort');
          aborted.name = 'AbortError';
          throw aborted;
        }
        await controls.onEvent({
          kind: 'completed', streamSeq: '1', content: { text: 'first timeout recovered', thought: '', toolCalls: [] }
        });
      }
    }, { onTransientTerminal: (event) => terminals.push(event) });
    assert.equal(blackHoleCalls, 2);
    assert.ok(terminals.some((entry) =>
      entry.event.content.terminalState === 'provider_transient_first_semantic_timeout'
    ));
  });
});

test('Provider 在 durable stream event 提交阻塞时仍保持 semantic idle watchdog', async () => {
  await withApp('provider-semantic-commit-stall', async (app, conversationId, turnId) => {
    const provider = controlPlane(app, {
      semanticTimeouts: { firstSemanticMs: 120, semanticIdleMs: 35 }
    });
    const request = await createRequest(app, conversationId, turnId, 'semantic-commit-stall');
    const originalCommit = app.database.commitModelStreamEvent.bind(app.database);
    let releaseCommit;
    let commitStarted;
    const commitStartedPromise = new Promise((resolve) => { commitStarted = resolve; });
    const releaseCommitPromise = new Promise((resolve) => { releaseCommit = resolve; });
    app.database.commitModelStreamEvent = async (input) => {
      if (input.modelRequestId !== request.modelRequestId) return originalCommit(input);
      commitStarted();
      await releaseCommitPromise;
      return originalCommit(input);
    };
    const startedAt = Date.now();
    const dispatch = provider.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest(_request, controls) {
        await controls.onEvent({
          kind: 'output_delta', streamSeq: '1',
          content: { type: 'thought_delta', text: 'checkpoint blocks' }
        });
      }
    }, { timeoutMs: 500 });
    try {
      await commitStartedPromise;
      await assert.rejects(dispatch, /no semantic progress for 35ms/);
    } finally {
      releaseCommit();
      app.database.commitModelStreamEvent = originalCommit;
    }
    assert.ok(Date.now() - startedAt < 300, 'semantic watchdog must win before the outer dispatch deadline');
  });
});

test('transient failures 达到冻结上限后才终止，408/425/429/5xx 均可分类为自动 retry', async () => {
  await withApp('provider-retry-exhaustion', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'retry-exhaustion');
    const provider = controlPlane(app);
    let calls = 0;
    await assert.rejects(provider.dispatch(request.modelRequestId, {
      providerId: 'provider-watchdog',
      async sendFullRequest() {
        calls += 1;
        throw new kernel.ProviderTransientError('temporary_service_error', `exhaust-${calls}`);
      }
    }), /exhaust-4/);
    assert.equal(calls, 4);
    const terminal = await get(app, 'ModelRequest', request.modelRequestId);
    assert.equal(terminal.status, 'terminal');
    assert.equal(terminal.terminal_state, 'provider_transient_temporary_service_error');
  });

  async function classifiedStatus(status) {
    const capability = {
      start(request, emit) {
        emit({
          type: 'llm:error',
          payload: { requestId: request.id, message: `HTTP ${status}`, rawError: { status } }
        });
      },
      compact() { throw new Error('unused'); }, abort() {}, cancelRetry() {}, dispose() {}, listModels: async () => []
    };
    const adapter = new kernel.LlmCapabilityFullRequestAdapter('provider-watchdog', capability);
    return adapter.sendFullRequest({
      kind: 'full-model-request', modelRequestId: `status-${status}`,
      conversationId: 'conversation-status',
      attemptSeq: '1', socketGeneration: '1', providerId: 'provider-watchdog', modelId: 'model-watchdog',
      authoritySnapshot: {
        model: { provider: 'openai-responses' },
        toolPolicy: { allowedTools: [], preset: 'custom', sourceConfigs: {} }
      },
      recipe: { tools: [] }, context: []
    }, { onEvent: async () => ({ accepted: true, checkpointed: true, terminal: false }) });
  }
  for (const status of [408, 425, 429, 500, 503]) {
    await assert.rejects(classifiedStatus(status), (error) => error instanceof kernel.ProviderTransientError);
  }
});

test('所有可恢复的 Responses 终态前关闭都会创建 durable Attempt 2 并自动恢复', async () => {
  for (const fixture of [
    { closeCode: 1000, reason: '' },
    { closeCode: 1001, reason: ' Going Away' },
    { closeCode: 1005, reason: ' No Status Received' },
    { closeCode: 1006, reason: ' Abnormal Closure' },
    { closeCode: 1008, reason: ' Policy Violation' },
    { closeCode: 1011, reason: ' Internal Error' },
    { closeCode: 1012, reason: ' Service Restart' },
    { closeCode: 1013, reason: ' Try Again Later' },
    { closeCode: 1014, reason: ' Bad Gateway' },
    { closeCode: 1015, reason: ' TLS Handshake' }
  ]) {
    await withApp(`provider-pre-terminal-close-${fixture.closeCode}`, async (app, conversationId, turnId) => {
      const request = await createRequest(app, conversationId, turnId, `pre-terminal-close-${fixture.closeCode}`);
      const message = `OpenAI Responses WebSocket closed before terminal event: ${fixture.closeCode}${fixture.reason}`;
      let calls = 0;
      const capability = {
        start(llmRequest, emit) {
          calls += 1;
          if (calls === 1) {
            emit({
              type: 'llm:error',
              payload: {
                requestId: llmRequest.id,
                message,
                rawError: {
                  name: 'WebSocketCloseError',
                  message,
                  closeCode: fixture.closeCode,
                  phase: 'awaiting_first_event',
                  receivedServerEvent: false,
                  retryable: false,
                  transportAttemptsExhausted: false
                }
              }
            });
            return;
          }
          emit({ type: 'llm:done', payload: { requestId: llmRequest.id, completedAt: Date.now() } });
        },
        compact() { throw new Error('unused'); },
        abort() {}, cancelRetry() {}, dispose() {}, listModels: async () => []
      };
      const adapter = new kernel.LlmCapabilityFullRequestAdapter('provider-watchdog', capability);
      await controlPlane(app, {
        semanticTimeouts: { firstSemanticMs: 500, semanticIdleMs: 500 }
      }).dispatch(request.modelRequestId, adapter);

      assert.equal(calls, 2);
      assert.equal((await get(app, 'ModelRequest', request.modelRequestId)).terminal_state, 'completed');
      const operation = (await list(app, 'Operation', {
        owner_kind: 'model_request', owner_id: request.modelRequestId
      }))[0];
      const attempts = (await list(app, 'Attempt', { operation_id: operation.id }))
        .slice()
        .sort((left, right) => Number(left.attempt_seq) - Number(right.attempt_seq));
      assert.equal(attempts.length, 2);
      assert.equal(attempts[0].status, 'transient_failed');
      assert.equal(attempts[1].status, 'completed');
    });
  }
});

test('配置的终态前关闭在文本、思考或工具输出后仍切换 Attempt，最终只采纳恢复输出', async () => {
  for (const fixture of [
    {
      closeCode: 1013,
      reason: 'upstream websocket disconnected; please reconnect',
      emitPartial(requestId, emit) {
        emit({ type: 'llm:delta', payload: { requestId, text: 'discarded text' } });
      }
    },
    {
      closeCode: 1006,
      reason: 'abnormal closure',
      emitPartial(requestId, emit) {
        emit({
          type: 'llm:thoughtDelta',
          payload: { requestId, text: 'discarded thought', thoughtStartedAt: Date.now(), thoughtElapsedMs: 1 }
        });
      }
    },
    {
      closeCode: 1008,
      reason: 'policy violation',
      emitPartial(requestId, emit) {
        emit({
          type: 'llm:toolCallDelta',
          payload: {
            requestId,
            calls: [{ id: 'discarded-call', name: 'echo', argumentsDelta: '{"partial":', streamIndex: '0' }]
          }
        });
      }
    }
  ]) {
    await withApp(`provider-close-after-output-${fixture.closeCode}`, async (app, conversationId, turnId) => {
      const request = await createRequest(app, conversationId, turnId, `close-after-output-${fixture.closeCode}`);
      const message = `OpenAI Responses WebSocket closed before terminal event: ${fixture.closeCode} ${fixture.reason}`;
      const terminals = [];
      let calls = 0;
      const adapter = new kernel.LlmCapabilityFullRequestAdapter('provider-watchdog', {
        start(llmRequest, emit) {
          calls += 1;
          if (calls === 1) {
            fixture.emitPartial(llmRequest.id, emit);
            emit({
              type: 'llm:error',
              payload: {
                requestId: llmRequest.id,
                message,
                rawError: {
                  name: 'WebSocketCloseError',
                  message,
                  closeCode: fixture.closeCode,
                  phase: 'streaming',
                  receivedServerEvent: true,
                  receivedSemanticOutput: true,
                  retryable: false,
                  transportAttemptsExhausted: false
                }
              }
            });
            return;
          }
          emit({ type: 'llm:delta', payload: { requestId: llmRequest.id, text: `recovered-${fixture.closeCode}` } });
          emit({ type: 'llm:done', payload: { requestId: llmRequest.id, completedAt: Date.now() } });
        },
        compact() { throw new Error('unused'); },
        abort() {}, cancelRetry() {}, dispose() {}, listModels: async () => []
      });

      await controlPlane(app).dispatch(request.modelRequestId, adapter, {
        onTransientTerminal: (terminal) => terminals.push(terminal)
      });

      assert.equal(calls, 2);
      const durableRequest = await get(app, 'ModelRequest', request.modelRequestId);
      assert.equal(durableRequest.terminal_state, 'completed');
      assert.equal(durableRequest.stream_stats_json.attemptSeq, '2');
      const operation = (await list(app, 'Operation', {
        owner_kind: 'model_request', owner_id: request.modelRequestId
      }))[0];
      const attempts = (await list(app, 'Attempt', { operation_id: operation.id }))
        .slice()
        .sort((left, right) => Number(left.attempt_seq) - Number(right.attempt_seq));
      assert.deepEqual(attempts.map((entry) => entry.status), ['transient_failed', 'completed']);
      assert.ok(terminals.some((terminal) =>
        terminal.attemptSeq === '1'
        && terminal.event.content.retrying === true
        && terminal.event.content.discardOutput === true
      ));
      const completed = await app.modelProvider.completedEvent(request.modelRequestId);
      assert.equal(completed.content.text, `recovered-${fixture.closeCode}`);
      assert.equal(completed.content.thought, '');
      assert.deepEqual(completed.content.toolCalls, []);
      const checkpoints = await list(app, 'ModelStreamCheckpoint', { model_request_id: request.modelRequestId });
      assert.ok(checkpoints.length > 0);
      assert.ok(checkpoints.every((checkpoint) => checkpoint.attempt_seq === 2n),
        'terminal prune must discard every failed-Attempt checkpoint');
    });
  }
});

test('response.created 后首语义前 EOF 仍会创建 durable Attempt 2 并自动恢复', async () => {
  await withApp('provider-created-before-eof', async (app, conversationId, turnId) => {
    const request = await createRequest(app, conversationId, turnId, 'created-before-eof');
    let calls = 0;
    const capability = {
      start(llmRequest, emit) {
        calls += 1;
        if (calls === 1) {
          emit({
            type: 'llm:error',
            payload: {
              requestId: llmRequest.id,
              message: 'OpenAI Responses WebSocket closed after response.created',
              rawError: {
                name: 'WebSocketCloseError',
                message: 'OpenAI Responses WebSocket closed before terminal event: 1000',
                closeCode: 1000,
                phase: 'streaming',
                receivedServerEvent: true,
                receivedSemanticOutput: false,
                retryable: true,
                transportAttemptsExhausted: false
              }
            }
          });
          return;
        }
        emit({ type: 'llm:done', payload: { requestId: llmRequest.id, completedAt: Date.now() } });
      },
      compact() { throw new Error('unused'); },
      abort() {}, cancelRetry() {}, dispose() {}, listModels: async () => []
    };
    const adapter = new kernel.LlmCapabilityFullRequestAdapter('provider-watchdog', capability);
    await controlPlane(app).dispatch(request.modelRequestId, adapter);

    assert.equal(calls, 2);
    assert.equal((await get(app, 'ModelRequest', request.modelRequestId)).terminal_state, 'completed');
    const operation = (await list(app, 'Operation', {
      owner_kind: 'model_request', owner_id: request.modelRequestId
    }))[0];
    const attempts = await list(app, 'Attempt', { operation_id: operation.id });
    assert.equal(attempts.length, 2);
    assert.equal(attempts.filter((entry) => entry.status === 'transient_failed').length, 1);
  });
});

test('0.1.35 retryable/error metadata映射为持久 retry authority，exhausted 与永久错误保持终态', async () => {
  async function runRawError(rawError) {
    const capability = {
      start(request, emit) {
        emit({ type: 'llm:error', payload: { requestId: request.id, message: rawError.message, rawError } });
      },
      compact() { throw new Error('unused'); },
      abort() {},
      cancelRetry() {},
      dispose() {},
      listModels: async () => []
    };
    const adapter = new kernel.LlmCapabilityFullRequestAdapter('provider-watchdog', capability);
    return adapter.sendFullRequest({
      kind: 'full-model-request',
      modelRequestId: `metadata-${Math.random()}`,
      conversationId: 'conversation-metadata',
      attemptSeq: '1', socketGeneration: '1',
      providerId: 'provider-watchdog', modelId: 'model-watchdog',
      authoritySnapshot: {
        model: { provider: 'openai-responses' },
        toolPolicy: { allowedTools: [], preset: 'custom', sourceConfigs: {} }
      },
      recipe: { tools: [] }, context: []
    }, { onEvent: async () => ({ accepted: true, checkpointed: true, terminal: false }) });
  }

  await assert.rejects(
    runRawError({ message: 'gateway reset', transport: 'websocket', retryable: true }),
    (error) => error instanceof kernel.ProviderTransientError && error.reason === 'connection_interrupted'
  );
  await assert.rejects(
    runRawError({ message: 'Unexpected server response: 429', transport: 'websocket' }),
    (error) => error instanceof kernel.ProviderTransientError && error.reason === 'rate_limited'
  );
  await assert.rejects(
    runRawError({
      message: 'OpenAI Responses WebSocket closed before terminal event: 1006 Abnormal Closure',
      closeCode: 1006,
      phase: 'awaiting_first_event',
      receivedServerEvent: false,
      receivedSemanticOutput: false,
      retryable: false,
      transportAttemptsExhausted: false
    }),
    (error) => error instanceof kernel.ProviderTransientError && error.reason === 'connection_interrupted'
  );
  await assert.rejects(
    runRawError({
      message: 'OpenAI Responses WebSocket closed before terminal event: 1000',
      closeCode: 1000,
      phase: 'awaiting_first_event',
      receivedServerEvent: false,
      receivedSemanticOutput: false,
      retryable: false,
      transportAttemptsExhausted: false
    }),
    (error) => error instanceof kernel.ProviderTransientError && error.reason === 'connection_interrupted'
  );
  await assert.rejects(
    runRawError({
      message: 'OpenAI Responses WebSocket closed before terminal event: 1000',
      closeCode: 1000,
      phase: 'streaming',
      receivedServerEvent: true,
      receivedSemanticOutput: false,
      retryable: true,
      transportAttemptsExhausted: false
    }),
    (error) => error instanceof kernel.ProviderTransientError
      && error.reason === 'connection_interrupted'
  );
  await assert.rejects(
    runRawError({
      message: 'OpenAI Responses WebSocket closed before terminal event: 1008 policy violation',
      closeCode: 1008,
      phase: 'streaming',
      receivedServerEvent: true,
      receivedSemanticOutput: true,
      retryable: false,
      transportAttemptsExhausted: false
    }),
    (error) => error instanceof kernel.ProviderTransientError
      && error.reason === 'connection_interrupted'
      && error.retryAfterOutput === true
  );
  await assert.rejects(
    runRawError({ message: 'transport exhausted', retryable: true, transportAttemptsExhausted: true }),
    (error) => !(error instanceof kernel.ProviderTransientError)
  );
  await assert.rejects(
    runRawError({ message: 'context length exceeded', status: 400, retryable: false }),
    (error) => !(error instanceof kernel.ProviderTransientError)
  );
});
