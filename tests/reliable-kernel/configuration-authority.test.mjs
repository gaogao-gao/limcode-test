import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { after, test } from 'node:test';

const require = createRequire(import.meta.url);
const Module = require('node:module');
const originalLoad = Module._load;
const vscode = createVscodeStub();
Module._load = function load(request, parent, isMain) {
  if (request === 'vscode') return vscode;
  return originalLoad.call(this, request, parent, isMain);
};
after(() => { Module._load = originalLoad; });

const { createVscodeStoragePaths } = require('../../dist/extension/backend/capabilities/vscodeStorage/paths.js');
const { createDefaultLlmProviderConfig } = require('../../dist/extension/backend/capabilities/vscodeStorage/llmProviderConfigs.js');
const { loadRecordStore } = require('../../dist/extension/backend/capabilities/vscodeStorage/recordStore.js');
const { VscodeConfigurationAuthority } = require('../../dist/extension/backend/reliableKernel/vscodeConfigurationAuthority.js');
const { frozenCompressionPolicy } = require('../../dist/extension/backend/reliableKernel/frozenAuthority.js');
const { resolveToolPolicyLayers } = require('../../dist/extension/shared/toolPolicyResolution.js');
const {
  createRemoteServerWorkEnvironmentRecord,
  workEnvironmentIdFromUri
} = require('../../dist/extension/shared/workEnvironmentCatalog.js');

async function saveLatestGlobalSettings(authority, section, settings) {
  const current = await authority.loadGlobalSettings(section);
  return authority.saveGlobalSettings(section, settings, current.revision);
}

test('ToolPolicy 层按能力上界收窄、深合并配置，并保持来源 deny 单调', () => {
  const resolved = resolveToolPolicyLayers([
    {
      scopeKind: 'global',
      policy: {
        id: 'global-policy',
        allowedTools: ['read', 'write', 'bash'],
        preset: 'yolo',
        toolConfigs: {
          bash: {
            config: { limits: { lines: 20, chars: 1000 }, cwd: 'global' },
            autoApproveExecution: false,
            display: { autoExpand: false, autoOpenDiffPreview: true }
          }
        },
        sourceConfigs: {
          exa: { enabled: false, disabledTools: ['exa_global_denied'] }
        }
      }
    },
    {
      scopeKind: 'agent',
      policy: {
        id: 'agent-policy',
        allowedTools: ['read', 'bash', 'skills'],
        preset: 'inherit',
        toolConfigs: {
          bash: {
            config: { limits: { chars: 500 }, cwd: 'agent' },
            autoApproveExecution: true,
            display: { autoExpand: true }
          }
        },
        sourceConfigs: {
          exa: { enabled: true, disabledTools: ['exa_agent_denied'] }
        }
      }
    },
    {
      scopeKind: 'workflow',
      policy: {
        id: 'workflow-policy',
        allowedTools: ['bash'],
        preset: 'inherit'
      }
    }
  ], ['read', 'write', 'bash', 'skills', 'delete']);

  assert.equal(resolved.id, 'workflow-policy');
  assert.equal(resolved.preset, 'yolo');
  assert.deepEqual(resolved.allowedTools, ['bash']);
  assert.deepEqual(resolved.toolConfigs.bash, {
    config: { limits: { lines: 20, chars: 500 }, cwd: 'agent' },
    autoApproveExecution: true,
    display: { autoExpand: true, autoOpenDiffPreview: true }
  });
  assert.deepEqual(resolved.sourceConfigs.exa, {
    enabled: false,
    disabledTools: ['exa_agent_denied', 'exa_global_denied']
  });
});

test('VscodeConfigurationAuthority 独立持久化配置记录/Link，并按 Run→Conversation→Workflow→Agent→Global 冻结 authority', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'limcode-configuration-authority-'));
  try {
    const paths = createVscodeStoragePaths(vscode.Uri.file(root));
    const authority = new VscodeConfigurationAuthority(() => paths);
    const provider = {
      ...createDefaultLlmProviderConfig({ name: '测试 Provider' }),
      id: 'provider:test',
      model: 'model:test',
      models: [{ id: 'model:test', name: '测试模型' }],
      systemPromptPrefix: '渠道默认前置要求',
      modelConfigs: [{
        id: 'model-config:test',
        modelId: 'model:test',
        toolCallFormat: 'function-call',
        openaiResponsesTransport: 'http',
        stream: true,
        retryOnError: true,
        retryMaxAttempts: 3,
        enableMultimodalTools: true,
        contextWindowTokens: 180_000,
        generationConfig: { maxOutputTokens: 24_000 },
        systemPromptPrefix: '模型专属前置要求',
        createdAt: 1,
        updatedAt: 1
      }]
    };
    await saveLatestGlobalSettings(authority, 'llmProviderConfigs', { configs: [provider] });
    await saveLatestGlobalSettings(authority, 'llm', { activeProviderConfigId: provider.id });

    const folderPath = path.join(root, 'workspace');
    await fs.mkdir(folderPath, { recursive: true });
    const folderUri = vscode.Uri.file(folderPath).toString();
    await authority.synchronizeWorkspaceFolders([{ uri: folderUri, name: 'Workspace', rootPath: folderPath, index: 0 }]);
    const workEnvironmentId = workEnvironmentIdFromUri(folderUri);
    const remoteEnvironment = await authority.mutations.upsertWorkEnvironment(createRemoteServerWorkEnvironmentRecord({
      id: 'work-env-remote-test',
      name: 'Remote Test',
      host: 'remote.test'
    }));

    const agent = await authority.mutations.createAgent({ name: '配置 Agent', kind: 'custom' });
    const workflow = await authority.mutations.createWorkflow({ name: '可靠 Workflow' });
    await authority.mutations.selectConversationWorkflow({
      conversationId: 'conversation:test',
      scopeKind: 'workflow',
      workflowId: workflow.id
    });
    await authority.mutations.setModelProfile({
      scopeKind: 'conversation',
      scopeId: 'conversation:test',
      name: '对话模型',
      providerConfigId: provider.id,
      provider: provider.provider,
      model: 'model:test'
    });
    await authority.mutations.setToolPolicy({
      scopeKind: 'global',
      name: '全局工具',
      allowedTools: ['read'],
      sourceConfigs: {
        'mcp-exa': { enabled: true, disabledTools: ['exa_hidden'] }
      }
    });
    await authority.mutations.setToolPolicy({
      scopeKind: 'workflow',
      scopeId: workflow.id,
      name: 'Workflow 工具',
      allowedTools: ['read', 'skills']
    });
    await authority.mutations.setPlanReviewPolicy({
      scopeKind: 'workflow',
      scopeId: workflow.id,
      mode: 'before_mutation',
      allowReadonlyBeforeApproval: true,
      requireForToolRiskLevels: ['write']
    });
    await authority.mutations.setSystemPrompt({ scopeKind: 'global', name: '全局规则', text: 'GLOBAL' });
    await authority.mutations.setSystemPrompt({ scopeKind: 'agent', scopeId: agent.id, name: 'Agent 规则', text: 'AGENT' });
    await authority.mutations.setSystemPrompt({ scopeKind: 'workflow', scopeId: workflow.id, name: '工作流规则', text: 'WORKFLOW' });
    await authority.mutations.setSystemPrompt({ scopeKind: 'conversation', scopeId: 'conversation:test', name: '对话规则', text: 'CONVERSATION' });
    await authority.mutations.setRuntimeContext({
      scopeKind: 'conversation',
      scopeId: 'conversation:test',
      template: 'ENV:\n{{$workEnvironment.current}}'
    });
    await authority.mutations.setWorkEnvironmentPolicy({
      scopeKind: 'conversation',
      scopeId: 'conversation:test',
      enabled: true,
      allowedWorkEnvironmentIds: [workEnvironmentId],
      defaultWorkEnvironmentId: workEnvironmentId
    });
    await authority.mutations.selectConversationWorkEnvironment('conversation:test', workEnvironmentId);

    const snapshot = await authority.configurationClientState();
    assert.equal(snapshot.agents.some((record) => record.id === agent.id), true);
    assert.equal(snapshot.workflows.some((record) => record.id === workflow.id), true);
    assert.equal(snapshot.workflows.some((record) => record.id === 'builtin:plan'), true);
    assert.equal(snapshot.conversationWorkflowSelections.length, 1);
    assert.equal(snapshot.conversationWorkEnvironmentLinks.length, 1);
    assert.equal(snapshot.workEnvironments.find((record) => record.id === workEnvironmentId)?.available, true);

    const compiled = await authority.compile({
      conversationId: 'conversation:test',
      turnId: 'turn:test',
      executorAgentId: agent.id,
      intentKind: 'input'
    });
    const frozen = JSON.parse(compiled.authoritySnapshot.content);
    assert.equal(frozen.model.providerConfigId, provider.id);
    assert.equal(frozen.model.modelId, 'model:test');
    assert.equal(frozen.model.systemPromptPrefix, '模型专属前置要求');
    assert.equal(frozen.model.maxOutputTokens, 24_000);
    assert.equal(frozen.compression.config.llmSummary.targetTokens, 8_000);
    assert.equal(frozen.compression.provider.contextWindowTokens, 180_000);
    assert.equal(frozen.compression.provider.maxOutputTokens, 16_000);
    assert.deepEqual(frozen.toolPolicy.allowedTools, ['read']);
    assert.deepEqual(frozen.toolPolicy.sourceConfigs, {
      'mcp-exa': { enabled: true, disabledTools: ['exa_hidden'] }
    });
    assert.equal(frozen.planReviewPolicy.mode, 'before_mutation');
    assert.deepEqual(frozen.planReviewPolicy.requireForToolRiskLevels, ['write']);
    assert.equal(
      frozen.systemPrompt.text,
      '[全局规则]\nGLOBAL\n\n[Agent 规则]\nAGENT\n\n[工作流规则]\nWORKFLOW\n\n[对话规则]\nCONVERSATION'
    );
    assert.equal(frozen.runtimeContext.template, 'ENV:\n{{$workEnvironment.current}}');
    assert.match(frozen.runtimeContext.text, /work-env-local-/);
    assert.doesNotMatch(frozen.runtimeContext.text, /work-env-remote-test/);
    assert.equal(frozen.workEnvironmentPolicy.enabled, true);
    assert.deepEqual(frozen.workEnvironmentPolicy.allowedWorkEnvironmentIds, [workEnvironmentId]);
    assert.equal(frozen.workEnvironmentPolicy.defaultWorkEnvironmentId, workEnvironmentId);

    await authority.mutations.clearWorkEnvironmentPolicy('conversation', 'conversation:test');
    const withoutEnvironmentPolicy = JSON.parse((await authority.compile({
      conversationId: 'conversation:test',
      turnId: 'turn:no-environment-policy',
      executorAgentId: agent.id,
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.equal(withoutEnvironmentPolicy.workEnvironmentPolicy.enabled, false);
    assert.match(withoutEnvironmentPolicy.runtimeContext.text, /work-env-local-/);
    assert.doesNotMatch(withoutEnvironmentPolicy.runtimeContext.text, new RegExp(remoteEnvironment.id));

    const changedProvider = {
      ...provider,
      modelConfigs: provider.modelConfigs.map((modelConfig) => ({
        ...modelConfig,
        systemPromptPrefix: '后来修改的模型要求',
        updatedAt: 2
      }))
    };
    await saveLatestGlobalSettings(authority, 'llmProviderConfigs', { configs: [changedProvider] });
    const afterSettingsChange = JSON.parse((await authority.compile({
      conversationId: 'conversation:test',
      turnId: 'turn:after-settings-change',
      executorAgentId: agent.id,
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.equal(frozen.model.systemPromptPrefix, '模型专属前置要求');
    assert.equal(afterSettingsChange.model.systemPromptPrefix, '后来修改的模型要求');

    await authority.mutations.clearToolPolicy('workflow', workflow.id);
    const inherited = JSON.parse((await authority.compile({
      conversationId: 'conversation:test',
      turnId: 'turn:inherited',
      executorAgentId: agent.id,
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.deepEqual(inherited.toolPolicy.allowedTools, ['read']);
    assert.deepEqual(inherited.toolPolicy.sourceConfigs, {
      'mcp-exa': { enabled: true, disabledTools: ['exa_hidden'] }
    });

    await authority.mutations.deleteWorkflow(workflow.id);
    const afterDelete = await authority.configurationClientState();
    assert.equal(afterDelete.workflows.some((record) => record.id === workflow.id), false);
    assert.equal(afterDelete.conversationWorkflowSelections.length, 0);
    assert.equal(afterDelete.systemPromptScopeLinks.some((link) => link.scopeKind === 'workflow' && link.scopeId === workflow.id), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('VscodeConfigurationAuthority 让 Agent 缺省 preset 继承全局 YOLO，同时保留能力上界与逐工具配置', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'limcode-configuration-yolo-inherit-'));
  try {
    const paths = createVscodeStoragePaths(vscode.Uri.file(root));
    const authority = new VscodeConfigurationAuthority(() => paths);
    const provider = {
      ...createDefaultLlmProviderConfig({ name: 'YOLO Provider' }),
      id: 'provider:yolo',
      model: 'model:yolo',
      models: [{ id: 'model:yolo', name: 'YOLO 模型' }],
      modelConfigs: []
    };
    await saveLatestGlobalSettings(authority, 'llmProviderConfigs', { configs: [provider] });
    await saveLatestGlobalSettings(authority, 'llm', { activeProviderConfigId: provider.id });
    await authority.mutations.setToolPolicy({
      scopeKind: 'global',
      name: '全局 YOLO',
      preset: 'yolo',
      allowedTools: ['read', 'write', 'edit', 'delete', 'bash', 'skills'],
      toolConfigs: {
        write: { config: {}, autoApproveExecution: true, autoApplyChange: true },
        edit: { config: {}, autoApproveExecution: true, autoApplyChange: true },
        delete: { config: {}, autoApproveExecution: true, autoApplyChange: true },
        bash: {
          config: { limits: { lines: 40, chars: 2000 }, cwd: 'global' },
          display: { autoExpand: false }
        }
      },
      sourceConfigs: { exa: { enabled: true, disabledTools: ['exa_global_denied'] } }
    });
    await authority.mutations.setToolPolicy({
      scopeKind: 'agent',
      scopeId: 'main',
      name: 'Main 上界',
      allowedTools: ['read', 'write', 'edit', 'delete', 'bash'],
      toolConfigs: {
        bash: {
          config: { limits: { chars: 500 }, cwd: 'agent' },
          display: { autoExpand: true }
        }
      },
      sourceConfigs: { exa: { enabled: true, disabledTools: ['exa_agent_denied'] } }
    });

    const frozen = JSON.parse((await authority.compile({
      conversationId: 'conversation:yolo',
      turnId: 'turn:yolo',
      executorAgentId: 'main',
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.equal(frozen.toolPolicy.preset, 'yolo');
    assert.deepEqual(frozen.toolPolicy.allowedTools, ['bash', 'delete', 'edit', 'read', 'write']);
    assert.equal(frozen.toolPolicy.toolConfigs.write.autoApproveExecution, true);
    assert.equal(frozen.toolPolicy.toolConfigs.edit.autoApplyChange, true);
    assert.equal(frozen.toolPolicy.toolConfigs.delete.autoApplyChange, true);
    assert.deepEqual(frozen.toolPolicy.toolConfigs.bash, {
      config: { limits: { lines: 40, chars: 500 }, cwd: 'agent' },
      display: { autoExpand: true }
    });
    assert.deepEqual(frozen.toolPolicy.sourceConfigs.exa, {
      enabled: true,
      disabledTools: ['exa_agent_denied', 'exa_global_denied']
    });
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('多个Host共享WorkEnvironment存储时只在本地投影当前Workspace可用性与有效策略', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'limcode-work-environment-rebind-'));
  try {
    const paths = createVscodeStoragePaths(vscode.Uri.file(root));
    const firstAuthority = new VscodeConfigurationAuthority(() => paths);
    const secondAuthority = new VscodeConfigurationAuthority(() => paths);
    const firstPath = path.join(root, 'workspace-first');
    const secondPath = path.join(root, 'workspace-second');
    await fs.mkdir(firstPath, { recursive: true });
    await fs.mkdir(secondPath, { recursive: true });
    const firstUri = vscode.Uri.file(firstPath).toString();
    const secondUri = vscode.Uri.file(secondPath).toString();
    const firstId = workEnvironmentIdFromUri(firstUri);
    const secondId = workEnvironmentIdFromUri(secondUri);

    const provider = {
      ...createDefaultLlmProviderConfig({ name: 'Workspace Provider' }),
      id: 'provider:workspace-isolation',
      model: 'model:workspace-isolation',
      models: [{ id: 'model:workspace-isolation', name: 'Workspace Model' }],
      modelConfigs: []
    };
    await saveLatestGlobalSettings(firstAuthority, 'llmProviderConfigs', { configs: [provider] });
    await saveLatestGlobalSettings(firstAuthority, 'llm', { activeProviderConfigId: provider.id });

    const eagerFirstAuthority = new VscodeConfigurationAuthority(() => paths, undefined, [{
      uri: firstUri,
      name: 'First',
      rootPath: firstPath,
      index: 0
    }]);
    const eagerFirstSnapshot = await eagerFirstAuthority.configurationClientState();
    assert.equal(eagerFirstSnapshot.workEnvironments.find((record) => record.id === firstId)?.available, true);
    assert.equal(eagerFirstSnapshot.workEnvironments.find((record) => record.id === secondId), undefined);

    await firstAuthority.synchronizeWorkspaceFolders([{ uri: firstUri, name: 'First', rootPath: firstPath, index: 0 }]);
    await firstAuthority.mutations.setWorkEnvironmentPolicy({
      scopeKind: 'global',
      enabled: false,
      allowedWorkEnvironmentIds: [firstId],
      defaultWorkEnvironmentId: firstId
    });
    await secondAuthority.synchronizeWorkspaceFolders([{ uri: secondUri, name: 'Second', rootPath: secondPath, index: 0 }]);

    const storedEnvironments = await loadRecordStore(
      paths.workEnvironmentsRootUri,
      paths.workEnvironmentsIndexUri,
      'workEnvironment'
    );
    assert.equal(storedEnvironments.find((record) => record.id === firstId)?.available, true);
    assert.equal(storedEnvironments.find((record) => record.id === secondId)?.available, true);
    const storedPolicies = await loadRecordStore(
      paths.workEnvironmentPoliciesRootUri,
      paths.workEnvironmentPoliciesIndexUri,
      'policy'
    );
    const storedPolicy = storedPolicies.find((record) => record.id === 'work-environment-policy:global:global');
    assert.deepEqual(storedPolicy?.allowedWorkEnvironmentIds, [firstId]);
    assert.equal(storedPolicy?.defaultWorkEnvironmentId, firstId);

    const firstSnapshot = await firstAuthority.configurationClientState();
    const firstPolicy = firstSnapshot.workEnvironmentPolicies.find((record) => record.id === 'work-environment-policy:global:global');
    assert.deepEqual(firstPolicy?.allowedWorkEnvironmentIds, [firstId]);
    assert.equal(firstPolicy?.defaultWorkEnvironmentId, firstId);
    assert.equal(firstSnapshot.workEnvironments.find((record) => record.id === firstId)?.available, true);
    assert.equal(firstSnapshot.workEnvironments.find((record) => record.id === secondId)?.available, false);

    const secondSnapshot = await secondAuthority.configurationClientState();
    const secondPolicy = secondSnapshot.workEnvironmentPolicies.find((record) => record.id === 'work-environment-policy:global:global');
    assert.equal(secondPolicy?.enabled, false);
    assert.deepEqual(secondPolicy?.allowedWorkEnvironmentIds, [firstId, secondId]);
    assert.equal(secondPolicy?.defaultWorkEnvironmentId, secondId);
    assert.equal(secondSnapshot.workEnvironments.find((record) => record.id === firstId)?.available, false);
    assert.equal(secondSnapshot.workEnvironments.find((record) => record.id === secondId)?.available, true);

    const secondFrozen = JSON.parse((await secondAuthority.compile({
      conversationId: 'conversation:second-workspace',
      turnId: 'turn:second-workspace',
      executorAgentId: 'main',
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.deepEqual(secondFrozen.workEnvironmentPolicy.allowedWorkEnvironmentIds, [secondId]);
    assert.equal(secondFrozen.workEnvironmentPolicy.defaultWorkEnvironmentId, secondId);

    const manualId = 'work-environment:shared-remote';
    await firstAuthority.mutations.upsertWorkEnvironment({
      id: manualId,
      kind: 'remoteServer',
      source: 'manual',
      name: 'Shared Remote',
      host: 'example.test',
      available: true,
      createdAt: 1,
      updatedAt: 1
    });
    await firstAuthority.mutations.setWorkEnvironmentPolicy({
      scopeKind: 'global',
      enabled: true,
      allowedWorkEnvironmentIds: [manualId],
      defaultWorkEnvironmentId: manualId
    });
    const manualSnapshot = await secondAuthority.configurationClientState();
    const manualPolicy = manualSnapshot.workEnvironmentPolicies.find((record) => record.id === 'work-environment-policy:global:global');
    assert.deepEqual(manualPolicy?.allowedWorkEnvironmentIds, [manualId]);
    assert.equal(manualPolicy?.defaultWorkEnvironmentId, manualId);
    assert.equal(manualSnapshot.workEnvironments.find((record) => record.id === manualId)?.available, true);

    const manualFrozen = JSON.parse((await secondAuthority.compile({
      conversationId: 'conversation:shared-remote',
      turnId: 'turn:shared-remote',
      executorAgentId: 'main',
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.deepEqual(manualFrozen.workEnvironmentPolicy.allowedWorkEnvironmentIds, [manualId]);
    assert.equal(manualFrozen.workEnvironmentPolicy.defaultWorkEnvironmentId, manualId);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('不相关的旧压缩配置不会阻塞 Agent、Workflow 与 ConfigurationSnapshot 投影', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'limcode-agent-projection-isolation-'));
  try {
    const paths = createVscodeStoragePaths(vscode.Uri.file(root));
    const authority = new VscodeConfigurationAuthority(() => paths);
    const compressionRoot = path.join(paths.settingsRootUri.fsPath, 'llm-compression-configs');
    const recordsRoot = path.join(compressionRoot, 'records');
    const recordFile = 'records/legacy-compression.json';
    const savedAt = new Date().toISOString();
    await fs.mkdir(recordsRoot, { recursive: true });
    await fs.writeFile(path.join(compressionRoot, 'index.json'), `${JSON.stringify({
      schemaVersion: 1,
      savedAt,
      records: [{ id: 'legacy-compression', file: recordFile, updatedAt: savedAt }]
    }, null, 2)}\n`);
    await fs.writeFile(path.join(compressionRoot, ...recordFile.split('/')), `${JSON.stringify({
      schemaVersion: 1,
      savedAt,
      config: {
        id: 'legacy-compression',
        name: 'Legacy compression',
        kind: 'segmented_summary',
        trigger: {
          mode: 'token_threshold',
          thresholdUnit: 'percent',
          thresholdPercent: 90,
          preserveLatestMessages: 8,
          reserveLatestUserMessageTokens: 20_000
        },
        llmSummary: { targetTokens: 2_000 },
        createdAt: 1,
        updatedAt: 1
      }
    }, null, 2)}\n`);

    const agents = await authority.agents();
    assert.ok(agents.some((agent) => agent.id === 'main'));
    const resolvedAgent = await authority.resolveAgent({ agentType: 'main' });
    assert.equal(resolvedAgent.agentId, 'main');
    const workflow = await authority.workflow('builtin:plan');
    assert.equal(workflow.id, 'builtin:plan');
    const snapshot = await authority.configurationClientState();
    assert.ok(snapshot.agents.some((agent) => agent.id === 'main'));
    assert.ok(snapshot.workflows.some((candidate) => candidate.id === 'builtin:plan'));
    await assert.rejects(
      authority.loadGlobalSettings('llmCompressionConfigs'),
      /removed message-count\/user-reserve fields/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('压缩配置 hard-cut 旧保留字段并冻结压缩 Provider 自己的窗口与输出上限', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'limcode-compression-config-cutover-'));
  try {
    const paths = createVscodeStoragePaths(vscode.Uri.file(root));
    const authority = new VscodeConfigurationAuthority(() => paths);
    const primary = {
      ...createDefaultLlmProviderConfig({ name: '主模型渠道' }),
      id: 'provider:primary-compression-cutover',
      model: 'model:primary-372k',
      models: [{ id: 'model:primary-372k', name: 'Primary 372K' }],
      contextWindowTokens: 372_000,
      generationConfig: { maxOutputTokens: 20_000 },
      modelConfigs: []
    };
    const summary = {
      ...createDefaultLlmProviderConfig({ name: '摘要渠道' }),
      id: 'provider:summary-compression-cutover',
      model: 'model:summary-64k',
      models: [{ id: 'model:summary-64k', name: 'Summary 64K' }],
      contextWindowTokens: 64_000,
      generationConfig: { maxOutputTokens: 6_000 },
      modelConfigs: []
    };
    await saveLatestGlobalSettings(authority, 'llmProviderConfigs', { configs: [primary, summary] });
    await saveLatestGlobalSettings(authority, 'llm', { activeProviderConfigId: primary.id });

    const legacyCompressionConfig = {
      id: 'compression-config:cutover',
      name: 'Hard cut compression',
      kind: 'llm_summary',
      trigger: {
        mode: 'token_threshold',
        thresholdUnit: 'tokens',
        thresholdTokens: 300_000,
        thresholdPercent: 80,
        preserveLatestMessages: 99,
        reserveLatestUserMessageTokens: 123_000
      },
      llmSummary: {
        providerConfigId: summary.id,
        model: summary.model,
        generationConfig: { maxOutputTokens: 12_000 }
      },
      createdAt: 1,
      updatedAt: 1
    };
    await assert.rejects(
      saveLatestGlobalSettings(authority, 'llmCompressionConfigs', {
        configs: [legacyCompressionConfig]
      }),
      /removed message-count\/user-reserve fields/
    );
    const compressionConfig = {
      ...legacyCompressionConfig,
      trigger: {
        mode: 'token_threshold',
        thresholdUnit: 'tokens',
        thresholdTokens: 300_000,
        thresholdPercent: 80
      }
    };
    const saved = await saveLatestGlobalSettings(authority, 'llmCompressionConfigs', {
      configs: [compressionConfig]
    });
    const normalizedConfig = saved.settings.configs[0];
    assert.equal(normalizedConfig.llmSummary.targetTokens, 8_000);
    assert.deepEqual(Object.keys(normalizedConfig.trigger).sort(), [
      'mode', 'thresholdPercent', 'thresholdTokens', 'thresholdUnit'
    ]);
    await saveLatestGlobalSettings(authority, 'llmCompression', {
      defaultConfigId: normalizedConfig.id,
      providerBindings: [],
      modelBindings: []
    });

    const frozen = JSON.parse((await authority.compile({
      conversationId: 'conversation:compression-cutover',
      turnId: 'turn:compression-cutover',
      executorAgentId: 'main',
      intentKind: 'input'
    })).authoritySnapshot.content);
    assert.equal(frozen.modelProfile.contextWindowTokens, 372_000);
    assert.equal(frozen.model.maxOutputTokens, 20_000);
    assert.equal(frozen.compression.config.llmSummary.targetTokens, 8_000);
    assert.equal(frozen.compression.provider.providerConfigId, summary.id);
    assert.equal(frozen.compression.provider.modelId, summary.model);
    assert.equal(frozen.compression.provider.contextWindowTokens, 64_000);
    assert.equal(frozen.compression.provider.maxOutputTokens, 12_000);
    assert.equal(Object.hasOwn(frozen.compression, 'preserveLatestMessages'), false);
    assert.equal(Object.hasOwn(frozen.compression.config.trigger, 'preserveLatestMessages'), false);
    assert.equal(Object.hasOwn(frozen.compression.config.trigger, 'reserveLatestUserMessageTokens'), false);
    assert.equal(frozenCompressionPolicy(frozen).provider.contextWindowTokens, 64_000);

    const incomplete = structuredClone(frozen);
    delete incomplete.compression.provider.contextWindowTokens;
    assert.throws(
      () => frozenCompressionPolicy(incomplete),
      /compression\.provider\.contextWindowTokens/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

function createVscodeStub() {
  const FileType = { Unknown: 0, File: 1, Directory: 2, SymbolicLink: 64 };
  class Uri {
    constructor(fsPath) {
      this.scheme = 'file';
      this.fsPath = path.resolve(fsPath);
      this.path = this.fsPath.split(path.sep).join('/');
    }
    static file(filePath) { return new Uri(filePath); }
    static joinPath(base, ...segments) { return new Uri(path.join(base.fsPath, ...segments)); }
    toString() { return `file://${this.path}`; }
  }
  return {
    Uri,
    FileType,
    workspace: {
      fs: {
        async createDirectory(uri) { await fs.mkdir(uri.fsPath, { recursive: true }); },
        async readFile(uri) { return fs.readFile(uri.fsPath); },
        async writeFile(uri, bytes) {
          await fs.mkdir(path.dirname(uri.fsPath), { recursive: true });
          await fs.writeFile(uri.fsPath, bytes);
        },
        async readDirectory(uri) {
          const entries = await fs.readdir(uri.fsPath, { withFileTypes: true });
          return entries.map((entry) => [
            entry.name,
            entry.isDirectory() ? FileType.Directory : entry.isFile() ? FileType.File : FileType.Unknown
          ]);
        },
        async delete(uri) { await fs.rm(uri.fsPath, { recursive: true, force: true }); },
        async stat(uri) {
          const stat = await fs.stat(uri.fsPath);
          return {
            type: stat.isDirectory() ? FileType.Directory : FileType.File,
            ctime: stat.ctimeMs,
            mtime: stat.mtimeMs,
            size: stat.size
          };
        }
      }
    }
  };
}
