import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import { mkdir } from 'node:fs/promises';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';
import { parentPort, threadId, workerData } from 'node:worker_threads';
import Database from 'better-sqlite3';
import type { RuntimeAllocatedSequence, RuntimeChange, RuntimeCommitResult, SnapshotBarrier } from './contracts';
import type { ContentObjectMetadata } from './contentAddressedStore';
import {
  approvedSubmitPlanTaskOperation,
  buildCurrentTurnTaskProjection,
  taskListOperationFromSettledArtifact,
  type CurrentTurnTaskOperationFact
} from './currentTurnTaskProjection';
import { preparedContentObjectSteps } from './contentObjectTransaction';
import {
  assertCurrentSchema,
  assertDatabaseBinding,
  configureReaderConnection,
  configureWriterConnection,
  initializeCurrentSchema,
  inspectDatabaseFoundation
} from './databaseSchema';
import {
  MODEL_STREAM_ACTIVE_CHECKPOINT_LIMIT,
  MODEL_STREAM_OUTPUT_DELTA_CHECKPOINT_LIMIT,
  MODEL_STREAM_TERMINAL_TAIL,
  type ClientKeysetPageInput,
  type ClientKeysetPageResult,
  type ClientVisibleMessageHistoryPageInput,
  type ClientVisibleMessageHistoryPageResult,
  type ChildConversationOriginCandidate,
  type ChildProcessCleanupMaterializationCandidate,
  type ClientProjectionSnapshot,
  type ConversationHistoryProjectionInput,
  type ConversationHistoryProjectionResult,
  type ContextContentMaterializationSnapshot,
  type ContextMaterializationRecord,
  type ContextMaterializationSnapshot,
  type DatabaseWorkerData,
  type DatabaseWorkerDiagnostics,
  type DatabaseWorkerRequest,
  type DatabaseWorkerResponse,
  type ExecutionLeaseFencePayload,
  type EffectReceiptReconciliationCandidate,
  type ModelStreamActivityInput,
  type ModelStreamActivityResult,
  type ModelStreamEventCommitInput,
  type ModelStreamEventCommitResult,
  type ModelRequestCancelInput,
  type ModelRequestCancelResult,
  type ProcessOutputRegistrationMismatch,
  type SerializedWorkerError,
  type ToolFactsSnapshot
} from './databaseWorkerProtocol';
import {
  CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE,
  CLIENT_MESSAGE_WINDOW_LIMIT,
  CLIENT_PAGE_MAX_BYTES,
  CLIENT_PAGE_MAX_ROWS,
  CLIENT_TOOL_EVENT_SUMMARY_LIMIT_PER_CALL,
  CLIENT_WINDOW_RECORD_SUMMARY_MAX_BYTES
} from './clientFeedBounds';
import {
  boundClientRecordSummary,
  settleClientWireResponseBytes
} from './clientWireData';
import {
  DOMAIN_REPOSITORIES,
  HISTORICAL_COPY_DOMAINS,
  assertRuntimeDomainUpdatePatch,
  type DomainRepository,
  type DomainRow,
  type EncodedRow,
  type RepositoryCheckpointPruneMutation,
  type RepositoryInsertMutation,
  type RepositoryListRead,
  type RepositoryMutation,
  type RepositoryRead,
  type RepositorySavepointOnError,
  type RepositoryTransactionStep
} from './repositories';

const CONTEXT_CAS_CACHE_MAX_ENTRIES = 4_096;
const CONTEXT_CAS_CACHE_MAX_BYTES = 32 * 1024 * 1024;
const TURN_INTENT_CONTENT_TYPE = 'application/vnd.limcode.turn-intent+json';
const CHILD_ACTIVITY_ARGUMENTS_MAX_BYTES = 64 * 1024;
const CHILD_ACTIVITY_SUMMARY_MAX_CHARACTERS = 180;

interface VerifiedContextCasCacheEntry {
  id: string;
  sha256: string;
  byteLength: bigint;
  storageKey: string;
  bytes: Buffer;
}

/**
 * Context materialization repeatedly reads immutable CAS objects while compiling adjacent model
 * rounds. Keep only verified bytes in a strict LRU budget so 1000-node histories do not issue 1000
 * filesystem reads on every round. Entries never cross a worker/RootBinding lifetime, and callers
 * receive copies in the packed transfer buffer rather than mutable cache Buffers.
 */
class VerifiedContextCasCache {
  private readonly entries = new Map<string, VerifiedContextCasCacheEntry>();
  private totalBytes = 0;
  private hits = 0;
  private misses = 0;
  private evictions = 0;

  public read(metadata: DomainRow, resolvedCasRootPath: string): Buffer {
    const identity = contextCasIdentity(metadata);
    const cached = this.entries.get(identity.id);
    if (cached) {
      assertSameContextCasIdentity(cached, identity);
      this.entries.delete(identity.id);
      this.entries.set(identity.id, cached);
      this.hits += 1;
      return cached.bytes;
    }
    this.misses += 1;
    const bytes = readVerifiedCasBytes(metadata, resolvedCasRootPath);
    if (bytes.length <= CONTEXT_CAS_CACHE_MAX_BYTES) {
      while (
        this.entries.size >= CONTEXT_CAS_CACHE_MAX_ENTRIES
        || this.totalBytes + bytes.length > CONTEXT_CAS_CACHE_MAX_BYTES
      ) {
        const oldestId = this.entries.keys().next().value as string | undefined;
        if (!oldestId) break;
        const oldest = this.entries.get(oldestId);
        this.entries.delete(oldestId);
        if (oldest) this.totalBytes -= oldest.bytes.length;
        this.evictions += 1;
      }
      const entry: VerifiedContextCasCacheEntry = { ...identity, bytes };
      this.entries.set(identity.id, entry);
      this.totalBytes += bytes.length;
    }
    return bytes;
  }

  public inspect(): DatabaseWorkerDiagnostics['contextCasCache'] {
    return {
      entries: this.entries.size,
      bytes: this.totalBytes,
      maxEntries: CONTEXT_CAS_CACHE_MAX_ENTRIES,
      maxBytes: CONTEXT_CAS_CACHE_MAX_BYTES,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions
    };
  }
}

const port = requireParentPort();
const data = workerData as DatabaseWorkerData;

void start().catch((error) => {
  post({ type: 'fatal', error: serializeError(error) });
  process.exitCode = 1;
});

async function start(): Promise<void> {
  if (data.mode === 'initialize') {
    await mkdir(data.binding.paths.casRootPath, { recursive: false });
    const database = new Database(data.binding.paths.databasePath, { fileMustExist: false });
    try {
      configureWriterConnection(database);
      initializeCurrentSchema(database, data.binding);
    } finally {
      database.close();
    }
    post({ type: 'ready', workerThreadId: threadId, mode: data.mode });
    port.close();
    return;
  }

  const writer = new Database(data.binding.paths.databasePath, { fileMustExist: true });
  configureWriterConnection(writer);
  assertCurrentSchema(writer, data.binding);
  configureTransactionChangeCapture(writer);
  const reader = new Database(data.binding.paths.databasePath, { readonly: true, fileMustExist: true });
  configureReaderConnection(reader);
  let commitSeq = 0n;
  let closed = false;
  const contextCasCache = new VerifiedContextCasCache();

  post({ type: 'ready', workerThreadId: threadId, mode: data.mode });
  port.on('message', (request: DatabaseWorkerRequest) => {
    if (closed) return;
    const receivedAtMs = Number.isFinite(request.metricEnqueuedAtMs)
      ? performance.now()
      : undefined;
    const respond = (
      response: Extract<DatabaseWorkerResponse, { type: 'response' }>,
      transferList: readonly ArrayBuffer[] = []
    ) => postMeasuredResponse(response, request.metricEnqueuedAtMs, receivedAtMs, transferList);
    try {
      if (request.kind === 'transaction') {
        assertDatabaseBinding(writer, data.binding);
        const result = executeTransaction(writer, request.steps, commitSeq + 1n);
        commitSeq += 1n;
        post({ type: 'commit', result });
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'snapshot') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeSnapshot(reader, request.reads, commitSeq);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'snapshotAll') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeSnapshotAll(reader, request.read, commitSeq);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'toolFactsSnapshot') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeToolFactsSnapshot(reader, request.toolCallId, commitSeq);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'processOutputRegistrationMismatches') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeProcessOutputRegistrationMismatches(reader);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'effectReceiptReconciliationCandidates') {
        assertDatabaseBinding(reader, data.binding);
        respond({ type: 'response', id: request.id, ok: true, result: executeEffectReceiptReconciliationCandidates(reader) });
        return;
      }
      if (request.kind === 'childConversationOriginCandidates') {
        assertDatabaseBinding(reader, data.binding);
        respond({ type: 'response', id: request.id, ok: true, result: executeChildConversationOriginCandidates(reader) });
        return;
      }
      if (request.kind === 'childProcessCleanupMaterializationCandidates') {
        assertDatabaseBinding(reader, data.binding);
        respond({ type: 'response', id: request.id, ok: true, result: executeChildProcessCleanupMaterializationCandidates(reader) });
        return;
      }
      if (request.kind === 'contextMaterialization') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeContextMaterialization(reader, request.rootId, commitSeq);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'contextContentMaterialization') {
        assertDatabaseBinding(reader, data.binding);
        const structure = executeContextMaterialization(reader, request.rootId, commitSeq);
        const attached = attachContextContent(structure, data.binding.paths.casRootPath, contextCasCache);
        respond({ type: 'response', id: request.id, ok: true, result: attached.result }, attached.transferList);
        return;
      }
      if (request.kind === 'modelStreamEvent') {
        assertDatabaseBinding(writer, data.binding);
        const result = executeModelStreamEvent(writer, request.input, commitSeq + 1n);
        if (result.commit) {
          commitSeq += 1n;
          post({ type: 'commit', result: result.commit });
        }
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'modelStreamActivity') {
        assertDatabaseBinding(writer, data.binding);
        const result = executeModelStreamActivity(writer, request.input, commitSeq + 1n);
        if (result.commit) {
          commitSeq += 1n;
          post({ type: 'commit', result: result.commit });
        }
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'cancelCurrentModelRequest') {
        assertDatabaseBinding(writer, data.binding);
        const result = executeCancelCurrentModelRequest(writer, request.input, commitSeq + 1n);
        if (result.commit) {
          commitSeq += 1n;
          post({ type: 'commit', result: result.commit });
        }
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'clientProjectionSnapshot') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeClientProjectionSnapshot(
          reader,
          request.activeConversationId,
          commitSeq
        );
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'clientKeysetPage') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeClientKeysetPage(reader, request.input);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'clientVisibleMessageHistoryPage') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeClientVisibleMessageHistoryPage(reader, request.input);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'conversationHistoryProjection') {
        assertDatabaseBinding(reader, data.binding);
        const result = executeConversationHistoryProjection(reader, request.input, commitSeq);
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'externalDataVersion') {
        assertDatabaseBinding(writer, data.binding);
        // SQLite changes this connection-local value only when another connection commits.
        // Reading it from the writer (rather than the separate reader) therefore excludes every
        // commit made by this RuntimeDatabase worker while still detecting other Extension Hosts.
        const result = BigInt(writer.pragma('data_version', { simple: true }) as number | bigint).toString();
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      if (request.kind === 'inspect') {
        assertDatabaseBinding(writer, data.binding);
        const result: DatabaseWorkerDiagnostics = {
          ...inspectDatabaseFoundation(writer),
          workerThreadId: threadId,
          hostBootId: data.hostBootId,
          writerConnectionCount: 1,
          readerConnectionCount: 1,
          readerJournalMode: String(reader.pragma('journal_mode', { simple: true })),
          readerForeignKeys: BigInt(reader.pragma('foreign_keys', { simple: true }) as number | bigint),
          readerBusyTimeoutMs: BigInt(reader.pragma('busy_timeout', { simple: true }) as number | bigint),
          currentCommitSeq: commitSeq.toString(),
          contextCasCache: contextCasCache.inspect()
        };
        respond({ type: 'response', id: request.id, ok: true, result });
        return;
      }
      assertDatabaseBinding(writer, data.binding);
      closed = true;
      reader.close();
      writer.close();
      respond({ type: 'response', id: request.id, ok: true, result: null });
      port.close();
    } catch (error) {
      respond({ type: 'response', id: request.id, ok: false, error: serializeError(error) });
    }
  });
}

function configureTransactionChangeCapture(database: Database.Database): void {
  database.exec(`
    CREATE TEMP TABLE runtime_transaction_change (
      sequence INTEGER PRIMARY KEY,
      domain TEXT NOT NULL,
      id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN ('upsert', 'remove'))
    )
  `);
  for (const schema of DOMAIN_REPOSITORIES.all().map((repository) => repository.schema)) {
    if (schema.client === 'none') continue;
    const domain = sqlText(schema.key);
    for (const operation of ['insert', 'update', 'delete'] as const) {
      const row = operation === 'delete' ? 'OLD' : 'NEW';
      const kind = operation === 'delete' ? 'remove' : 'upsert';
      database.exec(`
        CREATE TEMP TRIGGER ${quote(`capture_${schema.table}_${operation}`)}
        AFTER ${operation.toUpperCase()} ON ${quote(schema.table)}
        BEGIN
          INSERT INTO runtime_transaction_change (domain, id, kind)
          VALUES (${domain}, ${row}.id, '${kind}');
        END
      `);
    }
  }

  // Message 是独立事实，conversation membership/current revision 也是独立 Link。Link 改变时
  // 重新投影 Message 窗口记录；尚未组成完整窗口的 Message 只产生幂等 remove，不阻塞写事务。
  for (const relation of [
    { table: 'message_part_of_conversation', messageColumn: 'message_id' },
    { table: 'message_current_revision_link', messageColumn: 'message_id' }
  ]) {
    for (const operation of ['insert', 'update', 'delete'] as const) {
      const row = operation === 'delete' ? 'OLD' : 'NEW';
      const kind = operation === 'delete' ? 'remove' : 'upsert';
      database.exec(`
        CREATE TEMP TRIGGER ${quote(`capture_message_from_${relation.table}_${operation}`)}
        AFTER ${operation.toUpperCase()} ON ${quote(relation.table)}
        BEGIN
          INSERT INTO runtime_transaction_change (domain, id, kind)
          VALUES ('Message', ${row}.${quote(relation.messageColumn)}, '${kind}');
        END
      `);
    }
  }

  // RuntimeDeliveryInputLink 本身不是客户端领域；它改变的是 RuntimeDelivery 的派生
  // parent_handling_state，因此在同一 commit 中重新投影对应 Delivery。
  for (const operation of ['insert', 'update'] as const) {
    database.exec(`
      CREATE TEMP TRIGGER ${quote(`capture_runtime_delivery_from_input_link_${operation}`)}
      AFTER ${operation.toUpperCase()} ON runtime_delivery_input_link
      BEGIN
        INSERT INTO runtime_transaction_change (domain, id, kind)
        VALUES ('RuntimeDelivery', NEW.delivery_id, 'upsert');
      END
    `);
  }

  // ConversationContextHeadLink keeps its persisted client=none mapping. Its bounded current-root
  // summary is an independent derived view, captured transactionally like Message window and
  // RuntimeDelivery parent state without adding a second mutable authority.
  for (const operation of ['insert', 'update', 'delete'] as const) {
    const row = operation === 'delete' ? 'OLD' : 'NEW';
    const kind = operation === 'delete' ? 'remove' : 'upsert';
    database.exec(`
      CREATE TEMP TRIGGER ${quote(`capture_context_status_from_head_${operation}`)}
      AFTER ${operation.toUpperCase()} ON conversation_context_head_link
      BEGIN
        INSERT INTO runtime_transaction_change (domain, id, kind)
        VALUES ('ConversationContextStatus', ${row}.id, '${kind}');
      END
      `);
  }

  // CommandReceipt remains an internal epoch domain. The Webview receives only this narrow,
  // conversation-scoped derived fact so a lost one-shot TurnInputResult can still converge from
  // the durable Feed. Internal/callback/recovery source keys are never projected.
  database.exec(`
    CREATE TEMP TRIGGER capture_conversation_command_receipt_insert
    AFTER INSERT ON command_receipt
    WHEN NEW.source_kind = 'command' AND NEW.conversation_id IS NOT NULL
    BEGIN
      INSERT INTO runtime_transaction_change (domain, id, kind)
      VALUES ('ConversationCommandReceipt', NEW.id, 'upsert');
    END
  `);

  // Child ToolCall/ModelRequest rows stay isolated from the parent Conversation feed. Instead,
  // changes to the active child generation re-project one bounded, non-authoritative activity row
  // keyed by ChildExecution. This gives the parent UI live progress without merging transcripts.
  for (const table of ['tool_call', 'model_request'] as const) {
    for (const operation of ['insert', 'update'] as const) {
      database.exec(`
        CREATE TEMP TRIGGER ${quote(`capture_child_activity_from_${table}_${operation}`)}
        AFTER ${operation.toUpperCase()} ON ${quote(table)}
        BEGIN
          INSERT INTO runtime_transaction_change (domain, id, kind)
          SELECT 'ChildExecutionActivity', membership.child_execution_id, 'upsert'
            FROM child_execution_turn_link AS membership
           WHERE membership.turn_id = NEW.turn_id;
        END
      `);
    }
  }
  for (const operation of ['insert', 'update'] as const) {
    database.exec(`
      CREATE TEMP TRIGGER ${quote(`capture_child_activity_from_execution_${operation}`)}
      AFTER ${operation.toUpperCase()} ON child_execution
      BEGIN
        INSERT INTO runtime_transaction_change (domain, id, kind)
        VALUES ('ChildExecutionActivity', NEW.id, 'upsert');
      END
    `);
  }
  database.exec(`
    CREATE TEMP TRIGGER capture_child_activity_from_execution_delete
    AFTER DELETE ON child_execution
    BEGIN
      INSERT INTO runtime_transaction_change (domain, id, kind)
      VALUES ('ChildExecutionActivity', OLD.id, 'remove');
    END
  `);
  for (const operation of ['insert', 'update', 'delete'] as const) {
    const row = operation === 'delete' ? 'OLD' : 'NEW';
    database.exec(`
      CREATE TEMP TRIGGER ${quote(`capture_child_activity_from_active_turn_${operation}`)}
      AFTER ${operation.toUpperCase()} ON child_execution_active_turn_link
      BEGIN
        INSERT INTO runtime_transaction_change (domain, id, kind)
        VALUES ('ChildExecutionActivity', ${row}.child_execution_id, 'upsert');
      END
    `);
  }
}

function readTransactionChanges(database: Database.Database): RuntimeChange[] {
  const rows = database.prepare(`
    SELECT current.sequence, current.domain, current.id, current.kind
      FROM runtime_transaction_change AS current
      JOIN (
        SELECT domain, id, MAX(sequence) AS sequence
          FROM runtime_transaction_change
         GROUP BY domain, id
      ) AS latest
        ON latest.sequence = current.sequence
     ORDER BY current.sequence
  `).all() as Array<{ sequence: bigint; domain: string; id: string; kind: 'upsert' | 'remove' }>;
  const topology = new Map(DOMAIN_REPOSITORIES.all().map((repository, index) => [repository.schema.key, index]));
  topology.set('ConversationContextStatus', topology.size);
  topology.set('ConversationCommandReceipt', topology.size);
  topology.set('ChildExecutionActivity', topology.size);
  return rows
    .map((row) => {
      if (row.domain === 'ChildExecutionActivity') {
        if (row.kind === 'remove') return { ...row };
        const record = projectChildExecutionActivityRecord(database, row.id);
        return record
          ? { ...row, kind: 'upsert' as const, record }
          : { ...row, kind: 'remove' as const };
      }
      if (row.kind === 'remove') return { ...row };
      if (row.domain === 'ConversationContextStatus') {
        return { ...row, record: projectConversationContextStatusRecord(database, row.id) };
      }
      if (row.domain === 'ConversationCommandReceipt') {
        return { ...row, record: projectConversationCommandReceiptRecord(database, row.id) };
      }
      const repository = DOMAIN_REPOSITORIES.domain(row.domain);
      const raw = database.prepare(`SELECT * FROM ${quote(repository.schema.table)} WHERE id = ?`).get(row.id);
      if (!raw) throw new Error(`Committed upsert projection ${row.domain}/${row.id} is missing.`);
      let record = repository.codec.decode(raw as Record<string, unknown>);
      if (row.domain === 'TurnIntent') {
        const projected = projectQueuedTurnIntentRecord(database, row.id);
        if (!projected) return { ...row, kind: 'remove' as const };
        record = projected;
      }
      if (row.domain === 'Turn') {
        record = projectTurnClientRecord(database, row.id);
      }
      if (row.domain === 'Message') {
        const projected = projectMessageWindowRecord(database, row.id);
        if (!projected) return { ...row, kind: 'remove' as const };
        record = projected;
      }
      if (row.domain === 'CompressionBlock') {
        record = projectCompressionBlockRecord(database, row.id);
      }
      if (row.domain === 'AnswerBridge') {
        record = projectAnswerBridgeRecord(database, row.id);
      }
      if (row.domain === 'Process') {
        record = projectProcessRecord(database, row.id);
      }
      if (row.domain === 'RuntimeDelivery') {
        const links = database.prepare(`
          SELECT handled_at
            FROM runtime_delivery_input_link
           WHERE delivery_id = ?
           LIMIT 2
        `).all(row.id) as Array<{ handled_at: string | null }>;
        if (links.length > 1) throw new Error(`RuntimeDelivery ${row.id} has multiple input links.`);
        record.parent_handling_state = deriveCommittedParentHandling(record, links[0] ?? null);
      }
      return { ...row, record };
    })
    .sort((left, right) => {
      if (left.kind !== right.kind) return left.kind === 'remove' ? -1 : 1;
      const leftOrder = topology.get(left.domain);
      const rightOrder = topology.get(right.domain);
      if (leftOrder === undefined || rightOrder === undefined) throw new Error('Runtime change references an unknown domain.');
      const dependencyOrder = left.kind === 'remove' ? rightOrder - leftOrder : leftOrder - rightOrder;
      if (dependencyOrder !== 0) return dependencyOrder;
      return left.sequence < right.sequence ? -1 : left.sequence > right.sequence ? 1 : 0;
    })
    .map(({ sequence: _sequence, ...change }) => change);
}

function projectQueuedTurnIntentRecord(database: Database.Database, intentId: string): DomainRow | null {
  const rows = queryPlainRows(database, `
    SELECT intent.*,
           (
             SELECT CAST(revision.revision_seq AS TEXT)
               FROM turn_intent_revision AS revision
              WHERE revision.intent_id = intent.id
              ORDER BY revision.revision_seq DESC
              LIMIT 1
           ) AS current_revision_seq
      FROM turn_intent AS intent
     WHERE intent.id = @intentId
       AND intent.state = 'queued'
       AND intent.turn_id IS NULL
       AND NOT EXISTS (
         SELECT 1
           FROM child_execution_intent_link AS child_link
          WHERE child_link.turn_intent_id = intent.id
       )
     LIMIT 1
  `, { intentId });
  return rows[0] ?? null;
}

function projectConversationContextStatusRecord(database: Database.Database, headId: string): DomainRow {
  const record = queryPlainRows(database, `
    SELECT head.id,
           head.conversation_id,
           head.root_id,
           root.root_seq,
           root.segment_count,
           root.estimated_tokens,
           root.created_at AS root_created_at,
           head.updated_at
      FROM conversation_context_head_link AS head
      JOIN context_sequence_root AS root ON root.id = head.root_id
     WHERE head.id = @headId
     LIMIT 1
  `, { headId })[0];
  if (!record) throw new Error(`ConversationContextStatus ${headId} cannot resolve its current root.`);
  return record;
}

function projectConversationCommandReceiptRecord(database: Database.Database, receiptId: string): DomainRow {
  const record = queryPlainRows(database, `
    SELECT id,
           conversation_id,
           source_key AS command_id,
           created_at
      FROM command_receipt
     WHERE id = @receiptId
       AND source_kind = 'command'
       AND conversation_id IS NOT NULL
     LIMIT 1
  `, { receiptId })[0];
  if (!record) {
    throw new Error(`ConversationCommandReceipt ${receiptId} cannot resolve its durable command receipt.`);
  }
  return record;
}

function projectCompressionBlockRecord(database: Database.Database, blockId: string): DomainRow {
  const record = queryPlainRows(database, `
    SELECT block.*,
           COUNT(source.id) AS source_count,
           (
             SELECT revision.message_id
               FROM compression_block_source AS anchor_source
               JOIN context_segment_source AS segment_source
                 ON segment_source.segment_id = anchor_source.segment_id
                AND segment_source.source_kind = 'message_revision'
               JOIN message_revision AS revision ON revision.id = segment_source.source_id
               JOIN message_part_of_conversation AS anchor_membership
                 ON anchor_membership.message_id = revision.message_id
                AND anchor_membership.conversation_id = block.conversation_id
              WHERE anchor_source.compression_block_id = block.id
              ORDER BY anchor_source.position DESC, anchor_source.id DESC
              LIMIT 1
           ) AS anchor_message_id
      FROM compression_block AS block
      LEFT JOIN compression_block_source AS source ON source.compression_block_id = block.id
     WHERE block.id = @blockId
     GROUP BY block.id
     LIMIT 1
  `, { blockId })[0];
  if (!record) throw new Error(`CompressionBlock ${blockId} cannot resolve its bounded summary.`);
  return record;
}

function projectMessageWindowRecord(database: Database.Database, messageId: string): DomainRow | null {
  const rows = queryPlainRows(database, `
    SELECT message.id,
           membership.conversation_id,
           membership.message_seq,
           message.created_at,
           message.updated_at,
           message.deleted_at,
           revision.id AS revision_id,
           revision.revision_seq,
           revision.role,
           revision.content_object_id,
           content.content_type,
           content.byte_length
      FROM message
      JOIN message_part_of_conversation AS membership ON membership.message_id = message.id
      JOIN message_current_revision_link AS current_revision ON current_revision.message_id = message.id
      JOIN message_revision AS revision ON revision.id = current_revision.revision_id
      JOIN content_object AS content ON content.id = revision.content_object_id
     WHERE message.id = @messageId
     LIMIT 1
  `, { messageId });
  if (rows.length > 1) throw new Error(`Message ${messageId} has multiple client window projections.`);
  return rows[0] ?? null;
}

/**
 * A retry/edit-and-run Turn deliberately reuses the source Message instead of creating a second
 * input MessageTurnLink. Surface that already-durable TurnIntent relation in the bounded client
 * projection so process-local provider output has an exact timeline anchor before its final model
 * Message exists. Historical/terminal Turns do not need the enrichment.
 */
function projectTurnClientRecord(database: Database.Database, turnId: string): DomainRow {
  const raw = database.prepare('SELECT * FROM turn WHERE id = ?').get(turnId);
  if (!raw) throw new Error(`Turn ${turnId} does not exist.`);
  const record = DOMAIN_REPOSITORIES.codec('Turn').decode(raw as Record<string, unknown>);
  if (record.status !== 'active') return record;

  const sources = database.prepare(`
    SELECT content.*
      FROM turn_intent AS intent
      JOIN turn_intent_revision AS revision ON revision.intent_id = intent.id
      JOIN content_object AS content ON content.id = revision.content_object_id
     WHERE intent.turn_id = ?
       AND revision.revision_seq = (
         SELECT MAX(latest.revision_seq)
           FROM turn_intent_revision AS latest
          WHERE latest.intent_id = intent.id
       )
     LIMIT 2
  `).all(turnId) as Array<Record<string, unknown>>;
  if (sources.length > 1) throw new Error(`Turn ${turnId} has multiple admitted TurnIntents.`);
  if (sources.length === 0) return record;
  const metadata = DOMAIN_REPOSITORIES.codec('ContentObject').decode(sources[0]);
  if (metadata.content_type !== TURN_INTENT_CONTENT_TYPE) return record;

  const parsed = JSON.parse(
    readVerifiedCasBytes(metadata, path.resolve(data.binding.paths.casRootPath)).toString('utf8')
  ) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Turn ${turnId} has an invalid TurnIntent payload.`);
  }
  const intent = parsed as Record<string, unknown>;
  if (intent.kind !== 'retry') return record;
  const sourceMessageId = typeof intent.sourceMessageId === 'string'
    ? intent.sourceMessageId.trim()
    : '';
  // A retry of a synthetic partial ModelRequest can legitimately have only sourceModelRequestId.
  // It has no visible Message to anchor, so leave that Turn unchanged instead of inventing one.
  if (!sourceMessageId) return record;
  const sourceRevisionId = typeof intent.sourceMessageRevisionId === 'string'
    ? intent.sourceMessageRevisionId.trim()
    : typeof intent.editedMessageRevisionId === 'string'
      ? intent.editedMessageRevisionId.trim()
      : '';
  const membership = database.prepare(`
    SELECT membership.conversation_id
      FROM message_part_of_conversation AS membership
     WHERE membership.message_id = ?
     LIMIT 2
  `).all(sourceMessageId) as Array<{ conversation_id: string }>;
  if (membership.length !== 1 || membership[0]?.conversation_id !== record.conversation_id) {
    throw new Error(`Retry Turn ${turnId} source Message does not belong to its Conversation.`);
  }
  return {
    ...record,
    source_message_id: sourceMessageId,
    ...(sourceRevisionId ? { source_message_revision_id: sourceRevisionId } : {})
  };
}

function projectAnswerBridgeRecord(database: Database.Database, answerBridgeId: string): DomainRow {
  const rows = queryPlainRows(database, `
    SELECT bridge.*,
           submission.submission_seq AS current_submission_seq,
           submission.turn_id AS current_turn_id,
           submission.interrupted AS current_submission_interrupted,
           submission.created_at AS current_submission_created_at,
           payload.id AS current_payload_id,
           payload.title AS current_title,
           payload.byte_length AS current_byte_length
      FROM answer_bridge AS bridge
      LEFT JOIN answer_submission AS submission ON submission.id = bridge.current_submission_id
      LEFT JOIN answer_payload AS payload ON payload.submission_id = submission.id
     WHERE bridge.id = @answerBridgeId
     LIMIT 1
  `, { answerBridgeId });
  if (rows.length !== 1) throw new Error(`AnswerBridge ${answerBridgeId} does not exist.`);
  return rows[0];
}

/**
 * Small parent-facing view of what one child is doing right now. Child Message/ToolCall rows remain
 * outside the parent feed; this record deliberately contains only a bounded activity sentence.
 */
function projectChildExecutionActivityRecord(
  database: Database.Database,
  childExecutionId: string
): DomainRow | null {
  const owner = queryPlainRows(database, `
    SELECT child.id,
           child.status AS child_status,
           child.updated_at AS child_updated_at,
           active.turn_id,
           turn.status AS turn_status,
           turn.updated_at AS turn_updated_at
      FROM child_execution AS child
      LEFT JOIN child_execution_active_turn_link AS active
        ON active.child_execution_id = child.id
      LEFT JOIN turn ON turn.id = active.turn_id
     WHERE child.id = @childExecutionId
     LIMIT 1
  `, { childExecutionId })[0];
  if (!owner) return null;

  const turnId = typeof owner.turn_id === 'string' && owner.turn_status === 'active'
    ? owner.turn_id
    : undefined;
  const childStatus = String(owner.child_status);
  const base: DomainRow = {
    id: childExecutionId,
    child_execution_id: childExecutionId,
    ...(turnId ? { turn_id: turnId } : {}),
    updated_at: String(owner.turn_updated_at ?? owner.child_updated_at)
  };
  if (childStatus === 'interrupting') {
    return { ...base, kind: 'stopping', summary: '正在终止当前子树' };
  }
  if (!turnId) {
    return {
      ...base,
      kind: childStatus === 'starting' ? 'starting' : 'idle',
      summary: childStatus === 'starting' ? '正在启动' : '当前没有活动回合'
    };
  }

  const tool = queryPlainRows(database, `
    SELECT *
      FROM tool_call
     WHERE turn_id = @turnId
       AND status <> 'terminal'
     ORDER BY call_seq DESC, updated_at DESC, id DESC
     LIMIT 1
  `, { turnId })[0];
  if (tool) {
    return {
      ...base,
      kind: 'tool',
      tool_call_id: String(tool.id),
      tool_name: String(tool.tool_name),
      tool_status: String(tool.status),
      summary: childToolActivitySummary(database, tool),
      updated_at: String(tool.updated_at)
    };
  }

  const request = queryPlainRows(database, `
    SELECT *
      FROM model_request
     WHERE turn_id = @turnId
       AND status <> 'terminal'
     ORDER BY request_seq DESC, updated_at DESC, id DESC
     LIMIT 1
  `, { turnId })[0];
  if (request) {
    const status = String(request.status);
    return {
      ...base,
      kind: 'model',
      model_request_id: String(request.id),
      model_request_status: status,
      summary: status === 'pending' ? '正在准备模型请求' : '正在思考并生成下一步',
      updated_at: String(request.updated_at)
    };
  }
  return { ...base, kind: 'preparing', summary: '正在整理结果并准备下一步' };
}

function childToolActivitySummary(
  database: Database.Database,
  tool: Record<string, unknown>
): string {
  const toolName = String(tool.tool_name);
  const action = childToolAction(toolName);
  const prefix = tool.status === 'pending' ? `等待${action}` : `正在${action}`;
  const detail = childToolArgumentPreview(database, tool);
  return compactChildActivitySummary(detail ? `${prefix} · ${detail}` : prefix);
}

function childToolAction(toolName: string): string {
  switch (toolName) {
    case 'bash':
    case 'shell': return '运行命令';
    case 'read':
    case 'read_file': return '读取文件';
    case 'edit': return '编辑文件';
    case 'write': return '写入文件';
    case 'delete': return '删除文件';
    case 'run_agent': return '调度子 Agent';
    case 'submit_agent_answer': return '提交 Agent 回答';
    case 'read_agent_answer': return '读取 Agent 回答';
    case 'ask_user': return '请求用户输入';
    case 'skills': return '载入技能';
    case 'transfer': return '传输文件';
    case 'switch_work_environment': return '切换工作环境';
    case 'update_task_list': return '更新任务清单';
    default: return `调用 ${toolName}`;
  }
}

function childToolArgumentPreview(
  database: Database.Database,
  tool: Record<string, unknown>
): string | undefined {
  try {
    const contentObjectId = String(tool.arguments_object_id);
    const raw = database.prepare('SELECT * FROM content_object WHERE id = ?').get(contentObjectId);
    if (!raw) return undefined;
    const metadata = DOMAIN_REPOSITORIES.codec('ContentObject').decode(raw as Record<string, unknown>);
    if (
      typeof metadata.byte_length !== 'bigint'
      || metadata.byte_length > BigInt(CHILD_ACTIVITY_ARGUMENTS_MAX_BYTES)
    ) return undefined;
    const parsed = JSON.parse(readVerifiedCasBytes(
      metadata,
      path.resolve(data.binding.paths.casRootPath)
    ).toString('utf8')) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
    const args = parsed as Record<string, unknown>;
    const toolName = String(tool.tool_name);
    if ((toolName === 'bash' || toolName === 'shell') && typeof args.command === 'string') {
      return compactChildActivitySummary(args.command);
    }
    if (toolName === 'run_agent' && typeof args.prompt === 'string') {
      return compactChildActivitySummary(args.prompt);
    }
    if (toolName === 'skills' && typeof args.name === 'string') {
      return compactChildActivitySummary(args.name);
    }
    if (Array.isArray(args.paths)) {
      const paths = args.paths.filter((value): value is string => typeof value === 'string' && Boolean(value.trim()));
      if (paths.length > 0) {
        return compactChildActivitySummary(`${paths[0]}${paths.length > 1 ? ` +${paths.length - 1}` : ''}`);
      }
    }
    for (const key of ['path', 'query', 'pattern', 'title', 'question', 'explanation', 'summary']) {
      const value = args[key];
      if (typeof value === 'string' && value.trim()) return compactChildActivitySummary(value);
    }
  } catch {
    // Activity is a best-effort bounded view. Malformed/large arguments remain available only in
    // the child Conversation and must never make a Runtime transaction or parent feed fail.
  }
  return undefined;
}

function compactChildActivitySummary(value: string): string {
  const normalized = value.replace(/\s+/g, ' ').trim();
  return normalized.length > CHILD_ACTIVITY_SUMMARY_MAX_CHARACTERS
    ? `${normalized.slice(0, CHILD_ACTIVITY_SUMMARY_MAX_CHARACTERS - 1)}…`
    : normalized;
}

function projectProcessRecord(database: Database.Database, processId: string): DomainRow {
  const raw = database.prepare('SELECT * FROM process WHERE id = ?').get(processId);
  if (!raw) throw new Error(`Process ${processId} does not exist.`);
  const record = DOMAIN_REPOSITORIES.codec('Process').decode(raw as Record<string, unknown>);
  const source = database.prepare(`
    SELECT arguments.*
      FROM process_origin_link AS origin
      JOIN tool_call AS call ON call.id = origin.tool_call_id
      JOIN content_object AS arguments ON arguments.id = call.arguments_object_id
     WHERE origin.process_id = ?
     LIMIT 2
  `).all(processId) as Array<Record<string, unknown>>;
  if (source.length > 1) throw new Error(`Process ${processId} has multiple argument sources.`);
  let requestedBackground = false;
  let commandPreview: string | null = null;
  let argumentsProjectionState: 'ready' | 'missing' | 'error' = source.length === 1 ? 'ready' : 'missing';
  if (source.length === 1) {
    try {
      const metadata = DOMAIN_REPOSITORIES.codec('ContentObject').decode(source[0]);
      const parsed = JSON.parse(readVerifiedCasBytes(metadata, path.resolve(data.binding.paths.casRootPath)).toString('utf8')) as unknown;
      const argumentsRecord = parsed && typeof parsed === 'object' && !Array.isArray(parsed)
        ? parsed as Record<string, unknown>
        : undefined;
      requestedBackground = argumentsRecord?.foregroundWaitMs === 0;
      if (typeof argumentsRecord?.command === 'string' && argumentsRecord.command.length > 0) {
        commandPreview = argumentsRecord.command.length <= 240
          ? argumentsRecord.command
          : `${argumentsRecord.command.slice(0, 239)}…`;
      }
    } catch {
      argumentsProjectionState = 'error';
    }
  }
  const detached = database.prepare(`
    SELECT 1
      FROM operation
      JOIN attempt ON attempt.operation_id = operation.id
      JOIN effect_intent ON effect_intent.attempt_id = attempt.id
     WHERE operation.owner_kind = 'process'
       AND operation.owner_id = ?
       AND effect_intent.effect_kind = 'process_exit'
     LIMIT 1
  `).get(processId);
  return {
    ...record,
    background_kind: requestedBackground ? 'requested' : detached ? 'detached' : null,
    command_arguments_state: argumentsProjectionState,
    command_preview: commandPreview
  };
}

function deriveCommittedParentHandling(
  delivery: DomainRow,
  inputLink: { handled_at: string | null } | null
): 'unhandled' | 'handled' | 'not_applicable' {
  if (delivery.state === 'pending' || delivery.state === 'failed') return 'unhandled';
  if (delivery.state !== 'consumed') throw new Error(`RuntimeDelivery ${String(delivery.id)} has invalid state.`);
  if (delivery.phase === 'notify_only' && inputLink === null) return 'not_applicable';
  if ((delivery.phase === 'current_turn' || delivery.phase === 'next_turn') && inputLink) {
    return inputLink.handled_at === null ? 'unhandled' : 'handled';
  }
  throw new Error(`Consumed RuntimeDelivery ${String(delivery.id)} has an invalid InputLink combination.`);
}

function executeTransaction(
  database: Database.Database,
  steps: RepositoryTransactionStep[],
  nextCommitSeq: bigint
): RuntimeCommitResult {
  if (!Array.isArray(steps) || steps.length === 0) throw new Error('Runtime transaction requires at least one Repository step.');
  const allocatedSequences: RuntimeAllocatedSequence[] = [];
  let changes: RuntimeChange[] = [];
  database.exec('BEGIN IMMEDIATE');

  try {
    database.exec('DELETE FROM temp.runtime_transaction_change');
    executeSteps(database, steps, allocatedSequences);
    assertTouchedRuntimeAggregates(database, steps);
    changes = readTransactionChanges(database);
    database.exec('COMMIT');
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
  return { commitSeq: nextCommitSeq.toString(), changes, allocatedSequences };
}


function executeModelStreamEvent(
  database: Database.Database,
  input: ModelStreamEventCommitInput,
  nextCommitSeq: bigint
): ModelStreamEventCommitResult {
  const modelRequestId = requireRuntimeId(input.modelRequestId);
  const checkpointId = requireRuntimeId(input.checkpointId);
  const attemptSeq = requirePositiveInteger(input.attemptSeq, 'ModelStreamEvent.attemptSeq');
  const socketGeneration = requirePositiveInteger(input.socketGeneration, 'ModelStreamEvent.socketGeneration');
  const streamSeq = requirePositiveInteger(input.streamSeq, 'ModelStreamEvent.streamSeq');
  if (!['output_delta', 'output_item_done', 'terminal_summary'].includes(input.checkpointKind)) {
    throw new TypeError(`Unsupported ModelStream checkpoint kind: ${String(input.checkpointKind)}`);
  }
  if (typeof input.now !== 'string' || input.now.length === 0) throw new TypeError('ModelStreamEvent.now must be non-empty.');
  database.exec('BEGIN IMMEDIATE');
  try {
    database.exec('DELETE FROM temp.runtime_transaction_change');
    assertExecutionLeaseFence(database, input.executionFence);
    const requestRaw = database.prepare('SELECT * FROM model_request WHERE id = ?').get(modelRequestId);
    if (!requestRaw) throw new Error(`ModelRequest ${modelRequestId} does not exist.`);
    const request = DOMAIN_REPOSITORIES.codec('ModelRequest').decode(requestRaw as Record<string, unknown>);
    const existing = database.prepare(
      'SELECT model_request_id, attempt_seq, socket_generation, stream_seq, checkpoint_kind, content_object_id '
        + 'FROM model_stream_checkpoint WHERE id = ? LIMIT 1'
    ).get(checkpointId) as {
      model_request_id?: unknown;
      attempt_seq?: unknown;
      socket_generation?: unknown;
      stream_seq?: unknown;
      checkpoint_kind?: unknown;
      content_object_id?: unknown;
    } | undefined;
    if (existing) {
      if (
        existing.model_request_id !== modelRequestId
        || existing.attempt_seq !== attemptSeq
        || existing.socket_generation !== socketGeneration
        || existing.stream_seq !== streamSeq
        || existing.checkpoint_kind !== input.checkpointKind
        || existing.content_object_id !== input.contentObject.id
      ) {
        const error = new Error(`ModelStream checkpoint ${checkpointId} conflicts with an existing event identity.`) as Error & {
          code: string;
        };
        error.code = 'MODEL_STREAM_IDEMPOTENCY_CONFLICT';
        throw error;
      }
      database.exec('ROLLBACK');
      return {
        accepted: false,
        checkpointed: false,
        terminal: request.status === 'terminal',
        ignoredReason: 'duplicate'
      };
    }
    const fence = database.prepare(
      'SELECT id FROM model_stream_fence WHERE model_request_id = ? LIMIT 1'
    ).get(modelRequestId);
    const turn = database.prepare('SELECT status FROM turn WHERE id = ?').get(request.turn_id) as { status?: unknown } | undefined;
    if (fence || request.status === 'terminal' || turn?.status !== 'active') {
      database.exec('ROLLBACK');
      return { accepted: false, checkpointed: false, terminal: true, ignoredReason: 'terminal' };
    }
    if (request.status !== 'streaming') throw new Error(`ModelRequest ${modelRequestId} is not streaming.`);
    const identity = decodeModelStreamIdentity(request.stream_stats_json);
    if (identity.attemptSeq !== attemptSeq) {
      database.exec('ROLLBACK');
      return { accepted: false, checkpointed: false, terminal: false, ignoredReason: 'old-attempt' };
    }
    if (identity.socketGeneration !== socketGeneration) {
      database.exec('ROLLBACK');
      return { accepted: false, checkpointed: false, terminal: false, ignoredReason: 'old-socket-generation' };
    }
    const checkpointCountRow = database.prepare(`
      SELECT COUNT(*) AS count,
             COALESCE(SUM(CASE WHEN checkpoint_kind = 'output_delta' THEN 1 ELSE 0 END), 0) AS output_delta_count
        FROM model_stream_checkpoint
       WHERE model_request_id = ?
    `).get(modelRequestId) as { count: bigint; output_delta_count: bigint };
    if (
      typeof checkpointCountRow.count !== 'bigint'
      || typeof checkpointCountRow.output_delta_count !== 'bigint'
    ) throw new Error('ModelStream checkpoint counts were not INTEGER values.');
    if (
      input.checkpointKind === 'output_delta'
      && checkpointCountRow.output_delta_count >= BigInt(MODEL_STREAM_OUTPUT_DELTA_CHECKPOINT_LIMIT)
    ) {
      database.exec('ROLLBACK');
      return {
        accepted: true,
        checkpointed: false,
        terminal: false,
        ignoredReason: 'checkpoint-capacity'
      };
    }
    if (
      input.checkpointKind === 'output_item_done'
      && checkpointCountRow.count >= BigInt(MODEL_STREAM_ACTIVE_CHECKPOINT_LIMIT)
    ) {
      database.exec('ROLLBACK');
      return {
        accepted: true,
        checkpointed: false,
        terminal: false,
        ignoredReason: 'checkpoint-capacity'
      };
    }
    const contentId = requireRuntimeId(input.contentObject.id);
    assertPreparedContentInsert(input.contentObject, input.contentInsert);
    const contentSteps: RepositoryTransactionStep[] = preparedContentObjectSteps([{
      metadata: input.contentObject as ContentObjectMetadata,
      ...(input.contentInsert ? { insert: input.contentInsert } : {})
    }], 'model_stream_content');
    executeSteps(database, contentSteps, []);
    insertStreamFact(database, 'ModelStreamCheckpoint', {
      id: checkpointId,
      model_request_id: modelRequestId,
      attempt_seq: attemptSeq,
      socket_generation: socketGeneration,
      stream_seq: streamSeq,
      checkpoint_kind: input.checkpointKind,
      content_object_id: contentId,
      created_at: input.now
    });
    if (input.checkpointKind === 'terminal_summary') {
      const terminalFenceId = requireRuntimeId(input.terminalFenceId);
      if (!input.terminalStats || typeof input.terminalStats !== 'object' || Array.isArray(input.terminalStats)) {
        throw new TypeError('Completed ModelStream event requires terminalStats.');
      }
      const terminalIdentity = decodeModelStreamIdentity(input.terminalStats);
      if (terminalIdentity.attemptSeq !== attemptSeq || terminalIdentity.socketGeneration !== socketGeneration) {
        throw new Error('Completed ModelStream terminalStats do not match the active stream identity.');
      }
      const operation = database.prepare(
        "SELECT id FROM operation WHERE owner_kind = 'model_request' AND owner_id = ? LIMIT 1"
      ).get(modelRequestId) as { id?: unknown } | undefined;
      if (typeof operation?.id !== 'string') throw new Error(`ModelRequest ${modelRequestId} has no Operation.`);
      const attempt = database.prepare(
        'SELECT id FROM attempt WHERE operation_id = ? AND attempt_seq = ? LIMIT 1'
      ).get(operation.id, attemptSeq) as { id?: unknown } | undefined;
      if (typeof attempt?.id !== 'string') throw new Error(`ModelRequest ${modelRequestId} has no attempt ${attemptSeq}.`);
      insertStreamFact(database, 'ModelStreamFence', {
        id: terminalFenceId,
        model_request_id: modelRequestId,
        attempt_seq: attemptSeq,
        socket_generation: socketGeneration,
        terminal_stream_seq: streamSeq,
        outcome: 'completed',
        created_at: input.now
      });
      executeSteps(database, [
        DOMAIN_REPOSITORIES.domain('Attempt').update(attempt.id, {
          status: 'completed', updated_at: input.now, completed_at: input.now
        }),
        DOMAIN_REPOSITORIES.domain('Operation').update(operation.id, {
          status: 'completed', updated_at: input.now
        }),
        DOMAIN_REPOSITORIES.domain('ModelRequest').update(modelRequestId, {
          status: 'terminal',
          terminal_state: 'completed',
          usage_json: input.usage,
          stream_stats_json: input.terminalStats,
          updated_at: input.now
        })
      ], []);
      executeSteps(database, [
        DOMAIN_REPOSITORIES.domain('ModelStreamCheckpoint').pruneAfterTerminalFence(
          modelRequestId,
          attemptSeq,
          socketGeneration,
          checkpointId
        )
      ], []);
      assertModelRequestAggregate(database, modelRequestId);
    } else {
      if (input.terminalFenceId !== null || input.terminalStats !== null || input.usage !== null) {
        throw new TypeError('Non-terminal ModelStream event cannot carry terminal facts.');
      }
    }
    const changes = readTransactionChanges(database);
    database.exec('COMMIT');
    const commit: RuntimeCommitResult = {
      commitSeq: nextCommitSeq.toString(),
      changes,
      allocatedSequences: []
    };
    return {
      accepted: true,
      checkpointed: true,
      terminal: input.checkpointKind === 'terminal_summary',
      commit
    };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function executeModelStreamActivity(
  database: Database.Database,
  input: ModelStreamActivityInput,
  nextCommitSeq: bigint
): ModelStreamActivityResult {
  const modelRequestId = requireRuntimeId(input.modelRequestId);
  const attemptSeq = requirePositiveInteger(input.attemptSeq, 'ModelStreamActivity.attemptSeq');
  const socketGeneration = requirePositiveInteger(input.socketGeneration, 'ModelStreamActivity.socketGeneration');
  const streamSeq = requirePositiveInteger(input.streamSeq, 'ModelStreamActivity.streamSeq');
  if (!Number.isSafeInteger(input.observedAt) || input.observedAt <= 0) {
    throw new TypeError('ModelStreamActivity.observedAt must be a positive safe integer.');
  }
  if (typeof input.now !== 'string' || input.now.length === 0) {
    throw new TypeError('ModelStreamActivity.now must be non-empty.');
  }
  database.exec('BEGIN IMMEDIATE');
  try {
    database.exec('DELETE FROM temp.runtime_transaction_change');
    assertExecutionLeaseFence(database, input.executionFence);
    const requestRaw = database.prepare('SELECT * FROM model_request WHERE id = ?').get(modelRequestId);
    if (!requestRaw) throw new Error(`ModelRequest ${modelRequestId} does not exist.`);
    const request = DOMAIN_REPOSITORIES.codec('ModelRequest').decode(requestRaw as Record<string, unknown>);
    const fence = database.prepare(
      'SELECT id FROM model_stream_fence WHERE model_request_id = ? LIMIT 1'
    ).get(modelRequestId);
    const turn = database.prepare('SELECT status FROM turn WHERE id = ?').get(request.turn_id) as { status?: unknown } | undefined;
    if (fence || request.status === 'terminal' || turn?.status !== 'active') {
      database.exec('ROLLBACK');
      return { accepted: false, terminal: true };
    }
    if (request.status !== 'streaming') {
      database.exec('ROLLBACK');
      return { accepted: false, terminal: false };
    }
    const identity = decodeModelStreamIdentity(request.stream_stats_json);
    if (identity.attemptSeq !== attemptSeq || identity.socketGeneration !== socketGeneration) {
      database.exec('ROLLBACK');
      return { accepted: false, terminal: false };
    }
    const stats = request.stream_stats_json as Record<string, unknown>;
    const previousSeq = optionalDecimalInteger(stats.lastStreamSeq, 'lastStreamSeq') ?? 0n;
    const previousAt = optionalPositiveInteger(stats.lastStreamEventAt, 'lastStreamEventAt') ?? 0;
    const nextSeq = streamSeq > previousSeq ? streamSeq : previousSeq;
    const nextAt = input.observedAt > previousAt ? input.observedAt : previousAt;
    if (nextSeq === previousSeq && nextAt === previousAt) {
      database.exec('ROLLBACK');
      return { accepted: false, terminal: false };
    }
    executeSteps(database, [
      DOMAIN_REPOSITORIES.domain('ModelRequest').update(modelRequestId, {
        status: 'streaming',
        stream_stats_json: {
          ...stats,
          lastStreamSeq: nextSeq.toString(),
          lastStreamEventAt: nextAt
        },
        updated_at: input.now
      })
    ], []);
    const changes = readTransactionChanges(database);
    database.exec('COMMIT');
    return {
      accepted: true,
      terminal: false,
      commit: { commitSeq: nextCommitSeq.toString(), changes, allocatedSequences: [] }
    };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function executeCancelCurrentModelRequest(
  database: Database.Database,
  input: ModelRequestCancelInput,
  nextCommitSeq: bigint
): ModelRequestCancelResult {
  const modelRequestId = requireRuntimeId(input.modelRequestId);
  if (typeof input.terminalState !== 'string' || input.terminalState.length === 0) {
    throw new TypeError('ModelRequest cancellation terminalState must be non-empty.');
  }
  if (typeof input.now !== 'string' || input.now.length === 0) {
    throw new TypeError('ModelRequest cancellation time must be non-empty.');
  }
  database.exec('BEGIN IMMEDIATE');
  try {
    database.exec('DELETE FROM temp.runtime_transaction_change');
    assertExecutionLeaseFence(database, input.executionFence);
    const requestRaw = database.prepare('SELECT * FROM model_request WHERE id = ?').get(modelRequestId);
    if (!requestRaw) throw new Error(`ModelRequest ${modelRequestId} does not exist.`);
    const request = DOMAIN_REPOSITORIES.codec('ModelRequest').decode(requestRaw as Record<string, unknown>);
    const identity = decodeModelStreamIdentity(request.stream_stats_json);
    if (request.status === 'terminal') {
      database.exec('ROLLBACK');
      return {
        cancelled: false,
        terminalState: typeof request.terminal_state === 'string' ? request.terminal_state : null,
        attemptSeq: identity.attemptSeq.toString(),
        socketGeneration: identity.socketGeneration.toString()
      };
    }
    const fence = database.prepare(
      'SELECT id FROM model_stream_fence WHERE model_request_id = ? LIMIT 1'
    ).get(modelRequestId);
    if (fence) throw new Error(`Active ModelRequest ${modelRequestId} unexpectedly has a terminal fence.`);
    const operation = database.prepare(
      "SELECT id FROM operation WHERE owner_kind = 'model_request' AND owner_id = ? LIMIT 1"
    ).get(modelRequestId) as { id?: unknown } | undefined;
    if (typeof operation?.id !== 'string') throw new Error(`ModelRequest ${modelRequestId} has no Operation.`);
    const attempt = database.prepare(
      'SELECT id FROM attempt WHERE operation_id = ? AND attempt_seq = ? LIMIT 1'
    ).get(operation.id, identity.attemptSeq) as { id?: unknown } | undefined;
    if (typeof attempt?.id !== 'string') {
      throw new Error(`ModelRequest ${modelRequestId} has no current attempt ${identity.attemptSeq}.`);
    }
    executeSteps(database, [
      DOMAIN_REPOSITORIES.domain('Attempt').update(attempt.id, {
        status: 'cancelled', updated_at: input.now, completed_at: input.now
      }),
      DOMAIN_REPOSITORIES.domain('Operation').update(operation.id, {
        status: 'cancelled', updated_at: input.now
      }),
      DOMAIN_REPOSITORIES.domain('ModelRequest').update(modelRequestId, {
        status: 'terminal', terminal_state: input.terminalState, updated_at: input.now
      })
    ], []);
    assertModelRequestAggregate(database, modelRequestId);
    const changes = readTransactionChanges(database);
    database.exec('COMMIT');
    const commit: RuntimeCommitResult = {
      commitSeq: nextCommitSeq.toString(),
      changes,
      allocatedSequences: []
    };
    return {
      cancelled: true,
      terminalState: input.terminalState,
      attemptSeq: identity.attemptSeq.toString(),
      socketGeneration: identity.socketGeneration.toString(),
      commit
    };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function assertExecutionLeaseFence(
  database: Database.Database,
  fence: ExecutionLeaseFencePayload | undefined
): void {
  if (!fence) return;
  const id = requireRuntimeId(fence.id);
  const conversationId = requireRuntimeId(fence.conversationId);
  const turnId = requireRuntimeId(fence.turnId);
  const ownerId = requireRuntimeId(fence.ownerId);
  const hostBootId = requireRuntimeId(fence.hostBootId);
  const generation = requirePositiveInteger(fence.generation, 'ExecutionLeaseFence.generation');
  const row = database.prepare(
    'SELECT 1 AS present FROM execution_lease '
      + 'WHERE id = ? AND conversation_id = ? AND turn_id = ? AND owner_id = ? '
      + 'AND host_boot_id = ? AND generation = ? LIMIT 1'
  ).get(id, conversationId, turnId, ownerId, hostBootId, generation);
  if (row) return;
  const error = new Error(`ExecutionLease ${id} generation ${generation} no longer authorizes this write.`) as Error & {
    code: string;
  };
  error.code = 'RUNTIME_TRANSACTION_ASSERTION_FAILED';
  throw error;
}

function decodeModelStreamIdentity(value: unknown): {
  attemptSeq: bigint;
  socketGeneration: bigint;
  retryMaxAttempts?: number;
  retryDelayMs?: number;
  retryNotBeforeAt?: number;
} {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('ModelRequest.stream_stats_json must be an object.');
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  const allowedKeys = new Set([
    'attemptSeq',
    'socketGeneration',
    'retryReason',
    'retryMaxAttempts',
    'retryDelayMs',
    'retryNotBeforeAt',
    'providerStartedAt',
    'firstOutputAt',
    'completedAt',
    'streamOutputDurationMs',
    'lastStreamSeq',
    'lastStreamEventAt'
  ]);
  if (
    !keys.includes('attemptSeq')
    || !keys.includes('retryReason')
    || !keys.includes('socketGeneration')
    || keys.some((key) => !allowedKeys.has(key))
    || (
      record.retryReason !== null
      && record.retryReason !== 'connection_interrupted'
      && record.retryReason !== 'rate_limited'
      && record.retryReason !== 'temporary_service_error'
      && record.retryReason !== 'first_semantic_timeout'
      && record.retryReason !== 'stream_stalled'
      && record.retryReason !== 'compression_timeout'
    )
  ) throw new TypeError('ModelRequest.stream_stats_json has an invalid shape.');
  assertOptionalBoundedInteger(record.retryMaxAttempts, 'retryMaxAttempts', 1, 10);
  assertOptionalBoundedInteger(record.retryDelayMs, 'retryDelayMs', 0, Number.MAX_SAFE_INTEGER);
  assertOptionalBoundedInteger(record.retryNotBeforeAt, 'retryNotBeforeAt', 1, Number.MAX_SAFE_INTEGER);
  const attemptSeq = decimalRuntimeInteger(record.attemptSeq, 'stream_stats.attemptSeq');
  const retryMaxAttempts = typeof record.retryMaxAttempts === 'number'
    ? record.retryMaxAttempts
    : attemptSeq === 2n ? 1 : undefined;
  const retryDelayMs = typeof record.retryDelayMs === 'number' ? record.retryDelayMs : undefined;
  const retryNotBeforeAt = typeof record.retryNotBeforeAt === 'number' ? record.retryNotBeforeAt : undefined;
  assertOptionalStreamTiming(record.providerStartedAt, 'providerStartedAt');
  assertOptionalStreamTiming(record.firstOutputAt, 'firstOutputAt');
  assertOptionalStreamTiming(record.completedAt, 'completedAt');
  assertOptionalStreamTiming(record.streamOutputDurationMs, 'streamOutputDurationMs', true);
  optionalDecimalInteger(record.lastStreamSeq, 'lastStreamSeq');
  optionalPositiveInteger(record.lastStreamEventAt, 'lastStreamEventAt');
  return {
    attemptSeq,
    socketGeneration: decimalRuntimeInteger(record.socketGeneration, 'stream_stats.socketGeneration'),
    ...(retryMaxAttempts !== undefined ? { retryMaxAttempts } : {}),
    ...(retryDelayMs !== undefined ? { retryDelayMs } : {}),
    ...(retryNotBeforeAt !== undefined ? { retryNotBeforeAt } : {})
  };
}

function optionalDecimalInteger(value: unknown, label: string): bigint | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string' || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new TypeError(`ModelRequest.stream_stats_json.${label} must be a decimal integer string.`);
  }
  return BigInt(value);
}

function optionalNonNegativeInteger(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value === 'string' && /^(?:0|[1-9]\d*)$/.test(value)) {
    const parsed = Number(value);
    if (Number.isSafeInteger(parsed)) return parsed;
  }
  if (Number.isSafeInteger(value) && (value as number) >= 0) return value as number;
  throw new TypeError(`ModelRequest.stream_stats_json.${label} must be a non-negative safe integer.`);
}

function optionalPositiveInteger(value: unknown, label: string): number | undefined {
  const parsed = optionalNonNegativeInteger(value, label);
  if (parsed === undefined) return undefined;
  if (parsed <= 0) throw new TypeError(`ModelRequest.stream_stats_json.${label} must be positive.`);
  return parsed;
}

function assertOptionalBoundedInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number
): void {
  if (value === undefined) return;
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new TypeError(`ModelRequest.stream_stats_json.${label} must be an integer in [${minimum}, ${maximum}].`);
  }
}

function assertOptionalStreamTiming(value: unknown, label: string, allowZero = false): void {
  if (value === undefined) return;
  if (!Number.isSafeInteger(value) || (allowZero ? (value as number) < 0 : (value as number) <= 0)) {
    throw new TypeError(`ModelRequest.stream_stats_json.${label} must be a ${allowZero ? 'non-negative' : 'positive'} safe integer.`);
  }
}

function decimalRuntimeInteger(value: unknown, label: string): bigint {
  if (typeof value !== 'string' || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new TypeError(`${label} must be a decimal integer string.`);
  }
  return BigInt(value);
}

function requirePositiveInteger(value: unknown, label: string): bigint {
  if (typeof value !== 'bigint' || value <= 0n) throw new TypeError(`${label} must be a positive SQLite INTEGER.`);
  return value;
}

function executeSteps(
  database: Database.Database,
  steps: RepositoryTransactionStep[],
  allocatedSequences: RuntimeAllocatedSequence[]
): void {
  for (const step of steps) {
    if (step.kind === 'assert') {
      executeAssertion(database, step.domain, step.id, step.where);
      continue;
    }
    if (step.kind === 'assertAll') {
      executeAssertAll(database, step.domain, step.where, step.expected);
      continue;
    }
    if (step.kind === 'assertNone') {
      executeAssertNone(database, step.domain, step.where);
      continue;
    }
    if (step.kind === 'assertExactIds') {
      executeAssertExactIds(database, step.domain, step.where, step.expectedIds);
      continue;
    }
    if (step.kind !== 'savepoint') {
      executeMutation(database, step, allocatedSequences);
      continue;
    }
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(step.name)) throw new Error(`Invalid savepoint name: ${step.name}`);
    const marker = quote(step.name);
    const sequenceCount = allocatedSequences.length;
    database.exec(`SAVEPOINT ${marker}`);
    try {
      executeSteps(database, step.steps, allocatedSequences);
      database.exec(`RELEASE SAVEPOINT ${marker}`);
    } catch (error) {
      database.exec(`ROLLBACK TO SAVEPOINT ${marker}`);
      database.exec(`RELEASE SAVEPOINT ${marker}`);
      allocatedSequences.length = sequenceCount;
      if (!matchesSavepointContinuation(error, step.onError)) throw error;
    }
  }
}

function executeAssertion(
  database: Database.Database,
  domain: string,
  id: string,
  where: DomainRow
): void {
  const repository = DOMAIN_REPOSITORIES.domain(domain);
  const encoded = repository.codec.encodeWhere(where);
  const predicates = ['id = @__id'];
  const parameters: EncodedRow & { __id: string } = { __id: requireRuntimeId(id) };
  for (const [name, value] of Object.entries(encoded)) {
    if (name === 'id') throw new Error(`${repository.name} assertion id must be supplied separately.`);
    if (value === null) predicates.push(`${quote(name)} IS NULL`);
    else {
      predicates.push(`${quote(name)} = @${name}`);
      parameters[name] = value;
    }
  }
  const matched = database.prepare(
    `SELECT 1 AS matched FROM ${quote(repository.schema.table)} WHERE ${predicates.join(' AND ')} LIMIT 1`
  ).get(parameters);
  if (!matched) {
    const error = new Error(`${repository.name} transaction assertion failed for ${id}.`) as Error & { code: string };
    error.code = 'RUNTIME_TRANSACTION_ASSERTION_FAILED';
    throw error;
  }
}

function executeAssertAll(
  database: Database.Database,
  domain: string,
  where: DomainRow,
  expected: DomainRow
): void {
  const repository = DOMAIN_REPOSITORIES.domain(domain);
  const encodedWhere = repository.codec.encodeWhere(where);
  const encodedExpected = repository.codec.encodeWhere(expected);
  if (Object.keys(encodedExpected).length === 0) throw new Error(`${repository.name} assertAll requires expected fields.`);
  const predicates: string[] = [];
  const violations: string[] = [];
  const parameters: EncodedRow = {};
  for (const [name, value] of Object.entries(encodedWhere)) {
    if (value === null) predicates.push(`${quote(name)} IS NULL`);
    else {
      const parameter = `where_${name}`;
      predicates.push(`${quote(name)} = @${parameter}`);
      parameters[parameter] = value;
    }
  }
  for (const [name, value] of Object.entries(encodedExpected)) {
    if (value === null) violations.push(`${quote(name)} IS NOT NULL`);
    else {
      const parameter = `expected_${name}`;
      violations.push(`(${quote(name)} IS NULL OR ${quote(name)} != @${parameter})`);
      parameters[parameter] = value;
    }
  }
  const sql = `SELECT id FROM ${quote(repository.schema.table)}`
    + `${predicates.length ? ` WHERE ${predicates.join(' AND ')} AND (${violations.join(' OR ')})` : ` WHERE ${violations.join(' OR ')}`}`
    + ' LIMIT 1';
  const violating = database.prepare(sql).get(parameters) as { id?: unknown } | undefined;
  if (violating) {
    const error = new Error(`${repository.name} transaction assertAll failed for ${String(violating.id)}.`) as Error & { code: string };
    error.code = 'RUNTIME_TRANSACTION_ASSERTION_FAILED';
    throw error;
  }
}

function executeAssertExactIds(
  database: Database.Database,
  domain: string,
  where: DomainRow,
  expectedIds: readonly string[]
): void {
  const repository = DOMAIN_REPOSITORIES.domain(domain);
  const encoded = repository.codec.encodeWhere(where);
  const { predicates, parameters } = whereClause(encoded);
  if (predicates.length === 0) throw new Error(`${repository.name} assertExactIds requires predicates.`);
  const actualIds = (database.prepare(
    `SELECT id FROM ${quote(repository.schema.table)} WHERE ${predicates.join(' AND ')} ORDER BY id ASC`
  ).all(parameters) as Array<{ id: unknown }>).map((row) => requireRuntimeId(row.id));
  const expected = [...expectedIds].map(requireRuntimeId).sort();
  if (
    actualIds.length !== expected.length
    || actualIds.some((id, index) => id !== expected[index])
  ) {
    const error = new Error(`${repository.name} transaction assertExactIds failed.`) as Error & { code: string };
    error.code = 'RUNTIME_TRANSACTION_ASSERTION_FAILED';
    throw error;
  }
}

function executeAssertNone(database: Database.Database, domain: string, where: DomainRow): void {
  const repository = DOMAIN_REPOSITORIES.domain(domain);
  const encoded = repository.codec.encodeWhere(where);
  const { predicates, parameters } = whereClause(encoded);
  if (predicates.length === 0) throw new Error(`${repository.name} assertNone requires predicates.`);
  const matched = database.prepare(
    `SELECT id FROM ${quote(repository.schema.table)} WHERE ${predicates.join(' AND ')} LIMIT 1`
  ).get(parameters) as { id?: unknown } | undefined;
  if (matched) {
    const error = new Error(`${repository.name} transaction assertNone failed for ${String(matched.id)}.`) as Error & { code: string };
    error.code = 'RUNTIME_TRANSACTION_ASSERTION_FAILED';
    throw error;
  }
}

function executeMutation(
  database: Database.Database,
  mutation: RepositoryMutation,
  allocatedSequences: RuntimeAllocatedSequence[]
): void {
  const repository = DOMAIN_REPOSITORIES.domain(mutation.domain);
  const schema = repository.schema;
  const mutationKind = mutation.kind === 'deleteWhere' || mutation.kind === 'pruneModelStreamCheckpoints'
    ? 'delete'
    : mutation.kind;
  if (!schema.mutations.includes(mutationKind)) {
    throw new Error(`${schema.repository} does not allow ${mutationKind}.`);
  }

  if (mutation.kind === 'pruneModelStreamCheckpoints') {
    executeCheckpointPrune(database, mutation);
  } else if (mutation.kind === 'insert') {
    const historicalCopy = mutation.historicalCopy === true;
    if (historicalCopy && !HISTORICAL_COPY_DOMAINS.includes(schema.key)) {
      throw new Error(`${schema.key} does not permit historical copy inserts.`);
    }
    if (schema.key === 'ModelStreamCheckpoint' || (schema.key === 'ModelStreamFence' && !historicalCopy)) {
      throw new Error(`${schema.key} insert is limited to the fixed writer modelStreamEvent operation.`);
    }
    const allocatedRow = mutation.allocateSequence
      ? allocateNextSequence(database, repository, mutation)
      : mutation.row;
    const row = resolveMessageRevisionSequenceReference(allocatedRow, mutation, allocatedSequences);
    if (schema.key === 'ModelRequest') {
      if (historicalCopy) {
        if (row.status !== 'terminal' || row.terminal_state === null) {
          throw new Error('Fork copy ModelRequest must be terminal with a terminal_state.');
        }
      } else if (row.status !== 'prepared' || row.terminal_state !== null) {
        throw new Error('ModelRequest insert must start prepared and non-terminal.');
      }
      decodeModelStreamIdentity(row.stream_stats_json);
    }
    if (schema.key === 'Operation' && row.owner_kind === 'model_request') {
      if (historicalCopy) {
        if (row.status === 'pending') {
          throw new Error('Fork copy ModelRequest Operation must not be pending.');
        }
      } else if (row.status !== 'pending') {
        throw new Error('ModelRequest Operation must start pending.');
      }
    }
    if (schema.key === 'Attempt') {
      const operation = database.prepare('SELECT owner_kind FROM operation WHERE id = ?').get(row.operation_id) as {
        owner_kind?: unknown;
      } | undefined;
      if (operation?.owner_kind === 'model_request') {
        if (historicalCopy) {
          if (row.status === 'pending') throw new Error('Fork copy ModelRequest Attempt must not be pending.');
        } else if (row.status !== 'pending') {
          throw new Error('ModelRequest Attempt must start pending.');
        }
        if (typeof row.attempt_seq !== 'bigint' || row.attempt_seq < 1n || row.attempt_seq > 11n) {
          throw new Error('ModelRequest permits only attempt_seq 1 through 11.');
        }
      }
    }
    const encoded = repository.codec.encodeInsert(row);
    const id = requireEncodedId(encoded.id, schema.codec);
    if (mutation.allocateSequence) {
      const value = encoded[mutation.allocateSequence.column];
      if (typeof value !== 'bigint') throw new Error('Allocated sequence was not encoded as SQLite INTEGER.');
      allocatedSequences.push({
        domain: schema.key,
        id,
        column: mutation.allocateSequence.column,
        value: value.toString()
      });
    }
    if (schema.key === 'ContentObject') assertPublishedContentObject(encoded, data.binding.paths.casRootPath);
    const names = Object.keys(encoded);
    const sql = `INSERT INTO ${quote(schema.table)} (${names.map(quote).join(', ')}) VALUES (${names.map((name) => `@${name}`).join(', ')})`;
    database.prepare(sql).run(encoded);
  } else if (mutation.kind === 'update') {
    const id = requireRuntimeId(mutation.id);
    assertRuntimeDomainUpdatePatch(schema.key, mutation.patch);
    assertRuntimeStateTransition(database, schema.key, id, mutation.patch);
    const encoded = repository.codec.encodePatch(mutation.patch);
    const assignments = Object.keys(encoded).map((name) => `${quote(name)} = @${name}`);
    const result = database.prepare(`UPDATE ${quote(schema.table)} SET ${assignments.join(', ')} WHERE id = @__id`)
      .run({ ...encoded, __id: id });
    if (result.changes !== 1) throw new Error(`${schema.repository} update expected one row: ${id}`);
  } else if (mutation.kind === 'deleteWhere') {
    if (schema.key === 'ModelStreamCheckpoint') {
      throw new Error('ModelStreamCheckpoint rows can only be pruned by the fixed writer stream-finalization operation.');
    }
    const encoded = repository.codec.encodeWhere(mutation.where);
    const { predicates, parameters } = whereClause(encoded);
    if (predicates.length === 0) throw new Error(`${schema.repository}.deleteWhere requires predicates.`);
    const result = database.prepare(`DELETE FROM ${quote(schema.table)} WHERE ${predicates.join(' AND ')}`).run(parameters);
    if (result.changes > mutation.maxChanges) {
      throw new Error(`${schema.repository}.deleteWhere exceeded ${mutation.maxChanges} row.`);

    }
  } else {
    const id = requireRuntimeId(mutation.id);
    if (schema.key === 'ModelStreamCheckpoint') {
      throw new Error('ModelStreamCheckpoint rows can only be pruned by the fixed writer stream-finalization operation.');
    }
    const result = database.prepare(`DELETE FROM ${quote(schema.table)} WHERE id = ?`).run(id);
    if (result.changes !== 1) throw new Error(`${schema.repository} delete expected one row: ${id}`);
  }
}

function assertPreparedContentInsert(
  metadata: DomainRow,
  insert: RepositoryInsertMutation | undefined
): void {
  if (!insert) return;
  if (insert.domain !== 'ContentObject' || insert.allocateSequence || insert.messageRevisionSequenceReferenceId) {
    throw new Error('ModelStream contentInsert must be one plain ContentObject insert.');
  }
  const encodedMetadata = DOMAIN_REPOSITORIES.codec('ContentObject').encodeInsert(metadata);
  const encodedInsert = DOMAIN_REPOSITORIES.codec('ContentObject').encodeInsert(insert.row);
  for (const [column, value] of Object.entries(encodedMetadata)) {
    if (encodedInsert[column] !== value) {
      throw new Error(`ModelStream contentInsert does not match published metadata column ${column}.`);
    }
  }
}

function insertStreamFact(
  database: Database.Database,
  domain: 'ModelStreamCheckpoint' | 'ModelStreamFence',
  row: DomainRow
): void {
  const repository = DOMAIN_REPOSITORIES.domain(domain);
  const encoded = repository.codec.encodeInsert(row);
  const names = Object.keys(encoded);
  database.prepare(
    `INSERT INTO ${quote(repository.schema.table)} (${names.map(quote).join(', ')}) `
      + `VALUES (${names.map((name) => `@${name}`).join(', ')})`
  ).run(encoded);
}

function executeCheckpointPrune(
  database: Database.Database,
  mutation: RepositoryCheckpointPruneMutation
): void {
  const modelRequestId = requireRuntimeId(mutation.modelRequestId);
  const terminalCheckpointId = requireRuntimeId(mutation.terminalCheckpointId);
  const fence = database.prepare(`
    SELECT attempt_seq, socket_generation
      FROM model_stream_fence
     WHERE model_request_id = ?
     LIMIT 1
  `).get(modelRequestId) as { attempt_seq?: unknown; socket_generation?: unknown } | undefined;
  if (
    fence?.attempt_seq !== mutation.attemptSeq
    || fence.socket_generation !== mutation.socketGeneration
  ) throw new Error('ModelStream checkpoint prune requires the matching terminal fence identity.');
  const terminal = database.prepare(`
    SELECT model_request_id, attempt_seq, socket_generation, checkpoint_kind
      FROM model_stream_checkpoint
     WHERE id = ?
     LIMIT 1
  `).get(terminalCheckpointId) as {
    model_request_id?: unknown;
    attempt_seq?: unknown;
    socket_generation?: unknown;
    checkpoint_kind?: unknown;
  } | undefined;
  if (
    terminal?.model_request_id !== modelRequestId
    || terminal.attempt_seq !== mutation.attemptSeq
    || terminal.socket_generation !== mutation.socketGeneration
    || terminal.checkpoint_kind !== 'terminal_summary'
  ) throw new Error('ModelStream checkpoint prune requires the matching terminal summary.');
  const retained = database.prepare(`
    SELECT id
      FROM model_stream_checkpoint
     WHERE model_request_id = ?
       AND attempt_seq = ?
       AND socket_generation = ?
       AND checkpoint_kind != 'terminal_summary'
     ORDER BY stream_seq DESC
     LIMIT ?
  `).all(
    modelRequestId,
    mutation.attemptSeq,
    mutation.socketGeneration,
    BigInt(MODEL_STREAM_TERMINAL_TAIL)
  ) as Array<{ id: string }>;
  const keep = new Set([terminalCheckpointId, ...retained.map((row) => requireRuntimeId(row.id))]);
  const obsolete = database.prepare(
    'SELECT id FROM model_stream_checkpoint WHERE model_request_id = ?'
  ).all(modelRequestId) as Array<{ id: string }>;
  const deleteStatement = database.prepare('DELETE FROM model_stream_checkpoint WHERE id = ?');
  for (const row of obsolete) {
    const id = requireRuntimeId(row.id);
    if (!keep.has(id)) deleteStatement.run(id);
  }
}

function resolveMessageRevisionSequenceReference(
  row: DomainRow,
  mutation: RepositoryInsertMutation,
  allocatedSequences: RuntimeAllocatedSequence[]
): DomainRow {
  const messageRevisionId = mutation.messageRevisionSequenceReferenceId;
  if (!messageRevisionId) return row;
  if (
    mutation.domain !== 'ContextSegmentSource'
    || row.source_kind !== 'message_revision'
    || row.source_id !== messageRevisionId
    || 'source_revision' in row
  ) {
    throw new Error('Writer Message revision reference has an invalid ContextSegmentSource shape.');
  }
  const allocated = [...allocatedSequences].reverse().find((entry) =>
    entry.domain === 'MessageRevision'
    && entry.id === messageRevisionId
    && entry.column === 'revision_seq'
  );
  if (!allocated) {
    throw new Error(
      `ContextSegmentSource.source_revision references MessageRevision ${messageRevisionId} before its writer allocation.`
    );
  }
  return { ...row, source_revision: allocated.value };
}

function assertTouchedRuntimeAggregates(
  database: Database.Database,
  steps: readonly RepositoryTransactionStep[]
): void {
  const modelRequestIds = new Set<string>();
  const turnIds = new Set<string>();
  const visit = (step: RepositoryTransactionStep): void => {
    if (step.kind === 'savepoint') {
      step.steps.forEach(visit);
      return;
    }
    if (
      step.kind === 'assert'
      || step.kind === 'assertAll'
      || step.kind === 'assertNone'
      || step.kind === 'assertExactIds'
    ) return;
    if (step.domain === 'ModelRequest') {
      const id = step.kind === 'insert' ? step.row.id : 'id' in step ? step.id : null;
      if (typeof id === 'string') modelRequestIds.add(id);
    } else if (step.domain === 'Operation') {
      const id = step.kind === 'insert' ? step.row.id : 'id' in step ? step.id : null;
      if (typeof id === 'string') {
        const owner = database.prepare('SELECT owner_kind, owner_id FROM operation WHERE id = ?').get(id) as {
          owner_kind?: unknown;
          owner_id?: unknown;
        } | undefined;
        if (owner?.owner_kind === 'model_request' && typeof owner.owner_id === 'string') {
          modelRequestIds.add(owner.owner_id);
        }
      }
    } else if (step.domain === 'Attempt') {
      const id = step.kind === 'insert' ? step.row.id : 'id' in step ? step.id : null;
      if (typeof id === 'string') {
        const owner = database.prepare(`
          SELECT operation.owner_kind, operation.owner_id
            FROM attempt
            JOIN operation ON operation.id = attempt.operation_id
           WHERE attempt.id = ?
        `).get(id) as { owner_kind?: unknown; owner_id?: unknown } | undefined;
        if (owner?.owner_kind === 'model_request' && typeof owner.owner_id === 'string') {
          modelRequestIds.add(owner.owner_id);
        }
      }
    } else if (step.domain === 'Turn') {
      const id = step.kind === 'insert' ? step.row.id : 'id' in step ? step.id : null;
      if (typeof id === 'string') turnIds.add(id);
    } else if (step.domain === 'TurnTermination' && step.kind === 'insert') {
      if (typeof step.row.turn_id === 'string') turnIds.add(step.row.turn_id);
    }
  };
  steps.forEach(visit);
  for (const turnId of turnIds) {
    const turn = database.prepare('SELECT status FROM turn WHERE id = ?').get(turnId) as { status?: unknown } | undefined;
    if (turn?.status === 'active') continue;
    const requests = database.prepare('SELECT id FROM model_request WHERE turn_id = ?').all(turnId) as Array<{ id: string }>;
    for (const request of requests) modelRequestIds.add(requireRuntimeId(request.id));
  }
  for (const modelRequestId of modelRequestIds) assertModelRequestAggregate(database, modelRequestId);
}

function assertModelRequestAggregate(database: Database.Database, modelRequestId: string): void {
  const requestRaw = database.prepare('SELECT * FROM model_request WHERE id = ?').get(modelRequestId);
  if (!requestRaw) throw new Error(`ModelRequest ${modelRequestId} does not exist.`);
  const request = DOMAIN_REPOSITORIES.codec('ModelRequest').decode(requestRaw as Record<string, unknown>);
  const identity = decodeModelStreamIdentity(request.stream_stats_json);
  const operations = database.prepare(
    "SELECT id, status FROM operation WHERE owner_kind = 'model_request' AND owner_id = ?"
  ).all(modelRequestId) as Array<{ id: string; status: string }>;
  if (operations.length !== 1) throw new Error(`ModelRequest ${modelRequestId} must own exactly one Operation.`);
  const operation = operations[0];
  const attempts = database.prepare(
    'SELECT id, attempt_seq, status, completed_at FROM attempt WHERE operation_id = ? ORDER BY attempt_seq'
  ).all(operation.id) as Array<{ id: string; attempt_seq: bigint; status: string; completed_at: string | null }>;
  if (attempts.length < 1 || attempts.length > 11) {
    throw new Error(`ModelRequest ${modelRequestId} must have between one and eleven Attempts.`);
  }
  attempts.forEach((attempt, index) => {
    if (attempt.attempt_seq !== BigInt(index + 1)) {
      throw new Error(`ModelRequest ${modelRequestId} Attempt sequence is not contiguous.`);
    }
  });
  const currentAttempt = attempts.find((attempt) => attempt.attempt_seq === identity.attemptSeq);
  if (!currentAttempt) throw new Error(`ModelRequest ${modelRequestId} stream identity has no matching Attempt.`);
  if (identity.attemptSeq !== BigInt(attempts.length)) {
    throw new Error(`ModelRequest ${modelRequestId} current Attempt must be the contiguous tail.`);
  }
  const priorAttempts = attempts.slice(0, -1);
  if (priorAttempts.some((attempt) => attempt.status !== 'transient_failed' || attempt.completed_at === null)) {
    throw new Error(`ModelRequest ${modelRequestId} prior Attempts must be durably transient_failed.`);
  }
  const fence = database.prepare('SELECT * FROM model_stream_fence WHERE model_request_id = ?').get(modelRequestId) as {
    attempt_seq?: unknown;
    socket_generation?: unknown;
    outcome?: unknown;
  } | undefined;
  const status = String(request.status);
  const terminalState = request.terminal_state;
  if (status !== 'terminal' && terminalState !== null) {
    throw new Error(`Non-terminal ModelRequest ${modelRequestId} cannot carry terminal_state.`);
  }
  if (identity.attemptSeq > 1n && (
    identity.retryMaxAttempts === undefined
    || identity.attemptSeq - 1n > BigInt(identity.retryMaxAttempts)
  )) {
    throw new Error(`ModelRequest ${modelRequestId} current Attempt exceeds its frozen retry budget.`);
  }
  if (status === 'prepared') {
    if (
      identity.attemptSeq !== 1n
      || identity.socketGeneration !== 0n
      || operation.status !== 'pending'
      || currentAttempt.status !== 'pending'
      || fence
    ) throw new Error(`Prepared ModelRequest ${modelRequestId} aggregate is inconsistent.`);
    return;
  }
  if (status === 'streaming') {
    if (
      identity.socketGeneration <= 0n
      || operation.status !== 'running'
      || currentAttempt.status !== 'running'
      || fence
    ) throw new Error(`Streaming ModelRequest ${modelRequestId} aggregate is inconsistent.`);
    return;
  }
  if (status === 'retrying') {
    if (
      identity.attemptSeq < 2n
      || identity.attemptSeq > 11n
      || identity.socketGeneration !== 0n
      || identity.retryDelayMs === undefined
      || identity.retryNotBeforeAt === undefined
      || operation.status !== 'running'
      || currentAttempt.status !== 'pending'
      || priorAttempts.length !== Number(identity.attemptSeq - 1n)
      || fence
    ) throw new Error(`Retrying ModelRequest ${modelRequestId} aggregate is inconsistent.`);
    return;
  }
  if (status !== 'terminal' || typeof terminalState !== 'string' || terminalState.length === 0) {
    throw new Error(`ModelRequest ${modelRequestId} has an unsupported aggregate status.`);
  }
  if (terminalState === 'completed') {
    if (
      operation.status !== 'completed'
      || currentAttempt.status !== 'completed'
      || fence?.attempt_seq !== identity.attemptSeq
      || fence.socket_generation !== identity.socketGeneration
      || fence.outcome !== 'completed'
    ) throw new Error(`Completed ModelRequest ${modelRequestId} aggregate is inconsistent.`);
    return;
  }
  if (fence) throw new Error(`Non-completed ModelRequest ${modelRequestId} cannot have a terminal fence.`);
  if (
    !['cancelled', 'failed'].includes(currentAttempt.status)
    || operation.status !== currentAttempt.status
  ) throw new Error(`Terminal ModelRequest ${modelRequestId} aggregate is inconsistent.`);
}

function assertRuntimeStateTransition(
  database: Database.Database,
  domain: string,
  id: string,
  patch: DomainRow
): void {
  if (domain === 'RuntimeDelivery') {
    const current = database.prepare('SELECT state FROM runtime_delivery WHERE id = ?').get(id) as {
      state?: unknown;
    } | undefined;
    if (!current || typeof current.state !== 'string') {
      throw new Error(`RuntimeDeliveryRepository update expected one row: ${id}`);
    }
    const nextState = 'state' in patch ? String(patch.state) : current.state;
    const allowed: Record<string, readonly string[]> = {
      pending: ['pending', 'consumed', 'failed'],
      consumed: [],
      failed: []
    };
    if (!allowed[current.state]?.includes(nextState)) {
      throw new Error(`RuntimeDelivery state cannot transition from ${current.state} to ${nextState}.`);
    }
    return;
  }
  if (domain === 'RuntimeDeliveryInputLink') {
    const current = database.prepare('SELECT handled_at FROM runtime_delivery_input_link WHERE id = ?').get(id) as {
      handled_at?: unknown;
    } | undefined;
    if (!current) throw new Error(`RuntimeDeliveryInputLinkRepository update expected one row: ${id}`);
    const nextHandledAt = 'handled_at' in patch ? patch.handled_at : current.handled_at;
    if (current.handled_at !== null || typeof nextHandledAt !== 'string' || nextHandledAt.length === 0) {
      throw new Error('RuntimeDeliveryInputLink.handled_at may only transition once from NULL to a timestamp.');
    }
    return;
  }
  if (domain === 'ProcessCompletionDispatch') {
    assertClaimedOutboxTransition(database, domain, 'process_completion_dispatch', id, patch, 'completed_at', 'completed');
    return;
  }
  if (domain === 'RuntimeDeliveryWake') {
    assertClaimedOutboxTransition(database, domain, 'runtime_delivery_wake', id, patch, 'acknowledged_at', 'acknowledged');
    return;
  }
  if (domain === 'ModelRequest') {
    const raw = database.prepare('SELECT * FROM model_request WHERE id = ?').get(id);
    if (!raw) throw new Error(`ModelRequestRepository update expected one row: ${id}`);
    const current = DOMAIN_REPOSITORIES.codec('ModelRequest').decode(raw as Record<string, unknown>);
    const currentStatus = String(current.status);
    const nextStatus = 'status' in patch ? String(patch.status) : currentStatus;
    const allowed: Record<string, readonly string[]> = {
      prepared: ['prepared', 'streaming', 'terminal'],
      streaming: ['streaming', 'retrying', 'terminal'],
      retrying: ['retrying', 'streaming', 'terminal'],
      terminal: []
    };
    if (!allowed[currentStatus]?.includes(nextStatus)) {
      throw new Error(`ModelRequest status cannot transition from ${currentStatus} to ${nextStatus}.`);
    }
    const terminalState = 'terminal_state' in patch ? patch.terminal_state : current.terminal_state;
    if (nextStatus === 'terminal') {
      if (typeof terminalState !== 'string' || terminalState.length === 0) {
        throw new Error('Terminal ModelRequest requires terminal_state.');
      }
    } else if (terminalState !== null) {
      throw new Error('Non-terminal ModelRequest cannot have terminal_state.');
    }
    if ('stream_stats_json' in patch) {
      const currentIdentity = decodeModelStreamIdentity(current.stream_stats_json);
      const nextIdentity = decodeModelStreamIdentity(patch.stream_stats_json);
      const sameAttempt = nextIdentity.attemptSeq === currentIdentity.attemptSeq;
      const sameRetryBudget = currentIdentity.retryMaxAttempts === nextIdentity.retryMaxAttempts;
      const oneRetry = nextIdentity.attemptSeq === currentIdentity.attemptSeq + 1n
        && nextIdentity.attemptSeq <= 11n
        && nextIdentity.socketGeneration === 0n
        && nextIdentity.retryMaxAttempts !== undefined
        && nextIdentity.attemptSeq - 1n <= BigInt(nextIdentity.retryMaxAttempts)
        && (currentIdentity.retryMaxAttempts === undefined || sameRetryBudget);
      if (
        (!sameAttempt && !oneRetry)
        || (sameAttempt && (!sameRetryBudget || nextIdentity.socketGeneration < currentIdentity.socketGeneration))
      ) {
        throw new Error('ModelRequest stream identity cannot move backwards or skip a bounded retry transition.');
      }
    }
    return;
  }
  if (domain !== 'Attempt' && domain !== 'Operation') return;
  const table = domain === 'Attempt' ? 'attempt' : 'operation';
  const ownerJoin = domain === 'Attempt'
    ? 'SELECT current.status, owner.owner_kind FROM attempt AS current JOIN operation AS owner ON owner.id = current.operation_id WHERE current.id = ?'
    : 'SELECT current.status, current.owner_kind FROM operation AS current WHERE current.id = ?';
  const current = database.prepare(ownerJoin).get(id) as { status?: unknown; owner_kind?: unknown } | undefined;
  if (!current || current.owner_kind !== 'model_request') return;
  const currentStatus = String(current.status);
  const nextStatus = 'status' in patch ? String(patch.status) : currentStatus;
  const allowed = domain === 'Attempt'
    ? {
        pending: ['pending', 'running', 'cancelled', 'failed'],
        running: ['running', 'transient_failed', 'completed', 'cancelled', 'failed'],
        transient_failed: [], completed: [], cancelled: [], failed: []
      } as Record<string, readonly string[]>
    : {
        pending: ['pending', 'running', 'cancelled', 'failed'],
        running: ['running', 'completed', 'cancelled', 'failed'],
        completed: [], cancelled: [], failed: []
      } as Record<string, readonly string[]>;
  if (!allowed[currentStatus]?.includes(nextStatus)) {
    throw new Error(`${domain} status cannot transition from ${currentStatus} to ${nextStatus} for a ModelRequest.`);
  }
}

function assertClaimedOutboxTransition(
  database: Database.Database,
  domain: 'ProcessCompletionDispatch' | 'RuntimeDeliveryWake',
  table: 'process_completion_dispatch' | 'runtime_delivery_wake',
  id: string,
  patch: DomainRow,
  terminalTimestampColumn: 'completed_at' | 'acknowledged_at',
  successState: 'completed' | 'acknowledged'
): void {
  const raw = database.prepare(`SELECT * FROM ${table} WHERE id = ?`).get(id);
  if (!raw) throw new Error(`${domain}Repository update expected one row: ${id}`);
  const current = DOMAIN_REPOSITORIES.codec(domain).decode(raw as Record<string, unknown>);
  const currentState = String(current.state);
  const nextState = 'state' in patch ? String(patch.state) : currentState;
  const allowed: Record<string, readonly string[]> = {
    pending: ['claimed', 'dead_letter'],
    claimed: ['pending', successState, 'dead_letter'],
    [successState]: [],
    dead_letter: []
  };
  if (!allowed[currentState]?.includes(nextState)) {
    throw new Error(`${domain} state cannot transition from ${currentState} to ${nextState}.`);
  }
  const next = { ...current, ...patch };
  const owner = next.claim_owner_host_boot_id;
  const expiresAt = next.claim_expires_at;
  const generation = next.claim_generation;
  const attempts = next.attempt_count;
  const failures = next.failure_count;
  if (
    typeof generation !== 'bigint' || generation < 0n
    || typeof attempts !== 'bigint' || attempts < 0n
    || typeof failures !== 'bigint' || failures < 0n
  ) throw new Error(`${domain} counters must be non-negative integers.`);
  if (nextState === 'claimed') {
    if (
      typeof owner !== 'string' || owner.length === 0
      || typeof expiresAt !== 'string' || expiresAt.length === 0
      || generation !== (current.claim_generation as bigint) + 1n
      || attempts !== (current.attempt_count as bigint) + 1n
    ) throw new Error(`${domain} claim must atomically fence one owner generation and attempt.`);
  } else if (owner !== null || expiresAt !== null) {
    throw new Error(`${domain} non-claimed state cannot retain a claim owner or expiry.`);
  }
  const terminalTimestamp = next[terminalTimestampColumn];
  if (nextState === successState) {
    if (typeof terminalTimestamp !== 'string' || terminalTimestamp.length === 0) {
      throw new Error(`${domain} ${successState} state requires ${terminalTimestampColumn}.`);
    }
  } else if (terminalTimestamp !== null) {
    throw new Error(`${domain} ${terminalTimestampColumn} is only valid in ${successState} state.`);
  }
  if (nextState === 'dead_letter' && (typeof next.last_error !== 'string' || next.last_error.length === 0)) {
    throw new Error(`${domain} dead_letter state requires last_error.`);
  }
}

function allocateNextSequence(
  database: Database.Database,
  repository: DomainRepository,
  mutation: RepositoryInsertMutation
): DomainRow {
  const allocation = mutation.allocateSequence;
  if (!allocation) return mutation.row;
  const column = repository.codec.column(allocation.column);
  const isSequence = allocation.column.endsWith('_seq');
  const isPendingInputPosition = repository.schema.key === 'PendingTurnInput' && allocation.column === 'position';
  if (!column || column.type !== 'INTEGER' || (!isSequence && !isPendingInputPosition)) {
    throw new Error(`${repository.name}.${allocation.column} is not an allocatable writer INTEGER.`);
  }
  if (allocation.column in mutation.row) throw new Error(`${repository.name}.${allocation.column} was supplied and allocated.`);
  const scope = repository.codec.encodeWhere(allocation.scope);
  const encodedRowScope = repository.codec.encodeWhere(
    Object.fromEntries(Object.keys(scope).map((name) => [name, mutation.row[name]]))
  );
  for (const name of Object.keys(scope)) {
    if (scope[name] !== encodedRowScope[name]) {
      throw new Error(`${repository.name} sequence scope does not match the inserted row: ${name}`);
    }
  }
  const predicates: string[] = [];
  const parameters: EncodedRow = {};
  for (const [name, value] of Object.entries(scope)) {
    if (value === null) predicates.push(`${quote(name)} IS NULL`);
    else {
      predicates.push(`${quote(name)} = @${name}`);
      parameters[name] = value;
    }
  }
  const sql = `SELECT COALESCE(MAX(${quote(allocation.column)}), 0) + 1 AS next_value FROM ${quote(repository.schema.table)}${predicates.length ? ` WHERE ${predicates.join(' AND ')}` : ''}`;
  const result = database.prepare(sql).get(parameters) as { next_value: bigint };
  if (typeof result.next_value !== 'bigint' || result.next_value <= 0n) throw new Error('SQLite sequence allocation failed.');
  return { ...mutation.row, [allocation.column]: result.next_value };
}

function matchesSavepointContinuation(error: unknown, onError: RepositorySavepointOnError): boolean {
  if (onError === 'propagate') return false;
  const value = error as { code?: unknown; message?: unknown };
  if (
    typeof value.code !== 'string'
    || !['SQLITE_CONSTRAINT_UNIQUE', 'SQLITE_CONSTRAINT_PRIMARYKEY'].includes(value.code)
    || typeof value.message !== 'string'
  ) return false;
  const marker = 'UNIQUE constraint failed:';
  const markerIndex = value.message.indexOf(marker);
  if (markerIndex < 0) return false;
  const actualColumns = value.message
    .slice(markerIndex + marker.length)
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .sort();
  return onError.constraints.some((constraint) => {
    const table = DOMAIN_REPOSITORIES.domain(constraint.domain).schema.table;
    const expectedColumns = constraint.columns.map((column) => `${table}.${column}`).sort();
    return actualColumns.length === expectedColumns.length
      && actualColumns.every((column, index) => column === expectedColumns[index]);
  });
}

function whereClause(encoded: EncodedRow): { predicates: string[]; parameters: EncodedRow } {
  const predicates: string[] = [];
  const parameters: EncodedRow = {};
  for (const [name, value] of Object.entries(encoded)) {
    if (value === null) predicates.push(`${quote(name)} IS NULL`);
    else {
      predicates.push(`${quote(name)} = @${name}`);
      parameters[name] = value;
    }
  }
  return { predicates, parameters };
}

function executeContextMaterialization(
  database: Database.Database,
  rootId: string,
  commitSeq: bigint
): SnapshotBarrier<ContextMaterializationSnapshot> {
  const normalizedRootId = requireRuntimeId(rootId);
  database.exec('BEGIN');
  try {
    const rootRepository = DOMAIN_REPOSITORIES.domain('ContextSequenceRoot');
    const rawRoot = database.prepare('SELECT * FROM context_sequence_root WHERE id = ?').get(normalizedRootId);
    if (!rawRoot) throw new Error(`ContextSequenceRoot ${normalizedRootId} does not exist.`);
    const root = rootRepository.codec.decode(rawRoot as Record<string, unknown>);
    const rootNodeId = nullableRuntimeId(root.root_node_id, 'ContextSequenceRoot.root_node_id');
    const tailNodeId = nullableRuntimeId(root.tail_node_id, 'ContextSequenceRoot.tail_node_id');
    const tailCount = nonNegativeSafeInteger(root.tail_segment_count, 'ContextSequenceRoot.tail_segment_count');
    const segmentCount = nonNegativeSafeInteger(root.segment_count, 'ContextSequenceRoot.segment_count');
    let records: ContextMaterializationRecord[] = [];
    if (rootNodeId === null) {
      if (tailNodeId !== null || tailCount !== 0 || segmentCount !== 0) {
        throw new Error(`ContextSequenceRoot ${normalizedRootId} has an invalid empty shape.`);
      }
    } else {
      const rootRecord = readContextRecord(database, rootNodeId);
      if (rootRecord.segment.segment_kind === 'compression') {
        if (rootRecord.node.parent_node_id !== null) {
          throw new Error(`Compression root ${normalizedRootId} summary node must not have a parent.`);
        }
        if ((tailCount === 0) !== (tailNodeId === null)) {
          throw new Error(`Compression root ${normalizedRootId} tail pointer/count mismatch.`);
        }
        const tail = tailNodeId === null ? [] : readContextChain(database, tailNodeId, tailCount);
        records = [rootRecord, ...tail];
        if (records.length !== segmentCount) {
          throw new Error(`Compression root ${normalizedRootId} segment_count mismatch.`);
        }
      } else {
        if (tailNodeId !== null || tailCount !== 0) {
          throw new Error(`Ordinary root ${normalizedRootId} must not carry a compression tail.`);
        }
        records = readContextChain(database, rootNodeId, segmentCount);
        if (records[0]?.node.parent_node_id !== null) {
          throw new Error(`Ordinary root ${normalizedRootId} chain does not terminate at NULL.`);
        }
      }
    }
    database.exec('COMMIT');
    return { snapshotCommitSeq: commitSeq.toString(), snapshot: { root, records } };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function attachContextContent(
  barrier: SnapshotBarrier<ContextMaterializationSnapshot>,
  casRootPath: string,
  cache: VerifiedContextCasCache
): {
  result: SnapshotBarrier<ContextContentMaterializationSnapshot>;
  transferList: ArrayBuffer[];
} {
  const rootPath = path.resolve(casRootPath);
  const unique = new Map<string, Buffer>();
  let totalBytes = 0;
  for (const record of barrier.snapshot.records) {
    const metadata = record.contentObject;
    const id = requireRuntimeId(metadata.id);
    if (unique.has(id)) continue;
    const bytes = cache.read(metadata, rootPath);
    totalBytes += bytes.length;
    if (!Number.isSafeInteger(totalBytes)) throw new RangeError('Materialized Context bytes exceed the safe packed-buffer range.');
    unique.set(id, bytes);
  }
  const packed = new Uint8Array(totalBytes);
  const views = new Map<string, Uint8Array>();
  let offset = 0;
  for (const [id, bytes] of unique) {
    packed.set(bytes, offset);
    views.set(id, packed.subarray(offset, offset + bytes.length));
    offset += bytes.length;
  }
  return {
    result: {
      snapshotCommitSeq: barrier.snapshotCommitSeq,
      snapshot: {
        root: barrier.snapshot.root,
        records: barrier.snapshot.records.map((record) => ({
          ...record,
          content: views.get(requireRuntimeId(record.contentObject.id)) as Uint8Array
        }))
      }
    },
    transferList: packed.byteLength > 0 ? [packed.buffer] : []
  };
}

function contextCasIdentity(metadata: DomainRow): Omit<VerifiedContextCasCacheEntry, 'bytes'> {
  const id = requireRuntimeId(metadata.id);
  const sha256 = typeof metadata.sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.sha256)
    ? metadata.sha256
    : (() => { throw new Error(`ContentObject ${id} has an invalid sha256.`); })();
  const byteLength = typeof metadata.byte_length === 'bigint' && metadata.byte_length >= 0n
    ? metadata.byte_length
    : (() => { throw new Error(`ContentObject ${id} has an invalid byte length.`); })();
  const storageKey = `sha256/${sha256.slice(0, 2)}/${sha256}`;
  if (metadata.storage_key !== storageKey) throw new Error(`ContentObject ${id} storage key does not match sha256.`);
  return { id, sha256, byteLength, storageKey };
}

function assertSameContextCasIdentity(
  cached: VerifiedContextCasCacheEntry,
  current: Omit<VerifiedContextCasCacheEntry, 'bytes'>
): void {
  if (
    cached.sha256 !== current.sha256
    || cached.byteLength !== current.byteLength
    || cached.storageKey !== current.storageKey
  ) {
    throw new Error(`ContentObject ${current.id} metadata changed during one Runtime worker lifetime.`);
  }
}

function readVerifiedCasBytes(metadata: DomainRow, resolvedCasRootPath: string): Buffer {
  const id = requireRuntimeId(metadata.id);
  const sha256 = typeof metadata.sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.sha256)
    ? metadata.sha256
    : (() => { throw new Error(`ContentObject ${id} has an invalid sha256.`); })();
  const expectedKey = `sha256/${sha256.slice(0, 2)}/${sha256}`;
  if (metadata.storage_key !== expectedKey) throw new Error(`ContentObject ${id} storage key does not match sha256.`);
  if (!path.isAbsolute(resolvedCasRootPath)) throw new Error('CAS root must be resolved before verified reads.');
  // The path segments are derived only from a validated lowercase SHA-256, so no per-object resolve
  // or traversal check is needed on this 1000-record materialization hot path.
  const candidate = path.join(resolvedCasRootPath, 'sha256', sha256.slice(0, 2), sha256);
  const bytes = fs.readFileSync(candidate);
  if (typeof metadata.byte_length !== 'bigint' || BigInt(bytes.length) !== metadata.byte_length) {
    throw new Error(`ContentObject ${id} byte length mismatch.`);
  }
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) {
    throw new Error(`ContentObject ${id} digest mismatch.`);
  }
  return bytes;
}

function readContextChain(
  database: Database.Database,
  startNodeId: string,
  count: number
): ContextMaterializationRecord[] {
  if (count <= 0) throw new Error('Context chain with a start node requires a positive segment count.');
  const rows = database.prepare(`
    WITH RECURSIVE chain(id, parent_node_id, segment_id, created_at, depth) AS (
      SELECT id, parent_node_id, segment_id, created_at, 1
        FROM context_sequence_node
       WHERE id = @startNodeId
      UNION ALL
      SELECT parent.id, parent.parent_node_id, parent.segment_id, parent.created_at, chain.depth + 1
        FROM context_sequence_node AS parent
        JOIN chain ON parent.id = chain.parent_node_id
       WHERE chain.depth < @segmentCount
    )
    SELECT chain.id AS node_id,
           chain.parent_node_id AS node_parent_node_id,
           chain.segment_id AS node_segment_id,
           chain.created_at AS node_created_at,
           segment.id AS segment_id,
           segment.content_object_id AS segment_content_object_id,
           segment.segment_kind AS segment_kind,
           segment.created_at AS segment_created_at,
           content.id AS content_id,
           content.content_type AS content_type,
           content.sha256 AS content_sha256,
           content.byte_length AS content_byte_length,
           content.storage_key AS content_storage_key,
           content.created_at AS content_created_at,
           chain.depth AS depth
      FROM chain
      JOIN context_segment AS segment ON segment.id = chain.segment_id
      JOIN content_object AS content ON content.id = segment.content_object_id
     ORDER BY chain.depth DESC
  `).all({ startNodeId, segmentCount: BigInt(count) }) as Array<Record<string, unknown>>;
  if (rows.length !== count) throw new Error(`Context chain expected ${count} nodes, found ${rows.length}.`);
  return decodeContextRecords(database, rows);
}

function readContextRecord(database: Database.Database, nodeId: string): ContextMaterializationRecord {
  const row = database.prepare(`
    SELECT node.id AS node_id,
           node.parent_node_id AS node_parent_node_id,
           node.segment_id AS node_segment_id,
           node.created_at AS node_created_at,
           segment.id AS segment_id,
           segment.content_object_id AS segment_content_object_id,
           segment.segment_kind AS segment_kind,
           segment.created_at AS segment_created_at,
           content.id AS content_id,
           content.content_type AS content_type,
           content.sha256 AS content_sha256,
           content.byte_length AS content_byte_length,
           content.storage_key AS content_storage_key,
           content.created_at AS content_created_at,
           1 AS depth
      FROM context_sequence_node AS node
      JOIN context_segment AS segment ON segment.id = node.segment_id
      JOIN content_object AS content ON content.id = segment.content_object_id
     WHERE node.id = ?
  `).get(nodeId) as Record<string, unknown> | undefined;
  if (!row) throw new Error(`ContextSequenceNode ${nodeId} does not exist or has missing content.`);
  return decodeContextRecords(database, [row])[0];
}

function decodeContextRecords(
  database: Database.Database,
  rows: Array<Record<string, unknown>>
): ContextMaterializationRecord[] {
  const messageSegmentIds = rows
    .filter((row) => row.segment_kind === 'message')
    .map((row) => requireRuntimeId(row.segment_id));
  const roles = new Map<string, string[]>();
  for (let offset = 0; offset < messageSegmentIds.length; offset += 500) {
    const chunk = messageSegmentIds.slice(offset, offset + 500);
    const placeholders = chunk.map(() => '?').join(',');
    const sourceRows = database.prepare(`
      SELECT DISTINCT source.segment_id AS segment_id, revision.role AS role
        FROM context_segment_source AS source
        JOIN message_revision AS revision
          ON revision.id = source.source_id
         AND revision.revision_seq = source.source_revision
       WHERE source.source_kind = 'message_revision'
         AND source.segment_id IN (${placeholders})
       ORDER BY source.segment_id, source.id
    `).all(...chunk) as Array<{ segment_id: string; role: string }>;
    for (const source of sourceRows) {
      const segmentId = requireRuntimeId(source.segment_id);
      if (typeof source.role !== 'string' || source.role.length === 0) {
        throw new Error(`Message ContextSegment ${segmentId} has an invalid role.`);
      }
      const current = roles.get(segmentId) ?? [];
      current.push(source.role);
      roles.set(segmentId, current);
    }
  }
  return rows.map((row) => decodeContextRecord(row, roles.get(String(row.segment_id)) ?? []));
}

function decodeContextRecord(
  row: Record<string, unknown>,
  messageRoles: readonly string[]
): ContextMaterializationRecord {
  const segmentKind = typeof row.segment_kind === 'string' ? row.segment_kind : '';
  let messageRole: string | null = null;
  if (segmentKind === 'message') {
    if (messageRoles.length !== 1) {
      throw new Error(`Message ContextSegment ${String(row.segment_id)} must resolve exactly one immutable MessageRevision role.`);
    }
    messageRole = messageRoles[0];
  }
  return {
    node: DOMAIN_REPOSITORIES.codec('ContextSequenceNode').decode({
      id: row.node_id,
      parent_node_id: row.node_parent_node_id,
      segment_id: row.node_segment_id,
      created_at: row.node_created_at
    }),
    segment: DOMAIN_REPOSITORIES.codec('ContextSegment').decode({
      id: row.segment_id,
      content_object_id: row.segment_content_object_id,
      segment_kind: row.segment_kind,
      created_at: row.segment_created_at
    }),
    contentObject: DOMAIN_REPOSITORIES.codec('ContentObject').decode({
      id: row.content_id,
      content_type: row.content_type,
      sha256: row.content_sha256,
      byte_length: row.content_byte_length,
      storage_key: row.content_storage_key,
      created_at: row.content_created_at
    }),
    messageRole
  };
}

function nullableRuntimeId(value: unknown, label: string): string | null {
  if (value === null) return null;
  if (typeof value !== 'string' || value.length === 0) throw new TypeError(`${label} must be a non-empty string or NULL.`);
  return value;
}

function nonNegativeSafeInteger(value: unknown, label: string): number {
  if (typeof value !== 'bigint' || value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new TypeError(`${label} must be a non-negative safe SQLite INTEGER.`);
  }
  return Number(value);
}

function executeClientProjectionSnapshot(
  database: Database.Database,
  activeConversationId: string | null,
  commitSeq: bigint
): SnapshotBarrier<ClientProjectionSnapshot> {
  const conversationId = activeConversationId === null ? null : requireRuntimeId(activeConversationId);
  database.exec('BEGIN');
  try {
    const conversations = queryPlainRows(database, `
      SELECT id, title, status, created_at, updated_at
        FROM conversation
       ORDER BY updated_at DESC, id DESC
       LIMIT @limit
    `, { limit: BigInt(CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE) });
    const emptyWindow = {
      conversationId,
      messages: [],
      visibleMessageCount: '0',
      lastMessageSeq: '0',
      projectContexts: [],
      conversationProjectLinks: [],
      conversationReuseLinks: [],
      conversationBranchLinks: [],
      conversationOriginLinks: [],
      agentConversationLinks: [],
      commandReceipts: [],
      queuedTurnIntents: [],
      compressionBlocks: [],
      conversationContextStatuses: [],
      taskList: [],
      currentTaskList: null
    };
    const emptyTurns = {
      turns: [], executionLeases: [], turnTerminations: [], turnExecutorLinks: [], modelRequests: [],
      modelRequestMessageLinks: []
    };
    const emptyTools = {
      messageTurnLinks: [],
      toolCalls: [], toolCallSourceLinks: [], toolCallPolicySnapshots: [], toolCallEvents: [],
      toolExecutions: [], toolOutcomes: [], toolModelResults: [],
      toolResultArtifacts: [], interactionRequests: [], interactionOwnerLinks: [], interactionToolCallLinks: [],
      interactionResponses: [],
      fileChangeSets: [], fileChangeSetMembers: [], fileChangeDecisions: [], fileMutationReceipts: [],
      fileMutationReceiptMembers: [], processes: [], processOriginLinks: [], processOutputChunks: [],
      processReceipts: []
    };
    const emptySubagents = {
      childExecutions: [], childExecutionParentLinks: [], childExecutionTurnLinks: [],
      childExecutionActiveTurnLinks: [], childTurns: [], childExecutionLeases: [], childTurnTerminations: [], childTurnExecutorLinks: [],
      childExecutionActivities: [],
      answerBridges: [], answerSubmissions: [],
      runtimeInboxItems: [], runtimeDeliveries: []
    };
    if (conversationId === null) {
      database.exec('COMMIT');
      return {
        snapshotCommitSeq: commitSeq.toString(),
        snapshot: {
          navigationSummary: { conversations },
          activeConversationWindow: emptyWindow,
          activeTurnSummary: emptyTurns,
          activeToolAndInteractionSummary: emptyTools,
          subagentDeliverySummary: emptySubagents
        }
      };
    }

    const params = { conversationId, limit: BigInt(CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE) };
    const commandReceipts = queryPlainRows(database, `
      SELECT id,
             conversation_id,
             source_key AS command_id,
             created_at
        FROM command_receipt
       WHERE conversation_id = @conversationId
         AND source_kind = 'command'
       ORDER BY created_at DESC, id DESC
       LIMIT @limit
    `, params);
    const queuedTurnIntents = queryPlainRows(database, `
      SELECT intent.*,
             (
               SELECT CAST(revision.revision_seq AS TEXT)
                 FROM turn_intent_revision AS revision
                WHERE revision.intent_id = intent.id
                ORDER BY revision.revision_seq DESC
                LIMIT 1
             ) AS current_revision_seq
        FROM turn_intent AS intent
       WHERE intent.conversation_id = @conversationId
         AND intent.state = 'queued'
         AND intent.turn_id IS NULL
         AND NOT EXISTS (
           SELECT 1
             FROM child_execution_intent_link AS child_link
            WHERE child_link.turn_intent_id = intent.id
         )
       ORDER BY intent.created_at ASC, intent.id ASC
       LIMIT @limit
    `, params);
    const conversationProjectLinks = queryPlainRows(database, `
      SELECT * FROM conversation_project_link
       WHERE conversation_id = @conversationId
       ORDER BY updated_at DESC, id DESC LIMIT @limit
    `, params);
    const projectContexts = queryByIds(
      database,
      'project_context',
      'id',
      conversationProjectLinks.map((row) => String(row.project_context_id))
    );
    const rawMessageRows = queryPlainRows(database, `
      SELECT m.id,
             membership.conversation_id,
             membership.message_seq,
             m.created_at,
             m.updated_at,
             m.deleted_at,
             revision.id AS revision_id,
             revision.revision_seq,
             revision.role,
             revision.content_object_id,
             content.content_type,
             content.byte_length
        FROM message_part_of_conversation AS membership
        JOIN message AS m ON m.id = membership.message_id
        JOIN message_current_revision_link AS current_revision ON current_revision.message_id = m.id
        JOIN message_revision AS revision ON revision.id = current_revision.revision_id
        JOIN content_object AS content ON content.id = revision.content_object_id
       WHERE membership.conversation_id = @conversationId
         AND m.deleted_at IS NULL
         AND revision.role IN ('user', 'model')
       ORDER BY membership.message_seq DESC, m.id DESC
       LIMIT @messageLimit
    `, { conversationId, messageLimit: BigInt(CLIENT_MESSAGE_WINDOW_LIMIT) }).reverse();
    const messageSummary = database.prepare(`
      SELECT COUNT(CASE WHEN m.deleted_at IS NULL AND revision.role IN ('user', 'model') THEN 1 END) AS visible_message_count,
             COALESCE(MAX(membership.message_seq), 0) AS last_message_seq
        FROM message_part_of_conversation AS membership
        JOIN message AS m ON m.id = membership.message_id
        JOIN message_current_revision_link AS current_revision ON current_revision.message_id = m.id
        JOIN message_revision AS revision ON revision.id = current_revision.revision_id
       WHERE membership.conversation_id = ?
    `).get(conversationId) as { visible_message_count: bigint; last_message_seq: bigint };
    const visibleMessageCount = messageSummary.visible_message_count;
    let visibleFloor = visibleMessageCount - BigInt(rawMessageRows.length);
    let messageRows: Array<Record<string, unknown>> = rawMessageRows.map((row) => {
      visibleFloor += 1n;
      return { ...row, display_seq: visibleFloor };
    });
    const reuseLinks = queryPlainRows(database, `
      SELECT * FROM conversation_reuse_link
       WHERE conversation_id = @conversationId
       ORDER BY created_at DESC, id DESC LIMIT @limit
    `, params);
    const branchLinks = queryPlainRows(database, `
      SELECT * FROM conversation_branch_link
       WHERE target_conversation_id = @conversationId OR source_conversation_id = @conversationId
       ORDER BY created_at DESC, id DESC LIMIT @limit
    `, params);
    const originLinks = queryPlainRows(database, `
      SELECT * FROM conversation_origin_link
       WHERE conversation_id = @conversationId OR source_conversation_id = @conversationId
       ORDER BY created_at DESC, id DESC LIMIT @limit
    `, params);
    const agentConversationLinks = queryPlainRows(database, `
      SELECT * FROM agent_conversation_link
       WHERE conversation_id = @conversationId
       ORDER BY updated_at DESC, id DESC LIMIT @limit
    `, params);
    const compressionBlocks = queryPlainRows(database, `
      SELECT block.*,
             COUNT(source.id) AS source_count,
             (
               SELECT revision.message_id
                 FROM compression_block_source AS anchor_source
                 JOIN context_segment_source AS segment_source
                   ON segment_source.segment_id = anchor_source.segment_id
                  AND segment_source.source_kind = 'message_revision'
                 JOIN message_revision AS revision
                   ON revision.id = segment_source.source_id
                 JOIN message_part_of_conversation AS anchor_membership
                   ON anchor_membership.message_id = revision.message_id
                  AND anchor_membership.conversation_id = block.conversation_id
                WHERE anchor_source.compression_block_id = block.id
                ORDER BY anchor_source.position DESC, anchor_source.id DESC
                LIMIT 1
             ) AS anchor_message_id
        FROM compression_block AS block
        LEFT JOIN compression_block_source AS source
          ON source.compression_block_id = block.id
       WHERE block.conversation_id = @conversationId
       GROUP BY block.id
       ORDER BY block.created_at DESC, block.id DESC
       LIMIT @limit
    `, params).reverse();
    const conversationContextStatuses = queryPlainRows(database, `
      SELECT head.id,
             head.conversation_id,
             head.root_id,
             root.root_seq,
             root.segment_count,
             root.estimated_tokens,
             root.created_at AS root_created_at,
             head.updated_at
        FROM conversation_context_head_link AS head
        JOIN context_sequence_root AS root ON root.id = head.root_id
       WHERE head.conversation_id = @conversationId
       ORDER BY head.updated_at DESC, head.id DESC
       LIMIT 1
    `, params);
    const currentTaskList = projectCurrentTaskList(database, conversationId);
    let turns = queryClientRootTurns(database, conversationId)
      .map((turn) => turn.status === 'active'
        ? projectTurnClientRecord(database, String(turn.id))
        : turn);

    // Processes and child executions are independently visible summaries. Their active rows are
    // pinned even after their source Message leaves the normal 200-message suffix. The source
    // ToolCall bundle is added below so these roots never point at a clipped owner.
    let processRows = queryClientProcesses(database, conversationId)
      .map((row) => projectProcessRecord(database, String(row.id)));
    let processIds = processRows.map((row) => String(row.id));
    let processOriginLinks = queryAllByIds(database, 'process_origin_link', 'process_id', processIds);

    let childExecutions = queryClientChildExecutions(database, conversationId);
    let childIds = childExecutions.map((row) => String(row.id));
    let childParentLinks = queryAllByIds(database, 'child_execution_parent_link', 'child_execution_id', childIds);
    const childIdsSourcedFromActiveConversation = new Set(childExecutions
      .filter((row) => row.child_conversation_id !== conversationId)
      .map((row) => String(row.id)));

    const visibleSourceLinks = queryAllByIds(
      database,
      'tool_call_source_link',
      'message_id',
      messageRows.map((row) => String(row.id))
    );
    const pendingInteractionToolLinks = queryPlainRows(database, `
      SELECT tool_link.*
        FROM interaction_tool_call_link AS tool_link
        JOIN interaction_request AS request ON request.id = tool_link.request_id
        JOIN interaction_owner_link AS owner ON owner.request_id = request.id
        JOIN turn ON turn.id = owner.turn_id
       WHERE turn.conversation_id = @conversationId
         AND request.status = 'pending'
       ORDER BY request.created_at ASC, request.id ASC
    `, { conversationId });
    const nonterminalToolCalls = queryPlainRows(database, `
      SELECT call.*
        FROM tool_call AS call
        JOIN turn ON turn.id = call.turn_id
       WHERE turn.conversation_id = @conversationId
         AND call.status <> 'terminal'
       ORDER BY turn.created_at ASC, call.call_seq ASC, call.id ASC
    `, { conversationId });
    const toolCalls = mergeRowsById([
      ...nonterminalToolCalls,
      ...queryAllByIds(database, 'tool_call', 'id', [
        ...visibleSourceLinks.map((row) => String(row.tool_call_id)),
        ...processOriginLinks.map((row) => String(row.tool_call_id)),
        ...childParentLinks
          .filter((row) => childIdsSourcedFromActiveConversation.has(String(row.child_execution_id)))
          .map((row) => String(row.source_tool_call_id)),
        ...pendingInteractionToolLinks.map((row) => String(row.tool_call_id)),
        ...(currentTaskList ? [String(currentTaskList.sourceToolCallId)] : [])
      ])
    ]).sort(compareToolCallRows);
    const toolCallIds = toolCalls.map((row) => String(row.id));

    // Historical Process/Child summaries are retained only while their source ToolCall is in the
    // visible closure. Running/active roots were already included above and therefore remain pinned
    // even if their source Message is older than the ordinary suffix.
    processOriginLinks = mergeRowsById([
      ...processOriginLinks,
      ...queryAllByIds(database, 'process_origin_link', 'tool_call_id', toolCallIds)
    ]);
    processIds = [...new Set(processOriginLinks.map((row) => String(row.process_id)))];
    processRows = queryAllByIds(database, 'process', 'id', processIds)
      .map((row) => projectProcessRecord(database, String(row.id)))
      .sort((left, right) => String(right.started_at).localeCompare(String(left.started_at)) || String(right.id).localeCompare(String(left.id)));
    const processReceipts = queryAllByIds(database, 'process_receipt', 'process_id', processIds);

    childParentLinks = mergeRowsById([
      ...childParentLinks,
      ...queryAllByIds(database, 'child_execution_parent_link', 'source_tool_call_id', toolCallIds)
    ]);
    childIds = [...new Set(childParentLinks.map((row) => String(row.child_execution_id)))];
    childExecutions = mergeRowsById([
      ...childExecutions,
      ...queryAllByIds(database, 'child_execution', 'id', childIds)
    ]).sort((left, right) => String(right.created_at).localeCompare(String(left.created_at)) || String(right.id).localeCompare(String(left.id)));
    const childConversationIds = childExecutions.map((row) => String(row.child_conversation_id));
    const toolCallSourceLinks = queryAllByIds(database, 'tool_call_source_link', 'tool_call_id', toolCallIds);

    // Nonterminal/background/child roots may originate before the ordinary Message suffix. Pin
    // their exact visible source Message and recompute its absolute visible display rank.
    messageRows = mergeRowsById([
      ...messageRows,
      ...queryVisibleMessageRowsByIds(
        database,
        conversationId,
        toolCallSourceLinks.map((row) => String(row.message_id))
      )
    ]).sort(compareMessageWindowRows);

    const messageTurnLinks = queryAllByIds(
      database,
      'message_turn_link',
      'message_id',
      messageRows.map((row) => String(row.id))
    );
    const modelRequests = mergeRowsById([
      ...queryConversationModelRequests(database, conversationId, messageRows.map((row) => String(row.id))),
      ...queryAllByIds(
        database,
        'model_request',
        'id',
        toolCallSourceLinks.map((row) => String(row.model_request_id))
      )
    ]).sort(compareModelRequestRows);
    const modelRequestMessageLinks = queryAllByIds(
      database,
      'model_request_message_link',
      'model_request_id',
      modelRequests.map((row) => String(row.id))
    ).filter((row) => messageRows.some((message) => message.id === row.message_id));

    // A retained source bundle owns its Turn even when that Turn is older than the normal Turn
    // summary. This is the reverse closure missing from the old per-type query.
    turns = mergeRowsById([
      ...turns,
      ...queryAllByIds(database, 'turn', 'id', [
        ...toolCalls.map((row) => String(row.turn_id)),
        ...modelRequests.map((row) => String(row.turn_id)),
        ...messageTurnLinks.map((row) => String(row.turn_id))
      ])
    ]).sort(compareTurnRows);
    const turnIds = turns.map((row) => String(row.id));
    const leases = queryAllByIds(database, 'execution_lease', 'turn_id', turnIds);
    const terminations = queryAllByIds(database, 'turn_termination', 'turn_id', turnIds);
    const executorLinks = queryAllByIds(database, 'turn_executor_link', 'turn_id', turnIds);

    const toolCallPolicySnapshots = queryAllByIds(database, 'tool_call_policy_snapshot', 'tool_call_id', toolCallIds);
    const toolCallEvents = queryLatestToolCallEvents(database, toolCallIds);
    const toolExecutions = queryAllByIds(database, 'tool_execution', 'tool_call_id', toolCallIds);
    const toolOutcomes = queryAllByIds(database, 'tool_outcome', 'tool_call_id', toolCallIds);
    const toolModelResults = queryAllByIds(database, 'tool_model_result', 'tool_call_id', toolCallIds);
    const toolResultArtifacts = queryAllByIds(database, 'tool_result_artifact', 'tool_call_id', toolCallIds);
    const fileChangeSets = queryAllByIds(database, 'file_change_set', 'tool_call_id', toolCallIds);
    const fileChangeSetIds = fileChangeSets.map((row) => String(row.id));
    const fileChangeSetMembers = queryAllByIds(database, 'file_change_set_member', 'change_set_id', fileChangeSetIds);
    const fileChangeDecisions = queryAllByIds(database, 'file_change_decision', 'change_set_id', fileChangeSetIds);
    const fileMutationReceipts = queryAllByIds(database, 'file_mutation_receipt', 'change_set_id', fileChangeSetIds);
    const fileMutationReceiptMembers = queryAllByIds(
      database,
      'file_mutation_receipt_member',
      'receipt_id',
      fileMutationReceipts.map((row) => String(row.id))
    );
    const taskListCalls = queryPlainRows(database, `
      SELECT call.*
        FROM tool_call AS call
        JOIN turn ON turn.id = call.turn_id
       WHERE turn.conversation_id = @conversationId
         AND call.tool_name = 'update_task_list'
       ORDER BY turn.created_at DESC, call.call_seq DESC, call.id DESC
       LIMIT @limit
    `, params).reverse();
    const taskListOutcomes = queryAllByIds(
      database,
      'tool_outcome',
      'tool_call_id',
      taskListCalls.map((row) => String(row.id))
    );
    const taskList = taskListCalls
      .map((row) => {
        const outcome = taskListOutcomes.find((candidate) => candidate.tool_call_id === row.id) ?? null;
        return {
          tool_call_id: row.id,
          turn_id: row.turn_id,
          call_seq: row.call_seq,
          state: row.status,
          outcome: outcome?.status ?? null,
          ...taskListProjectionFromOutcome(database, outcome, String(row.id))
        };
      });
    const selectedInteractionToolCallLinks = mergeRowsById([
      ...pendingInteractionToolLinks,
      ...queryAllByIds(database, 'interaction_tool_call_link', 'tool_call_id', toolCallIds)
    ]);
    const pendingInteractionRequests = queryPlainRows(database, `
      SELECT request.*
        FROM interaction_request AS request
        JOIN interaction_owner_link AS owner ON owner.request_id = request.id
        JOIN turn ON turn.id = owner.turn_id
       WHERE turn.conversation_id = @conversationId
         AND request.status = 'pending'
       ORDER BY request.created_at ASC, request.id ASC
    `, { conversationId });
    const interactionRequestIds = [...new Set([
      ...selectedInteractionToolCallLinks.map((row) => String(row.request_id)),
      ...pendingInteractionRequests.map((row) => String(row.id))
    ])];
    const interactionRequests = mergeRowsById([
      ...pendingInteractionRequests,
      ...queryAllByIds(database, 'interaction_request', 'id', interactionRequestIds)
    ]);
    const interactionOwnerLinks = queryAllByIds(database, 'interaction_owner_link', 'request_id', interactionRequestIds);
    const allInteractionToolCallLinks = queryAllByIds(database, 'interaction_tool_call_link', 'request_id', interactionRequestIds)
      .filter((row) => toolCallIds.includes(String(row.tool_call_id)));
    const interactionResponses = queryAllByIds(database, 'interaction_response', 'request_id', interactionRequestIds);

    const projectedAgentConversationLinks = [
      ...agentConversationLinks,
      ...queryAllByIds(database, 'agent_conversation_link', 'conversation_id', childConversationIds)
    ].filter((row, index, rows) => rows.findIndex((candidate) => candidate.id === row.id) === index);
    const childActiveLinks = queryAllByIds(database, 'child_execution_active_turn_link', 'child_execution_id', childIds);
    const childTurnLinks = queryLatestChildExecutionTurnLinks(database, childIds);
    const childTurnIds = [...new Set([
      ...childTurnLinks.map((row) => String(row.turn_id)),
      ...childActiveLinks.map((row) => String(row.turn_id))
    ])];
    const childTurns = queryAllByIds(database, 'turn', 'id', childTurnIds);
    const childExecutionLeases = queryAllByIds(database, 'execution_lease', 'turn_id', childTurnIds);
    const childTurnTerminations = queryAllByIds(database, 'turn_termination', 'turn_id', childTurnIds);
    const childTurnExecutorLinks = queryAllByIds(database, 'turn_executor_link', 'turn_id', childTurnIds);
    const childExecutionActivities = childExecutions.flatMap((child) => {
      const activity = projectChildExecutionActivityRecord(database, String(child.id));
      return activity ? [activity] : [];
    });
    const answerBridges = queryAllByIds(database, 'answer_bridge', 'child_execution_id', childIds)
      .map((bridge) => projectAnswerBridgeRecord(database, String(bridge.id)));
    const bridgeIds = answerBridges.map((row) => String(row.id));
    const answerSubmissions = queryAllByIds(
      database,
      'answer_submission',
      'id',
      answerBridges.flatMap((row) => row.current_submission_id ? [String(row.current_submission_id)] : [])
    );
    const deliveries = queryClientRuntimeDeliveries(database, conversationId);
    const deliveryIds = deliveries.map((row) => String(row.id));
    const deliveryInputLinks = queryAllByIds(database, 'runtime_delivery_input_link', 'delivery_id', deliveryIds);
    const projectedDeliveries = deliveries.map((delivery) => {
      const matching = deliveryInputLinks.filter((link) => link.delivery_id === delivery.id);
      if (matching.length > 1) throw new Error(`RuntimeDelivery ${String(delivery.id)} has multiple input links.`);
      return {
        ...delivery,
        parent_handling_state: deriveCommittedParentHandling(
          delivery,
          matching[0] ? { handled_at: matching[0].handled_at as string | null } : null
        )
      };
    });
    const inboxIds = deliveries.map((row) => String(row.inbox_item_id));
    const inboxItems = queryAllByIds(database, 'runtime_inbox_item', 'id', inboxIds);

    const snapshot: ClientProjectionSnapshot = {
      navigationSummary: { conversations },
      activeConversationWindow: {
        conversationId,
        messages: messageRows,
        visibleMessageCount,
        lastMessageSeq: messageSummary.last_message_seq,
        projectContexts,
        conversationProjectLinks,
        conversationReuseLinks: reuseLinks,
        conversationBranchLinks: branchLinks,
        conversationOriginLinks: originLinks,
        agentConversationLinks: projectedAgentConversationLinks,
        commandReceipts,
        queuedTurnIntents,
        compressionBlocks,
        conversationContextStatuses,
        taskList,
        currentTaskList
      },
      activeTurnSummary: {
        turns,
        executionLeases: leases,
        turnTerminations: terminations,
        turnExecutorLinks: executorLinks,
        modelRequests,
        modelRequestMessageLinks
      },
      activeToolAndInteractionSummary: {
        messageTurnLinks,
        toolCalls,
        toolCallSourceLinks,
        toolCallPolicySnapshots,
        toolCallEvents,
        toolExecutions,
        toolOutcomes,
        toolModelResults,
        toolResultArtifacts,
        interactionRequests,
        interactionOwnerLinks,
        interactionToolCallLinks: allInteractionToolCallLinks,
        interactionResponses,
        fileChangeSets,
        fileChangeSetMembers,
        fileChangeDecisions,
        fileMutationReceipts,
        fileMutationReceiptMembers,
        processes: processRows,
        processOriginLinks,
        // Output bytes and chunk continuity are materialized through the pageable CAS detail
        // reader. Projecting an arbitrary prefix here would make a 200-row window look complete.
        processOutputChunks: [],
        processReceipts
      },
      subagentDeliverySummary: {
        childExecutions,
        childExecutionParentLinks: childParentLinks,
        childExecutionTurnLinks: childTurnLinks,
        childExecutionActiveTurnLinks: childActiveLinks,
        childTurns,
        childExecutionLeases,
        childTurnTerminations,
        childTurnExecutorLinks,
        childExecutionActivities,
        answerBridges,
        answerSubmissions,
        runtimeInboxItems: inboxItems,
        runtimeDeliveries: projectedDeliveries
      }
    };
    database.exec('COMMIT');
    return { snapshotCommitSeq: commitSeq.toString(), snapshot };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function taskListProjectionFromOutcome(
  database: Database.Database,
  outcome: Record<string, unknown> | null,
  toolCallId: string
): { mode?: string; items: unknown[] | null; detail_on_demand: boolean } {
  if (!outcome || outcome.content_object_id === null) return { items: null, detail_on_demand: false };
  if (outcome.status !== 'succeeded') return { items: null, detail_on_demand: false };
  const contentObjectId = requireRuntimeId(outcome.content_object_id);
  const raw = database.prepare('SELECT * FROM content_object WHERE id = ?').get(contentObjectId);
  if (!raw) throw new Error(`Task-list ToolOutcome ${toolCallId} references missing ContentObject ${contentObjectId}.`);
  const metadata = DOMAIN_REPOSITORIES.codec('ContentObject').decode(raw as Record<string, unknown>);
  if (
    typeof metadata.byte_length !== 'bigint'
    || metadata.byte_length > BigInt(CLIENT_WINDOW_RECORD_SUMMARY_MAX_BYTES * 16)
  ) {
    return { items: null, detail_on_demand: true };
  }
  const bytes = readVerifiedCasBytes(metadata, path.resolve(data.binding.paths.casRootPath));
  let value: unknown;
  try {
    value = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Task-list ToolOutcome ${toolCallId} content is not JSON: ${String(error)}`);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Task-list ToolOutcome ${toolCallId} content is not an object.`);
  }
  let operation;
  try {
    operation = taskListOperationFromSettledArtifact(value, toolCallId);
  } catch {
    return { items: null, detail_on_demand: true };
  }
  if (!operation) return { items: null, detail_on_demand: false };
  return {
    mode: operation.mode,
    items: operation.items,
    detail_on_demand: false
  };
}

function projectCurrentTaskList(
  database: Database.Database,
  conversationId: string
): Record<string, unknown> | null {
  const latestTurn = queryPlainRows(database, `
    SELECT turn.id
      FROM turn
     WHERE turn.conversation_id = @conversationId
     ORDER BY turn.created_at DESC, turn.id DESC
     LIMIT 1
  `, { conversationId })[0];
  if (!latestTurn) return null;
  const turnId = String(latestTurn.id);
  const calls = queryPlainRows(database, `
    SELECT call.id,
           call.turn_id,
           call.call_seq,
           call.tool_name,
           call.arguments_object_id,
           artifact.content_object_id AS artifact_content_object_id,
           source.message_id,
           source.provider_ordinal
      FROM tool_call AS call
      JOIN tool_result_artifact AS artifact
        ON artifact.tool_call_id = call.id
       AND artifact.role = 'no_effect_result'
      JOIN operation AS task_operation
        ON task_operation.tool_call_id = call.id
       AND task_operation.status = 'succeeded'
      JOIN tool_call_source_link AS source ON source.tool_call_id = call.id
     WHERE call.turn_id = @turnId
       AND call.tool_name IN ('update_task_list', 'submit_plan')
     ORDER BY call.call_seq ASC,
              call.id ASC
  `, { turnId });
  if (calls.length === 0) return null;

  // The task panel is an optional client projection. A pre-hard-cut or malformed artifact must
  // never prevent the bounded Conversation Feed from opening; omit the card without interpreting
  // the old shape. ModelRequest recipe reads remain strict in readCurrentTurnTaskCard().
  try {
    const operations: CurrentTurnTaskOperationFact[] = [];
    for (const call of calls) {
      const toolCallId = String(call.id);
      const artifact = readTaskProjectionJson(
        database,
        call.artifact_content_object_id,
        `Task-list ToolResultArtifact ${toolCallId}`
      );
      if (call.tool_name === 'update_task_list') {
        const operation = taskListOperationFromSettledArtifact(artifact, toolCallId);
        if (!operation) continue;
        operations.push({
          toolCallId,
          callSeq: String(call.call_seq),
          toolName: 'update_task_list',
          operation,
          sourceMessageId: String(call.message_id)
        });
        continue;
      }
      const argumentsValue = readTaskProjectionJson(
        database,
        call.arguments_object_id,
        `submit_plan ToolCall ${toolCallId} arguments`
      );
      const operation = approvedSubmitPlanTaskOperation({
        argumentsValue,
        resultArtifactValue: artifact,
        toolCallId
      });
      if (!operation) continue;
      operations.push({
        toolCallId,
        callSeq: String(call.call_seq),
        toolName: 'submit_plan',
        operation,
        planApproved: true,
        sourceMessageId: String(call.message_id)
      });
    }
    const projection = buildCurrentTurnTaskProjection({ turnId, operations });
    if (!projection) return null;
    return {
      conversationId,
      revision: projection.revision,
      operationCount: projection.operationCount,
      sourceToolCallId: projection.sourceToolCallId,
      sourceTurnId: projection.turnId,
      ...(projection.sourceMessageId ? { sourceMessageId: projection.sourceMessageId } : {}),
      baselineToolCallId: projection.baselineToolCallId,
      items: projection.snapshot.items,
      stats: projection.snapshot.stats,
      ...(projection.snapshot.activeItem ? { activeItem: projection.snapshot.activeItem } : {})
    };
  } catch {
    return null;
  }
}

function readTaskProjectionJson(
  database: Database.Database,
  contentObjectIdValue: unknown,
  label: string
): unknown {
  const contentObjectId = requireRuntimeId(contentObjectIdValue);
  const raw = database.prepare('SELECT * FROM content_object WHERE id = ?').get(contentObjectId);
  if (!raw) throw new Error(`${label} references missing ContentObject ${contentObjectId}.`);
  const metadata = DOMAIN_REPOSITORIES.codec('ContentObject').decode(raw as Record<string, unknown>);
  const bytes = readVerifiedCasBytes(metadata, path.resolve(data.binding.paths.casRootPath));
  try {
    return JSON.parse(bytes.toString('utf8')) as unknown;
  } catch (error) {
    throw new Error(`${label} content is not JSON: ${String(error)}`);
  }
}

function executeConversationHistoryProjection(
  database: Database.Database,
  input: ConversationHistoryProjectionInput,
  commitSeq: bigint
): ConversationHistoryProjectionResult {
  if (!Number.isSafeInteger(input.limit) || input.limit < 1 || input.limit > CLIENT_PAGE_MAX_ROWS) {
    throw new RangeError(`Conversation history page limit must be from 1 to ${CLIENT_PAGE_MAX_ROWS}.`);
  }
  if ((input.afterUpdatedAt === undefined) !== (input.afterId === undefined)) {
    throw new TypeError('Conversation history cursor requires both afterUpdatedAt and afterId.');
  }
  if (input.scopeKind === 'project' && !input.projectFolderUri?.trim()) {
    throw new TypeError('Project conversation history requires projectFolderUri.');
  }
  const dataVersion = BigInt(database.pragma('data_version', { simple: true }) as number | bigint);
  const snapshotCommitSeq = `${commitSeq.toString()}:${dataVersion.toString()}`;
  const cursorReset = input.expectedCommitSeq !== undefined && input.expectedCommitSeq !== snapshotCommitSeq;
  const useCursor = !cursorReset && input.afterUpdatedAt !== undefined;
  const scope = conversationHistoryScopeSql(input, 'conversation');
  const cursorSql = useCursor
    ? `AND (conversation.updated_at < @afterUpdatedAt
         OR (conversation.updated_at = @afterUpdatedAt AND conversation.id < @afterId))`
    : '';
  database.exec('BEGIN');
  try {
    const seedCandidates = queryPlainRows(database, `
      SELECT conversation.id, conversation.title, conversation.status,
             conversation.created_at, conversation.updated_at
        FROM conversation
       WHERE ${scope.sql}
             ${cursorSql}
       ORDER BY conversation.updated_at DESC, conversation.id DESC
       LIMIT @seedLimit
    `, {
      ...scope.params,
      ...(useCursor ? { afterUpdatedAt: input.afterUpdatedAt!, afterId: input.afterId! } : {}),
      seedLimit: BigInt(input.limit + 1)
    });
    const hasMore = seedCandidates.length > input.limit;
    const seedRows = seedCandidates.slice(0, input.limit);
    const totalRow = database.prepare(`
      SELECT COUNT(*) AS total FROM conversation WHERE ${scope.sql}
    `).get(scope.params) as { total: bigint };
    const seedIds = seedRows.map((row) => String(row.id));
    if (seedIds.length === 0) {
      database.exec('COMMIT');
      return {
        snapshotCommitSeq,
        cursorReset,
        seedRows: [], conversations: [], origins: [], turns: [], leases: [], agentLinks: [],
        messageSummaries: [], previewTargets: [], titleTargets: [], childExecutions: [], activeChildTurnLinks: [],
        answerBridges: [], inboxItems: [], deliveries: [], deliveryWakes: [], deliveryInputLinks: [],
        projectContexts: [], conversationProjectLinks: [], total: Number(totalRow.total), hasMore: false
      };
    }
    const seedParameters = Object.fromEntries(seedIds.map((id, index) => [`seed${index}`, id]));
    const seedValues = seedIds.map((_id, index) => `(@seed${index})`).join(',');
    const conversations = queryPlainRows(database, `
      WITH seed(id) AS (
        VALUES ${seedValues}
      ), page_conversation(id) AS (
        SELECT id FROM seed
        UNION
        SELECT origin.source_conversation_id
          FROM conversation_origin_link AS origin
          JOIN seed ON seed.id = origin.conversation_id
         WHERE origin.source_conversation_id IS NOT NULL
      )
      SELECT conversation.id, conversation.title, conversation.status,
             conversation.created_at, conversation.updated_at
        FROM conversation
        JOIN page_conversation ON page_conversation.id = conversation.id
       ORDER BY conversation.updated_at DESC, conversation.id DESC
    `, seedParameters);
    const conversationIds = conversations.map((row) => String(row.id));
    const origins = queryAllByIds(database, 'conversation_origin_link', 'conversation_id', conversationIds)
      .filter((row) => row.source_conversation_id === null || conversationIds.includes(String(row.source_conversation_id)));
    const turns = queryAllByIds(database, 'turn', 'conversation_id', conversationIds);
    const turnIds = turns.map((row) => String(row.id));
    const leases = queryAllByIds(database, 'execution_lease', 'turn_id', turnIds);
    const agentLinks = queryAllByIds(database, 'agent_conversation_link', 'conversation_id', conversationIds);
    const conversationProjectLinks = queryAllByIds(database, 'conversation_project_link', 'conversation_id', conversationIds);
    const projectContexts = queryAllByIds(
      database,
      'project_context',
      'id',
      conversationProjectLinks.map((row) => String(row.project_context_id))
    );
    const messageSummaries = queryConversationMessageSummaries(database, conversationIds);
    const latestVisible = queryLatestVisibleRevisions(database, conversationIds);
    const firstUser = queryFirstUserRevisions(database, conversationIds);
    const targetRevisionIds = [...new Set([
      ...latestVisible.map((row) => String(row.revision_id)),
      ...firstUser.map((row) => String(row.revision_id))
    ])];
    const revisions = queryAllByIds(database, 'message_revision', 'id', targetRevisionIds);
    const revisionById = new Map(revisions.map((row) => [String(row.id), row]));
    const contents = queryAllByIds(database, 'content_object', 'id', revisions.map((row) => String(row.content_object_id)));
    const contentById = new Map(contents.map((row) => [String(row.id), row]));
    const projectionTargets = (rows: Array<Record<string, unknown>>) => rows.flatMap((row) => {
      const revisionId = String(row.revision_id);
      const revision = revisionById.get(revisionId);
      const content = revision ? contentById.get(String(revision.content_object_id)) : undefined;
      return content ? [{ conversationId: String(row.conversation_id), revisionId, content }] : [];
    });
    const previewTargets = projectionTargets(latestVisible);
    const titleTargets = projectionTargets(firstUser);
    const childExecutions = queryAllByIds(database, 'child_execution', 'child_conversation_id', conversationIds);
    const childIds = childExecutions.map((row) => String(row.id));
    const activeChildTurnLinks = queryAllByIds(database, 'child_execution_active_turn_link', 'child_execution_id', childIds);
    const answerBridges = queryAllByIds(database, 'answer_bridge', 'child_execution_id', childIds);
    const submissionIds = answerBridges.flatMap((row) => row.current_submission_id === null
      ? []
      : [String(row.current_submission_id)]);
    const inboxItems = queryAllByIds(database, 'runtime_inbox_item', 'source_id', submissionIds)
      .filter((row) => row.source_kind === 'answer_submission');
    const deliveries = queryAllByIds(database, 'runtime_delivery', 'inbox_item_id', inboxItems.map((row) => String(row.id)));
    const deliveryIds = deliveries.map((row) => String(row.id));
    const deliveryWakes = queryAllByIds(database, 'runtime_delivery_wake', 'delivery_id', deliveryIds);
    const deliveryInputLinks = queryAllByIds(database, 'runtime_delivery_input_link', 'delivery_id', deliveryIds);
    database.exec('COMMIT');
    return {
      snapshotCommitSeq,
      cursorReset,
      seedRows,
      conversations,
      origins,
      turns,
      leases,
      agentLinks,
      messageSummaries,
      previewTargets,
      titleTargets,
      childExecutions,
      activeChildTurnLinks,
      answerBridges,
      inboxItems,
      deliveries,
      deliveryWakes,
      deliveryInputLinks,
      projectContexts,
      conversationProjectLinks,
      total: Number(totalRow.total),
      hasMore
    };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function conversationHistoryScopeSql(
  input: ConversationHistoryProjectionInput,
  alias: string
): { sql: string; params: Record<string, string> } {
  if (input.scopeKind === 'all') return { sql: '1 = 1', params: {} };
  if (input.scopeKind === 'unbound') {
    return {
      sql: `NOT EXISTS (
        SELECT 1 FROM conversation_project_link AS scope_link
         WHERE scope_link.conversation_id = ${alias}.id AND scope_link.role = 'primary'
      )`,
      params: {}
    };
  }
  return {
    sql: `EXISTS (
      SELECT 1
        FROM conversation_project_link AS scope_link
        JOIN project_context AS scope_project ON scope_project.id = scope_link.project_context_id
       WHERE scope_link.conversation_id = ${alias}.id
         AND scope_link.role = 'primary'
         AND scope_project.uri = @projectFolderUri
    )`,
    params: { projectFolderUri: input.projectFolderUri!.trim() }
  };
}

function executeClientKeysetPage(
  database: Database.Database,
  input: ClientKeysetPageInput
): ClientKeysetPageResult {
  if (!Number.isSafeInteger(input.limit) || input.limit <= 0 || input.limit > CLIENT_PAGE_MAX_ROWS) {
    throw new RangeError(`Client keyset page limit must be from 1 to ${CLIENT_PAGE_MAX_ROWS}.`);
  }
  if ((input.afterSortKey === undefined) !== (input.afterId === undefined)) {
    throw new TypeError('Client keyset cursor requires both afterSortKey and afterId.');
  }
  const afterId = input.afterId === undefined ? undefined : requireRuntimeId(input.afterId);
  database.exec('BEGIN');
  try {
    let candidates: Array<Record<string, unknown>>;
    let sortKey: (row: Record<string, unknown>) => string;
    if (input.query === 'conversation') {
      if (input.sortId !== 'created_at+id') throw new TypeError('Conversation keyset sortId must be created_at+id.');
      const afterSortKey = input.afterSortKey;
      candidates = queryPlainRows(database, `
        SELECT id, title, status, created_at, updated_at
          FROM conversation
         ${afterSortKey === undefined ? '' : 'WHERE created_at > @afterSortKey OR (created_at = @afterSortKey AND id > @afterId)'}
         ORDER BY created_at ASC, id ASC
         LIMIT @limit
      `, {
        ...(afterSortKey === undefined ? {} : { afterSortKey, afterId: afterId! }),
        limit: BigInt(input.limit + 1)
      });
      sortKey = (row) => String(row.created_at);
    } else if (input.query === 'message') {
      if (input.sortId !== 'message_seq') throw new TypeError('Message keyset sortId must be message_seq.');
      const conversationId = requireRuntimeId(input.conversationId);
      const afterSortKey = input.afterSortKey === undefined
        ? undefined
        : requireNonNegativeIntegerString(input.afterSortKey, 'afterSortKey');
      candidates = queryPlainRows(database, `
        SELECT message.id,
               membership.conversation_id,
               membership.message_seq,
               message.created_at,
               message.updated_at,
               message.deleted_at,
               revision.id AS revision_id,
               revision.revision_seq,
               revision.role,
               revision.content_object_id,
               content.content_type,
               content.byte_length
          FROM message_part_of_conversation AS membership
          JOIN message ON message.id = membership.message_id
          JOIN message_current_revision_link AS current_revision ON current_revision.message_id = message.id
          JOIN message_revision AS revision ON revision.id = current_revision.revision_id
          JOIN content_object AS content ON content.id = revision.content_object_id
         WHERE membership.conversation_id = @conversationId
           ${afterSortKey === undefined ? '' : 'AND (membership.message_seq > @afterSortKey OR (membership.message_seq = @afterSortKey AND message.id > @afterId))'}
         ORDER BY membership.message_seq ASC, message.id ASC
         LIMIT @limit
      `, {
        conversationId,
        ...(afterSortKey === undefined ? {} : { afterSortKey: BigInt(afterSortKey), afterId: afterId! }),
        limit: BigInt(input.limit + 1)
      });
      sortKey = (row) => String(row.message_seq);
    } else {
      throw new TypeError(`Unsupported client keyset query: ${String(input.query)}.`);
    }

    let hasMore = candidates.length > input.limit;
    const boundedCandidates = candidates.slice(0, input.limit);
    const rows: Array<Record<string, unknown>> = [];
    for (const candidate of boundedCandidates) {
      const tentative = [...rows, candidate];
      const last = tentative[tentative.length - 1];
      const responseProbe = {
        rows: tentative,
        nextSortKey: sortKey(last),
        nextId: String(last.id),
        hasMore: true
      };
      if (wireJsonBytes(responseProbe) > CLIENT_PAGE_MAX_BYTES) {
        hasMore = true;
        break;
      }
      rows.push(candidate);
    }
    if (rows.length === 0 && boundedCandidates.length > 0) {
      throw new Error('A single client keyset summary exceeds maxPageBytes.');
    }
    const last = rows[rows.length - 1];
    const result: ClientKeysetPageResult = {
      rows,
      ...(last ? { nextSortKey: sortKey(last), nextId: String(last.id) } : {}),
      hasMore: hasMore || rows.length < boundedCandidates.length,
      responseBytes: 0
    };
    result.responseBytes = wireJsonBytes(result);
    if (result.responseBytes > CLIENT_PAGE_MAX_BYTES) throw new Error('Client keyset response exceeds maxPageBytes.');
    database.exec('COMMIT');
    return result;
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

/**
 * Reads the visible timeline strictly before one durable Message membership cursor. The page is
 * self-contained for rendering: Message bodies remain in CAS detail authority, while the bounded
 * causal summaries needed to associate Turns, model requests, tools, interactions and effects are
 * returned beside the Message anchors.
 */
function executeClientVisibleMessageHistoryPage(
  database: Database.Database,
  input: ClientVisibleMessageHistoryPageInput
): ClientVisibleMessageHistoryPageResult {
  if (!Number.isSafeInteger(input.limit) || input.limit <= 0 || input.limit > CLIENT_PAGE_MAX_ROWS) {
    throw new RangeError(`Visible Message history page limit must be from 1 to ${CLIENT_PAGE_MAX_ROWS}.`);
  }
  const conversationId = requireRuntimeId(input.conversationId);
  const beforeMessageSeq = requireNonNegativeIntegerString(input.beforeMessageSeq, 'beforeMessageSeq');
  if (beforeMessageSeq === '0') throw new RangeError('beforeMessageSeq must be positive.');
  const beforeId = requireRuntimeId(input.beforeId);

  database.exec('BEGIN');
  try {
    const candidates = queryPlainRows(database, `
      WITH visible_messages AS (
        SELECT message.id,
               membership.conversation_id,
               membership.message_seq,
               message.created_at,
               message.updated_at,
               message.deleted_at,
               revision.id AS revision_id,
               revision.revision_seq,
               revision.role,
               revision.content_object_id,
               content.content_type,
               content.byte_length,
               ROW_NUMBER() OVER (
                 ORDER BY membership.message_seq ASC, message.id ASC
               ) AS display_seq
          FROM message_part_of_conversation AS membership
          JOIN message ON message.id = membership.message_id
          JOIN message_current_revision_link AS current_revision
            ON current_revision.message_id = message.id
          JOIN message_revision AS revision ON revision.id = current_revision.revision_id
          JOIN content_object AS content ON content.id = revision.content_object_id
         WHERE membership.conversation_id = @conversationId
           AND message.deleted_at IS NULL
           AND revision.role IN ('user', 'model')
      )
      SELECT *
        FROM visible_messages
       WHERE message_seq < @beforeMessageSeq
          OR (message_seq = @beforeMessageSeq AND id < @beforeId)
       ORDER BY message_seq DESC, id DESC
       LIMIT @limit
    `, {
      conversationId,
      beforeMessageSeq: BigInt(beforeMessageSeq),
      beforeId,
      limit: BigInt(input.limit + 1)
    });

    if (candidates.length === 0) {
      const empty: ClientVisibleMessageHistoryPageResult = {
        records: {},
        hasMore: false,
        responseBytes: 0
      };
      settleClientWireResponseBytes(empty);
      database.exec('COMMIT');
      return empty;
    }

    const maximumCount = Math.min(input.limit, candidates.length);
    const materialize = (count: number): ClientVisibleMessageHistoryPageResult => {
      const messages = candidates.slice(0, count).reverse();
      const oldest = messages[0];
      const rawRecords = buildClientVisibleMessageHistoryRecords(database, messages);
      const result: ClientVisibleMessageHistoryPageResult = {
        records: Object.fromEntries(Object.entries(rawRecords).map(([domain, rows]) => [
          domain,
          rows.map((row) => boundClientRecordSummary(row))
        ])),
        nextBeforeMessageSeq: String(oldest.message_seq),
        nextBeforeId: String(oldest.id),
        hasMore: candidates.length > count,
        responseBytes: 0
      };
      settleClientWireResponseBytes(result);
      return result;
    };

    // Causal closure size varies by Message, so choose the largest anchor prefix that still fits
    // one bounded response. Reducing count drops the oldest/farthest anchors, preserving keyset
    // continuity from the caller's cursor.
    let lower = 1;
    let upper = maximumCount;
    let selected: ClientVisibleMessageHistoryPageResult | undefined;
    while (lower <= upper) {
      const count = Math.floor((lower + upper) / 2);
      const candidate = materialize(count);
      if (candidate.responseBytes <= CLIENT_PAGE_MAX_BYTES) {
        selected = candidate;
        lower = count + 1;
      } else {
        upper = count - 1;
      }
    }
    if (!selected) throw new Error('A single visible Message history summary exceeds maxPageBytes.');
    if (selected.responseBytes > CLIENT_PAGE_MAX_BYTES) {
      throw new Error('Visible Message history response exceeds maxPageBytes.');
    }
    database.exec('COMMIT');
    return selected;
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function buildClientVisibleMessageHistoryRecords(
  database: Database.Database,
  messages: readonly Record<string, unknown>[]
): Record<string, DomainRow[]> {
  const records: Record<string, DomainRow[]> = {};
  const include = (domain: string, rows: readonly Record<string, unknown>[]): void => {
    const unique = mergeRowsById(rows);
    if (unique.length > 0) records[domain] = unique;
  };
  const messageIds = messages.map((row) => String(row.id));
  include('Message', messages);

  const messageTurnLinks = queryAllByIds(database, 'message_turn_link', 'message_id', messageIds);
  const requestMessageLinks = queryAllByIds(database, 'model_request_message_link', 'message_id', messageIds);
  const sourceLinks = queryAllByIds(database, 'tool_call_source_link', 'message_id', messageIds);
  include('MessageTurnLink', messageTurnLinks);
  include('ModelRequestMessageLink', requestMessageLinks);
  include('ToolCallSourceLink', sourceLinks);

  const requestIds = [...new Set([
    ...requestMessageLinks.map((row) => String(row.model_request_id)),
    ...sourceLinks.map((row) => String(row.model_request_id))
  ])];
  const modelRequests = queryAllByIds(database, 'model_request', 'id', requestIds)
    .sort(compareModelRequestRows);
  include('ModelRequest', modelRequests);

  const toolCallIds = [...new Set(sourceLinks.map((row) => String(row.tool_call_id)))];
  const toolCalls = queryAllByIds(database, 'tool_call', 'id', toolCallIds)
    .sort(compareToolCallRows);
  include('ToolCall', toolCalls);
  include('ToolCallPolicySnapshot', queryAllByIds(database, 'tool_call_policy_snapshot', 'tool_call_id', toolCallIds));
  include('ToolCallEvent', queryLatestToolCallEvents(database, toolCallIds));
  include('ToolExecution', queryAllByIds(database, 'tool_execution', 'tool_call_id', toolCallIds));
  include('ToolOutcome', queryAllByIds(database, 'tool_outcome', 'tool_call_id', toolCallIds));
  include('ToolModelResult', queryAllByIds(database, 'tool_model_result', 'tool_call_id', toolCallIds));
  include('ToolResultArtifact', queryAllByIds(database, 'tool_result_artifact', 'tool_call_id', toolCallIds));

  const interactionToolLinks = queryAllByIds(database, 'interaction_tool_call_link', 'tool_call_id', toolCallIds);
  const interactionRequestIds = [...new Set(interactionToolLinks.map((row) => String(row.request_id)))];
  const interactionRequests = queryAllByIds(database, 'interaction_request', 'id', interactionRequestIds);
  const interactionOwnerLinks = queryAllByIds(database, 'interaction_owner_link', 'request_id', interactionRequestIds);
  include('InteractionToolCallLink', interactionToolLinks);
  include('InteractionRequest', interactionRequests);
  include('InteractionOwnerLink', interactionOwnerLinks);
  include('InteractionResponse', queryAllByIds(database, 'interaction_response', 'request_id', interactionRequestIds));

  const fileChangeSets = queryAllByIds(database, 'file_change_set', 'tool_call_id', toolCallIds);
  const fileChangeSetIds = fileChangeSets.map((row) => String(row.id));
  const fileMutationReceipts = queryAllByIds(database, 'file_mutation_receipt', 'change_set_id', fileChangeSetIds);
  include('FileChangeSet', fileChangeSets);
  include('FileChangeSetMember', queryAllByIds(database, 'file_change_set_member', 'change_set_id', fileChangeSetIds));
  include('FileChangeDecision', queryAllByIds(database, 'file_change_decision', 'change_set_id', fileChangeSetIds));
  include('FileMutationReceipt', fileMutationReceipts);
  include('FileMutationReceiptMember', queryAllByIds(
    database,
    'file_mutation_receipt_member',
    'receipt_id',
    fileMutationReceipts.map((row) => String(row.id))
  ));

  const processOriginLinks = queryAllByIds(database, 'process_origin_link', 'tool_call_id', toolCallIds);
  const processIds = [...new Set(processOriginLinks.map((row) => String(row.process_id)))];
  const processes = queryAllByIds(database, 'process', 'id', processIds)
    .map((row) => projectProcessRecord(database, String(row.id)));
  include('ProcessOriginLink', processOriginLinks);
  include('Process', processes);
  include('ProcessReceipt', queryAllByIds(database, 'process_receipt', 'process_id', processIds));

  const childParentLinks = queryAllByIds(database, 'child_execution_parent_link', 'source_tool_call_id', toolCallIds);
  const childExecutionIds = [...new Set(childParentLinks.map((row) => String(row.child_execution_id)))];
  const childExecutions = queryAllByIds(database, 'child_execution', 'id', childExecutionIds);
  const childActiveLinks = queryAllByIds(database, 'child_execution_active_turn_link', 'child_execution_id', childExecutionIds);
  const childTurnLinks = queryLatestChildExecutionTurnLinks(database, childExecutionIds);
  const childTurnIds = [...new Set([
    ...childActiveLinks.map((row) => String(row.turn_id)),
    ...childTurnLinks.map((row) => String(row.turn_id))
  ])];
  include('ChildExecutionParentLink', childParentLinks);
  include('ChildExecution', childExecutions);
  include('ChildExecutionActiveTurnLink', childActiveLinks);
  include('ChildExecutionTurnLink', childTurnLinks);
  include('AgentConversationLink', queryAllByIds(
    database,
    'agent_conversation_link',
    'conversation_id',
    childExecutions.map((row) => String(row.child_conversation_id))
  ));

  const answerBridges = queryAllByIds(database, 'answer_bridge', 'child_execution_id', childExecutionIds)
    .map((row) => projectAnswerBridgeRecord(database, String(row.id)));
  include('AnswerBridge', answerBridges);
  include('AnswerSubmission', queryAllByIds(
    database,
    'answer_submission',
    'id',
    answerBridges.flatMap((row) => row.current_submission_id ? [String(row.current_submission_id)] : [])
  ));

  const rootTurnIds = [
    ...messageTurnLinks.map((row) => String(row.turn_id)),
    ...modelRequests.map((row) => String(row.turn_id)),
    ...toolCalls.map((row) => String(row.turn_id)),
    ...interactionOwnerLinks.map((row) => String(row.turn_id)),
    ...childParentLinks.flatMap((row) => row.parent_turn_id ? [String(row.parent_turn_id)] : [])
  ];
  const turnIds = [...new Set([...rootTurnIds, ...childTurnIds])];
  const turns = queryAllByIds(database, 'turn', 'id', turnIds)
    .map((row) => row.status === 'active' ? projectTurnClientRecord(database, String(row.id)) : row)
    .sort(compareTurnRows);
  include('Turn', turns);
  include('ExecutionLease', queryAllByIds(database, 'execution_lease', 'turn_id', turnIds));
  include('TurnTermination', queryAllByIds(database, 'turn_termination', 'turn_id', turnIds));
  include('TurnExecutorLink', queryAllByIds(database, 'turn_executor_link', 'turn_id', turnIds));
  return records;
}

function queryPlainRows(
  database: Database.Database,
  sql: string,
  parameters: Record<string, string | bigint> = {}
): Array<Record<string, unknown>> {
  return database.prepare(sql).all(parameters) as Array<Record<string, unknown>>;
}

function mergeRowsById(rows: readonly Record<string, unknown>[]): Array<Record<string, unknown>> {
  const byId = new Map<string, Record<string, unknown>>();
  for (const row of rows) byId.set(String(row.id), row);
  return [...byId.values()];
}

function compareRuntimeRowInteger(left: unknown, right: unknown): number {
  const leftValue = typeof left === 'bigint' ? left : BigInt(String(left ?? 0));
  const rightValue = typeof right === 'bigint' ? right : BigInt(String(right ?? 0));
  return leftValue < rightValue ? -1 : leftValue > rightValue ? 1 : 0;
}

function compareMessageWindowRows(left: Record<string, unknown>, right: Record<string, unknown>): number {
  return compareRuntimeRowInteger(left.message_seq, right.message_seq)
    || String(left.id).localeCompare(String(right.id));
}

function compareTurnRows(left: Record<string, unknown>, right: Record<string, unknown>): number {
  return String(left.created_at).localeCompare(String(right.created_at))
    || String(left.id).localeCompare(String(right.id));
}

function compareModelRequestRows(left: Record<string, unknown>, right: Record<string, unknown>): number {
  return String(left.created_at).localeCompare(String(right.created_at))
    || compareRuntimeRowInteger(left.request_seq, right.request_seq)
    || String(left.id).localeCompare(String(right.id));
}

function compareToolCallRows(left: Record<string, unknown>, right: Record<string, unknown>): number {
  return String(left.created_at).localeCompare(String(right.created_at))
    || compareRuntimeRowInteger(left.call_seq, right.call_seq)
    || String(left.id).localeCompare(String(right.id));
}

function queryClientRootTurns(
  database: Database.Database,
  conversationId: string
): Array<Record<string, unknown>> {
  return queryPlainRows(database, `
    WITH recent_turns AS (
      SELECT id
        FROM turn
       WHERE conversation_id = @conversationId
       ORDER BY created_at DESC, id DESC
       LIMIT @limit
    )
    SELECT turn.*
      FROM turn
     WHERE turn.conversation_id = @conversationId
       AND (turn.status = 'active' OR turn.id IN (SELECT id FROM recent_turns))
     ORDER BY turn.created_at ASC, turn.id ASC
  `, { conversationId, limit: BigInt(CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE) });
}

function queryClientProcesses(
  database: Database.Database,
  conversationId: string
): Array<Record<string, unknown>> {
  return queryPlainRows(database, `
    SELECT process.*
      FROM process
      JOIN process_origin_link AS origin ON origin.process_id = process.id
      JOIN tool_call ON tool_call.id = origin.tool_call_id
      JOIN turn ON turn.id = tool_call.turn_id
     WHERE turn.conversation_id = @conversationId
       AND process.status = 'running'
     ORDER BY process.started_at DESC, process.id DESC
  `, { conversationId });
}

function queryClientChildExecutions(
  database: Database.Database,
  conversationId: string
): Array<Record<string, unknown>> {
  return queryPlainRows(database, `
    SELECT DISTINCT child.*
      FROM child_execution AS child
      JOIN child_execution_parent_link AS parent_link
        ON parent_link.child_execution_id = child.id
      LEFT JOIN tool_call AS source_call ON source_call.id = parent_link.source_tool_call_id
      LEFT JOIN turn AS source_turn ON source_turn.id = source_call.turn_id
     WHERE child.child_conversation_id = @conversationId
        OR (
          source_turn.conversation_id = @conversationId
          AND child.status NOT IN ('closed', 'needs_human')
        )
     ORDER BY child.created_at DESC, child.id DESC
  `, { conversationId });
}

function queryClientRuntimeDeliveries(
  database: Database.Database,
  conversationId: string
): Array<Record<string, unknown>> {
  return queryPlainRows(database, `
    WITH recent_deliveries AS (
      SELECT id
        FROM runtime_delivery
       WHERE target_conversation_id = @conversationId
       ORDER BY created_at DESC, id DESC
       LIMIT @limit
    )
    SELECT delivery.*
      FROM runtime_delivery AS delivery
     WHERE delivery.target_conversation_id = @conversationId
       AND (
         delivery.state IN ('pending', 'failed')
         OR EXISTS (
           SELECT 1
             FROM runtime_delivery_input_link AS input_link
            WHERE input_link.delivery_id = delivery.id
              AND input_link.handled_at IS NULL
         )
         OR delivery.id IN (SELECT id FROM recent_deliveries)
       )
     ORDER BY delivery.created_at DESC, delivery.id DESC
  `, { conversationId, limit: BigInt(CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE) });
}

function queryVisibleMessageRowsByIds(
  database: Database.Database,
  conversationId: string,
  messageIds: readonly string[]
): Array<Record<string, unknown>> {
  const unique = [...new Set(messageIds)];
  const rows: Array<Record<string, unknown>> = [];
  for (let offset = 0; offset < unique.length; offset += 350) {
    const chunk = unique.slice(offset, offset + 350);
    if (chunk.length === 0) continue;
    const parameters: Record<string, string | bigint> = { conversationId };
    const placeholders = chunk.map((id, index) => {
      parameters[`message${index}`] = id;
      return `@message${index}`;
    });
    rows.push(...queryPlainRows(database, `
      WITH visible_messages AS (
        SELECT message.id,
               membership.conversation_id,
               membership.message_seq,
               message.created_at,
               message.updated_at,
               message.deleted_at,
               revision.id AS revision_id,
               revision.revision_seq,
               revision.role,
               revision.content_object_id,
               content.content_type,
               content.byte_length,
               ROW_NUMBER() OVER (
                 ORDER BY membership.message_seq ASC, message.id ASC
               ) AS display_seq
          FROM message_part_of_conversation AS membership
          JOIN message ON message.id = membership.message_id
          JOIN message_current_revision_link AS current_revision ON current_revision.message_id = message.id
          JOIN message_revision AS revision ON revision.id = current_revision.revision_id
          JOIN content_object AS content ON content.id = revision.content_object_id
         WHERE membership.conversation_id = @conversationId
           AND message.deleted_at IS NULL
           AND revision.role IN ('user', 'model')
      )
      SELECT *
        FROM visible_messages
       WHERE id IN (${placeholders.join(',')})
       ORDER BY message_seq ASC, id ASC
    `, parameters));
  }
  return mergeRowsById(rows);
}

function queryLatestToolCallEvents(
  database: Database.Database,
  toolCallIds: readonly string[]
): Array<Record<string, unknown>> {
  const unique = [...new Set(toolCallIds)];
  const rows: Array<Record<string, unknown>> = [];
  for (let offset = 0; offset < unique.length; offset += 350) {
    const chunk = unique.slice(offset, offset + 350);
    if (chunk.length === 0) continue;
    const parameters: Record<string, string | bigint> = {
      eventLimit: BigInt(CLIENT_TOOL_EVENT_SUMMARY_LIMIT_PER_CALL)
    };
    const placeholders = chunk.map((id, index) => {
      parameters[`tool${index}`] = id;
      return `@tool${index}`;
    });
    rows.push(...queryPlainRows(database, `
      SELECT *
        FROM (
          SELECT event.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY event.tool_call_id
                   ORDER BY event.event_seq DESC, event.id DESC
                 ) AS client_tail_ordinal
            FROM tool_call_event AS event
           WHERE event.tool_call_id IN (${placeholders.join(',')})
        )
       WHERE client_tail_ordinal <= @eventLimit
       ORDER BY tool_call_id ASC, event_seq ASC, id ASC
    `, parameters).map(({ client_tail_ordinal: _ordinal, ...row }) => row));
  }
  return rows;
}

function queryLatestChildExecutionTurnLinks(
  database: Database.Database,
  childExecutionIds: readonly string[]
): Array<Record<string, unknown>> {
  const unique = [...new Set(childExecutionIds)];
  const rows: Array<Record<string, unknown>> = [];
  for (let offset = 0; offset < unique.length; offset += 350) {
    const chunk = unique.slice(offset, offset + 350);
    if (chunk.length === 0) continue;
    const parameters: Record<string, string | bigint> = {};
    const placeholders = chunk.map((id, index) => {
      parameters[`child${index}`] = id;
      return `@child${index}`;
    });
    rows.push(...queryPlainRows(database, `
      SELECT id, child_execution_id, turn_seq, turn_id, created_at
        FROM (
          SELECT link.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY link.child_execution_id
                   ORDER BY link.turn_seq DESC, link.id DESC
                 ) AS client_ordinal
            FROM child_execution_turn_link AS link
           WHERE link.child_execution_id IN (${placeholders.join(',')})
        )
       WHERE client_ordinal = 1
       ORDER BY child_execution_id ASC, turn_seq ASC, id ASC
    `, parameters));
  }
  return rows;
}

function queryByIds(
  database: Database.Database,
  table: string,
  column: string,
  ids: readonly string[]
): Array<Record<string, unknown>> {
  if (!/^[a-z][a-z0-9_]*$/.test(table) || !/^[a-z][a-z0-9_]*$/.test(column)) {
    throw new Error('Fixed client projection contains an unsafe identifier.');
  }
  const unique = [...new Set(ids)].slice(0, CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE);
  if (unique.length === 0) return [];
  const parameters: Record<string, string | bigint> = {
    limit: BigInt(CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE)
  };
  const placeholders = unique.map((id, index) => {
    parameters[`id${index}`] = id;
    return `@id${index}`;
  });
  const parentPriority = unique
    .map((_id, index) => `WHEN @id${index} THEN ${index}`)
    .join(' ');
  return queryPlainRows(database, `
    SELECT ${quote(table)}.*
      FROM ${quote(table)}
     WHERE ${quote(column)} IN (${placeholders.join(',')})
     ORDER BY (
       CASE ${quote(column)} ${parentPriority} ELSE ${unique.length} END
       + (ROW_NUMBER() OVER (
           PARTITION BY ${quote(column)}
           ORDER BY id DESC
         ) - 1) * 8
     ) ASC,
     id DESC
     LIMIT @limit
  `, parameters);
}

/** History-page closure is already scope-bounded; do not apply the client-feed global row cap. */
function queryAllByIds(
  database: Database.Database,
  table: string,
  column: string,
  ids: readonly string[]
): Array<Record<string, unknown>> {
  if (!/^[a-z][a-z0-9_]*$/.test(table) || !/^[a-z][a-z0-9_]*$/.test(column)) {
    throw new Error('Conversation history projection contains an unsafe identifier.');
  }
  const unique = [...new Set(ids)];
  const rows: Array<Record<string, unknown>> = [];
  for (let offset = 0; offset < unique.length; offset += 400) {
    const chunk = unique.slice(offset, offset + 400);
    const parameters: Record<string, string | bigint> = {};
    const placeholders = chunk.map((id, index) => {
      parameters[`id${index}`] = id;
      return `@id${index}`;
    });
    rows.push(...queryPlainRows(database, `
      SELECT * FROM ${quote(table)}
       WHERE ${quote(column)} IN (${placeholders.join(',')})
       ORDER BY id ASC
    `, parameters));
  }
  return rows;
}

function queryConversationMessageSummaries(
  database: Database.Database,
  conversationIds: readonly string[]
): Array<Record<string, unknown>> {
  return queryConversationIdChunks(database, conversationIds, (placeholders) => `
    SELECT membership.conversation_id, COUNT(*) AS message_count
      FROM message_part_of_conversation AS membership
      JOIN message ON message.id = membership.message_id
      JOIN message_current_revision_link AS current ON current.message_id = message.id
      JOIN message_revision AS revision ON revision.id = current.revision_id
     WHERE membership.conversation_id IN (${placeholders})
       AND message.deleted_at IS NULL
       AND revision.role IN ('user', 'model')
     GROUP BY membership.conversation_id
  `);
}

function queryFirstUserRevisions(
  database: Database.Database,
  conversationIds: readonly string[]
): Array<Record<string, unknown>> {
  return queryConversationIdChunks(database, conversationIds, (placeholders) => `
    SELECT conversation_id, revision_id
      FROM (
        SELECT membership.conversation_id,
               revision.id AS revision_id,
               ROW_NUMBER() OVER (
                 PARTITION BY membership.conversation_id
                 ORDER BY membership.message_seq ASC, membership.message_id ASC
               ) AS ordinal
          FROM message_part_of_conversation AS membership
          JOIN message ON message.id = membership.message_id
          JOIN message_current_revision_link AS current ON current.message_id = message.id
          JOIN message_revision AS revision ON revision.id = current.revision_id
         WHERE membership.conversation_id IN (${placeholders})
           AND message.deleted_at IS NULL
           AND revision.role = 'user'
      )
     WHERE ordinal = 1
  `);
}

function queryLatestVisibleRevisions(
  database: Database.Database,
  conversationIds: readonly string[]
): Array<Record<string, unknown>> {
  return queryConversationIdChunks(database, conversationIds, (placeholders) => `
    SELECT conversation_id, revision_id
      FROM (
        SELECT membership.conversation_id,
               revision.id AS revision_id,
               ROW_NUMBER() OVER (
                 PARTITION BY membership.conversation_id
                 ORDER BY membership.message_seq DESC, membership.message_id DESC
               ) AS ordinal
          FROM message_part_of_conversation AS membership
          JOIN message ON message.id = membership.message_id
          JOIN message_current_revision_link AS current ON current.message_id = message.id
          JOIN message_revision AS revision ON revision.id = current.revision_id
         WHERE membership.conversation_id IN (${placeholders})
           AND message.deleted_at IS NULL
           AND revision.role IN ('user', 'model')
      )
     WHERE ordinal = 1
  `);
}

function queryConversationIdChunks(
  database: Database.Database,
  conversationIds: readonly string[],
  sql: (placeholders: string) => string
): Array<Record<string, unknown>> {
  const rows: Array<Record<string, unknown>> = [];
  const unique = [...new Set(conversationIds)];
  for (let offset = 0; offset < unique.length; offset += 400) {
    const chunk = unique.slice(offset, offset + 400);
    const parameters: Record<string, string | bigint> = {};
    const placeholders = chunk.map((id, index) => {
      parameters[`conversation${index}`] = id;
      return `@conversation${index}`;
    }).join(',');
    rows.push(...queryPlainRows(database, sql(placeholders), parameters));
  }
  return rows;
}

function queryConversationModelRequests(
  database: Database.Database,
  conversationId: string,
  visibleMessageIds: readonly string[]
): Array<Record<string, unknown>> {
  const parameters: Record<string, string | bigint> = {
    conversationId
  };
  const visiblePlaceholders = [...new Set(visibleMessageIds)].map((messageId, index) => {
    parameters[`visibleMessage${index}`] = messageId;
    return `@visibleMessage${index}`;
  });
  const visibleLinkClause = visiblePlaceholders.length > 0
    ? `OR EXISTS (
         SELECT 1
           FROM model_request_message_link AS visible_link
          WHERE visible_link.model_request_id = request.id
            AND visible_link.message_id IN (${visiblePlaceholders.join(',')})
       )`
    : '';
  return queryPlainRows(database, `
    WITH independent_request_tail AS (
      SELECT request.id
        FROM model_request AS request
        JOIN turn AS owner_turn ON owner_turn.id = request.turn_id
       WHERE owner_turn.conversation_id = @conversationId
         AND (
           request.status <> 'terminal'
           OR NOT EXISTS (
             SELECT 1
               FROM model_request_message_link AS any_link
              WHERE any_link.model_request_id = request.id
           )
         )
       ORDER BY owner_turn.created_at DESC, request.request_seq DESC, request.id DESC
       LIMIT ${CLIENT_ACTIVE_RECORD_LIMIT_PER_TYPE}
    )
    SELECT request.*
      FROM model_request AS request
      JOIN turn AS owner_turn ON owner_turn.id = request.turn_id
     WHERE owner_turn.conversation_id = @conversationId
       AND (
         request.id IN (SELECT id FROM independent_request_tail)
         ${visibleLinkClause}
       )
     ORDER BY owner_turn.created_at DESC, request.request_seq DESC, request.id DESC
  `, parameters).reverse();
}

function wireJsonBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value, (_key, nested) =>
    typeof nested === 'bigint' ? nested.toString() : nested
  ), 'utf8');
}

function requireNonNegativeIntegerString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new TypeError(`${label} must be a decimal integer string.`);
  }
  return value;
}

function executeSnapshot(
  database: Database.Database,
  reads: RepositoryRead[],
  commitSeq: bigint
): SnapshotBarrier<Array<DomainRow | DomainRow[] | null>> {
  if (!Array.isArray(reads)) throw new TypeError('Snapshot reads must be an array.');
  database.exec('BEGIN');
  try {
    const snapshot = reads.map((read) => executeRead(database, read));
    database.exec('COMMIT');
    return { snapshotCommitSeq: commitSeq.toString(), snapshot };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function executeSnapshotAll(
  database: Database.Database,
  read: RepositoryListRead,
  commitSeq: bigint
): SnapshotBarrier<DomainRow[]> {
  if (read.orderBy?.column !== 'id' || read.orderBy.direction !== 'asc') {
    throw new TypeError('snapshotAll requires id ascending order.');
  }
  database.exec('BEGIN');
  try {
    const rows: DomainRow[] = [];
    let afterId = read.afterId;
    for (;;) {
      const page = executeRead(database, { ...read, ...(afterId ? { afterId } : {}) });
      if (!Array.isArray(page)) throw new TypeError('snapshotAll list did not return rows.');
      rows.push(...page);
      if (page.length < read.limit) break;
      const lastId = page[page.length - 1]?.id;
      if (typeof lastId !== 'string' || lastId.length === 0 || lastId === afterId) {
        throw new Error('snapshotAll pagination did not advance.');
      }
      afterId = lastId;
    }
    database.exec('COMMIT');
    return { snapshotCommitSeq: commitSeq.toString(), snapshot: rows };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function executeToolFactsSnapshot(
  database: Database.Database,
  toolCallIdInput: string,
  commitSeq: bigint
): SnapshotBarrier<ToolFactsSnapshot> {
  const toolCallId = requireRuntimeId(toolCallIdInput);
  database.exec('BEGIN');
  try {
    const toolCallRead = executeRead(
      database,
      DOMAIN_REPOSITORIES.domain('ToolCall').get(toolCallId)
    );
    if (Array.isArray(toolCallRead)) throw new TypeError('ToolCall fixed read returned a list.');
    const executionsRead = executeRead(
      database,
      DOMAIN_REPOSITORIES.domain('ToolExecution').list({
        where: { tool_call_id: toolCallId },
        limit: 2
      })
    );
    if (!Array.isArray(executionsRead)) throw new TypeError('ToolExecution fixed read did not return a list.');

    let turn: DomainRow | null = null;
    let leases: DomainRow[] = [];
    let conversation: DomainRow | null = null;
    if (toolCallRead) {
      const turnId = requireRuntimeId(toolCallRead.turn_id);
      const turnRead = executeRead(database, DOMAIN_REPOSITORIES.domain('Turn').get(turnId));
      if (Array.isArray(turnRead)) throw new TypeError('Turn fixed read returned a list.');
      turn = turnRead;
      const leasesRead = executeRead(
        database,
        DOMAIN_REPOSITORIES.domain('ExecutionLease').list({
          where: { turn_id: turnId },
          limit: 2
        })
      );
      if (!Array.isArray(leasesRead)) throw new TypeError('ExecutionLease fixed read did not return a list.');
      leases = leasesRead;
      if (turn) {
        const conversationId = requireRuntimeId(turn.conversation_id);
        const conversationRead = executeRead(
          database,
          DOMAIN_REPOSITORIES.domain('Conversation').get(conversationId)
        );
        if (Array.isArray(conversationRead)) throw new TypeError('Conversation fixed read returned a list.');
        conversation = conversationRead;
      }
    }
    database.exec('COMMIT');
    return {
      snapshotCommitSeq: commitSeq.toString(),
      snapshot: {
        toolCall: toolCallRead,
        executions: executionsRead,
        turn,
        leases,
        conversation
      }
    };
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function executeProcessOutputRegistrationMismatches(
  database: Database.Database
): ProcessOutputRegistrationMismatch[] {
  const rows = database.prepare(`
    SELECT process.id AS process_id,
           CAST(process.retained_chunks AS TEXT) AS expected_chunks,
           CAST(COUNT(chunk.id) AS TEXT) AS registered_chunks,
           CAST(process.retained_bytes AS TEXT) AS expected_bytes,
           CAST(COALESCE(SUM(chunk.byte_length), 0) AS TEXT) AS registered_bytes
      FROM process
      LEFT JOIN process_output_chunk AS chunk ON chunk.process_id = process.id
     WHERE process.status IN ('exited', 'cancelled', 'timed_out', 'output_limit_exceeded')
     GROUP BY process.id, process.retained_chunks, process.retained_bytes
    HAVING COUNT(chunk.id) <> process.retained_chunks
        OR COALESCE(SUM(chunk.byte_length), 0) <> process.retained_bytes
        OR (process.retained_chunks > 0 AND MIN(chunk.chunk_seq) <> 1)
        OR (process.retained_chunks > 0 AND MAX(chunk.chunk_seq) <> process.retained_chunks)
     ORDER BY process.id ASC
  `).all() as Array<{
    process_id: string;
    expected_chunks: string;
    registered_chunks: string;
    expected_bytes: string;
    registered_bytes: string;
  }>;
  return rows.map((row) => ({
    processId: requireRuntimeId(row.process_id),
    expectedChunks: requireNonNegativeIntegerString(row.expected_chunks, 'Process.retained_chunks'),
    registeredChunks: requireNonNegativeIntegerString(row.registered_chunks, 'registered ProcessOutputChunk count'),
    expectedBytes: requireNonNegativeIntegerString(row.expected_bytes, 'Process.retained_bytes'),
    registeredBytes: requireNonNegativeIntegerString(row.registered_bytes, 'registered ProcessOutputChunk bytes')
  }));
}

function executeEffectReceiptReconciliationCandidates(
  database: Database.Database
): EffectReceiptReconciliationCandidate[] {
  const rows = database.prepare(`
    SELECT intent.id AS effect_intent_id, receipt.id AS effect_receipt_id
      FROM tool_call AS tool_call_row INDEXED BY ix_tool_call_02
      CROSS JOIN operation AS operation_row
      CROSS JOIN attempt AS attempt_row
      CROSS JOIN effect_intent AS intent
      CROSS JOIN effect_receipt AS receipt
     WHERE tool_call_row.status IN ('pending', 'executing', 'waiting_approval', 'waiting_answer')
       AND operation_row.tool_call_id = tool_call_row.id
       AND attempt_row.operation_id = operation_row.id
       AND intent.attempt_id = attempt_row.id
       AND receipt.attempt_id = attempt_row.id
       AND intent.dispatch_state = 'receipt_written'
       AND intent.effect_kind <> 'subagent_spawn'
       AND (
         operation_row.status IN ('pending', 'executing', 'waiting_answer')
         OR tool_call_row.status = 'executing'
       )
    UNION ALL
    SELECT intent.id AS effect_intent_id, receipt.id AS effect_receipt_id
      FROM operation AS operation_row INDEXED BY ix_operation_03
      CROSS JOIN attempt AS attempt_row
      CROSS JOIN effect_intent AS intent
      CROSS JOIN effect_receipt AS receipt
     WHERE operation_row.tool_call_id IS NULL
       AND operation_row.status IN ('pending', 'executing', 'waiting_answer')
       AND attempt_row.operation_id = operation_row.id
       AND intent.attempt_id = attempt_row.id
       AND receipt.attempt_id = attempt_row.id
       AND intent.dispatch_state = 'receipt_written'
       AND intent.effect_kind <> 'subagent_spawn'
     ORDER BY effect_intent_id ASC
  `).all() as Array<{ effect_intent_id: string; effect_receipt_id: string }>;
  return rows.map((row) => ({
    effectIntentId: requireRuntimeId(row.effect_intent_id),
    effectReceiptId: requireRuntimeId(row.effect_receipt_id)
  }));
}

function executeChildConversationOriginCandidates(
  database: Database.Database
): ChildConversationOriginCandidate[] {
  const rows = database.prepare(`
    SELECT child.id AS child_execution_id
      FROM child_execution AS child
      LEFT JOIN conversation_origin_link AS origin
        ON origin.conversation_id = child.child_conversation_id
     WHERE origin.id IS NULL
     ORDER BY child.id ASC
  `).all() as Array<{ child_execution_id: string }>;
  return rows.map((row) => ({ childExecutionId: requireRuntimeId(row.child_execution_id) }));
}

function executeChildProcessCleanupMaterializationCandidates(
  database: Database.Database
): ChildProcessCleanupMaterializationCandidate[] {
  const rows = database.prepare(`
    SELECT turn_link.id AS turn_link_id,
           turn_link.interruption_request_id AS interruption_request_id,
           turn_link.turn_id AS turn_id,
           source_link.id AS source_link_id,
           source_link.process_id AS process_id
      FROM child_interruption_turn_link AS turn_link
      JOIN process_completion_source_link AS source_link
        ON source_link.source_turn_id = turn_link.turn_id
      LEFT JOIN child_interruption_process_cleanup AS cleanup
        ON cleanup.interruption_request_id = turn_link.interruption_request_id
       AND cleanup.process_id = source_link.process_id
     WHERE cleanup.id IS NULL
     ORDER BY turn_link.id ASC, source_link.id ASC
  `).all() as Array<{
    turn_link_id: string;
    interruption_request_id: string;
    turn_id: string;
    source_link_id: string;
    process_id: string;
  }>;
  return rows.map((row) => ({
    turnLinkId: requireRuntimeId(row.turn_link_id),
    interruptionRequestId: requireRuntimeId(row.interruption_request_id),
    turnId: requireRuntimeId(row.turn_id),
    sourceLinkId: requireRuntimeId(row.source_link_id),
    processId: requireRuntimeId(row.process_id)
  }));
}

function executeRead(database: Database.Database, read: RepositoryRead): DomainRow | DomainRow[] | null {
  const repository = DOMAIN_REPOSITORIES.domain(read.domain);
  const schema = repository.schema;
  if (read.kind === 'get') {
    const row = database.prepare(`SELECT * FROM ${quote(schema.table)} WHERE id = ?`).get(requireRuntimeId(read.id));
    return row ? repository.codec.decode(row as Record<string, unknown>) : null;
  }
  if (!Number.isSafeInteger(read.limit) || read.limit <= 0 || read.limit > 1000) {
    throw new RangeError('Repository list limit must be an integer from 1 to 1000.');
  }
  const encodedWhere = repository.codec.encodeWhere(read.where ?? {});
  const predicates: string[] = [];
  const parameters: EncodedRow & { __after_id?: string; __limit?: bigint } = {};
  for (const [name, value] of Object.entries(encodedWhere)) {
    if (value === null) predicates.push(`${quote(name)} IS NULL`);
    else {
      predicates.push(`${quote(name)} = @${name}`);
      parameters[name] = value;
    }
  }
  const orderColumn = read.orderBy?.column ?? 'id';
  if (!repository.codec.hasColumn(orderColumn)) throw new Error(`${schema.repository} cannot order by ${orderColumn}.`);
  const direction = read.orderBy?.direction === 'desc' ? 'DESC' : 'ASC';
  if (read.afterId !== undefined) {
    if (orderColumn !== 'id' || direction !== 'ASC') throw new TypeError('Repository afterId pagination requires id ascending order.');
    parameters.__after_id = requireRuntimeId(read.afterId);
    predicates.push(`${quote('id')} > @__after_id`);
  }
  parameters.__limit = BigInt(read.limit);
  const sql = `SELECT * FROM ${quote(schema.table)}${predicates.length ? ` WHERE ${predicates.join(' AND ')}` : ''} ORDER BY ${quote(orderColumn)} ${direction} LIMIT @__limit`;
  return (database.prepare(sql).all(parameters) as Array<Record<string, unknown>>).map((row) => repository.codec.decode(row));
}

function assertPublishedContentObject(row: EncodedRow, casRootPath: string): void {
  const digest = row.sha256;
  const storageKey = row.storage_key;
  const byteLength = row.byte_length;
  if (typeof digest !== 'string' || !/^[a-f0-9]{64}$/.test(digest)) throw new Error('ContentObject.sha256 must be lowercase SHA-256.');
  const expectedKey = `sha256/${digest.slice(0, 2)}/${digest}`;
  if (storageKey !== expectedKey) throw new Error('ContentObject.storage_key does not match its digest.');
  if (typeof byteLength !== 'bigint' || byteLength < 0n) throw new Error('ContentObject.byte_length must be non-negative.');
  const absolutePath = path.resolve(casRootPath, ...expectedKey.split('/'));
  if (!absolutePath.startsWith(`${path.resolve(casRootPath)}${path.sep}`)) throw new Error('ContentObject CAS path escapes the active root.');
  const stat = fs.statSync(absolutePath);
  if (!stat.isFile() || BigInt(stat.size) !== byteLength) throw new Error('ContentObject CAS file is missing or has the wrong length.');
}

function postMeasuredResponse(
  response: Extract<DatabaseWorkerResponse, { type: 'response' }>,
  enqueuedAtMs: number | undefined,
  receivedAtMs: number | undefined,
  transferList: readonly ArrayBuffer[] = []
): void {
  if (receivedAtMs === undefined || enqueuedAtMs === undefined || !Number.isFinite(enqueuedAtMs)) {
    post(response, transferList);
    return;
  }
  post({
    ...response,
    timing: {
      queueWaitMs: Math.max(0, receivedAtMs - enqueuedAtMs),
      executeDurationMs: Math.max(0, performance.now() - receivedAtMs)
    }
  }, transferList);
}

function post(message: DatabaseWorkerResponse, transferList: readonly ArrayBuffer[] = []): void {
  port.postMessage(message, transferList);
}

function requireParentPort(): NonNullable<typeof parentPort> {
  if (!parentPort) throw new Error('SQLite database worker requires parentPort.');
  return parentPort;
}

function serializeError(error: unknown): SerializedWorkerError {
  const value = error as { name?: unknown; message?: unknown; stack?: unknown; code?: unknown };
  return {
    name: typeof value?.name === 'string' ? value.name : 'Error',
    message: typeof value?.message === 'string' ? value.message : String(error),
    ...(typeof value?.stack === 'string' ? { stack: value.stack } : {}),
    ...(typeof value?.code === 'string' ? { code: value.code } : {})
  };
}

function requireEncodedId(value: EncodedRow[string], codecName: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${codecName}.id must be a non-empty string.`);
  return value;
}

function requireRuntimeId(value: unknown): string {
  if (typeof value !== 'string' || value.length === 0) throw new TypeError('Runtime row id must be a non-empty string.');
  return value;
}

function sqlText(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

function quote(identifier: string): string {
  if (!/^[a-z][a-z0-9_]*$/.test(identifier)) throw new Error(`Unsafe SQLite identifier: ${identifier}`);
  return `"${identifier}"`;
}
