import { createHash } from 'node:crypto';
import type { MessageContent } from '../../shared/protocol';
import { mapSettledWithBoundedConcurrency } from '../capabilities/boundedConcurrency';
import { classifyCommandCall } from '../world/modules/tools/definitions/command';
import type { RuntimeDeliveryControlPlane } from './answerDelivery';
import { AutomaticRuntimeDeliveryRouter } from './automaticRuntimeDelivery';
import { ContentAddressedStore, type ContentObjectMetadata } from './contentAddressedStore';
import { ContextSequenceControlPlane } from './contextSequence';
import { estimateStoredMessageContentTokens } from './contextTokenEstimator';
import {
  compareGuidancePositions,
  initialGuidancePosition,
  parseInputTurnIntentEnvelope,
  TURN_INTENT_ENVELOPE_CONTENT_TYPE
} from './guidanceIntent';
import {
  EffectControlPlane,
  type FrozenToolCallPolicyDecision,
  type ToolOutcomeStatus,
  type ToolTerminalResult
} from './effectControlPlane';
import {
  ModelRequestPreflightError,
  ModelProviderControlPlane,
  modelRequestIdFor,
  type FullRequestProviderAdapter,
  type ProviderStreamEvent,
  type StreamEventResult
} from './modelProviderControlPlane';
import {
  readCurrentTurnTaskCard,
  shouldInjectTurnTaskCard,
  type TurnTaskCardReminderState
} from './currentTurnTaskProjection';
import { canonicalPlainJson, normalizePlainJson, type PlainJsonValue } from './plainJson';
import { DOMAIN_REPOSITORIES, type DomainRow } from './repositories';
import { listAllDomainRows } from './repositoryPagination';
import { RuntimeDatabase } from './runtimeDatabase';
import {
  isTurnTerminalGuidanceConflictError,
  isTurnTerminalInputConflictError,
  TurnControlPlane,
  type TurnInputCommand
} from './turnControlPlane';
import { assistantMessageIdFor, TurnOutputControlPlane } from './turnOutput';
import { ExecutionHandoffError, isExecutionHandoffError } from './executionLeaseFence';
import type {
  CoordinateCompressionCommand,
  CoordinateCompressionResult
} from './contextCompressionCoordinator';

export interface ReliableAgentToolDefinition {
  name: string;
  description: string;
  parameters: PlainJsonValue;
  /** Credential-free source identity used for frozen dynamic MCP policy evaluation. */
  source?: PlainJsonValue;
  /** Plain definition facts frozen into the ModelRequest recipe. */
  metadata?: PlainJsonValue;
  defaultConfig?: PlainJsonValue;
}

export interface ReliableAgentProviderRegistry {
  resolve(providerId: string): Promise<FullRequestProviderAdapter> | FullRequestProviderAdapter;
  dispose?(): Promise<void> | void;
}

export interface ReliableAgentCompressionCoordinator {
  coordinate(command: CoordinateCompressionCommand): Promise<CoordinateCompressionResult>;
}

export interface ReliableAgentToolDispatchInput {
  turnId: string;
  modelRequestId: string;
  toolCallId: string;
  providerCallId?: string;
  toolName: string;
  arguments: PlainJsonValue;
}

export interface ReliableAgentToolPause {
  disposition: 'paused';
  toolCallId: string;
  reason: 'awaiting_user' | 'awaiting_approval' | 'awaiting_plan_review' | 'awaiting_child' | 'background_process' | 'converging';
  resumeKey?: string;
}

export interface ReliableAgentToolSettled {
  disposition: 'settled';
  toolCallId: string;
  status: ToolOutcomeStatus;
}

/** Dispatcher owns capability-specific EffectIntent/Receipt semantics and may durably pause the Turn. */
export interface ReliableAgentToolDispatcher {
  /** turnId selects definitions through that Turn's immutable authority snapshot. */
  definitions(turnId?: string): Promise<ReliableAgentToolDefinition[]> | ReliableAgentToolDefinition[];
  /** Compiles display/gate/scheduling for a Provider batch from one immutable authority read. */
  freezeCalls?(inputs: ReadonlyArray<ReliableAgentToolDispatchInput & {
    definition: ReliableAgentToolDefinition;
  }>): Promise<FrozenToolCallPolicyDecision[]>;
  /** Compiles display/gate/scheduling from the immutable Turn authority and frozen recipe definition. */
  freezeCall?(input: ReliableAgentToolDispatchInput & {
    definition: ReliableAgentToolDefinition;
  }): Promise<FrozenToolCallPolicyDecision>;
  /** Dispatches one already-frozen parallel group while sharing read-only preflight/finalization work. */
  dispatchBatch?(inputs: readonly ReliableAgentToolDispatchInput[]): Promise<Array<ToolTerminalResult | ReliableAgentToolPause | ReliableAgentToolSettled>>;
  dispatch(input: ReliableAgentToolDispatchInput): Promise<ToolTerminalResult | ReliableAgentToolPause | ReliableAgentToolSettled>;
  /** Prewired cancellation boundary; Runner may invoke it without knowing capability internals. */
  cancelActive?(input: { turnId: string; reason: string }): Promise<void> | void;
  /** Host handoff aborts local waits without inventing a user cancellation or terminal Turn. */
  quiesceTurn?(input: { turnId: string; reason: ExecutionHandoffError }): Promise<void> | void;
  quiesce?(reason: ExecutionHandoffError): Promise<void> | void;
  /** Closes capability-specific durable waits before a terminal Turn asserts every ToolCall is terminal. */
  cancelWaiting?(input: { turnId: string; sourceKey: string; reason: string }): Promise<void>;
  dispose?(): Promise<void> | void;
}

export interface ReliableAgentTransientEvent {
  conversationId: string;
  turnId: string;
  modelRequestId: string;
  requestSeq: string;
  providerId: string;
  modelId: string;
  attemptSeq: string;
  socketGeneration: string;
  /** Durable commit frontier visible before this Provider socket was dispatched. */
  afterCommitSeq: string;
  event: ProviderStreamEvent;
  observedAt: string;
}

export interface ReliableAgentTransientObserver {
  observe(event: ReliableAgentTransientEvent): void;
}

export type ReliableAgentLifecycleStage =
  | 'drive_started'
  | 'round_facts_ready'
  | 'provider_dispatch_started'
  | 'provider_output_ready'
  | 'assistant_commit_started'
  | 'assistant_commit_completed'
  | 'tool_dispatch_started'
  | 'tool_dispatch_completed'
  | 'tool_model_result_committed'
  | 'terminal_prefix_scanned'
  | 'context_tool_pair_committed'
  | 'turn_terminal_started'
  | 'turn_terminal_completed'
  | 'open_tasks_at_final'
  | 'drive_failed'
  | 'failure_terminal_started'
  | 'failure_terminal_completed'
  | 'failure_terminal_failed';

/** Bounded metadata-only diagnostics. No prompt, model output, tool arguments, or credentials are exposed. */
export interface ReliableAgentLifecycleEvent {
  turnId: string;
  stage: ReliableAgentLifecycleStage;
  observedAt: string;
  /** Durable ModelRequest sequence. Decimal string because runtime sequences must not cross JS number. */
  round?: string;
  modelRequestId?: string;
  toolCallId?: string;
  /** Present when one dispatcher call owns a Provider parallel group. */
  toolBatchSize?: number;
  schedulingMode?: 'parallel' | 'serial';
  terminalCallsScanned?: number;
  terminalPrefixCursor?: number;
  contextPairCount?: number;
  contextTransactionCount?: number;
  openTaskCount?: number;
  taskCardSha256?: string;
  activeChildCount?: number;
  runningProcessCount?: number;
  errorName?: string;
  errorMessage?: string;
}

export interface ReliableAgentLifecycleObserver {
  observe(event: ReliableAgentLifecycleEvent): void;
}

export interface ReliableAgentLoopResult {
  turnId: string;
  terminalStatus: 'completed' | 'failed' | 'interrupted' | 'waiting';
  modelRequestIds: string[];
  assistantMessageIds: string[];
  toolCallIds: string[];
  waitingToolCallId?: string;
}

interface NormalizedToolCall {
  providerCallId?: string;
  providerOrdinal: number;
  name: string;
  arguments: PlainJsonValue;
  thoughtSignature?: string;
}

interface NormalizedProviderOutput {
  content: MessageContent;
  toolCalls: NormalizedToolCall[];
  usage?: PlainJsonValue;
}

interface FrozenProviderToolCall extends NormalizedToolCall {
  toolCallId: string;
  policy: FrozenToolCallPolicyDecision;
}

interface FrozenCurrentTurnInputReference {
  kind: 'current_turn_input';
  messageId: string;
  messageRevisionId: string;
  contentObjectId: string;
  estimatedTokens: number;
  reinject: boolean;
}

interface CurrentTurnRequestState {
  reference?: FrozenCurrentTurnInputReference;
  compressionBoundaryId?: string;
}

interface FrozenRuntimeStatusCard {
  kind: 'runtime_status_card';
  activeChildCount: number;
  runningProcessCount: number;
  children: Array<{ childExecutionId: string; answerBridgeId?: string; status: string }>;
  processes: Array<{ processId: string; status: 'running' }>;
  card: string;
}

const MESSAGE_CONTENT_TYPE = 'application/vnd.limcode.message+json';
const RUNTIME_STATUS_RECIPE_LIMIT = 32;
const RUNTIME_STATUS_CARD_LIMIT = 4;

/**
 * 单 Turn 的可靠 Agent loop。每轮都冻结 Context root/authority，Provider 完成摘要先落 SQLite/CAS，
 * 再幂等提交 assistant Message；工具结果按 call_seq 持久化并追加 Context tool_pair。
 */
export class ReliableAgentLoop {
  private readonly context: ContextSequenceControlPlane;
  private readonly automaticDeliveries: AutomaticRuntimeDeliveryRouter;
  private readonly now: () => string;
  private readonly reconcileCommittedToolCall:
    | ((toolCallId: string) => Promise<ToolTerminalResult | null>)
    | undefined;

  public constructor(
    private readonly database: RuntimeDatabase,
    private readonly contentStore: ContentAddressedStore,
    private readonly turns: TurnControlPlane,
    private readonly turnOutput: TurnOutputControlPlane,
    private readonly modelProvider: ModelProviderControlPlane,
    private readonly effects: EffectControlPlane,
    private readonly runtimeDeliveries: RuntimeDeliveryControlPlane,
    private readonly providers: ReliableAgentProviderRegistry,
    private readonly compressionCoordinator: ReliableAgentCompressionCoordinator,
    private readonly tools: ReliableAgentToolDispatcher,
    private readonly transientObserver?: ReliableAgentTransientObserver,
    private readonly lifecycleObserver?: ReliableAgentLifecycleObserver,
    options: {
      now?: () => string;
      reconcileCommittedToolCall?: (toolCallId: string) => Promise<ToolTerminalResult | null>;
    } = {}
  ) {
    this.now = options.now ?? (() => new Date().toISOString());
    this.reconcileCommittedToolCall = options.reconcileCommittedToolCall;
    this.context = new ContextSequenceControlPlane(database, contentStore, options);
    this.automaticDeliveries = new AutomaticRuntimeDeliveryRouter(database);
  }

  public async runInput(command: TurnInputCommand): Promise<ReliableAgentLoopResult> {
    const started = await this.turns.input(command);
    const turnId = requireId(started.turnId, 'Turn input result.turnId');
    return this.drive(turnId);
  }

  /** Safe for explicit recovery/re-entry; every round and output identity is deterministic. */
  public async drive(turnIdInput: string): Promise<ReliableAgentLoopResult> {
    const turnId = requireId(turnIdInput, 'turnId');
    const modelRequestIds: string[] = [];
    const assistantMessageIds: string[] = [];
    const toolCallIds: string[] = [];
    this.observeLifecycle({ turnId, stage: 'drive_started' });

    try {
      // ModelRequest.request_seq is the durable loop frontier. A re-entry deliberately starts at
      // the last committed request so a crash between Provider completion, assistant commit, tool
      // settlement and Context append replays that one round idempotently. Only after the replayed
      // round is complete do we advance to request_seq + 1. There is no process-local round cap:
      // safety limits belong to explicit token/cost/time policy, never an invisible failed Turn.
      let requestSequence = await this.resumeRequestSequence(turnId);
      agentRounds: for (;;) {
        const round = requestSequence.toString();
        let facts = await this.readRoundFacts(turnId);
        await this.cancelSupersededCompressionRequests(
          turnId,
          requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id')
        );
        this.observeLifecycle({ turnId, stage: 'round_facts_ready', round });
        if (facts.turn.status !== 'active') {
          return {
            turnId,
            terminalStatus: await this.readLoopTerminalStatus(turnId, facts.turn),
            modelRequestIds,
            assistantMessageIds,
            toolCallIds
          };
        }
        if (await this.terminateIfRequested(turnId, `round:${round}:before-model-request`)) {
          return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
        }
        const idempotencyKey = `agent-loop:${turnId}:round:${round}`;
        const expectedModelRequestId = modelRequestIdFor(turnId, idempotencyKey);
        let request = await this.maybeGet('ModelRequest', expectedModelRequestId);
        if (!request) {
          // Runtime input is admitted only at a new request boundary. On recovery an existing
          // request may still be waiting for its tool results; inserting runtime_context before
          // those results would split the atomic assistant-tool/result pair.
          if (await this.absorbRuntimeDeliveryInputs(turnId) > 0) {
            facts = await this.readRoundFacts(turnId);
          }
          const toolDefinitions = await this.tools.definitions(turnId);
          let frozenRecipe = await this.freezeOrdinaryRequestRecipe({
            turnId,
            round,
            headRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
            tools: toolDefinitions
          });
          let preview = await this.modelProvider.previewOrdinaryRequest({
            turnId,
            contextRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
            authoritySnapshotId: requireId(facts.authority.id, 'AuthoritySnapshot.id'),
            recipe: frozenRecipe,
            idempotencyKey
          });
          let previewAdapter = await this.providers.resolve(preview.providerId);
          if (previewAdapter.providerId !== preview.providerId) {
            throw new Error(`Provider registry returned ${previewAdapter.providerId} for ${preview.providerId}.`);
          }
          let budget = this.modelProvider.budgetFullRequest(preview, previewAdapter);
          // The adapter estimate protects the physical request limit, but it is only heuristic.
          // Always let the coordinator compare the same frozen head with Provider-observed usage;
          // otherwise an underestimated budget can suppress the only authoritative level trigger.
          const compression = await this.compressionCoordinator.coordinate({
            turnId,
            authoritySnapshotId: requireId(facts.authority.id, 'AuthoritySnapshot.id'),
            headRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
            trigger: 'auto',
            requestBudget: budget,
            protectedCurrentInputTokens: currentInputReferenceTokens(frozenRecipe)
          });
          if (compression.status === 'error') {
            throw new ModelRequestPreflightError(
              compression.code,
              `${compression.code}: ${compression.message}`,
              compression.estimatedTokens,
              compression.limitTokens
            );
          }
          if (compression.status === 'compressed') {
            facts = await this.readRoundFacts(turnId);
            // Delivery that became model-visible while Compact was running belongs after the
            // canonical output. It is absorbed only after the new head CAS has succeeded.
            if (await this.absorbRuntimeDeliveryInputs(turnId) > 0) {
              facts = await this.readRoundFacts(turnId);
            }
            frozenRecipe = await this.freezeOrdinaryRequestRecipe({
              turnId,
              round,
              headRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
              tools: toolDefinitions
            });
            preview = await this.modelProvider.previewOrdinaryRequest({
              turnId,
              contextRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
              authoritySnapshotId: requireId(facts.authority.id, 'AuthoritySnapshot.id'),
              recipe: frozenRecipe,
              idempotencyKey
            });
            previewAdapter = await this.providers.resolve(preview.providerId);
            if (previewAdapter.providerId !== preview.providerId) {
              throw new Error(`Provider registry returned ${previewAdapter.providerId} for ${preview.providerId}.`);
            }
            budget = this.modelProvider.budgetFullRequest(preview, previewAdapter);
          }
          if (!budget.canSend) {
            throw new ModelRequestPreflightError(
              'request_still_too_large',
              `request_still_too_large: rebuilt request is ${budget.estimatedFullInputTokens} tokens, `
                + `safe limit is ${budget.estimatedInputLimitTokens}.`,
              budget.estimatedFullInputTokens,
              budget.estimatedInputLimitTokens
            );
          }
          const created = await this.modelProvider.createModelRequest({
            turnId,
            contextRootId: requireId(facts.head.root_id, 'ConversationContextHeadLink.root_id'),
            authoritySnapshotId: requireId(facts.authority.id, 'AuthoritySnapshot.id'),
            recipe: frozenRecipe,
            projectedEstimatedTokens: budget.estimatedFullInputTokens,
            idempotencyKey
          });
          if (created.modelRequestId !== expectedModelRequestId) {
            throw new Error('ModelProvider returned an unexpected stable ModelRequest identity.');
          }
          request = await this.requireExisting('ModelRequest', expectedModelRequestId);
        }
        const modelRequestRecipe = await this.assertModelRequestRound(
          request,
          requestSequence,
          expectedModelRequestId
        );
        const modelRequestId = expectedModelRequestId;
        modelRequestIds.push(modelRequestId);
        if (await this.terminateIfRequested(turnId, `round:${round}:model-request:${modelRequestId}`)) {
          return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
        }
        let output: NormalizedProviderOutput;
        if (request.status === 'terminal') {
          output = await this.readTerminalProviderOutput(modelRequestId);
        } else {
          this.observeLifecycle({ turnId, stage: 'provider_dispatch_started', round, modelRequestId });
          output = await this.dispatchAndCapture(
            requireId(facts.turn.conversation_id, 'Turn.conversation_id'),
            turnId,
            modelRequestId,
            request
          );
        }
        this.observeLifecycle({ turnId, stage: 'provider_output_ready', round, modelRequestId });
        if (await this.terminateIfRequested(turnId, `round:${round}:provider-output:${modelRequestId}`)) {
          return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
        }
        if (output.toolCalls.length === 0) {
          const fence = await this.automaticDeliveries.establishFinalOutputFence({ turnId, modelRequestId });
          if (!fence.established) {
            if (await this.absorbRuntimeDeliveryInputs(turnId) > 0) {
              requestSequence += 1n;
              continue agentRounds;
            }
            const latestTurn = await this.requireExisting('Turn', turnId);
            if (latestTurn.status !== 'active') {
              return {
                turnId,
                terminalStatus: await this.readLoopTerminalStatus(turnId, latestTurn),
                modelRequestIds,
                assistantMessageIds,
                toolCallIds
              };
            }
            throw new Error(`Turn ${turnId} could not establish final-output authority.`);
          }
          if (await this.terminateIfRequested(turnId, `round:${round}:final-output-fenced:${modelRequestId}`)) {
            return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
          }
        }
        this.observeLifecycle({ turnId, stage: 'assistant_commit_started', round, modelRequestId });
        const message = await this.turnOutput.appendAssistantMessage({
          turnId,
          modelRequestId,
          sourceKey: modelRequestId,
          content: JSON.stringify(providerOutputMessage(output)),
          contentType: MESSAGE_CONTENT_TYPE
        });
        assistantMessageIds.push(message.messageId);
        this.observeLifecycle({ turnId, stage: 'assistant_commit_completed', round, modelRequestId });

        if (output.toolCalls.length === 0) {
          // The final-output fence was committed before this visible Message. Automatic runtime
          // input must now target a new Turn; extending this Turn would rewrite a displayed final.
          for (;;) {
            if (await this.terminateIfRequested(turnId, `round:${round}:before-complete:${modelRequestId}`)) {
              return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
            }
            this.observeLifecycle({ turnId, stage: 'turn_terminal_started', round, modelRequestId });
            try {
              await this.turns.terminal({
                source: { kind: 'internal', key: `agent-loop:${turnId}:complete:${modelRequestId}` },
                turnId,
                terminalStatus: 'completed',
                reason: 'model_completed_without_tool_calls'
              });
            } catch (error) {
              if (isTurnTerminalInputConflictError(error)) {
                if (await this.terminateIfRequested(turnId, `round:${round}:final-output-fenced:${modelRequestId}`)) {
                  return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
                }
                throw new Error(
                  `Runtime input crossed final-output fence for Turn ${turnId}: ${errorMessage(error)}`
                );
              }
              throw error;
            }
            this.observeLifecycle({ turnId, stage: 'turn_terminal_completed', round, modelRequestId });
            this.observeOpenTasksAtFinal(turnId, round, modelRequestId, modelRequestRecipe);
            const terminalTurn = await this.requireExisting('Turn', turnId);
            return {
              turnId,
              terminalStatus: await this.readLoopTerminalStatus(turnId, terminalTurn),
              modelRequestIds,
              assistantMessageIds,
              toolCallIds
            };
          }
        }

        const batch = await this.prepareProviderToolBatch({
          turnId,
          modelRequestId,
          messageId: message.messageId,
          output,
          recipe: modelRequestRecipe
        });
        toolCallIds.push(...batch.map((call) => call.toolCallId));
        if (await this.terminateIfRequested(
          turnId,
          `round:${round}:created-tool-batch`,
          batch[0]?.toolCallId
        )) {
          return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
        }
        const batchDispatch = await this.dispatchProviderToolBatch({
          conversationId: requireId(facts.turn.conversation_id, 'Turn.conversation_id'),
          turnId,
          round,
          modelRequestId,
          calls: batch
        });
        if (batchDispatch.status === 'interrupted') {
          return { turnId, terminalStatus: 'interrupted', modelRequestIds, assistantMessageIds, toolCallIds };
        }
        if (batchDispatch.status === 'waiting') {
          return {
            turnId,
            terminalStatus: 'waiting',
            modelRequestIds,
            assistantMessageIds,
            toolCallIds,
            waitingToolCallId: batchDispatch.toolCallId
          };
        }
        if (await this.completeForQueuedBoundaryInput({
          turnId,
          conversationId: requireId(facts.turn.conversation_id, 'Turn.conversation_id'),
          round,
          modelRequestId
        })) {
          const terminalTurn = await this.requireExisting('Turn', turnId);
          return {
            turnId,
            terminalStatus: await this.readLoopTerminalStatus(turnId, terminalTurn),
            modelRequestIds,
            assistantMessageIds,
            toolCallIds
          };
        }
        requestSequence += 1n;
      }
    } catch (error) {
      // Host shutdown / lease replacement is a recoverable transport handoff. Recording a failed
      // Turn here would destroy the exact durable frontier the next Host needs to resume.
      if (isExecutionHandoffError(error)) throw error;
      let interruptionCheckError: unknown;
      try {
        if (await this.terminateIfRequested(turnId, 'drive-interrupted')) {
          const terminalTurn = await this.requireExisting('Turn', turnId);
          return {
            turnId,
            terminalStatus: await this.readLoopTerminalStatus(turnId, terminalTurn),
            modelRequestIds,
            assistantMessageIds,
            toolCallIds
          };
        }
      } catch (checkError) {
        interruptionCheckError = checkError;
      }
      // A durable interrupt wins over the transport AbortError that it deliberately caused. Only a
      // genuine unrequested failure is allowed to enter the drive_failed terminal path.
      this.observeLifecycle({ turnId, stage: 'drive_failed', ...errorDiagnostic(error) });
      try {
        if (interruptionCheckError !== undefined) throw interruptionCheckError;
        this.observeLifecycle({ turnId, stage: 'failure_terminal_started' });
        if (!await this.terminateIfRequested(turnId, 'drive-failed')) {
          await this.failActiveTurn(turnId, error);
        }
        this.observeLifecycle({ turnId, stage: 'failure_terminal_completed' });
      } catch (terminalError) {
        this.observeLifecycle({ turnId, stage: 'failure_terminal_failed', ...errorDiagnostic(terminalError) });
        const combined = new Error(`Reliable Agent Turn ${turnId} failed and could not record terminal state.`) as Error & {
          originalError?: unknown;
          terminalError?: unknown;
        };
        combined.originalError = error;
        combined.terminalError = terminalError;
        throw combined;
      }
      const terminalTurn = await this.requireExisting('Turn', turnId);
      return {
        turnId,
        terminalStatus: await this.readLoopTerminalStatus(turnId, terminalTurn),
        modelRequestIds,
        assistantMessageIds,
        toolCallIds
      };
    }
  }

  private async freezeOrdinaryRequestRecipe(input: {
    turnId: string;
    round: string;
    headRootId: string;
    tools: readonly ReliableAgentToolDefinition[];
  }): Promise<PlainJsonValue> {
    const [currentTurnState, runtimeStatusCard, turnTaskCard, previousTaskCard] = await Promise.all([
      this.readCurrentTurnInputReference(input.turnId, input.headRootId),
      this.readRuntimeStatusCard(input.turnId),
      readCurrentTurnTaskCard(this.database, this.contentStore, input.turnId),
      this.readPreviousTaskCardReminderStateForRound(input.turnId, input.round)
    ]);
    const boundaryKey = currentTurnState.compressionBoundaryId ?? 'pre-compression';
    const turnTaskCardReminderEnabled = turnTaskCard
      ? shouldInjectTurnTaskCard({
          revision: turnTaskCard.revision,
          cardSha256: turnTaskCard.cardSha256,
          boundaryKey
        }, previousTaskCard)
      : false;
    return normalizePlainJson({
      kind: 'reliable-agent-turn',
      projectionRevision: '2026-08-09',
      round: input.round,
      tools: input.tools,
      ...(currentTurnState.reference ? { currentTurnInput: currentTurnState.reference } : {}),
      ...(turnTaskCard ? {
        turnTaskCard,
        turnTaskCardBoundaryKey: boundaryKey,
        turnTaskCardReminderEnabled
      } : {}),
      ...(runtimeStatusCard ? { runtimeStatusCard } : {})
    }, 'Reliable Agent recipe');
  }

  private async readPreviousTaskCardReminderStateForRound(
    turnId: string,
    round: string
  ): Promise<TurnTaskCardReminderState | undefined> {
    const currentRound = requirePositiveInteger(round, 'ModelRequest recipe.round');
    if (currentRound <= 1n) return undefined;
    const previousId = modelRequestIdFor(
      turnId,
      `agent-loop:${turnId}:round:${(currentRound - 1n).toString()}`
    );
    const previousRequest = await this.maybeGet('ModelRequest', previousId);
    if (!previousRequest) return undefined;
    const recipe = (await this.readModelRequestRecipes([previousRequest])).get(previousId);
    if (!recipe || recipe.kind !== 'reliable-agent-turn') return undefined;
    const task = asRecord(recipe.turnTaskCard);
    const revision = typeof task?.revision === 'string' ? task.revision : undefined;
    const cardSha256 = typeof task?.cardSha256 === 'string' ? task.cardSha256 : undefined;
    const boundaryKey = typeof recipe.turnTaskCardBoundaryKey === 'string'
      ? recipe.turnTaskCardBoundaryKey
      : 'pre-compression';
    if (!revision || !cardSha256) return undefined;
    return { revision, cardSha256, boundaryKey };
  }

  private async readCurrentTurnInputReference(
    turnId: string,
    headRootId: string
  ): Promise<CurrentTurnRequestState> {
    const current = await this.context.materializeStructure(requireId(headRootId, 'headRootId'));
    const firstSegment = current.records[0]?.segment;
    const compressionBoundaryId = firstSegment?.segment_kind === 'compression'
      ? requireId(firstSegment.id, 'ContextSegment.id')
      : undefined;
    const inputLinks = await this.list('MessageTurnLink', { turn_id: turnId, role: 'input' }, 2);
    if (inputLinks.length === 0) return { compressionBoundaryId };
    if (inputLinks.length !== 1) throw new Error(`Turn ${turnId} must have at most one input Message.`);
    const messageId = requireId(inputLinks[0].message_id, 'MessageTurnLink.message_id');
    const currentLinks = await this.list('MessageCurrentRevisionLink', { message_id: messageId }, 2);
    if (currentLinks.length !== 1) throw new Error(`Input Message ${messageId} must have one current revision.`);
    const messageRevisionId = requireId(
      currentLinks[0].revision_id,
      'MessageCurrentRevisionLink.revision_id'
    );
    const snapshot = await this.database.snapshot([
      DOMAIN_REPOSITORIES.domain('MessageRevision').get(messageRevisionId),
      DOMAIN_REPOSITORIES.domain('ContextSegmentSource').list({
        where: { source_kind: 'message_revision', source_id: messageRevisionId }, limit: 8
      })
    ]);
    const revision = requireRow(snapshot.snapshot[0], `MessageRevision ${messageRevisionId}`);
    if (revision.message_id !== messageId || revision.role !== 'user') {
      throw new Error(`Current input revision ${messageRevisionId} conflicts with Turn ${turnId}.`);
    }
    const sources = rows(snapshot.snapshot[1]);
    const currentSegmentIds = new Set(current.records.map((record) =>
      requireId(record.segment.id, 'ContextSegment.id')
    ));
    const presentInCurrentWindow = sources.some((source) =>
      currentSegmentIds.has(requireId(source.segment_id, 'ContextSegmentSource.segment_id'))
    );
    const contentObjectId = requireId(revision.content_object_id, 'MessageRevision.content_object_id');
    const contentObject = await this.requireExisting('ContentObject', contentObjectId) as unknown as ContentObjectMetadata;
    const content = await this.contentStore.read(contentObject);
    return {
      ...(compressionBoundaryId ? { compressionBoundaryId } : {}),
      reference: {
        kind: 'current_turn_input',
        messageId,
        messageRevisionId,
        contentObjectId,
        estimatedTokens: estimateStoredMessageContentTokens(content, contentObject.content_type),
        reinject: !presentInCurrentWindow
      }
    };
  }

  private async readRuntimeStatusCard(turnId: string): Promise<FrozenRuntimeStatusCard | undefined> {
    const [childLinks, processLinks] = await Promise.all([
      listAllDomainRows(this.database, 'ChildExecutionParentLink', { parent_turn_id: turnId }),
      listAllDomainRows(this.database, 'ProcessCompletionSourceLink', { source_turn_id: turnId })
    ]);
    const orderedChildLinks = childLinks
      .sort((left, right) => String(left.child_execution_id).localeCompare(String(right.child_execution_id)));
    const orderedProcessLinks = processLinks
      .sort((left, right) => String(left.process_id).localeCompare(String(right.process_id)));
    const factReads = [
      ...orderedChildLinks.map((link) =>
        DOMAIN_REPOSITORIES.domain('ChildExecution').get(
          requireId(link.child_execution_id, 'ChildExecutionParentLink.child_execution_id')
        )
      ),
      ...orderedProcessLinks.map((link) =>
        DOMAIN_REPOSITORIES.domain('Process').get(
          requireId(link.process_id, 'ProcessCompletionSourceLink.process_id')
        )
      )
    ];
    const facts = factReads.length === 0 ? null : await this.database.snapshot(factReads);
    const activeChildren: Array<{ childExecutionId: string; status: string }> = [];
    for (let index = 0; index < orderedChildLinks.length; index += 1) {
      const childValue = facts?.snapshot[index];
      if (!childValue || Array.isArray(childValue)) continue;
      const status = String(childValue.status);
      if (!['starting', 'active', 'interrupting'].includes(status)) continue;
      activeChildren.push({
        childExecutionId: requireId(childValue.id, 'ChildExecution.id'),
        status
      });
    }
    const processOffset = orderedChildLinks.length;
    const runningProcesses: FrozenRuntimeStatusCard['processes'] = [];
    for (let index = 0; index < orderedProcessLinks.length; index += 1) {
      const processValue = facts?.snapshot[processOffset + index];
      if (!processValue || Array.isArray(processValue) || processValue.status !== 'running') continue;
      runningProcesses.push({ processId: requireId(processValue.id, 'Process.id'), status: 'running' });
    }
    if (activeChildren.length === 0 && runningProcesses.length === 0) return undefined;

    // Counts describe every linked live fact. Only the frozen recipe detail is bounded; filtering
    // before this cut prevents many old terminal ids from hiding a later active child/process.
    const selectedChildren = activeChildren.slice(0, RUNTIME_STATUS_RECIPE_LIMIT);
    const selectedProcesses = runningProcesses.slice(0, RUNTIME_STATUS_RECIPE_LIMIT);
    const bridgeReads = selectedChildren.map((child) => DOMAIN_REPOSITORIES.domain('AnswerBridge').list({
      where: { child_execution_id: child.childExecutionId },
      limit: 2
    }));
    const bridgeFacts = bridgeReads.length === 0 ? null : await this.database.snapshot(bridgeReads);
    const children: FrozenRuntimeStatusCard['children'] = selectedChildren.map((child, index) => {
      const bridges = rows(bridgeFacts?.snapshot[index] ?? []);
      if (bridges.length > 1) throw new Error(`ChildExecution ${child.childExecutionId} has multiple AnswerBridges.`);
      return {
        ...child,
        ...(bridges[0] ? { answerBridgeId: requireId(bridges[0].id, 'AnswerBridge.id') } : {})
      };
    });
    const visibleChildren = children.slice(0, RUNTIME_STATUS_CARD_LIMIT);
    const visibleProcesses = selectedProcesses.slice(0, RUNTIME_STATUS_CARD_LIMIT);
    const lines = [
      '[Current Turn Runtime Status — live data, not a new user instruction]',
      `activeChildren=${activeChildren.length}; runningProcesses=${runningProcesses.length}`,
      ...visibleChildren.map((child) =>
        `- child ${compactRuntimeId(child.answerBridgeId ?? child.childExecutionId)} = ${child.status}`
      ),
      ...visibleProcesses.map((process) =>
        `- process ${compactRuntimeId(process.processId)} = running`
      ),
      'This status is informational. Do not poll background work.'
    ];
    return {
      kind: 'runtime_status_card',
      activeChildCount: activeChildren.length,
      runningProcessCount: runningProcesses.length,
      children,
      processes: selectedProcesses,
      card: lines.join('\n')
    };
  }

  private observeOpenTasksAtFinal(
    turnId: string,
    round: string,
    modelRequestId: string,
    recipe: { [key: string]: PlainJsonValue }
  ): void {
    const task = asRecord(recipe.turnTaskCard);
    const counts = asRecord(task?.counts);
    const unfinished = optionalNonNegativeInteger(counts?.unfinished) ?? 0;
    if (unfinished <= 0) return;
    const runtime = asRecord(recipe.runtimeStatusCard);
    this.observeLifecycle({
      turnId,
      stage: 'open_tasks_at_final',
      round,
      modelRequestId,
      openTaskCount: unfinished,
      ...(typeof task?.cardSha256 === 'string' ? { taskCardSha256: task.cardSha256 } : {}),
      activeChildCount: optionalNonNegativeInteger(runtime?.activeChildCount) ?? 0,
      runningProcessCount: optionalNonNegativeInteger(runtime?.runningProcessCount) ?? 0
    });
  }

  private async prepareProviderToolBatch(input: {
    turnId: string;
    modelRequestId: string;
    messageId: string;
    output: NormalizedProviderOutput;
    recipe?: { [key: string]: PlainJsonValue };
  }): Promise<FrozenProviderToolCall[]> {
    const definitions = await this.readModelRequestToolDefinitions(input.modelRequestId, input.recipe);
    const definitionsByName = new Map(definitions.map((definition) => [definition.name, definition]));
    const existingLinks = await listAllDomainRows(
      this.database,
      'ToolCallSourceLink',
      { model_request_id: input.modelRequestId }
    );
    if (existingLinks.length !== 0 && existingLinks.length !== input.output.toolCalls.length) {
      throw new Error(`ModelRequest ${input.modelRequestId} has an incomplete durable ToolCall batch.`);
    }
    const existingByOrdinal = new Map(existingLinks.map((link) => [
      requireNonNegativeSafeNumber(link.provider_ordinal, 'ToolCallSourceLink.provider_ordinal'),
      link
    ]));
    const calls: Array<FrozenProviderToolCall | undefined> = new Array(input.output.toolCalls.length);
    const pending: Array<{
      index: number;
      call: NormalizedProviderOutput['toolCalls'][number];
      toolCallId: string;
      dispatchInput: ReliableAgentToolDispatchInput & { definition: ReliableAgentToolDefinition };
    }> = [];
    for (let index = 0; index < input.output.toolCalls.length; index += 1) {
      const call = input.output.toolCalls[index];
      const toolCallId = providerToolCallId(input.modelRequestId, call);
      const existingLink = existingByOrdinal.get(call.providerOrdinal);
      if (existingLink) {
        if (
          existingLink.tool_call_id !== toolCallId
          || existingLink.message_id !== input.messageId
          || existingLink.provider_call_id !== (call.providerCallId ?? null)
          || existingLink.thought_signature !== (call.thoughtSignature ?? null)
        ) throw new Error(`Provider ToolCall source replay conflicts at ordinal ${call.providerOrdinal}.`);
        const rows = await this.list('ToolCallPolicySnapshot', { tool_call_id: toolCallId }, 2);
        if (rows.length !== 1) throw new Error(`ToolCall ${toolCallId} must have one frozen policy snapshot.`);
        calls[index] = { ...call, toolCallId, policy: frozenPolicyFromRow(rows[0]) };
        continue;
      }
      const definition = definitionsByName.get(call.name) ?? unknownToolDefinition(call.name);
      pending.push({
        index,
        call,
        toolCallId,
        dispatchInput: {
          turnId: input.turnId,
          modelRequestId: input.modelRequestId,
          toolCallId,
          ...(call.providerCallId ? { providerCallId: call.providerCallId } : {}),
          toolName: call.name,
          arguments: call.arguments,
          definition
        }
      });
    }
    const pendingPolicies = this.tools.freezeCalls
      ? await this.tools.freezeCalls(pending.map((entry) => entry.dispatchInput))
      : await Promise.all(pending.map((entry) => this.tools.freezeCall
          ? this.tools.freezeCall(entry.dispatchInput)
          : Promise.resolve(fallbackFrozenToolPolicy(entry.dispatchInput.definition, entry.call.arguments))));
    if (pendingPolicies.length !== pending.length) {
      throw new Error('Tool dispatcher freezeCalls result length does not match the Provider batch.');
    }
    for (let index = 0; index < pending.length; index += 1) {
      const entry = pending[index];
      calls[entry.index] = {
        ...entry.call,
        toolCallId: entry.toolCallId,
        policy: pendingPolicies[index]
      };
    }
    const frozenCalls = calls.map((call, index) => {
      if (!call) throw new Error(`Provider ToolCall ${index} lacks a frozen policy.`);
      return call;
    });
    const batchId = stableId('tool_call_batch', input.modelRequestId);
    await this.effects.createToolCallBatch({
      source: { kind: 'callback', key: `agent-loop:${input.modelRequestId}:tool-batch` },
      batchId,
      turnId: input.turnId,
      modelRequestId: input.modelRequestId,
      messageId: input.messageId,
      entries: frozenCalls.map((call) => ({
        toolCallId: call.toolCallId,
        toolName: call.name,
        arguments: call.arguments,
        ...(call.providerCallId ? { providerCallId: call.providerCallId } : {}),
        providerOrdinal: call.providerOrdinal,
        ...(call.thoughtSignature ? { thoughtSignature: call.thoughtSignature } : {}),
        policy: call.policy
      }))
    });
    return frozenCalls;
  }

  private async dispatchProviderToolBatch(input: {
    conversationId: string;
    turnId: string;
    round: string;
    modelRequestId: string;
    calls: readonly FrozenProviderToolCall[];
  }): Promise<{ status: 'completed' } | { status: 'waiting' | 'interrupted'; toolCallId: string }> {
    let cursor = 0;
    let terminalPrefixCursor = 0;
    while (cursor < input.calls.length) {
      const first = input.calls[cursor];
      if (await this.terminateIfRequested(
        input.turnId,
        `round:${input.round}:before-tool-batch:${cursor + 1}`,
        first.toolCallId
      )) return { status: 'interrupted', toolCallId: first.toolCallId };
      let end = cursor + 1;
      if (first.policy.schedulingMode === 'parallel') {
        while (end < input.calls.length && input.calls[end].policy.schedulingMode === 'parallel') end += 1;
      }
      const group = input.calls.slice(cursor, end);
      const batchFinalized = await this.dispatchProviderToolGroup({
        turnId: input.turnId,
        round: input.round,
        modelRequestId: input.modelRequestId,
        calls: group
      });
      if (!batchFinalized) await this.effects.finalizeReadyInOrder(input.turnId);
      terminalPrefixCursor = await this.appendTerminalToolPairsInOrder({
        conversationId: input.conversationId,
        turnId: input.turnId,
        round: input.round,
        modelRequestId: input.modelRequestId,
        calls: input.calls,
        terminalPrefixCursor,
        terminalPrefixLimit: end
      });

      if (await this.terminateIfRequested(
        input.turnId,
        `round:${input.round}:after-tool-batch:${end}`,
        group[group.length - 1].toolCallId
      )) return { status: 'interrupted', toolCallId: group[group.length - 1].toolCallId };
      if (terminalPrefixCursor < end) {
        return { status: 'waiting', toolCallId: input.calls[terminalPrefixCursor].toolCallId };
      }
      cursor = end;
    }
    return { status: 'completed' };
  }

  private async dispatchProviderToolGroup(input: {
    turnId: string;
    round: string;
    modelRequestId: string;
    calls: readonly FrozenProviderToolCall[];
  }): Promise<boolean> {
    if (!this.tools.dispatchBatch) {
      const outcomes = await mapSettledWithBoundedConcurrency(
        input.calls,
        4,
        async (call) => this.dispatchProviderToolCall({
          turnId: input.turnId,
          round: input.round,
          modelRequestId: input.modelRequestId,
          call
        })
      );
      const rejected = outcomes.find(
        (outcome): outcome is PromiseRejectedResult => outcome.status === 'rejected'
      );
      if (rejected) throw rejected.reason;
      return false;
    }
    const schedulingMode = input.calls[0]?.policy.schedulingMode ?? 'serial';
    for (const call of input.calls) {
      this.observeLifecycle({
        turnId: input.turnId,
        stage: 'tool_dispatch_started',
        round: input.round,
        modelRequestId: input.modelRequestId,
        toolCallId: call.toolCallId,
        toolBatchSize: input.calls.length,
        schedulingMode
      });
    }
    const dispatched = await this.tools.dispatchBatch(input.calls.map((call) => ({
      turnId: input.turnId,
      modelRequestId: input.modelRequestId,
      toolCallId: call.toolCallId,
      ...(call.providerCallId ? { providerCallId: call.providerCallId } : {}),
      toolName: call.name,
      arguments: call.arguments
    })));
    if (dispatched.length !== input.calls.length) {
      throw new Error('Tool dispatcher dispatchBatch result length does not match the provider group.');
    }
    for (let index = 0; index < dispatched.length; index += 1) {
      if (isToolPause(dispatched[index])) continue;
      this.observeLifecycle({
        turnId: input.turnId,
        stage: 'tool_dispatch_completed',
        round: input.round,
        modelRequestId: input.modelRequestId,
        toolCallId: input.calls[index].toolCallId,
        toolBatchSize: input.calls.length,
        schedulingMode
      });
    }
    return true;
  }

  private async dispatchProviderToolCall(input: {
    turnId: string;
    round: string;
    modelRequestId: string;
    call: FrozenProviderToolCall;
  }): Promise<ToolTerminalResult | ReliableAgentToolPause | ReliableAgentToolSettled> {
    const existing = await this.effects.readTerminalResult(input.call.toolCallId, false);
    if (existing) return existing;
    try {
      this.observeLifecycle({
        turnId: input.turnId,
        stage: 'tool_dispatch_started',
        round: input.round,
        modelRequestId: input.modelRequestId,
        toolCallId: input.call.toolCallId
      });
      const dispatched = await this.tools.dispatch({
        turnId: input.turnId,
        modelRequestId: input.modelRequestId,
        toolCallId: input.call.toolCallId,
        ...(input.call.providerCallId ? { providerCallId: input.call.providerCallId } : {}),
        toolName: input.call.name,
        arguments: input.call.arguments
      });
      if (!isToolPause(dispatched)) {
        this.observeLifecycle({
          turnId: input.turnId,
          stage: 'tool_dispatch_completed',
          round: input.round,
          modelRequestId: input.modelRequestId,
          toolCallId: input.call.toolCallId
        });
      }
      return dispatched;
    } catch (error) {
      // Host handoff is not a tool failure. The durable ToolCall/Effect frontier deliberately
      // remains incomplete so the next lease generation can recover it; materializing a failed
      // ToolOutcome here would both lie to the model and race a still-running detached process.
      if (isExecutionHandoffError(error)) throw error;
      await this.effects.finalizeReadyInOrder(input.turnId);
      const terminal = await this.effects.readTerminalResult(input.call.toolCallId, false);
      if (terminal) return terminal;
      const operations = await listAllDomainRows(this.database, 'Operation', {
        tool_call_id: input.call.toolCallId
      });
      if (operations.length > 0) {
        return {
          disposition: 'paused',
          toolCallId: input.call.toolCallId,
          reason: 'background_process',
          resumeKey: input.call.toolCallId
        };
      }
      const failed = await this.effects.settleWithoutEffect({
        source: { kind: 'internal', key: `agent-loop:${input.call.toolCallId}:dispatcher-failed` },
        toolCallId: input.call.toolCallId,
        status: 'failed',
        detail: { error: error instanceof Error ? error.message : String(error) }
      });
      return failed.terminal ?? {
        disposition: 'settled',
        toolCallId: input.call.toolCallId,
        status: failed.status
      };
    }
  }

  private async appendTerminalToolPairsInOrder(input: {
    conversationId: string;
    turnId: string;
    round: string;
    modelRequestId: string;
    calls: readonly FrozenProviderToolCall[];
    terminalPrefixCursor: number;
    terminalPrefixLimit: number;
  }): Promise<number> {
    if (
      !Number.isSafeInteger(input.terminalPrefixCursor)
      || input.terminalPrefixCursor < 0
      || input.terminalPrefixCursor > input.calls.length
    ) throw new RangeError('terminalPrefixCursor is outside the Provider ToolCall batch.');
    if (
      !Number.isSafeInteger(input.terminalPrefixLimit)
      || input.terminalPrefixLimit < input.terminalPrefixCursor
      || input.terminalPrefixLimit > input.calls.length
    ) throw new RangeError('terminalPrefixLimit is outside the dispatched Provider ToolCall prefix.');
    let cursor = input.terminalPrefixCursor;
    let scanned = 0;
    const pairs: Array<{
      toolCallId: string;
      toolModelResultId: string;
      providerCallId?: string;
    }> = [];
    const terminalResults = await this.effects.readTerminalResults(
      input.calls.slice(cursor, input.terminalPrefixLimit).map((call) => call.toolCallId),
      false
    );
    for (const terminal of terminalResults) {
      const call = input.calls[cursor];
      scanned += 1;
      if (!terminal) break;
      this.observeLifecycle({
        turnId: input.turnId,
        stage: 'tool_model_result_committed',
        round: input.round,
        modelRequestId: input.modelRequestId,
        toolCallId: call.toolCallId
      });
      pairs.push({
        toolCallId: call.toolCallId,
        toolModelResultId: terminal.toolModelResultId,
        ...(call.providerCallId ? { providerCallId: call.providerCallId } : {})
      });
      cursor += 1;
    }
    this.observeLifecycle({
      turnId: input.turnId,
      stage: 'terminal_prefix_scanned',
      round: input.round,
      modelRequestId: input.modelRequestId,
      terminalCallsScanned: scanned,
      terminalPrefixCursor: cursor
    });
    if (pairs.length === 0) return cursor;
    const appended = await this.context.appendToolPairsInOrderBatch({
      conversationId: input.conversationId,
      pairs
    });
    this.observeLifecycle({
      turnId: input.turnId,
      stage: 'context_tool_pair_committed',
      round: input.round,
      modelRequestId: input.modelRequestId,
      toolCallId: pairs[pairs.length - 1].toolCallId,
      contextPairCount: pairs.length,
      contextTransactionCount: appended.transactionCount
    });
    return cursor;
  }

  private async readModelRequestToolDefinitions(
    modelRequestId: string,
    frozenRecipe?: { [key: string]: PlainJsonValue }
  ): Promise<ReliableAgentToolDefinition[]> {
    const recipe = frozenRecipe ?? await this.readModelRequestRecipe(modelRequestId);
    if (!Array.isArray(recipe.tools)) throw new TypeError('ModelRequest recipe.tools must be an array.');
    return recipe.tools.map((value, index) => normalizeFrozenToolDefinition(value, index));
  }

  private async dispatchAndCapture(
    conversationId: string,
    turnId: string,
    modelRequestId: string,
    request: DomainRow
  ): Promise<NormalizedProviderOutput> {
    const providerId = requireText(request.provider_id, 'ModelRequest.provider_id');
    const modelId = requireText(request.model_id, 'ModelRequest.model_id');
    const requestSeq = requirePositiveInteger(request.request_seq, 'ModelRequest.request_seq').toString();
    const adapter = await this.providers.resolve(providerId);
    if (adapter.providerId !== providerId) throw new Error(`Provider registry returned ${adapter.providerId} for ${providerId}.`);
    const dispatchBarrier = await this.database.snapshot([]);
    const wrapped: FullRequestProviderAdapter = {
      providerId,
      ...(adapter.estimateFullRequestInput
        ? { estimateFullRequestInput: (fullRequest) => adapter.estimateFullRequestInput!(fullRequest) }
        : {}),
      sendFullRequest: (fullRequest, controls) => adapter.sendFullRequest(fullRequest, {
        signal: controls.signal,
        onEvent: async (event): Promise<StreamEventResult> => {
          const observe = (): void => this.observeTransientEvent({
            conversationId,
            turnId,
            modelRequestId,
            requestSeq,
            providerId,
            modelId,
            attemptSeq: fullRequest.attemptSeq,
            socketGeneration: fullRequest.socketGeneration,
            afterCommitSeq: dispatchBarrier.snapshotCommitSeq,
            event: {
              kind: event.kind,
              streamSeq: event.streamSeq,
              content: event.content,
              ...(event.usage !== undefined ? { usage: event.usage } : {}),
              ...(event.timing !== undefined ? { timing: event.timing } : {})
            },
            observedAt: this.timestamp()
          });
          // Streaming deltas are intentionally low-latency. A terminal visual state, however,
          // must never outrun the durable terminal checkpoint it claims to represent.
          if (event.kind === 'completed') {
            const result = await controls.onEvent(event);
            observe();
            return result;
          }
          observe();
          return controls.onEvent(event);
        }
      })
    };
    const streamStats = asRecord(request.stream_stats_json);
    const reconnect = reliableDecimal(streamStats?.socketGeneration) > 0n || request.status === 'streaming';
    await this.modelProvider.dispatch(modelRequestId, wrapped, {
      ...(reconnect ? { reconnect: true } : {}),
      onTransientTerminal: (terminal) => this.observeTransientEvent({
        conversationId,
        turnId,
        modelRequestId,
        requestSeq,
        providerId,
        modelId,
        attemptSeq: terminal.attemptSeq,
        socketGeneration: terminal.socketGeneration,
        afterCommitSeq: dispatchBarrier.snapshotCommitSeq,
        event: terminal.event,
        observedAt: this.timestamp()
      })
    });
    // The terminal CAS checkpoint is the only final-output authority. The transient collector exists
    // solely to drive low-latency UI observation and must never become a second durable result path.
    return this.readTerminalProviderOutput(modelRequestId);
  }

  private async readLoopTerminalStatus(
    turnId: string,
    turn: DomainRow
  ): Promise<ReliableAgentLoopResult['terminalStatus']> {
    if (turn.status !== 'terminated') return 'failed';
    const terminations = await this.list('TurnTermination', { turn_id: turnId }, 2);
    if (terminations.length !== 1) {
      throw new Error(`Terminated Turn ${turnId} must have exactly one TurnTermination.`);
    }
    if (terminations[0].terminal_status === 'completed') return 'completed';
    if (terminations[0].terminal_status === 'interrupted') return 'interrupted';
    return 'failed';
  }

  private async readTerminalProviderOutput(modelRequestId: string): Promise<NormalizedProviderOutput> {
    const checkpoints = await this.list('ModelStreamCheckpoint', { model_request_id: modelRequestId }, 512);
    const terminal = checkpoints
      .filter((row) => row.checkpoint_kind === 'terminal_summary')
      .sort((left, right) => compareInteger(right.stream_seq, left.stream_seq))[0];
    if (!terminal) throw new Error(`Terminal ModelRequest ${modelRequestId} has no terminal summary checkpoint.`);
    const metadata = await this.requireExisting('ContentObject', requireId(terminal.content_object_id, 'ModelStreamCheckpoint.content_object_id'));
    const bytes = await this.contentStore.read(metadata as unknown as ContentObjectMetadata);
    const envelope = normalizePlainJson(JSON.parse(bytes.toString('utf8')), 'Model terminal checkpoint');
    const record = requireRecord(envelope, 'Model terminal checkpoint');
    if (record.kind !== 'completed') throw new Error('Model terminal checkpoint is not a completed event.');
    return normalizeProviderOutput(record.content);
  }

  private async readRoundFacts(turnId: string): Promise<{ turn: DomainRow; authority: DomainRow; head: DomainRow }> {
    const turn = await this.requireExisting('Turn', turnId);
    const conversationId = requireId(turn.conversation_id, 'Turn.conversation_id');
    const snapshot = await this.database.snapshot([
      DOMAIN_REPOSITORIES.domain('AuthoritySnapshot').list({ where: { turn_id: turnId }, limit: 2 }),
      DOMAIN_REPOSITORIES.domain('ConversationContextHeadLink').list({ where: { conversation_id: conversationId }, limit: 2 })
    ]);
    const authorities = rows(snapshot.snapshot[0]);
    const heads = rows(snapshot.snapshot[1]);
    if (authorities.length !== 1) throw new Error(`Turn ${turnId} must have exactly one AuthoritySnapshot.`);
    if (heads.length !== 1) throw new Error(`Conversation ${conversationId} must have exactly one Context head.`);
    return { turn, authority: authorities[0], head: heads[0] };
  }

  /**
   * Returns the last durable request sequence, or 1 for a fresh Turn. Replaying the last sequence
   * is required: request existence alone does not prove that its assistant Message, every tool
   * result and every Context tool_pair were committed before a crash.
   */
  private async resumeRequestSequence(turnId: string): Promise<bigint> {
    const requests = (await listAllDomainRows(this.database, 'ModelRequest', { turn_id: turnId }))
      .sort((left, right) => compareInteger(left.request_seq, right.request_seq));
    const recipes = await this.readModelRequestRecipes(requests);
    let expectedPhysicalSequence = 1n;
    let normalRound = 0n;
    for (const request of requests) {
      const actual = requirePositiveInteger(request.request_seq, 'ModelRequest.request_seq');
      if (actual !== expectedPhysicalSequence) {
        throw new Error(`Turn ${turnId} ModelRequest sequence is not contiguous at ${expectedPhysicalSequence.toString()}.`);
      }
      const requestId = requireId(request.id, 'ModelRequest.id');
      const recipe = recipes.get(requestId);
      if (!recipe) throw new Error(`ModelRequest ${requestId} recipe batch lost its request.`);
      if (recipe.kind === 'reliable-agent-turn') {
        normalRound += 1n;
        const round = requirePositiveInteger(recipe.round, 'ModelRequest recipe.round');
        if (round !== normalRound) {
          throw new Error(`Turn ${turnId} ordinary ModelRequest round is not contiguous at ${normalRound.toString()}.`);
        }
        const expectedId = modelRequestIdFor(turnId, `agent-loop:${turnId}:round:${normalRound.toString()}`);
        if (request.id !== expectedId) {
          throw new Error(`Turn ${turnId} ordinary ModelRequest ${normalRound.toString()} has an invalid identity.`);
        }
      } else if (recipe.kind !== 'reliable-context-compression') {
        throw new Error(`Turn ${turnId} ModelRequest ${String(request.id)} has unsupported recipe kind ${String(recipe.kind)}.`);
      }
      expectedPhysicalSequence += 1n;
    }
    return normalRound === 0n ? 1n : normalRound;
  }

  private async cancelSupersededCompressionRequests(turnId: string, currentHeadRootId: string): Promise<void> {
    const requests = await listAllDomainRows(this.database, 'ModelRequest', { turn_id: turnId });
    const activeRequests = requests.filter((request) => request.status !== 'terminal');
    const recipes = await this.readModelRequestRecipes(activeRequests);
    for (const request of activeRequests) {
      const requestId = requireId(request.id, 'ModelRequest.id');
      const recipe = recipes.get(requestId);
      if (!recipe) throw new Error(`ModelRequest ${requestId} recipe batch lost its request.`);
      if (recipe.kind !== 'reliable-context-compression') continue;
      const sourceRootId = requireId(recipe.sourceRootId, 'Compression recipe.sourceRootId');
      if (sourceRootId === currentHeadRootId) continue;
      await this.modelProvider.cancel(requestId, 'compression-source-head-superseded-before-recovery');
    }
  }

  private async assertModelRequestRound(
    request: DomainRow,
    expected: bigint,
    expectedId: string
  ): Promise<{ [key: string]: PlainJsonValue }> {
    if (request.id !== expectedId) throw new Error('ModelProvider returned an unexpected stable ModelRequest identity.');
    const recipe = (await this.readModelRequestRecipes([request])).get(expectedId);
    if (!recipe) throw new Error(`ModelRequest ${expectedId} recipe batch lost its request.`);
    const actual = requirePositiveInteger(recipe.round, 'ModelRequest recipe.round');
    if (recipe.kind !== 'reliable-agent-turn' || actual !== expected) {
      throw new Error(
        `ModelRequest ${expectedId} recipe round ${actual.toString()} does not match durable round ${expected.toString()}.`
      );
    }
    return recipe;
  }

  private async readModelRequestRecipe(modelRequestId: string): Promise<{ [key: string]: PlainJsonValue }> {
    const request = await this.requireExisting('ModelRequest', modelRequestId);
    const recipe = (await this.readModelRequestRecipes([request])).get(modelRequestId);
    if (!recipe) throw new Error(`ModelRequest ${modelRequestId} recipe batch lost its request.`);
    return recipe;
  }

  /** Preserves the full-history recipe audit while collapsing its SQLite and CAS round trips. */
  private async readModelRequestRecipes(
    requests: readonly DomainRow[]
  ): Promise<Map<string, { [key: string]: PlainJsonValue }>> {
    if (requests.length === 0) return new Map();
    const indexed = requests.map((request) => ({
      requestId: requireId(request.id, 'ModelRequest.id'),
      recipeObjectId: requireId(request.recipe_object_id, 'ModelRequest.recipe_object_id')
    }));
    const recipeObjectIds = [...new Set(indexed.map((entry) => entry.recipeObjectId))];
    const snapshot = await this.database.snapshot(recipeObjectIds.map((id) =>
      DOMAIN_REPOSITORIES.domain('ContentObject').get(id)
    ));
    if (snapshot.snapshot.length !== recipeObjectIds.length) {
      throw new Error('ModelRequest recipe metadata batch returned the wrong result count.');
    }
    const metadata = snapshot.snapshot.map((value, index) => {
      if (!value || Array.isArray(value)) {
        throw new Error(`ContentObject ${recipeObjectIds[index]} does not exist.`);
      }
      return value as unknown as ContentObjectMetadata;
    });
    const bytes = await this.contentStore.readMany(metadata);
    if (bytes.length !== recipeObjectIds.length) {
      throw new Error('ModelRequest recipe CAS batch returned the wrong result count.');
    }
    const recipeByObjectId = new Map(recipeObjectIds.map((id, index) => [
      id,
      requireRecord(
        normalizePlainJson(JSON.parse(bytes[index].toString('utf8')), 'ModelRequest recipe'),
        'ModelRequest recipe'
      )
    ]));
    return new Map(indexed.map(({ requestId, recipeObjectId }) => {
      const recipe = recipeByObjectId.get(recipeObjectId);
      if (!recipe) throw new Error(`ContentObject ${recipeObjectId} recipe batch lost its content.`);
      return [requestId, recipe];
    }));
  }

  private async requireTerminalToolResult(toolCallId: string): Promise<ToolTerminalResult> {
    const terminal = await this.effects.readTerminalResult(toolCallId, true);
    if (!terminal) throw new Error(`ToolCall ${toolCallId} has no terminal model result.`);
    return terminal;
  }

  /**
   * A Provider round containing tools owns the whole tool batch. Once that response boundary is
   * durably complete, hand off before issuing another Provider request when either ordinary user
   * guidance or an internal RuntimeDelivery continuation is already queued. RuntimeDelivery used
   * to be skipped here because its no-message TurnIntent is encoded as kind=retry; consequently a
   * late Subagent answer could wait through every subsequent round of the active Turn.
   */
  private async completeForQueuedBoundaryInput(input: {
    turnId: string;
    conversationId: string;
    round: string;
    modelRequestId: string;
  }): Promise<boolean> {
    for (;;) {
      const queuedInput = await this.oldestQueuedBoundaryIntent(input.conversationId);
      if (!queuedInput) return false;
      const queuedIntentId = requireId(queuedInput.intent.id, 'Queued boundary TurnIntent.id');
      if (await this.terminateIfRequested(
        input.turnId,
        `round:${input.round}:before-queued-input-handoff:${queuedIntentId}`
      )) return true;
      this.observeLifecycle({
        turnId: input.turnId,
        stage: 'turn_terminal_started',
        round: input.round,
        modelRequestId: input.modelRequestId
      });
      try {
        await this.turns.terminal({
          source: {
            kind: 'internal',
            key: `agent-loop:${input.turnId}:queued-input-handoff:${queuedIntentId}:${input.modelRequestId}`
          },
          turnId: input.turnId,
          terminalStatus: 'completed',
          reason: queuedInput.kind === 'runtime_continuation'
            ? 'queued_runtime_delivery_after_response_boundary'
            : 'queued_guidance_after_tool_batch',
          handoffQueuedIntentId: queuedIntentId,
          handoffQueuedIntentRevisionIds: queuedInput.revisionIds
        });
        this.observeLifecycle({
          turnId: input.turnId,
          stage: 'turn_terminal_completed',
          round: input.round,
          modelRequestId: input.modelRequestId
        });
        return true;
      } catch (error) {
        if (isTurnTerminalGuidanceConflictError(error)) continue;
        if (isTurnTerminalInputConflictError(error)) {
          if (await this.terminateIfRequested(
            input.turnId,
            `round:${input.round}:queued-input-handoff-conflict:${queuedIntentId}`
          )) return true;
          if (await this.absorbRuntimeDeliveryInputs(input.turnId) > 0) continue;
        }
        throw error;
      }
    }
  }

  private async oldestQueuedBoundaryIntent(conversationId: string): Promise<{
    intent: DomainRow;
    kind: 'guidance' | 'runtime_continuation';
    revisionIds: string[];
  } | null> {
    const [queued, childIntentLinks] = await Promise.all([
      listAllDomainRows(this.database, 'TurnIntent', { conversation_id: conversationId }),
      listAllDomainRows(this.database, 'ChildExecutionIntentLink', { state: 'pending' })
    ]);
    const childIntentIds = new Set(childIntentLinks.map((link) =>
      requireId(link.turn_intent_id, 'ChildExecutionIntentLink.turn_intent_id')
    ));
    const candidates = queued
      .filter((intent) => intent.state === 'queued' && intent.turn_id === null)
      .filter((intent) => !childIntentIds.has(requireId(intent.id, 'TurnIntent.id')));
    const boundaryInputs: Array<{
      intent: DomainRow;
      kind: 'guidance' | 'runtime_continuation';
      position: string;
      hold: 'none' | 'paused';
      revisionIds: string[];
    }> = [];
    for (const candidate of candidates) {
      const intentId = requireId(candidate.id, 'TurnIntent.id');
      const revisions = await listAllDomainRows(this.database, 'TurnIntentRevision', { intent_id: intentId });
      if (revisions.length === 0) throw new Error(`Queued TurnIntent ${intentId} has no frozen revision.`);
      const current = [...revisions].sort((left, right) =>
        compareInteger(right.revision_seq, left.revision_seq)
      )[0]!;
      const contentObject = await this.requireExisting(
        'ContentObject',
        requireId(current.content_object_id, 'TurnIntentRevision.content_object_id')
      );
      if (contentObject.content_type !== TURN_INTENT_ENVELOPE_CONTENT_TYPE) {
        boundaryInputs.push({
          intent: candidate,
          kind: 'guidance',
          position: initialGuidancePosition(requireText(candidate.created_at, 'TurnIntent.created_at')),
          hold: 'none',
          revisionIds: revisions.map((revision) => requireId(revision.id, 'TurnIntentRevision.id'))
        });
        continue;
      }
      const metadata = contentObject as unknown as ContentObjectMetadata;
      const envelopeValue = JSON.parse(
        (await this.contentStore.read(metadata)).toString('utf8')
      ) as unknown;
      const envelope = parseInputTurnIntentEnvelope(envelopeValue);
      if (envelope) {
        boundaryInputs.push({
          intent: candidate,
          kind: 'guidance',
          position: envelope.guidance.position,
          hold: envelope.guidance.hold,
          revisionIds: revisions.map((revision) => requireId(revision.id, 'TurnIntentRevision.id'))
        });
        continue;
      }
      if (!isQueuedRuntimeContinuationEnvelope(envelopeValue)) continue;
      boundaryInputs.push({
        intent: candidate,
        kind: 'runtime_continuation',
        position: initialGuidancePosition(requireText(candidate.created_at, 'TurnIntent.created_at')),
        hold: 'none',
        revisionIds: revisions.map((revision) => requireId(revision.id, 'TurnIntentRevision.id'))
      });
    }
    return boundaryInputs
      .filter((entry) => entry.hold === 'none')
      .sort((left, right) => compareGuidancePositions(left.position, right.position)
        || String(left.intent.created_at).localeCompare(String(right.intent.created_at))
        || String(left.intent.id).localeCompare(String(right.intent.id)))[0] ?? null;
  }

  private async absorbRuntimeDeliveryInputs(turnId: string): Promise<number> {
    const deliveries = (await listAllDomainRows(this.database, 'RuntimeDelivery', {
      target_turn_id: turnId,
      phase: 'current_turn',
      state: 'pending'
    })).sort((left, right) =>
      String(left.created_at).localeCompare(String(right.created_at))
      || String(left.id).localeCompare(String(right.id))
    );
    for (const delivery of deliveries) {
      await this.runtimeDeliveries.advance(requireId(delivery.id, 'RuntimeDelivery.id'));
    }
    const pending = (await listAllDomainRows(this.database, 'PendingTurnInput', {
      turn_id: turnId,
      state: 'pending',
      input_kind: 'runtime_delivery'
    }))
      .sort((left, right) => compareInteger(left.position, right.position));
    if (pending.length === 0) return 0;
    const turn = await this.requireExisting('Turn', turnId);
    if (turn.status !== 'active') return 0;
    const conversationId = requireId(turn.conversation_id, 'Turn.conversation_id');
    let absorbed = 0;
    for (const input of pending) {
      const inputId = requireId(input.id, 'PendingTurnInput.id');
      const contentObjectId = requireId(input.content_object_id, 'PendingTurnInput.content_object_id');
      const metadata = await this.requireExisting('ContentObject', contentObjectId) as unknown as ContentObjectMetadata;
      const content = await this.contentStore.read(metadata);
      const projection = await this.runtimeDeliveries.projectInputForModel({
        pendingTurnInputId: inputId,
        contentObjectId,
        content,
        contentType: requireText(metadata.content_type, 'ContentObject.content_type')
      });
      if (!projection) {
        await this.runtimeDeliveries.markInputHandled(inputId);
        absorbed += 1;
        continue;
      }
      await this.context.appendContent({
        conversationId,
        segmentKind: 'runtime_context',
        source: {
          sourceKind: 'runtime_context',
          sourceId: inputId,
          // Runtime context identity is carried by the stable PendingTurnInput id. Unlike a
          // MessageRevision or tool call sequence it has no revision axis; ContextSequence's
          // source contract therefore requires the sentinel revision 0.
          sourceRevision: 0n
        },
        content: projection.content,
        contentType: projection.contentType
      });
      await this.runtimeDeliveries.markInputHandled(inputId);
      absorbed += 1;
    }
    return absorbed;
  }

  private async terminateIfRequested(
    turnId: string,
    stage: string,
    pendingToolCallId?: string
  ): Promise<boolean> {
    for (;;) {
      const terminationFacts = await this.readTurnTerminationFacts(turnId);
      const turn = terminationFacts.turn;
      if (turn.status !== 'active') return turn.status === 'terminated';
      const request = terminationFacts.pending
        .sort((left, right) => compareInteger(left.position, right.position))[0];
      if (!request) return false;

      // Cancellation may race between ModelRequest/ToolCall creation and external dispatch. Close
      // every durable wait, then absorb any concurrently delivered runtime context before the
      // interrupted terminal writer ACKs termination inputs and releases the exact lease.
      await this.modelProvider.cancelTurnDispatches(turnId, `termination request observed at ${stage}`);
      await this.tools.cancelWaiting?.({
        turnId,
        sourceKey: `agent-loop:${turnId}:termination-request:${request.id}`,
        reason: `Turn observed ${String(request.input_kind)} at ${stage}.`
      });
      await this.closeInterruptedToolContext(turnId, requireId(request.id, 'PendingTurnInput.id'), pendingToolCallId);
      await this.absorbRuntimeDeliveryInputs(turnId);
      try {
        await this.turns.terminal({
          source: { kind: 'internal', key: `agent-loop:${turnId}:termination-request:${request.id}` },
          turnId,
          terminalStatus: 'interrupted',
          reason: `Executor observed ${String(request.input_kind)} at ${stage}.`
        });
        return true;
      } catch (error) {
        if (isTurnTerminalInputConflictError(error)) continue;
        throw error;
      }
    }
  }

  /**
   * A committed assistant message may contain several function calls while only the first call has
   * reached a durable user/file wait. Before terminating the Turn, materialize and cancel every
   * call represented by that committed message, then append each terminal tool_pair in provider
   * order. This keeps the next Provider request canonical after interruption and is safe to replay.
   */
  private async closeInterruptedToolContext(
    turnId: string,
    terminationRequestId: string,
    pendingToolCallId?: string
  ): Promise<void> {
    const turn = await this.requireExisting('Turn', turnId);
    const conversationId = requireId(turn.conversation_id, 'Turn.conversation_id');
    const representedToolCallIds = new Set<string>();
    const requests = (await listAllDomainRows(this.database, 'ModelRequest', {
      turn_id: turnId,
      status: 'terminal',
      terminal_state: 'completed'
    }))
      .sort((left, right) => compareInteger(left.request_seq, right.request_seq));

    for (const request of requests) {
      const modelRequestId = requireId(request.id, 'ModelRequest.id');
      const committedAssistant = await this.maybeGet('Message', assistantMessageIdFor(turnId, modelRequestId));
      if (!committedAssistant) continue;
      const output = await this.readTerminalProviderOutput(modelRequestId);
      if (output.toolCalls.length === 0) continue;
      const batch = await this.prepareProviderToolBatch({
        turnId,
        modelRequestId,
        messageId: requireId(committedAssistant.id, 'Assistant Message.id'),
        output
      });
      for (const call of batch) {
        const toolCallId = call.toolCallId;
        representedToolCallIds.add(toolCallId);
        let terminal = await this.effects.readTerminalResult(toolCallId, false);
        if (!terminal) {
          await this.cancelUndispatchedToolEffects(
            toolCallId,
            `agent-loop:${turnId}:termination-request:${terminationRequestId}`
          );
          const operations = await listAllDomainRows(this.database, 'Operation', { tool_call_id: toolCallId });
          if (operations.length === 0) {
            const settled = await this.effects.settleWithoutEffect({
              source: {
                kind: 'internal',
                key: `agent-loop:${turnId}:termination-request:${terminationRequestId}:cancel-tool:${toolCallId}`
              },
              toolCallId,
              status: 'cancelled',
              detail: { reason: 'turn_termination_requested' }
            });
            terminal = settled.terminal ?? null;
          } else {
            await this.effects.finalizeReadyInOrder(turnId);
            terminal = await this.effects.readTerminalResult(toolCallId, false)
              ?? await this.reconcileCommittedToolCall?.(toolCallId)
              ?? await this.effects.finalizeTerminalOperationsWithFallback({
                source: {
                  kind: 'internal',
                  key: `agent-loop:${turnId}:termination-request:${terminationRequestId}:close-effect:${toolCallId}`
                },
                toolCallId,
                detail: { reason: 'turn_termination_requested_after_effect_terminal' }
              });
          }
          terminal ??= await this.requireTerminalToolResult(toolCallId);
        }
        await this.appendTerminalToolPairOnce({
          conversationId,
          toolCallId,
          toolModelResultId: terminal.toolModelResultId,
          ...(call.providerCallId ? { providerCallId: call.providerCallId } : {})
        });
      }
    }

    if (pendingToolCallId && !representedToolCallIds.has(pendingToolCallId)
      && !await this.effects.readTerminalResult(pendingToolCallId, false)) {
      await this.cancelUndispatchedToolEffects(
        pendingToolCallId,
        `agent-loop:${turnId}:termination-request:${terminationRequestId}`
      );
      const operations = await listAllDomainRows(this.database, 'Operation', { tool_call_id: pendingToolCallId });
      if (operations.length === 0) {
        await this.effects.settleWithoutEffect({
          source: {
            kind: 'internal',
            key: `agent-loop:${turnId}:termination-request:${terminationRequestId}:cancel-unrepresented-tool:${pendingToolCallId}`
          },
          toolCallId: pendingToolCallId,
          status: 'cancelled',
          detail: { reason: 'turn_termination_requested' }
        });
      } else {
        await this.effects.finalizeReadyInOrder(turnId);
        const terminal = await this.effects.readTerminalResult(pendingToolCallId, false)
          ?? await this.reconcileCommittedToolCall?.(pendingToolCallId)
          ?? await this.effects.finalizeTerminalOperationsWithFallback({
            source: {
              kind: 'internal',
              key: `agent-loop:${turnId}:termination-request:${terminationRequestId}:close-effect:${pendingToolCallId}`
            },
            toolCallId: pendingToolCallId,
            detail: { reason: 'turn_termination_requested_after_effect_terminal' }
          });
        if (!terminal) await this.requireTerminalToolResult(pendingToolCallId);
      }
    }
  }

  /** A prepared Effect may be cancelled; a dispatched Effect must first produce/recover a Receipt. */
  private async cancelUndispatchedToolEffects(toolCallId: string, sourcePrefix: string): Promise<void> {
    const operations = await listAllDomainRows(this.database, 'Operation', { tool_call_id: toolCallId });
    for (const operation of operations) {
      if (isTerminalToolStatus(operation.status)) continue;
      const attempts = await listAllDomainRows(this.database, 'Attempt', { operation_id: operation.id });
      for (const attempt of attempts) {
        const intents = await this.list('EffectIntent', { attempt_id: attempt.id }, 2);
        if (intents.length > 1) throw new Error(`Attempt ${String(attempt.id)} has multiple EffectIntents.`);
        if (intents[0]?.dispatch_state !== 'pending') continue;
        await this.effects.cancelPendingEffect({
          source: {
            kind: 'internal',
            key: `${sourcePrefix}:cancel-before-dispatch:${String(intents[0].id)}`
          },
          effectIntentId: requireId(intents[0].id, 'EffectIntent.id'),
          detail: { reason: 'turn_termination_requested_before_effect_dispatch' }
        });
      }
    }
    const remaining = await listAllDomainRows(this.database, 'Operation', { tool_call_id: toolCallId });
    const unresolved = remaining.filter((operation) => !isTerminalToolStatus(operation.status));
    if (unresolved.length > 0) {
      throw new Error(
        `ToolCall ${toolCallId} still has non-terminal Operations after cancellation: ${unresolved
          .map((operation) => `${String(operation.id)}=${String(operation.status)}`)
          .join(', ')}.`
      );
    }
  }

  private async appendTerminalToolPairOnce(input: {
    conversationId: string;
    toolCallId: string;
    toolModelResultId: string;
    providerCallId?: string;
  }): Promise<void> {
    const sources = await this.list('ContextSegmentSource', {
      source_kind: 'tool_model_result',
      source_id: input.toolModelResultId
    }, 2);
    if (sources.length > 1) {
      throw new Error(`ToolModelResult ${input.toolModelResultId} has multiple Context occurrences.`);
    }
    if (sources.length === 1) {
      const callSources = await this.list('ContextSegmentSource', {
        segment_id: requireId(sources[0].segment_id, 'ContextSegmentSource.segment_id'),
        source_kind: 'tool_call'
      }, 2);
      if (callSources.length !== 1 || callSources[0].source_id !== input.toolCallId) {
        throw new Error(`ToolModelResult ${input.toolModelResultId} is linked to a conflicting Context tool pair.`);
      }
      return;
    }
    await this.context.appendToolPair(input);
  }

  private async failActiveTurn(turnId: string, error: unknown): Promise<void> {
    const reason = errorMessage(error);
    for (;;) {
      const turn = await this.maybeGet('Turn', turnId);
      if (!turn || turn.status !== 'active') return;
      if (await this.terminateIfRequested(turnId, 'failure-terminal')) return;
      await this.absorbRuntimeDeliveryInputs(turnId);
      try {
        await this.turns.terminal({
          source: {
            kind: 'internal',
            key: `agent-loop:${turnId}:failed:${stableDigest(reason)}`
          },
          turnId,
          terminalStatus: 'failed',
          reason
        });
        return;
      } catch (terminalError) {
        if (isTurnTerminalInputConflictError(terminalError)) continue;
        throw terminalError;
      }
    }
  }

  private async maybeGet(domain: string, id: string): Promise<DomainRow | null> {
    const snapshot = await this.database.snapshot([DOMAIN_REPOSITORIES.domain(domain).get(id)]);
    return snapshot.snapshot[0] as DomainRow | null;
  }

  private async requireExisting(domain: string, id: string): Promise<DomainRow> {
    const row = await this.maybeGet(domain, id);
    if (!row) throw new Error(`${domain} ${id} does not exist.`);
    return row;
  }

  private async list(domain: string, where: DomainRow, limit: number): Promise<DomainRow[]> {
    const snapshot = await this.database.snapshot([DOMAIN_REPOSITORIES.domain(domain).list({ where, limit })]);
    return rows(snapshot.snapshot[0]);
  }

  private async readTurnTerminationFacts(
    turnId: string
  ): Promise<{ turn: DomainRow; pending: DomainRow[] }> {
    const snapshot = await this.database.snapshot([
      DOMAIN_REPOSITORIES.domain('Turn').get(turnId),
      ...TERMINATION_INPUT_KINDS.map((inputKind) =>
        DOMAIN_REPOSITORIES.domain('PendingTurnInput').list({
          where: { turn_id: turnId, state: 'pending', input_kind: inputKind },
          orderBy: { column: 'position', direction: 'asc' },
          limit: 1
        }))
    ]);
    const turn = snapshot.snapshot[0];
    if (!turn || Array.isArray(turn)) throw new Error(`Turn ${turnId} does not exist.`);
    return {
      turn,
      pending: snapshot.snapshot.slice(1).flatMap((value) => rows(value))
    };
  }

  private observeLifecycle(event: Omit<ReliableAgentLifecycleEvent, 'observedAt'>): void {
    if (!this.lifecycleObserver) return;
    try {
      this.lifecycleObserver.observe({ ...event, observedAt: this.timestamp() });
    } catch {
      // Diagnostics must never become a second control path or break the Agent loop.
    }
  }

  private observeTransientEvent(event: ReliableAgentTransientEvent): void {
    if (!this.transientObserver) return;
    try {
      this.transientObserver.observe(event);
    } catch {
      // A memory-only low-latency overlay must never become a Provider/Turn control path.
    }
  }

  private timestamp(): string {
    return requireText(this.now(), 'clock result');
  }
}

const TERMINATION_INPUT_KINDS = [
  'interrupt_request',
  'interrupt_current_turn',
  'termination_request'
] as const;

function isTerminalToolStatus(value: unknown): boolean {
  return ['succeeded', 'failed', 'partial', 'rejected', 'cancelled', 'conflict', 'outcome_unknown']
    .includes(String(value));
}

function providerOutputMessage(output: NormalizedProviderOutput): MessageContent {
  return output.content;
}

function normalizeProviderOutput(value: PlainJsonValue): NormalizedProviderOutput {
  const record = requireRecord(value, 'Provider completed MessageContent');
  if (record.role !== 'model' || !Array.isArray(record.parts)) {
    throw new TypeError('Provider completed MessageContent must contain model parts.');
  }
  const content = normalizePlainJson(record, 'Provider completed MessageContent') as unknown as MessageContent;
  const calls: PlainJsonValue[] = [];
  for (const part of content.parts) {
    if (!('functionCall' in part)) continue;
    calls.push(normalizePlainJson({
      ...(part.id ? { id: part.id } : {}),
      ordinal: calls.length,
      name: part.functionCall.name,
      arguments: part.functionCall.args,
      ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {})
    }, `Provider completed function call ${calls.length}`));
  }
  return { content, toolCalls: normalizeToolCalls(calls) };
}

function normalizeToolCalls(value: unknown): NormalizedToolCall[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new TypeError('Provider toolCalls must be an array.');
  const normalized: NormalizedToolCall[] = [];
  const byProviderCallId = new Map<string, { signature: string; index: number }>();
  const byExplicitOrdinal = new Map<number, number>();
  const explicitOrdinalByIndex: Array<number | undefined> = [];
  value.forEach((entry, index) => {
    const record = requireRecord(entry as PlainJsonValue, `Provider toolCall ${index}`);
    const name = requireText(record.name, `Provider toolCall ${index}.name`);
    let argumentsValue = record.arguments;
    if (typeof record.argumentsJson === 'string') {
      argumentsValue = normalizePlainJson(JSON.parse(record.argumentsJson), `Provider toolCall ${index}.argumentsJson`);
    }
    const providerCallId = optionalText(record.id);
    const explicitOrdinal = optionalNonNegativeInteger(record.ordinal);
    const providerOrdinal = explicitOrdinal ?? index;
    const call: NormalizedToolCall = {
      ...(providerCallId ? { providerCallId } : {}),
      providerOrdinal,
      name,
      arguments: normalizePlainJson(argumentsValue ?? {}, `Provider toolCall ${index}.arguments`),
      ...(optionalText(record.thoughtSignature) ? { thoughtSignature: optionalText(record.thoughtSignature) } : {})
    };
    const signature = canonicalPlainJson({
      name: call.name,
      arguments: call.arguments
    }, 'Provider toolCall signature');
    const idIdentity = providerCallId ? byProviderCallId.get(providerCallId) : undefined;
    const ordinalIdentity = explicitOrdinal === undefined ? undefined : byExplicitOrdinal.get(explicitOrdinal);
    if (idIdentity !== undefined && ordinalIdentity !== undefined && idIdentity.index !== ordinalIdentity) {
      throw new Error(
        `Provider tool call id ${providerCallId} and ordinal ${explicitOrdinal} identify different calls.`
      );
    }
    const existingIndex = idIdentity?.index ?? ordinalIdentity;
    if (existingIndex !== undefined) {
      const prior = normalized[existingIndex];
      if (prior.providerCallId && providerCallId && prior.providerCallId !== providerCallId) {
        throw new Error(
          `Provider reused tool call ordinal ${explicitOrdinal} for ids ${prior.providerCallId} and ${providerCallId}.`
        );
      }
      const priorSignature = canonicalPlainJson({
        name: prior.name,
        arguments: prior.arguments
      }, 'Provider prior toolCall signature');
      if (priorSignature !== signature) {
        const identity = providerCallId ? `id ${providerCallId}` : `ordinal ${explicitOrdinal}`;
        throw new Error(`Provider reused tool call ${identity} with conflicting content.`);
      }
      const priorExplicitOrdinal = explicitOrdinalByIndex[existingIndex];
      if (
        explicitOrdinal !== undefined
        && priorExplicitOrdinal !== undefined
        && priorExplicitOrdinal !== explicitOrdinal
      ) {
        throw new Error(`Provider reused tool call id ${providerCallId} with ordinal ${explicitOrdinal}.`);
      }
      if (prior.thoughtSignature && call.thoughtSignature && prior.thoughtSignature !== call.thoughtSignature) {
        const identity = providerCallId ? `id ${providerCallId}` : `ordinal ${explicitOrdinal}`;
        throw new Error(`Provider reused tool call ${identity} with conflicting thoughtSignature.`);
      }
      if (!prior.providerCallId && providerCallId) {
        prior.providerCallId = providerCallId;
        byProviderCallId.set(providerCallId, { signature, index: existingIndex });
      }
      if (priorExplicitOrdinal === undefined && explicitOrdinal !== undefined) {
        prior.providerOrdinal = explicitOrdinal;
        explicitOrdinalByIndex[existingIndex] = explicitOrdinal;
        byExplicitOrdinal.set(explicitOrdinal, existingIndex);
      }
      if (!prior.thoughtSignature && call.thoughtSignature) prior.thoughtSignature = call.thoughtSignature;
      return;
    }
    if (providerCallId) byProviderCallId.set(providerCallId, { signature, index: normalized.length });
    if (explicitOrdinal !== undefined) byExplicitOrdinal.set(explicitOrdinal, normalized.length);
    explicitOrdinalByIndex.push(explicitOrdinal);
    normalized.push(call);
  });
  const ordinals = new Set<number>();
  for (const call of normalized) {
    if (ordinals.has(call.providerOrdinal)) {
      throw new Error(`Provider repeated tool call ordinal ${call.providerOrdinal}.`);
    }
    ordinals.add(call.providerOrdinal);
  }
  return normalized;
}

function isToolPause(
  value: ToolTerminalResult | ReliableAgentToolPause | ReliableAgentToolSettled
): value is ReliableAgentToolPause {
  return 'disposition' in value && value.disposition === 'paused';
}

function providerToolCallId(modelRequestId: string, call: NormalizedToolCall): string {
  return stableId(
    'tool_call',
    modelRequestId,
    String(call.providerOrdinal),
    call.providerCallId ?? call.name
  );
}

function normalizeFrozenToolDefinition(value: PlainJsonValue, index: number): ReliableAgentToolDefinition {
  const record = requireRecord(value, `ModelRequest recipe.tools[${index}]`);
  return {
    name: requireText(record.name, `ModelRequest recipe.tools[${index}].name`),
    description: optionalText(record.description),
    parameters: normalizePlainJson(record.parameters ?? {}, `ModelRequest recipe.tools[${index}].parameters`),
    ...(record.source !== undefined
      ? { source: normalizePlainJson(record.source, `ModelRequest recipe.tools[${index}].source`) }
      : {}),
    ...(record.metadata !== undefined
      ? { metadata: normalizePlainJson(record.metadata, `ModelRequest recipe.tools[${index}].metadata`) }
      : {}),
    ...(record.defaultConfig !== undefined
      ? { defaultConfig: normalizePlainJson(record.defaultConfig, `ModelRequest recipe.tools[${index}].defaultConfig`) }
      : {})
  };
}

function unknownToolDefinition(name: string): ReliableAgentToolDefinition {
  return {
    name,
    description: '',
    parameters: {},
    metadata: { defaultEnabled: false }
  };
}

function fallbackFrozenToolPolicy(
  definition: ReliableAgentToolDefinition,
  argumentsValue: PlainJsonValue
): FrozenToolCallPolicyDecision {
  const metadata = asRecord(definition.metadata);
  const args = asRecord(argumentsValue);
  const requestedScheduling = args?.scheduling === 'parallel' || args?.scheduling === 'serial'
    ? args.scheduling
    : undefined;
  const trustedCommand = definition.name === 'bash' || definition.name === 'shell'
    ? classifyCommandCall(argumentsValue)
    : undefined;
  const backendParallel = trustedCommand
    ? trustedCommand.parallelSafe
    : metadata?.readonly === true || metadata?.riskLevel === 'read';
  const schedulingMode = requestedScheduling === 'serial'
    ? 'serial'
    : requestedScheduling === 'parallel' || backendParallel ? 'parallel' : 'serial';
  const supportsChangeApply = metadata?.supportsChangeApply === true;
  const automaticChangeApply = supportsChangeApply && metadata?.defaultAutoApplyChange === true;
  const configuredDelay = optionalNonNegativeInteger(metadata?.defaultAutoApplyChangeDelaySeconds) ?? 0;
  return {
    displayAutoExpand: metadata?.defaultAutoExpand === true,
    displayAutoOpenDiff: metadata?.defaultAutoOpenDiffPreview === true,
    executionGate: ['ask_user', 'submit_plan'].includes(definition.name)
      || metadata?.defaultAutoApproveExecution !== false
      ? 'automatic'
      : 'approval_required',
    changeApplyMode: supportsChangeApply
      ? automaticChangeApply ? 'automatic' : 'manual'
      : 'unsupported',
    changeApplyDelaySeconds: automaticChangeApply ? Math.min(configuredDelay, 600) : 0,
    autoSubmitResult: metadata?.defaultAutoSubmitResult !== false,
    schedulingMode,
    schedulingReason: requestedScheduling === 'serial'
      ? 'model_selected_serial'
      : requestedScheduling === 'parallel'
        ? 'model_selected_parallel'
      : backendParallel
        ? trustedCommand?.reason ?? 'frozen_readonly_metadata'
        : trustedCommand?.reason ?? 'frozen_default_serial'
  };
}

function frozenPolicyFromRow(row: DomainRow): FrozenToolCallPolicyDecision {
  const executionGate = String(row.execution_gate);
  const changeApplyMode = String(row.change_apply_mode);
  const schedulingMode = String(row.scheduling_mode);
  if (!['automatic', 'approval_required'].includes(executionGate)) {
    throw new TypeError(`Invalid frozen Tool execution gate: ${executionGate}.`);
  }
  if (!['automatic', 'manual', 'unsupported'].includes(changeApplyMode)) {
    throw new TypeError(`Invalid frozen Tool change-apply mode: ${changeApplyMode}.`);
  }
  if (!['parallel', 'serial'].includes(schedulingMode)) {
    throw new TypeError(`Invalid frozen Tool scheduling mode: ${schedulingMode}.`);
  }
  return {
    ...(typeof row.summary === 'string' ? { summary: row.summary } : {}),
    displayAutoExpand: row.display_auto_expand === 1n,
    displayAutoOpenDiff: row.display_auto_open_diff === 1n,
    executionGate: executionGate as FrozenToolCallPolicyDecision['executionGate'],
    changeApplyMode: changeApplyMode as FrozenToolCallPolicyDecision['changeApplyMode'],
    changeApplyDelaySeconds: requireNonNegativeSafeNumber(
      row.change_apply_delay_seconds,
      'ToolCallPolicySnapshot.change_apply_delay_seconds'
    ),
    autoSubmitResult: row.auto_submit_result === 1n,
    schedulingMode: schedulingMode as FrozenToolCallPolicyDecision['schedulingMode'],
    ...(typeof row.scheduling_reason === 'string' ? { schedulingReason: row.scheduling_reason } : {})
  };
}

function requireNonNegativeSafeNumber(value: unknown, label: string): number {
  const bigint = typeof value === 'bigint'
    ? value
    : typeof value === 'number' && Number.isSafeInteger(value) ? BigInt(value)
      : typeof value === 'string' && /^(?:0|[1-9]\d*)$/.test(value) ? BigInt(value)
        : -1n;
  if (bigint < 0n || bigint > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new TypeError(`${label} must be a non-negative safe integer.`);
  }
  return Number(bigint);
}

function rows(value: DomainRow | DomainRow[] | null): DomainRow[] {
  if (!Array.isArray(value)) throw new TypeError('Repository list did not return rows.');
  return value;
}

function requireRow(value: DomainRow | DomainRow[] | null, label: string): DomainRow {
  if (!value || Array.isArray(value)) throw new Error(`${label} does not exist.`);
  return value;
}

function requireRecord(value: PlainJsonValue, label: string): { [key: string]: PlainJsonValue } {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object.`);
  return value;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

/**
 * RuntimeDelivery continuations are frozen as no-message retry envelopes so they can inherit the
 * source Turn authority without fabricating a user message. Explicit user retries always carry
 * rewind lineage, while maintenance retries carry runtimeMaintenance; neither may be treated as a
 * response-boundary notification handoff.
 */
function isQueuedRuntimeContinuationEnvelope(value: unknown): boolean {
  const record = asRecord(value);
  return record?.kind === 'retry'
    && typeof record.sourceTurnId === 'string'
    && record.sourceTurnId.trim().length > 0
    && record.runtimeMaintenance === undefined
    && record.sourceMessageId === undefined
    && record.sourceMessageRevisionId === undefined
    && record.sourceModelRequestId === undefined
    && record.messageContentObjectId === undefined;
}

function optionalText(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function reliableDecimal(value: unknown): bigint {
  if (typeof value === 'bigint' && value >= 0n) return value;
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === 'string' && /^(?:0|[1-9]\d*)$/.test(value)) return BigInt(value);
  return 0n;
}

function requirePositiveInteger(value: unknown, label: string): bigint {
  const normalized = reliableDecimal(value);
  if (normalized < 1n) throw new TypeError(`${label} must be a positive integer.`);
  return normalized;
}

function optionalNonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

function currentInputReferenceTokens(recipe: PlainJsonValue): number {
  const record = asRecord(recipe);
  const currentInput = asRecord(record?.currentTurnInput);
  return optionalNonNegativeInteger(currentInput?.estimatedTokens) ?? 0;
}

function compactRuntimeId(value: string): string {
  const normalized = requireId(value, 'runtime status id');
  return normalized.length <= 160 ? normalized : `${normalized.slice(0, 159)}…`;
}

function compareInteger(left: unknown, right: unknown): number {
  const a = reliableDecimal(left);
  const b = reliableDecimal(right);
  return a < b ? -1 : a > b ? 1 : 0;
}

function stableId(kind: string, ...parts: string[]): string {
  return `rk_${kind}_${createHash('sha256').update(JSON.stringify(parts)).digest('hex').slice(0, 32)}`;
}

function stableDigest(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 16);
}

function errorDiagnostic(error: unknown): Pick<ReliableAgentLifecycleEvent, 'errorName' | 'errorMessage'> {
  return {
    errorName: error instanceof Error ? error.name : 'NonError',
    errorMessage: errorMessage(error, 500)
  };
}

function errorMessage(error: unknown, maxLength = 2_000): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.length <= maxLength ? message : `${message.slice(0, Math.max(0, maxLength - 3))}...`;
}

function requireId(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new TypeError(`${label} must be a non-empty id.`);
  return value;
}

function requireText(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) throw new TypeError(`${label} must be non-empty.`);
  return value.trim();
}
