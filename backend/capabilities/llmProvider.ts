import { createHash } from 'crypto';
import {
  groupAtomicMessageContents,
  isModelToolResponseMultimodalMimeType
} from '../reliableKernel/modelFacingContextProjection';
import { decodeCanonicalBase64 } from './canonicalBase64';
import { estimateTokenCount, sliceByTokens } from 'tokenx';
import { mapWithBoundedConcurrency } from './boundedConcurrency';
import { createProxyFetch } from './proxyFetch';
import { createTerminalValidatedFetch } from './terminalValidatedFetch';
import { createLlmStreamEventBatcher } from './llmStreamEventBatcher';
import { LIMCODE_OPENAI_RESPONSES_WS_IMPLEMENTATION } from './openAIResponsesWebSocketIdentity';
import type {
  LimCodeOpenAIResponsesStreamChunk,
  OpenAIResponsesFormatAdapter,
  OpenAIResponsesWebSocketDecision,
  OpenAIResponsesWebSocketPhase,
  OpenAIResponsesWebSocketPhaseKind,
  OpenAIResponsesWebSocketStreamOptions,
  OpenAIResponsesWebSocketTimeoutPhase
} from './openAIResponsesWebSocketSession';
import { LlmEventType } from '../world/modules/llm/events';
import { ATTACHMENT_OBSERVATION_PROMPT_REVISION } from '../world/modules/llm/contracts';
import type {
  LlmCompactDryRunResult,
  LlmCompactRequest,
  LlmCompactResult,
  LlmAttachmentObservation,
  LlmAttachmentObservationRequirement,
  LlmDryRunOptions,
  LlmDryRunResult,
  LlmResolveInvocationRequest,
  LlmStartRequest,
  LlmModelSettings,
  ToolSchema
} from '../world/modules/llm/contracts';
import type { Emit, LlmCapability } from './types';
import {
  isFileDataPart,
  isFunctionCallPart,
  isFunctionResponsePart,
  DEFAULT_LLM_COMPRESSION_SUMMARY_SYSTEM_PROMPT,
  DEFAULT_LLM_COMPRESSION_SUMMARY_USER_PROMPT,
  DEFAULT_LLM_COMPRESSION_OUTPUT_RESERVE_TOKENS,
  DEFAULT_LLM_COMPRESSION_SUMMARY_TARGET_TOKENS,
  DEFAULT_LLM_CONTEXT_WINDOW_TOKENS,
  DEFAULT_SEGMENTED_SUMMARY_SYSTEM_PROMPT,
  DEFAULT_SEGMENTED_SUMMARY_USER_PROMPT,
  isInlineDataPart,
  DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
  DEFAULT_LLM_RETRY_ON_ERROR,
  isTextPart,
  isVisibleTextPart,
  isProviderContextPart,
  createDefaultLlmPromptCacheConfig,
  defaultLlmPromptCacheModeForProvider,
  defaultLlmPromptCacheTtlForProvider,
  isPromptCacheSupportedProvider
} from '../../shared/protocol';
import {
  normalizeAttachmentObservationRequirement,
  normalizeLlmAttachmentObservation,
  renderAttachmentObservationStateContent
} from '../reliableKernel/attachmentObservations';
import {
  geminiThinkingCapabilityForModel,
  isGeminiThinkingLevelSupported
} from '../../shared/geminiThinking';
import type {
  ContentPart,
  FunctionCallPart,
  FunctionResponsePart,
  InlineDataPart,
  LlmCompressionConfigRecord,
  LlmGenerationConfigRecord,
  LlmInvocationSettingsSnapshotRecord,
  LlmProviderConfigRecord,
  LlmProviderHeadersRecord,
  LlmProviderKind,
  LlmProviderModelRecord,
  LlmOpenAIResponsesTransport,
  LlmPromptCacheConfigRecord,
  LlmPromptCacheMode,
  LlmPromptCacheTtl,
  LlmRequestBodyRecord,
  LlmToolCallFormat,
  LlmRawErrorInfoRecord,
  LlmUsageMetadataRecord,
  MessageContent,
  ModelOutputItemReference
} from '../../shared/protocol';

export const DEFAULT_LLM_BASE_URL = 'https://api.openai.com/v1';
const COMPRESSION_DEBUG_PREFIX = '[LimCode][CompressionDebug]';
type MaybeProvider<T, TArg = void> = T | undefined | ((arg: TArg) => T | undefined | Promise<T | undefined>);
type LlmSettingsRequest = LlmStartRequest | LlmCompactRequest | LlmResolveInvocationRequest | undefined;
type LlmCompressionSettingsProvider = (request: LlmCompactRequest) => LlmCompressionConfigRecord | undefined | Promise<LlmCompressionConfigRecord | undefined>;

type UnifiedModule = typeof import('unified-llm-provider');
type UnifiedContent = import('unified-llm-provider').Content;
type UnifiedPart = import('unified-llm-provider').Part;
type UnifiedLLMRequest = import('unified-llm-provider').LLMRequest;
type UnifiedLLMResponse = import('unified-llm-provider').LLMResponse;
type UnifiedLLMCompactResponse = import('unified-llm-provider').LLMCompactResponse;
type UnifiedLLMStreamChunk = import('unified-llm-provider').LLMStreamChunk;
type UnifiedFunctionDeclaration = import('unified-llm-provider').FunctionDeclaration;
type UnifiedModelCatalogEntry = import('unified-llm-provider').ModelCatalogEntry;

interface UnifiedDryRunResult {
  url: string;
  method: 'POST';
  stream: boolean;
  headers: Record<string, string>;
  body: unknown;
  bodyText: string;
  curl: string;
  providerName: string;
  inputFormat: string;
  outputFormat: string;
  timestamp: number;
}

interface UnifiedDryRunCapable {
  dryRun(request: unknown, options?: { inputFormat?: string; outputFormat?: string; stream?: boolean; curl?: { includeApiKey?: boolean; prettyBody?: boolean } }): Promise<UnifiedDryRunResult>;
  compactDryRun?(request: unknown, options?: {
    inputFormat?: string;
    outputFormat?: string;
    requestBody?: LlmRequestBodyRecord;
    curl?: { includeApiKey?: boolean; prettyBody?: boolean };
  }): Promise<UnifiedDryRunResult>;
}

interface UnifiedChatProvider extends UnifiedDryRunCapable {
  chat<T>(request: unknown, options: {
    inputFormat: 'unified';
    outputFormat: 'unified';
    signal?: AbortSignal;
  }): Promise<T>;
  chatStream<T>(request: unknown, options: {
    inputFormat: 'unified';
    outputFormat: 'unified';
    signal?: AbortSignal;
  }): AsyncIterable<T>;
}

export interface LlmProviderTransportTrace {
  requestId: string;
  conversationId: string;
  phase: OpenAIResponsesWebSocketPhaseKind
    | 'continuation_decision'
    | 'http_fallback'
    | 'http_cooldown';
  observedAt: number;
  sessionKeyHash: string;
  connectionGeneration: number;
  elapsedMs?: number;
  connectionReused?: boolean;
  connectionReason?: OpenAIResponsesWebSocketDecision['connectionReason'];
  mode?: OpenAIResponsesWebSocketDecision['mode'];
  reason?: string;
  timeoutPhase?: OpenAIResponsesWebSocketTimeoutPhase;
  fullInputItemCount?: number;
  sentInputItemCount?: number;
  responseCreateFrameSha256?: string;
  responseCreateFrameBytes?: number;
  responseCreateSeq?: number;
}

export interface LlmProviderOptions {
  settings: MaybeProvider<LlmProviderConfigRecord, LlmSettingsRequest>;
  proxy?: MaybeProvider<string>;
  compressionSettings?: LlmCompressionSettingsProvider;
  activeCompressionSettings?: (request?: { conversationId?: string; providerConfigId?: string; model?: string }) => LlmCompressionConfigRecord | undefined | Promise<LlmCompressionConfigRecord | undefined>;

  headers?: MaybeProvider<Record<string, string>>;
  resolveAttachment?: (input: { attachmentId?: string; sourcePath?: string; mimeType?: string; name?: string }) => Promise<InlineDataPart | undefined>;
  onTransportTrace?: (trace: LlmProviderTransportTrace) => void;
}
interface RetryControl {
  cancelRequested: boolean;
  wakeRetryWait?: () => void;
}

interface LlmAttemptFailure {
  message: string;
  rawError?: LlmRawErrorInfoRecord;
  createdAt?: number;
  streamOutputDurationMs?: number;
}

interface LlmAttemptRetryRecoveryNotice {
  retryAttempt: number;
  retryMaxAttempts: number;
}

interface LlmAttemptTimingState {
  firstStreamChunkAt?: number;
  firstStreamChunkMark?: number;
  streamTimingChunkCount: number;
}

const THOUGHT_PROGRESS_INTERVAL_MS = 500;
const OPENAI_RESPONSES_WS_RETRY_BUDGET_MS = 120_000;
const OPENAI_RESPONSES_HTTP_COOLDOWN_MS = 60_000;
const openAIResponsesHttpCooldowns = new Map<string, number>();
type OpenAIResponsesWebSocketSessionModule = typeof import('./openAIResponsesWebSocketSession');
let loadedOpenAIResponsesWebSocketSession: OpenAIResponsesWebSocketSessionModule | undefined;
let loadingOpenAIResponsesWebSocketSession: Promise<OpenAIResponsesWebSocketSessionModule> | undefined;

async function openAIResponsesWebSocketSession(): Promise<OpenAIResponsesWebSocketSessionModule> {
  if (loadedOpenAIResponsesWebSocketSession) return loadedOpenAIResponsesWebSocketSession;
  loadingOpenAIResponsesWebSocketSession ??= import('./openAIResponsesWebSocketSession');
  loadedOpenAIResponsesWebSocketSession = await loadingOpenAIResponsesWebSocketSession;
  return loadedOpenAIResponsesWebSocketSession;
}

function resetLoadedOpenAIResponsesWebSocketSessions(): void {
  loadedOpenAIResponsesWebSocketSession?.resetOpenAIResponsesWebSocketSessions();
}

class LlmAttemptFailureError extends Error {
  public constructor(public readonly failure: LlmAttemptFailure) {
    super(failure.message);
    this.name = 'LlmAttemptFailureError';
  }
}



/**
 * LLM capability 只维护 unified/Gemini-like 请求。
 * provider 真实 wire format 交给 unified-llm-provider 的 provider/format registry 处理。
 */
export function createLlmProviderCapability(options: LlmProviderOptions): LlmCapability {
  const controllers = new Map<string, AbortController>();
  const retryControls = new Map<string, RetryControl>();
  const resolvedRuntimeSettingsByInvocationId = new Map<string, LlmProviderConfigRecord>();

  return {
    resolveInvocation(request, emit) {
      void resolveLlmInvocationProvider(request, emit, options, resolvedRuntimeSettingsByInvocationId);
    },
    start(request, emit) {
      controllers.get(request.id)?.abort(createAbortError(`Superseded LLM request: ${request.id}`));
      retryControls.get(request.id)?.wakeRetryWait?.();

      const controller = new AbortController();
      const retryControl: RetryControl = { cancelRequested: false };
      controllers.set(request.id, controller);
      retryControls.set(request.id, retryControl);

      void startLlmProvider(request, emit, options, controller.signal, resolvedRuntimeSettingsByInvocationId, retryControl)
        .finally(() => {
          if (controllers.get(request.id) === controller) {
            controllers.delete(request.id);
          }
          if (retryControls.get(request.id) === retryControl) retryControls.delete(request.id);
          if (request.invocationId) resolvedRuntimeSettingsByInvocationId.delete(request.invocationId);
        });
    },
    compact(request, emit) {
      const previous = controllers.get(request.id);
      if (previous) {
        logCompressionDebug('capability.compact.supersede', compactRequestDebugInfo(request));
        retryControls.get(request.id)?.wakeRetryWait?.();
        previous.abort(createAbortError(`Superseded LLM compact request: ${request.id}`));
      }
      const controller = new AbortController();
      const retryControl: RetryControl = { cancelRequested: false };
      controllers.set(request.id, controller);
      retryControls.set(request.id, retryControl);
      logCompressionDebug('capability.compact.start', compactRequestDebugInfo(request));
      void compactLlmProvider(request, emit, options, controller.signal, retryControl)
        .finally(() => {
          const stillActive = controllers.get(request.id) === controller;
          logCompressionDebug('capability.compact.finally', {
            ...compactRequestDebugInfo(request),
            stillActive,
            signalAborted: controller.signal.aborted,
            abortReason: abortReasonText(controller.signal.reason)
          });
          if (stillActive) controllers.delete(request.id);
          if (retryControls.get(request.id) === retryControl) retryControls.delete(request.id);
        });
    },
    dryRun(request, dryRunOptions) {
      return dryRunLlmProvider(request, options, dryRunOptions, resolvedRuntimeSettingsByInvocationId);
    },
    dryRunCompact(request, dryRunOptions) {
      return dryRunCompactLlmProvider(request, options, dryRunOptions);
    },
    listModels(config) {
      return listLlmProviderModels(config, options);
    },
    cancelRetry(requestId) {
      const control = retryControls.get(requestId);
      if (!control) return;
      control.cancelRequested = true;
      control.wakeRetryWait?.();
    },
    abort(requestId) {
      const control = retryControls.get(requestId);
      if (control) control.cancelRequested = true;
      control?.wakeRetryWait?.();
      const controller = controllers.get(requestId);
      if (!controller) return;
      controllers.delete(requestId);
      controller.abort(createAbortError(`Aborted LLM request: ${requestId}`));
    },
    dispose() {
      for (const control of retryControls.values()) {
        control.cancelRequested = true;
        control.wakeRetryWait?.();
      }
      retryControls.clear();
      for (const [requestId, controller] of controllers) {
        controller.abort(createAbortError(`Disposed LLM capability during request: ${requestId}`));
      }
      controllers.clear();
      resolvedRuntimeSettingsByInvocationId.clear();
      resetLoadedOpenAIResponsesWebSocketSessions();
      openAIResponsesHttpCooldowns.clear();
    }
  };
}

export async function startLlmProvider(
  request: LlmStartRequest,
  emit: Emit,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  resolvedRuntimeSettingsByInvocationId?: Map<string, LlmProviderConfigRecord>,
  retryControl: RetryControl = { cancelRequested: false }
): Promise<void> {
  const streamEvents = createLlmStreamEventBatcher(emit, {
    onTerminalMetrics: (metrics) => {
      if (metrics.rawDeltaEvents === 0) return;
      console.log('[LimCode][LlmStreamAggregation]', JSON.stringify({
        requestId: request.id,
        ...metrics,
        reductionRatio: Number((metrics.emittedDeltaEvents / metrics.rawDeltaEvents).toFixed(4))
      }));
    }
  });
  const streamEmit = streamEvents.emit;
  try {
    const settings = await resolveRuntimeSettings(request, options, resolvedRuntimeSettingsByInvocationId);
    emitLlmStarted(streamEmit, request.id, request.invocationId, resolveModelDisplayName(settings));

    const unified = await importUnifiedLlmProvider();
    const registry = unified.createBootstrapExtensionRegistry();
    const proxy = normalizeOptionalString(await resolveMaybe(options.proxy));
    const proxyFetch = proxy ? createProxyFetch(proxy) : undefined;
    const providerFetch = createTerminalValidatedFetch(proxyFetch ?? fetch, settings.provider);
    const headers = mergeHeaders(await resolveMaybe(options.headers), settings.headers);
    const requestBody = requestBodyWithOpenAIPromptCacheKey(settings, request.conversationId);
    if (proxy) console.log(`[LimCode] LLM proxy enabled: ${proxy}`);
    const providerConfig = {
      provider: settings.provider,
      model: settings.model,
      apiKey: settings.apiKey,
      baseUrl: settings.baseUrl,
      ...(settings.contextWindowTokens ? { contextWindow: settings.contextWindowTokens } : {}),
      ...(headers ? { headers } : {}),
      ...(requestBody ? { requestBody } : {}),
      ...unifiedPromptCacheConfigEntry(settings, requestBody),
      ...openAIResponsesWebSocketConfigEntry(settings, request.conversationId),
      ...(proxy ? { proxy } : {}),
      fetch: providerFetch
    };
    const provider = installGeminiProviderCompatibility(
      unified.createLLMFromConfig(providerConfig, registry.llmProviders) as UnifiedChatProvider,
      settings.provider,
      settings.model
    );
    const httpFallbackProvider = isOpenAIResponsesWebSocketMode(settings)
      ? unified.createLLMFromConfig({
          ...providerConfig,
          transport: undefined,
          webSocketSessionKey: undefined
        }, registry.llmProviders) as UnifiedChatProvider
      : undefined;

    const retryEnabled = settings.retryOnError !== false;
    const maxRetries = normalizeRetryMaxAttempts(settings.retryMaxAttempts) ?? DEFAULT_LLM_RETRY_MAX_ATTEMPTS;
    let retryCount = 0;
    let sawRetry = false;

    while (true) {
      try {
        await runLlmAttempt(
          request,
          streamEmit,
          settings,
          provider,
          httpFallbackProvider,
          unified,
          options,
          signal,
          sawRetry ? { retryAttempt: retryCount, retryMaxAttempts: maxRetries } : undefined,
          proxy
        );
        return;
      } catch (error) {
        if (isRequestAbort(signal)) return;
        const failure = failureFromCaughtError(error);
        const nextRetryCount = retryCount + 1;
        const canRetry = retryEnabled
          && !retryControl.cancelRequested
          && (maxRetries === -1 || nextRetryCount <= maxRetries);

        if (!canRetry) {
          if (retryControl.cancelRequested && retryCount > 0) {
            emitLlmRetryCancelled(streamEmit, request.id, failure.message, retryCount, maxRetries, failure.rawError);
          }
          emitLlmError(streamEmit, request.id, failure.message, failure.rawError, {
            retryAttempt: retryCount || undefined,
            retryMaxAttempts: retryEnabled ? maxRetries : 0,
            createdAt: failure.createdAt,
            streamOutputDurationMs: failure.streamOutputDurationMs
          });
          return;
        }

        sawRetry = true;
        retryCount = nextRetryCount;
        const retryDelayMs = retryDelayForAttempt(retryCount);
        emitLlmRetryScheduled(streamEmit, request.id, failure.message, failure.rawError, retryCount, maxRetries, retryDelayMs);
        const shouldRetry = await waitForRetryDelay(retryDelayMs, retryControl, signal);
        if (!shouldRetry) {
          emitLlmRetryCancelled(streamEmit, request.id, failure.message, retryCount, maxRetries, failure.rawError);
          emitLlmError(streamEmit, request.id, failure.message, failure.rawError, {
            retryAttempt: retryCount,
            retryMaxAttempts: maxRetries,
            createdAt: failure.createdAt,
            streamOutputDurationMs: failure.streamOutputDurationMs
          });
          return;
        }
        emitLlmRetryStarted(streamEmit, request.id, failure.message, failure.rawError, retryCount, maxRetries);
      }
    }
  } catch (error) {
    if (isRequestAbort(signal)) return;
    const failure = failureFromCaughtError(error);
    emitLlmError(streamEmit, request.id, failure.message, failure.rawError, {
      createdAt: failure.createdAt,
      streamOutputDurationMs: failure.streamOutputDurationMs
    });
  } finally {
    streamEvents.dispose();
  }
}

async function runLlmAttempt(
  request: LlmStartRequest,
  emit: Emit,
  settings: LlmProviderConfigRecord,
  provider: UnifiedChatProvider,
  httpFallbackProvider: UnifiedChatProvider | undefined,
  unified: UnifiedModule,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  retryRecoveryNotice?: LlmAttemptRetryRecoveryNotice,
  proxy?: string
): Promise<void> {
  const preparedRequest = await prepareLlmStartRequestMultimodal(request, options);
  const unifiedRequest = toUnifiedRequest(preparedRequest, settings.generationConfig, settings.provider);
  const forceStreaming = isOpenAIResponsesWebSocketMode(settings);
  if (settings.stream === false && !forceStreaming) {
    const response = await provider.chat<UnifiedLLMResponse>(unifiedRequest, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      signal
    });
    if (signal?.aborted) throw createAbortError(`Aborted LLM request: ${request.id}`);
    if (hasUnifiedError(response)) {
      throw new LlmAttemptFailureError(failureFromProviderError(response.error, { rawResponse: response.rawResponse }));
    }
    emitRetryRecovered(request.id, emit, retryRecoveryNotice);
    emitUnifiedResponse(request.id, response, emit);
    const completedAt = Date.now();
    emit({
      type: LlmEventType.Done,
      payload: {
        requestId: request.id,
        createdAt: completedAt,
        completedAt,
        streamOutputDurationMs: 0,
        ...(usageMetadataFromCompact(response.usageMetadata) ? { usageMetadata: usageMetadataFromCompact(response.usageMetadata) } : {})
      }
    });
    return;
  }

  let latestUsageMetadata: LlmUsageMetadataRecord | undefined;
  let authoritativeCompletedContent: MessageContent | undefined;
  const timing: LlmAttemptTimingState = { streamTimingChunkCount: 0 };
  let activeThoughtBlock: ActiveThoughtBlock | undefined;
  let retryRecoveryPending = retryRecoveryNotice !== undefined;
  try {
    const stream: AsyncIterable<UnifiedLLMStreamChunk> = forceStreaming
      ? streamOpenAIResponsesWithLimCodeSession({
          request,
          settings,
          provider,
          httpFallbackProvider,
          unified,
          unifiedRequest,
          signal,
          retryRecoveryNotice,
          proxy,
          onTransportTrace: options.onTransportTrace
        })
      : provider.chatStream<UnifiedLLMStreamChunk>(unifiedRequest, {
          inputFormat: 'unified',
          outputFormat: 'unified',
          signal
        });
    for await (const chunk of stream) {
      if (signal?.aborted) throw createAbortError(`Aborted LLM request: ${request.id}`);
      if (hasUnifiedError(chunk)) {
        const failure = failureFromProviderError(chunk.error, {
          rawChunk: (chunk as { rawChunk?: unknown }).rawChunk ?? chunk,
          ...createDoneTiming(timing.firstStreamChunkAt, Date.now(), timing.firstStreamChunkMark, nowMonotonicMs(), timing.streamTimingChunkCount)
        });
        throw new LlmAttemptFailureError(failure);
      }
      const chunkAt = Date.now();
      const chunkMark = nowMonotonicMs();
      if (retryRecoveryPending && hasStreamTimingChunk(chunk)) {
        emitRetryRecovered(request.id, emit, retryRecoveryNotice);
        retryRecoveryPending = false;
      }
      activeThoughtBlock = emitThoughtDeltas(request.id, activeThoughtBlock, chunk, chunkAt, emit);
      if (activeThoughtBlock && shouldCloseThoughtBlock(chunk)) activeThoughtBlock = finishThoughtBlock(request.id, activeThoughtBlock, chunkAt, emit);
      const chunkUsageMetadata = usageMetadataFromChunk(chunk);
      if (chunkUsageMetadata) latestUsageMetadata = mergeUsageMetadata(latestUsageMetadata, chunkUsageMetadata);
      const completedContent = (chunk as LimCodeOpenAIResponsesStreamChunk).completedContent;
      if (completedContent) authoritativeCompletedContent = fromUnifiedCompletedContent(completedContent);
      if (hasStreamTimingChunk(chunk)) {
        timing.firstStreamChunkAt ??= chunkAt;
        timing.firstStreamChunkMark ??= chunkMark;
        timing.streamTimingChunkCount += 1;
      }
      emitUnifiedChunk(request.id, chunk, emit);
    }
  } catch (error) {
    const aborted = isRequestAbort(signal);
    if (activeThoughtBlock) activeThoughtBlock = aborted
      ? disposeThoughtBlock(activeThoughtBlock)
      : finishThoughtBlock(request.id, activeThoughtBlock, Date.now(), emit);
    if (aborted) throw createAbortError(`Aborted LLM request: ${request.id}`);
    const failure = failureFromCaughtError(error);
    const failureTiming = createDoneTiming(
      timing.firstStreamChunkAt,
      Date.now(),
      timing.firstStreamChunkMark,
      nowMonotonicMs(),
      timing.streamTimingChunkCount
    );
    throw new LlmAttemptFailureError({
      ...failure,
      createdAt: failureTiming.createdAt,
      ...(failureTiming.streamOutputDurationMs !== undefined
        ? { streamOutputDurationMs: failureTiming.streamOutputDurationMs }
        : {})
    });
  }

  if (signal?.aborted) throw createAbortError(`Aborted LLM request: ${request.id}`);
  const finishedAt = Date.now();
  const finishedMark = nowMonotonicMs();
  if (activeThoughtBlock) finishThoughtBlock(request.id, activeThoughtBlock, finishedAt, emit);
  if (retryRecoveryPending) emitRetryRecovered(request.id, emit, retryRecoveryNotice);
  emit({
    type: LlmEventType.Done,
    payload: {
      requestId: request.id,
      ...(authoritativeCompletedContent ? { content: authoritativeCompletedContent } : {}),
      ...createDoneTiming(timing.firstStreamChunkAt, finishedAt, timing.firstStreamChunkMark, finishedMark, timing.streamTimingChunkCount),
      completedAt: finishedAt,
      ...(latestUsageMetadata ? { usageMetadata: latestUsageMetadata } : {})
    }
  });
}

async function* streamOpenAIResponsesWithLimCodeSession(input: {
  request: LlmStartRequest;
  settings: LlmProviderConfigRecord;
  provider: UnifiedChatProvider;
  httpFallbackProvider?: UnifiedChatProvider;
  unified: UnifiedModule;
  unifiedRequest: UnifiedLLMRequest;
  signal?: AbortSignal;
  retryRecoveryNotice?: LlmAttemptRetryRecoveryNotice;
  proxy?: string;
  onTransportTrace?: (trace: LlmProviderTransportTrace) => void;
}): AsyncGenerator<LimCodeOpenAIResponsesStreamChunk> {
  const conversationId = requireOpenAIResponsesWebSocketConversationId(input.request.conversationId);
  const sessionKey = createOpenAIResponsesWebSocketSessionKey(input.settings, conversationId);
  const now = Date.now();
  for (const [key, expiresAt] of openAIResponsesHttpCooldowns) {
    if (expiresAt <= now) openAIResponsesHttpCooldowns.delete(key);
  }
  const cooldownUntil = openAIResponsesHttpCooldowns.get(sessionKey) ?? 0;
  if (input.httpFallbackProvider && cooldownUntil > now) {
    reportTransportPolicyTrace(input, sessionKey, 'http_cooldown', 'temporary_ws_cooldown');
    yield* input.httpFallbackProvider.chatStream<LimCodeOpenAIResponsesStreamChunk>(input.unifiedRequest, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      signal: input.signal
    });
    return;
  }
  const reliableAttempt = input.request.reliableProviderAttempt;
  const retryTimeBudgetExhausted = reliableAttempt?.requestCreatedAt !== undefined
    && reliableAttempt.attemptSeq > 1
    && now - reliableAttempt.requestCreatedAt >= OPENAI_RESPONSES_WS_RETRY_BUDGET_MS;
  if (input.httpFallbackProvider && retryTimeBudgetExhausted) {
    openAIResponsesHttpCooldowns.set(sessionKey, now + OPENAI_RESPONSES_HTTP_COOLDOWN_MS);
    reportTransportPolicyTrace(input, sessionKey, 'http_fallback', 'ws_retry_time_budget_exhausted');
    yield* input.httpFallbackProvider.chatStream<LimCodeOpenAIResponsesStreamChunk>(input.unifiedRequest, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      signal: input.signal
    });
    return;
  }
  if (cooldownUntil > 0) openAIResponsesHttpCooldowns.delete(sessionKey);

  const dryRun = await input.provider.dryRun(input.unifiedRequest, {
    inputFormat: 'unified',
    outputFormat: 'unified',
    stream: true
  });
  const format = new input.unified.OpenAIResponsesFormat(input.settings.model) as OpenAIResponsesFormatAdapter;
  const continuation = openAIResponsesContinuationHint(input.request, input.unifiedRequest);
  try {
    const { streamOpenAIResponsesWebSocketSession } = await openAIResponsesWebSocketSession();
    yield* streamOpenAIResponsesWebSocketSession({
      sessionKey,
      url: dryRun.url,
      headers: dryRun.headers,
      body: dryRun.body,
      format,
      ...(continuation ? { continuation } : {}),
      forceNewConnection: reliableAttempt?.attemptSeq !== undefined
        && reliableAttempt.attemptSeq > 1,
      signal: input.signal,
      proxy: input.proxy,
      onDecision: (decision) => {
        reportTransportTrace(input, {
          requestId: input.request.id,
          conversationId,
          phase: 'continuation_decision',
          observedAt: Date.now(),
          sessionKeyHash: decision.sessionKeyHash,
          connectionGeneration: decision.connectionGeneration,
          connectionReused: decision.connectionReused,
          connectionReason: decision.connectionReason,
          mode: decision.mode,
          reason: decision.reason,
          fullInputItemCount: decision.fullInputItemCount,
          sentInputItemCount: decision.sentInputItemCount
        });
        console.log('[LimCode][OpenAIResponsesWS]', JSON.stringify({
          implementation: LIMCODE_OPENAI_RESPONSES_WS_IMPLEMENTATION,
          requestId: input.request.id,
          conversationId: input.request.conversationId ?? '',
          sessionKeyHash: decision.sessionKeyHash,
          connectionGeneration: decision.connectionGeneration,
          connectionReused: decision.connectionReused,
          connectionReason: decision.connectionReason,
          mode: decision.mode,
          reason: decision.reason,
          fullInputItemCount: decision.fullInputItemCount,
          sentInputItemCount: decision.sentInputItemCount,
          fullInputFingerprint: decision.fullInputFingerprint,
          sentInputFingerprint: decision.sentInputFingerprint,
          baselineFingerprint: decision.baselineFingerprint
        }));
      },
      onPhase: (phase) => reportTransportTrace(input, traceFromWebSocketPhase(input, phase))
    });
    openAIResponsesHttpCooldowns.delete(sessionKey);
  } catch (error) {
    if (!input.httpFallbackProvider || !shouldFallbackOpenAIResponsesToHttp(input, error)) throw error;
    openAIResponsesHttpCooldowns.set(sessionKey, Date.now() + OPENAI_RESPONSES_HTTP_COOLDOWN_MS);
    reportTransportPolicyTrace(input, sessionKey, 'http_fallback', 'ws_retry_budget_exhausted');
    yield* input.httpFallbackProvider.chatStream<LimCodeOpenAIResponsesStreamChunk>(input.unifiedRequest, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      signal: input.signal
    });
  }
}

function openAIResponsesContinuationHint(
  request: LlmStartRequest,
  unifiedRequest: UnifiedLLMRequest
): OpenAIResponsesWebSocketStreamOptions['continuation'] | undefined {
  const metadata = request.openAIResponsesContinuation;
  if (!metadata) return undefined;
  const kinds = metadata.volatileTailContentKinds;
  if (!Array.isArray(kinds) || kinds.length > unifiedRequest.contents.length) {
    return {
      volatileTailContents: [],
      volatileTailContentKinds: [],
      forceFullReason: 'invalid_volatile_tail_boundary'
    };
  }
  return {
    volatileTailContents: unifiedRequest.contents.slice(unifiedRequest.contents.length - kinds.length),
    volatileTailContentKinds: [...kinds]
  };
}

function shouldFallbackOpenAIResponsesToHttp(
  input: {
    request: LlmStartRequest;
    retryRecoveryNotice?: LlmAttemptRetryRecoveryNotice;
  },
  error: unknown
): boolean {
  const raw = rawErrorFromUnknown(error);
  if (findNestedMetadata(raw, 'receivedSemanticOutput') === true) return false;
  if (findNestedMetadata(raw, 'retryable') === false) return false;

  const status = findNestedNumber(raw, 'status');
  const signature = JSON.stringify(raw).toLowerCase();
  const recoverable = findNestedMetadata(raw, 'retryable') === true
    || status === 408
    || status === 425
    || status === 429
    || (status !== undefined && status >= 500 && status <= 599)
    || /econnreset|econnrefused|enotfound|enetunreach|ehostunreach|etimedout|eai_again|network_changed|socket hang up|network error|fetch failed|websocket closed before|timed? out|unexpected server response:\s*(?:408|425|429|5\d\d)\b/.test(signature);
  if (!recoverable) return false;

  const attempt = input.request.reliableProviderAttempt;
  const attemptLimitReached = attempt !== undefined && attempt.attemptSeq >= attempt.maxAttempts;
  const elapsedBudgetReached = attempt?.requestCreatedAt !== undefined
    && Date.now() - attempt.requestCreatedAt >= OPENAI_RESPONSES_WS_RETRY_BUDGET_MS;
  const legacyLimitReached = input.retryRecoveryNotice !== undefined
    && input.retryRecoveryNotice.retryAttempt >= input.retryRecoveryNotice.retryMaxAttempts;
  return attemptLimitReached || elapsedBudgetReached || legacyLimitReached;
}

function reportTransportPolicyTrace(
  input: {
    request: LlmStartRequest;
    onTransportTrace?: (trace: LlmProviderTransportTrace) => void;
  },
  sessionKey: string,
  phase: 'http_fallback' | 'http_cooldown',
  reason: string
): void {
  reportTransportTrace(input, {
    requestId: input.request.id,
    conversationId: input.request.conversationId ?? '',
    phase,
    observedAt: Date.now(),
    sessionKeyHash: createHash('sha256').update(sessionKey).digest('hex').slice(0, 12),
    connectionGeneration: 0,
    reason
  });
}

function findNestedMetadata(
  value: unknown,
  key: string,
  depth = 0,
  seen = new Set<object>()
): boolean | undefined {
  if (depth > 8 || value === null || typeof value !== 'object' || seen.has(value)) return undefined;
  seen.add(value);
  if (isRecord(value) && typeof value[key] === 'boolean') return value[key] as boolean;
  for (const child of Array.isArray(value) ? value : Object.values(value)) {
    const nested = findNestedMetadata(child, key, depth + 1, seen);
    if (nested !== undefined) return nested;
  }
  return undefined;
}

function findNestedNumber(
  value: unknown,
  key: string,
  depth = 0,
  seen = new Set<object>()
): number | undefined {
  if (depth > 8 || value === null || typeof value !== 'object' || seen.has(value)) return undefined;
  seen.add(value);
  if (isRecord(value) && typeof value[key] === 'number' && Number.isFinite(value[key])) return value[key] as number;
  for (const child of Array.isArray(value) ? value : Object.values(value)) {
    const nested = findNestedNumber(child, key, depth + 1, seen);
    if (nested !== undefined) return nested;
  }
  return undefined;
}

function traceFromWebSocketPhase(
  input: { request: LlmStartRequest },
  phase: OpenAIResponsesWebSocketPhase
): LlmProviderTransportTrace {
  return {
    requestId: input.request.id,
    conversationId: input.request.conversationId ?? '',
    phase: phase.phase,
    observedAt: phase.observedAt,
    sessionKeyHash: phase.sessionKeyHash,
    connectionGeneration: phase.connectionGeneration,
    ...(phase.elapsedMs !== undefined ? { elapsedMs: phase.elapsedMs } : {}),
    ...(phase.connectionReused !== undefined ? { connectionReused: phase.connectionReused } : {}),
    ...(phase.connectionReason ? { connectionReason: phase.connectionReason } : {}),
    ...(phase.mode ? { mode: phase.mode } : {}),
    ...(phase.reason ? { reason: phase.reason } : {}),
    ...(phase.timeoutPhase ? { timeoutPhase: phase.timeoutPhase } : {}),
    ...(phase.responseCreateFrameSha256
      ? { responseCreateFrameSha256: phase.responseCreateFrameSha256 }
      : {}),
    ...(phase.responseCreateFrameBytes !== undefined
      ? { responseCreateFrameBytes: phase.responseCreateFrameBytes }
      : {}),
    ...(phase.responseCreateSeq !== undefined
      ? { responseCreateSeq: phase.responseCreateSeq }
      : {})
  };
}

function reportTransportTrace(
  input: { onTransportTrace?: (trace: LlmProviderTransportTrace) => void },
  trace: LlmProviderTransportTrace
): void {
  try {
    input.onTransportTrace?.(trace);
  } catch {
    // Observability is best-effort and must not become Provider authority.
  }
}

function emitRetryRecovered(requestId: string, emit: Emit, notice: LlmAttemptRetryRecoveryNotice | undefined): void {
  if (!notice) return;
  emitLlmRetryRecovered(emit, requestId, '自动重试成功。', notice.retryAttempt, notice.retryMaxAttempts);
}

function hasUnifiedError(value: unknown): value is { error: unknown; rawResponse?: unknown; rawChunk?: unknown } {
  return isRecord(value) && value.error !== undefined && value.error !== null;
}

function failureFromCaughtError(error: unknown): LlmAttemptFailure {
  if (error instanceof LlmAttemptFailureError) return error.failure;
  const rawError = rawErrorFromUnknown(error);
  return { message: messageFromRawError(rawError), rawError, createdAt: Date.now() };
}

function failureFromProviderError(error: unknown, extras: Record<string, unknown> = {}): LlmAttemptFailure {
  const rawError = rawErrorFromUnknown(error, extras);
  return {
    message: messageFromRawError(rawError),
    rawError,
    createdAt: typeof extras.createdAt === 'number' ? extras.createdAt : Date.now(),
    ...(typeof extras.streamOutputDurationMs === 'number' ? { streamOutputDurationMs: extras.streamOutputDurationMs } : {})
  };
}

export function rawErrorFromUnknown(error: unknown, extras: Record<string, unknown> = {}): LlmRawErrorInfoRecord {
  const base = toPlainJsonLike(error);
  const baseRecord = isRecord(base) ? base : { data: base };
  const merged: LlmRawErrorInfoRecord = { ...baseRecord };
  for (const [key, value] of Object.entries(extras)) {
    if (value !== undefined && !isSensitiveLlmErrorField(key)) merged[key] = toPlainJsonLike(value);
  }
  if (typeof merged.message !== 'string') {
    const message = error instanceof Error ? error.message : typeof error === 'string' ? error : undefined;
    if (message) merged.message = message;
  }
  return merged;
}

function messageFromRawError(rawError: LlmRawErrorInfoRecord): string {
  return summarizeLlmRawError(rawError);
}

export function summarizeLlmRawError(rawError: LlmRawErrorInfoRecord): string {
  const summary = summarizeLlmRawErrorBase(rawError);
  const evidence = wireInvariantEvidence(rawError);
  return evidence && remoteReportsMissingToolResultId(rawError)
    ? `${summary} Local wire invariant passed before fetch; ${evidence}.`
    : summary;
}

function summarizeLlmRawErrorBase(rawError: LlmRawErrorInfoRecord): string {
  const direct = specificErrorMessage(rawError.message);
  if (direct) return direct;
  for (const candidate of [
    rawError.rawBody,
    rawError.rawChunk,
    rawError.rawResponse,
    rawError.data,
    rawError.bodyText
  ]) {
    const message = nestedMessage(candidate);
    if (message) return message;
  }
  const bodyText = specificErrorMessage(rawError.bodyText);
  if (bodyText) return bodyText;
  const dataText = specificErrorMessage(rawError.data);
  if (dataText) return dataText;
  const kind = typeof rawError.kind === 'string' && rawError.kind.trim() ? rawError.kind.trim() : 'llm_error';
  const status = typeof rawError.status === 'number' ? ` HTTP ${rawError.status}` : '';
  return `LLM 请求失败：${kind}${status}`;
}

function wireInvariantEvidence(rawError: LlmRawErrorInfoRecord): string | undefined {
  const headers = isRecord(rawError.headers) ? rawError.headers : undefined;
  const value = headers?.['x-limcode-wire-invariant'];
  return typeof value === 'string' && /^passed; body_sha256=[a-f0-9]{64}$/.test(value)
    ? value.slice('passed; '.length)
    : undefined;
}

function remoteReportsMissingToolResultId(rawError: LlmRawErrorInfoRecord): boolean {
  const text = stringifyJson(toPlainJsonLike(rawError));
  return /(?:missing|required)[^\n]{0,160}(?:tool_call_id|call_id|tool_use_id|functionResponse)|(?:tool_call_id|call_id|tool_use_id|functionResponse)[^\n]{0,160}(?:missing|required)/i.test(text);
}

function nestedMessage(value: unknown, depth = 0, seen = new Set<object>()): string | undefined {
  if (depth > 8 || value === null || value === undefined) return undefined;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return undefined;
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        const nested = nestedMessage(parsed, depth + 1, seen);
        if (nested) return nested;
      } catch {
        // Keep the original non-JSON text as a final specific-message candidate.
      }
    }
    return specificErrorMessage(trimmed);
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const message = nestedMessage(item, depth + 1, seen);
      if (message) return message;
    }
    return undefined;
  }
  if (!isRecord(value) || seen.has(value)) return undefined;
  seen.add(value);

  for (const key of ['message', 'detail', 'error_description', 'reason']) {
    const message = specificErrorMessage(value[key]);
    if (message) return message;
  }
  for (const key of [
    'error',
    'response',
    'cause',
    'details',
    'incomplete_details',
    'rawBody',
    'rawChunk',
    'rawResponse',
    'data'
  ]) {
    const message = nestedMessage(value[key], depth + 1, seen);
    if (message) return message;
  }
  return undefined;
}

function specificErrorMessage(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed || isGenericLlmErrorLabel(trimmed)) return undefined;
  return truncateForSummary(trimmed);
}

function isGenericLlmErrorLabel(value: string): boolean {
  return new Set([
    'error',
    'stream_error',
    'upstream_error',
    'http_error',
    'response_error',
    'decode_error',
    'stream_read_error',
    'stream_parse_error',
    'llm_error'
  ]).has(value.trim().toLowerCase());
}

function truncateForSummary(value: string): string {
  const limit = 600;
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function retryDelayForAttempt(retryAttempt: number): number {
  const base = Math.min(8_000, 500 * (2 ** Math.max(0, retryAttempt - 1)));
  return Math.max(0, Math.round(base * (0.75 + Math.random() * 0.25)));
}

function waitForRetryDelay(delayMs: number, control: RetryControl, signal?: AbortSignal): Promise<boolean> {
  if (control.cancelRequested) return Promise.resolve(false);
  if (signal?.aborted) return Promise.reject(createAbortError('Aborted LLM retry wait.'));
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const previousWake = control.wakeRetryWait;
    const cleanup = () => {
      if (timeout !== undefined) clearTimeout(timeout);
      if (signal) signal.removeEventListener('abort', onAbort);
      control.wakeRetryWait = previousWake;
    };
    const settle = (value: boolean) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(createAbortError('Aborted LLM retry wait.'));
    };
    control.wakeRetryWait = () => {
      previousWake?.();
      settle(false);
    };
    timeout = setTimeout(() => settle(!control.cancelRequested), delayMs);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function toPlainJsonLike(value: unknown, seen = new WeakSet<object>()): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'symbol' || typeof value === 'function') return String(value);
  if (value instanceof Error) {
    if (seen.has(value)) return '[Circular]';
    seen.add(value);
    const source = value as Error & Record<string, unknown> & { cause?: unknown };
    const result: Record<string, unknown> = {
      name: value.name,
      message: value.message,
      stack: value.stack
    };
    if (source.cause !== undefined) result.cause = toPlainJsonLike(source.cause, seen);
    for (const [key, child] of Object.entries(source)) {
      if (key === 'name' || key === 'message' || key === 'stack' || key === 'cause') continue;
      if (isSensitiveLlmErrorField(key)) continue;
      result[key] = toPlainJsonLike(child, seen);
    }
    return result;
  }
  if (typeof Headers !== 'undefined' && value instanceof Headers) {
    return plainRecordFromEntries(value.entries(), seen);
  }
  if (typeof value !== 'object') return value;
  if (seen.has(value)) return '[Circular]';
  seen.add(value);
  if (Array.isArray(value)) return value.map((item) => toPlainJsonLike(item, seen));
  if (typeof (value as { entries?: unknown }).entries === 'function' && typeof (value as { forEach?: unknown }).forEach === 'function') {
    try {
      return plainRecordFromEntries((value as { entries(): Iterable<[string, unknown]> }).entries(), seen);
    } catch {
      // fall through
    }
  }
  const result: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (isSensitiveLlmErrorField(key)) continue;
    result[key] = toPlainJsonLike(child, seen);
  }
  return result;
}

function plainRecordFromEntries(entries: Iterable<[string, unknown]>, seen: WeakSet<object>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, child] of entries) {
    if (isSensitiveLlmErrorField(key)) continue;
    result[key] = toPlainJsonLike(child, seen);
  }
  return result;
}

function isSensitiveLlmErrorField(key: string): boolean {
  const normalized = key.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
  if (!normalized) return false;
  return normalized === 'authorization'
    || normalized.endsWith('authorization')
    || normalized === 'cookie'
    || normalized.endsWith('cookie')
    || normalized === 'auth'
    || normalized === 'credentials'
    || normalized === 'credential'
    || normalized === 'password'
    || normalized.endsWith('password')
    || normalized === 'passwd'
    || normalized === 'secret'
    || normalized.endsWith('secret')
    || normalized.endsWith('secretkey')
    || normalized === 'privatekey'
    || normalized.endsWith('privatekey')
    || normalized === 'accesskey'
    || normalized.endsWith('accesskey')
    || normalized === 'apikey'
    || normalized.endsWith('apikey')
    || normalized === 'xapikey'
    || normalized === 'token'
    || normalized.endsWith('authtoken')
    || normalized.endsWith('accesstoken')
    || normalized.endsWith('refreshtoken')
    || normalized.endsWith('sessiontoken')
    || normalized.endsWith('bearertoken')
    || normalized.endsWith('idtoken');
}




export async function resolveLlmInvocationProvider(
  request: LlmResolveInvocationRequest,
  emit: Emit,
  options: LlmProviderOptions,
  resolvedRuntimeSettingsByInvocationId?: Map<string, LlmProviderConfigRecord>
): Promise<void> {
  try {
    const settings = normalizeSettings(await resolveMaybe(options.settings, request));
    const compressionConfig = await options.activeCompressionSettings?.({ conversationId: request.conversationId, providerConfigId: settings.id, model: settings.model });
    resolvedRuntimeSettingsByInvocationId?.set(request.invocationId, settings);
    emit({ type: LlmEventType.InvocationResolved, payload: { invocationId: request.invocationId, requestId: request.requestId, settings: snapshotFromSettings(settings, compressionConfig), resolvedAt: Date.now() } });
  } catch (error) {
    emit({ type: LlmEventType.InvocationResolveError, payload: { invocationId: request.invocationId, requestId: request.requestId, message: error instanceof Error ? error.message : String(error), resolvedAt: Date.now() } });
  }
}

export async function dryRunLlmProvider(request: LlmStartRequest, options: LlmProviderOptions, dryRunOptions: LlmDryRunOptions = {}, resolvedRuntimeSettingsByInvocationId?: Map<string, LlmProviderConfigRecord>): Promise<LlmDryRunResult> {
  const settings = await resolveRuntimeSettings(request, options, resolvedRuntimeSettingsByInvocationId);
  const apiKeyAvailable = !!settings.apiKey;
  const runtimeSettings = apiKeyAvailable ? settings : { ...settings, apiKey: 'limcode-dry-run-placeholder-key' };

  const unified = await importUnifiedLlmProvider();
  const registry = unified.createBootstrapExtensionRegistry();
  const proxy = normalizeOptionalString(await resolveMaybe(options.proxy));
  const proxyFetch = proxy ? createProxyFetch(proxy) : undefined;
  const providerFetch = createTerminalValidatedFetch(proxyFetch ?? fetch, runtimeSettings.provider);
  const headers = mergeHeaders(await resolveMaybe(options.headers), runtimeSettings.headers);
  const requestBody = requestBodyWithOpenAIPromptCacheKey(runtimeSettings, request.conversationId);
  const provider = installGeminiProviderCompatibility(unified.createLLMFromConfig({
    provider: runtimeSettings.provider,
    model: runtimeSettings.model,
    apiKey: runtimeSettings.apiKey,
    baseUrl: runtimeSettings.baseUrl,
    ...(runtimeSettings.contextWindowTokens ? { contextWindow: runtimeSettings.contextWindowTokens } : {}),
    ...(headers ? { headers } : {}),
    ...(requestBody ? { requestBody } : {}),
    ...unifiedPromptCacheConfigEntry(runtimeSettings, requestBody),
    ...openAIResponsesWebSocketConfigEntry(runtimeSettings, request.conversationId),
    ...(proxy ? { proxy } : {}),
    fetch: providerFetch
  }, registry.llmProviders) as UnifiedChatProvider, runtimeSettings.provider, runtimeSettings.model);

  const dryRun = (provider as unknown as Partial<UnifiedDryRunCapable>).dryRun;
  if (typeof dryRun !== 'function') {
    throw new Error('当前 unified-llm-provider 版本不支持 provider.dryRun，请更新依赖。');
  }

  const preparedRequest = await prepareLlmStartRequestMultimodal(request, options);
  const webSocketMode = isOpenAIResponsesWebSocketMode(runtimeSettings);
  const result = await dryRun.call(provider, toUnifiedRequest(
    preparedRequest,
    runtimeSettings.generationConfig,
    runtimeSettings.provider
  ), {
    inputFormat: 'unified',
    outputFormat: 'unified',
    stream: runtimeSettings.stream !== false || webSocketMode,
    curl: { includeApiKey: dryRunOptions.includeApiKey === true, prettyBody: true }
  });

  if (webSocketMode) {
    const displayResult = openAIResponsesWebSocketDryRunResult(result, dryRunOptions.includeApiKey === true);
    return formatUnifiedDryRunResult(displayResult, runtimeSettings, unified, dryRunOptions, apiKeyAvailable, displayResult.maskedCurl);
  }
  return formatUnifiedDryRunResult(result, runtimeSettings, unified, dryRunOptions, apiKeyAvailable);
}

export async function dryRunCompactLlmProvider(
  request: LlmCompactRequest,
  options: LlmProviderOptions,
  dryRunOptions: LlmDryRunOptions = {}
): Promise<LlmCompactDryRunResult> {
  const methodConfig = normalizeCompressionConfig(
    request.methodConfigSnapshot ?? await options.compressionSettings?.(request),
    request.methodKind
  );
  if (methodConfig.kind === 'disabled') throw new Error('当前压缩方法已关闭。');
  const generatedAt = Date.now();
  if (methodConfig.kind === 'openai_responses_compact') {
    const observationContract = normalizeCompressionAttachmentObservationContract(request);
    if (observationContract.requirements.length > 0) {
      throw new TypeError('Provider-native Compact cannot carry text-summary Attachment observations.');
    }
    const call = await dryRunOpenAIResponsesCompact(request, methodConfig, options, dryRunOptions);
    return {
      kind: 'provider_requests',
      methodKind: methodConfig.kind,
      calls: [{ ...call, id: `${request.id}:compact`, label: 'Responses Compact', ordinal: 0 }],
      generatedAt
    };
  }
  if (methodConfig.kind === 'deterministic_summary' || methodConfig.kind === 'manual_summary') {
    await prepareCompressionMediaSemantics(request, methodConfig, options);
    return {
      kind: 'no_provider_call',
      methodKind: methodConfig.kind,
      calls: [],
      note: methodConfig.kind === 'manual_summary'
        ? '该方法只生成本地可编辑摘要，不会调用 Provider；媒体必须已有持久化 observation。'
        : '该方法使用确定性本地摘要，不会调用 Provider；媒体必须已有持久化 observation。',
      generatedAt
    };
  }

  const resolved = await resolveSummaryProvider(request, methodConfig, options, { allowPlaceholderApiKey: true });
  if (!resolved.provider) throw new Error('无法构造压缩 dry-run Provider。');
  const mediaSemantics = await prepareCompressionMediaSemanticsDryRun(
    request,
    methodConfig,
    options,
    resolved
  );
  const semanticRequest = mediaSemantics.request;
  if (summaryDeltaContents(semanticRequest).length === 0) {
    return {
      kind: mediaSemantics.observationCalls.length > 0 ? 'provider_requests' : 'no_provider_call',
      methodKind: methodConfig.kind,
      calls: [],
      note: '没有新的摘要源；沿用并收口现有 replacement summary。',
      generatedAt
    };
  }

  const summaryCalls: SummaryProviderCall[] = methodConfig.kind === 'segmented_summary'
    ? buildSegmentedSummaryProviderCalls(semanticRequest, methodConfig, resolved.settings)
    : [buildSummaryProviderCall(semanticRequest, methodConfig, resolved.settings)];
  if (methodConfig.kind === 'llm_summary' && !isSummaryProviderCallWithinWindow(summaryCalls[0]!, resolved.settings)) {
    throw new Error('compression_request_too_large: summary input exceeds the frozen Provider input limit.');
  }
  const dryRunCalls = [...mediaSemantics.observationCalls, ...summaryCalls];
  const unified = await importUnifiedLlmProvider();
  const providerDryRun = (resolved.provider as unknown as Partial<UnifiedDryRunCapable>).dryRun;
  if (typeof providerDryRun !== 'function') throw new Error('当前 unified-llm-provider 版本不支持 provider.dryRun。');
  const results = [] as LlmCompactDryRunResult['calls'];
  for (const [ordinal, call] of dryRunCalls.entries()) {
    const result = await providerDryRun.call(resolved.provider, call.request, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      stream: resolved.stream,
      curl: { includeApiKey: dryRunOptions.includeApiKey === true, prettyBody: true }
    });
    results.push({
      ...formatUnifiedDryRunResult(result, resolved.settings, unified, dryRunOptions, resolved.apiKeyAvailable),
      id: ordinal < mediaSemantics.observationCalls.length
        ? `${request.id}:attachment-observation:${ordinal}`
        : `${request.id}:summary:${ordinal - mediaSemantics.observationCalls.length}`,
      label: call.label,
      ordinal
    });
  }
  return {
    kind: 'provider_requests',
    methodKind: methodConfig.kind,
    calls: results,
    ...(methodConfig.kind === 'segmented_summary' || mediaSemantics.observationCalls.length > 0 ? {
      note: [
        ...(mediaSemantics.observationCalls.length > 0
          ? ['前置请求逐个分析缺失的 F 附件；后续 summary dry-run 使用明确的 observation 占位值。']
          : []),
        ...(methodConfig.kind === 'segmented_summary'
          ? ['仅展示可预先确定的 leaf summary requests；后续 hierarchy merge requests 依赖前序 Provider 摘要，运行时动态构造。']
          : [])
      ].join(' ')
    } : {}),
    generatedAt
  };
}

async function dryRunOpenAIResponsesCompact(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  dryRunOptions: LlmDryRunOptions
): Promise<LlmDryRunResult> {
  const preparedContext = await prepareNativeCompactContentsMultimodal(request.contents, options);
  const normalizedContext = assertCanonicalProviderToolContext(preparedContext);
  const settings = await resolveCompactProviderSettings(request, methodConfig, normalizedContext, options);
  if (settings.provider !== 'openai-responses') throw new Error('OpenAI 原生压缩仅支持 openai-responses 渠道格式。');
  const apiKeyAvailable = !!settings.apiKey;
  const runtimeSettings = apiKeyAvailable ? settings : { ...settings, apiKey: 'limcode-dry-run-placeholder-key' };
  const unified = await importUnifiedLlmProvider();
  const registry = unified.createBootstrapExtensionRegistry();
  const proxy = normalizeOptionalString(await resolveMaybe(options.proxy));
  const proxyFetch = proxy ? createProxyFetch(proxy) : undefined;
  const providerFetch = createTerminalValidatedFetch(proxyFetch ?? fetch, runtimeSettings.provider);
  const headers = mergeHeaders(await resolveMaybe(options.headers), runtimeSettings.headers);
  const requestBody = requestBodyWithOpenAIPromptCacheKey(runtimeSettings, request.conversationId);
  const provider = unified.createLLMFromConfig({
    provider: runtimeSettings.provider,
    model: runtimeSettings.model,
    apiKey: runtimeSettings.apiKey,
    baseUrl: runtimeSettings.baseUrl,
    ...(runtimeSettings.contextWindowTokens ? { contextWindow: runtimeSettings.contextWindowTokens } : {}),
    ...(headers ? { headers } : {}),
    ...(requestBody ? { requestBody } : {}),
    ...unifiedPromptCacheConfigEntry(runtimeSettings, requestBody),
    ...(proxy ? { proxy } : {}),
    fetch: providerFetch
  }, registry.llmProviders) as unknown as Partial<UnifiedDryRunCapable>;
  if (typeof provider.compactDryRun !== 'function') {
    throw new Error('当前 unified-llm-provider 版本不支持 provider.compactDryRun。');
  }
  const result = await provider.compactDryRun(
    { contents: normalizedContext.flatMap((content) => toUnifiedContents(content, 'openai-responses')) },
    {
      inputFormat: 'unified',
      outputFormat: 'unified',
      ...(requestBody ? { requestBody } : {}),
      curl: { includeApiKey: dryRunOptions.includeApiKey === true, prettyBody: true }
    }
  );
  return formatUnifiedDryRunResult(result, runtimeSettings, unified, dryRunOptions, apiKeyAvailable);
}

function formatUnifiedDryRunResult(
  result: UnifiedDryRunResult,
  settings: LlmProviderConfigRecord,
  unified: UnifiedModule,
  options: LlmDryRunOptions,
  apiKeyAvailable: boolean,
  maskedCurlOverride?: string
): LlmDryRunResult {
  return {
    provider: settings.provider,
    model: settings.model,
    providerName: result.providerName,
    url: result.url,
    method: result.method,
    stream: result.stream,
    headers: result.headers,
    body: result.body,
    bodyText: result.bodyText,
    curl: result.curl,
    maskedCurl: maskedCurlOverride ?? unified.formatRequestAsCurl(result.url, result.headers, result.body, { includeApiKey: false, prettyBody: true }),
    inputFormat: result.inputFormat,
    outputFormat: result.outputFormat,
    generatedAt: result.timestamp,
    maskedSecrets: options.includeApiKey !== true || !apiKeyAvailable,
    apiKeyAvailable
  };
}

export async function listLlmProviderModels(config: LlmProviderConfigRecord, options: LlmProviderOptions): Promise<LlmProviderModelRecord[]> {
  const settings = normalizeSettings(config);

  const unified = await importUnifiedLlmProvider();
  const headers = mergeHeaders(await resolveMaybe(options.headers), settings.headers);
  const result = await unified.listAvailableModels({
    provider: settings.provider,
    apiKey: settings.apiKey,
    baseUrl: settings.baseUrl,
    ...(headers ? { headers } : {}),
    outputFormat: 'unified'
  });

  return result.models.map(modelCatalogEntryToRecord);
}

export type LlmCompressionMethodHandler = (
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal
) => Promise<LlmCompactResult>;

const compressionMethodHandlers = new Map<LlmCompressionConfigRecord['kind'], LlmCompressionMethodHandler>();

export function registerLlmCompressionMethod(kind: LlmCompressionConfigRecord['kind'], handler: LlmCompressionMethodHandler): void {
  compressionMethodHandlers.set(kind, handler);
}

function ensureDefaultCompressionMethodsRegistered(): void {
  if (compressionMethodHandlers.size > 0) return;
  registerLlmCompressionMethod('openai_responses_compact', compactWithOpenAIResponses);
  registerLlmCompressionMethod('llm_summary', compactWithSummary);
  registerLlmCompressionMethod('segmented_summary', compactWithSegmentedSummary);
  registerLlmCompressionMethod('deterministic_summary', compactWithSummary);
  registerLlmCompressionMethod('manual_summary', compactWithSummary);
}

export async function compactLlmProvider(
  request: LlmCompactRequest,
  emit: Emit,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  retryControl: RetryControl = { cancelRequested: false }
): Promise<void> {
  logCompressionDebug('provider.compact.begin', { ...compactRequestDebugInfo(request), signalAborted: signal?.aborted === true });
  try {
    ensureDefaultCompressionMethodsRegistered();
    const methodConfig = normalizeCompressionConfig(
      request.methodConfigSnapshot ?? await options.compressionSettings?.(request),
      request.methodKind
    );
    logCompressionDebug('provider.compact.methodResolved', {
      ...compactRequestDebugInfo(request),
      methodConfigId: methodConfig.id,
      methodConfigKind: methodConfig.kind,
      signalAborted: signal?.aborted === true
    });
    if (methodConfig.kind === 'disabled') {
      throw new Error('当前压缩方法已关闭。');
    }

    const handler = compressionMethodHandlers.get(methodConfig.kind);
    if (!handler) throw new Error(`未注册的压缩方法：${methodConfig.kind}`);

    // Freeze resolved media once for the whole native compact operation. Capability retries must
    // replay identical bytes even when the original reference was a mutable local sourcePath.
    const handlerRequest = methodConfig.kind === 'openai_responses_compact'
      ? { ...request, contents: await prepareNativeCompactContentsMultimodal(request.contents, options) }
      : request;

    const retrySettings = await resolveCompactRetrySettings(request, methodConfig, options);
    // segmented_summary is a bounded multi-call operation. Retrying the whole handler would replay
    // already-paid leaf calls, so recovery must happen at the durable ModelRequest boundary instead.
    const retryEnabled = retrySettings?.retryOnError !== false
      && isRetryCapableCompressionMethod(methodConfig.kind)
      && methodConfig.kind !== 'segmented_summary';
    const maxRetries = normalizeRetryMaxAttempts(retrySettings?.retryMaxAttempts) ?? DEFAULT_LLM_RETRY_MAX_ATTEMPTS;
    let retryCount = 0;
    let sawRetry = false;

    while (true) {
      try {
        const result = await handler(handlerRequest, methodConfig, options, signal);
        logCompressionDebug('provider.compact.done', {
          ...compactRequestDebugInfo(request),
          resultId: result.id,
          resultContentCount: result.contents.length,
          resultMethodKind: result.methodConfig?.kind,
          retryCount,
          signalAborted: signal?.aborted === true
        });

        if (sawRetry) emitLlmRetryRecovered(emit, request.id, '自动重试成功，压缩已恢复。', retryCount, maxRetries);
        emitCompactDone(emit, request, result, Date.now());
        return;
      } catch (error) {
        if (isRequestAbort(signal)) {
          logCompressionDebug('provider.compact.cancelledByRequestAbort', {
            ...compactRequestDebugInfo(request),
            error: errorDebugInfo(error),
            abortReason: abortReasonText(signal?.reason)
          });
          return;
        }

        const failure = failureFromCaughtError(error);
        const nextRetryCount = retryCount + 1;
        const canRetry = retryEnabled
          && isRetryableCompactFailure(error, failure)
          && !retryControl.cancelRequested
          && (maxRetries === -1 || nextRetryCount <= maxRetries);

        if (!canRetry) {
          if (retryControl.cancelRequested && retryCount > 0) {
            emitLlmRetryCancelled(emit, request.id, failure.message, retryCount, maxRetries, failure.rawError);
          }
          emitCompactError(emit, request, failure, Date.now(), {
            retryAttempt: retryCount || undefined,
            retryMaxAttempts: retryEnabled ? maxRetries : 0
          });
          return;
        }

        sawRetry = true;
        retryCount = nextRetryCount;
        const retryDelayMs = retryDelayForAttempt(retryCount);
        logCompressionDebug('provider.compact.retryScheduled', {
          ...compactRequestDebugInfo(request),
          message: failure.message,
          retryCount,
          maxRetries,
          retryDelayMs
        });
        emitLlmRetryScheduled(emit, request.id, failure.message, failure.rawError, retryCount, maxRetries, retryDelayMs);
        const shouldRetry = await waitForRetryDelay(retryDelayMs, retryControl, signal);
        if (!shouldRetry) {
          emitLlmRetryCancelled(emit, request.id, failure.message, retryCount, maxRetries, failure.rawError);
          emitCompactError(emit, request, failure, Date.now(), {
            retryAttempt: retryCount,
            retryMaxAttempts: maxRetries
          });
          return;
        }
        emitLlmRetryStarted(emit, request.id, failure.message, failure.rawError, retryCount, maxRetries);
      }
    }
  } catch (error) {
    if (isRequestAbort(signal)) {
      logCompressionDebug('provider.compact.cancelledByRequestAbort', {
        ...compactRequestDebugInfo(request),
        error: errorDebugInfo(error),
        abortReason: abortReasonText(signal?.reason)
      });
      return;
    }
    const failure = failureFromCaughtError(error);
    emitCompactError(emit, request, failure, Date.now());
  }
}

function emitCompactDone(emit: Emit, request: LlmCompactRequest, result: LlmCompactResult, completedAt: number): void {
  logCompressionDebug('provider.compact.emitDone', { ...compactRequestDebugInfo(request), completedAt });
  emit({
    type: LlmEventType.CompactDone,
    payload: {
      requestId: request.id,
      blockId: request.blockId,
      conversationId: request.conversationId,
      result,
      completedAt
    }
  });
}

function emitCompactError(
  emit: Emit,
  request: LlmCompactRequest,
  failure: LlmAttemptFailure,
  completedAt: number,
  extra: { retryAttempt?: number; retryMaxAttempts?: number } = {}
): void {
  logCompressionDebug('provider.compact.emitError', {
    ...compactRequestDebugInfo(request),
    message: failure.message,
    rawError: failure.rawError,
    completedAt,
    retryAttempt: extra.retryAttempt,
    retryMaxAttempts: extra.retryMaxAttempts
  });
  emit({
    type: LlmEventType.CompactError,
    payload: {
      requestId: request.id,
      blockId: request.blockId,
      conversationId: request.conversationId,
      message: failure.message,
      ...(failure.rawError ? { rawError: failure.rawError } : {}),
      ...(extra.retryAttempt !== undefined ? { retryAttempt: extra.retryAttempt } : {}),
      ...(extra.retryMaxAttempts !== undefined ? { retryMaxAttempts: extra.retryMaxAttempts } : {}),
      completedAt
    }
  });
}

async function resolveCompactRetrySettings(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions
): Promise<LlmProviderConfigRecord | undefined> {
  if (!isRetryCapableCompressionMethod(methodConfig.kind)) return undefined;
  const model = compressionMethodModelOverride(methodConfig);
  return resolveRuntimeSettings({
    id: request.id,
    contents: request.contents,
    tools: [],
    conversationId: request.conversationId,
    ...(model ? { model } : {})
  }, options);
}

function isRetryCapableCompressionMethod(kind: LlmCompressionConfigRecord['kind']): boolean {
  return kind === 'openai_responses_compact' || kind === 'llm_summary' || kind === 'segmented_summary';
}

function compressionMethodModelOverride(methodConfig: LlmCompressionConfigRecord): LlmModelSettings | undefined {
  if (methodConfig.kind === 'openai_responses_compact') {
    const providerConfigId = methodConfig.openaiResponsesCompact?.providerConfigId?.trim();
    const model = methodConfig.openaiResponsesCompact?.model?.trim();
    return providerConfigId || model ? { ...(providerConfigId ? { providerConfigId } : {}), model: model || '' } : undefined;
  }
  if (methodConfig.kind === 'llm_summary' || methodConfig.kind === 'segmented_summary') {
    const providerConfigId = methodConfig.llmSummary?.providerConfigId?.trim();
    const model = methodConfig.llmSummary?.model?.trim();
    return providerConfigId || model ? { ...(providerConfigId ? { providerConfigId } : {}), model: model || '' } : undefined;
  }
  return undefined;
}

function isRetryableCompactFailure(error: unknown, failure: LlmAttemptFailure): boolean {
  const text = `${failure.message}\n${errorSearchText(error)}`.toLowerCase();
  return !(
    text.includes('当前压缩方法已关闭')
    || text.includes('未注册的压缩方法')
    || text.includes('缺少 llm api key')
    || text.includes('openai 原生压缩仅支持')
    || text.includes('media_size_unknown')
    || text.includes('media_semantics_unavailable')
    || text.includes('compression_request_too_large')
    || text.includes('compression_source_too_large')
  );
}



async function resolveCompactProviderSettings(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  contents: MessageContent[],
  options: LlmProviderOptions
): Promise<LlmProviderConfigRecord> {
  const modelOverride = methodConfig.openaiResponsesCompact?.model?.trim();
  const providerConfigId = methodConfig.openaiResponsesCompact?.providerConfigId?.trim();
  return resolveRuntimeSettings({
    id: request.id,
    contents,
    tools: [],
    conversationId: request.conversationId,
    ...(request.settingsSnapshot ? { settingsSnapshot: request.settingsSnapshot } : {}),
    model: {
      ...(providerConfigId ? { providerConfigId } : {}),
      model: modelOverride || ''
    }
  }, options);
}

async function compactWithOpenAIResponses(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal
): Promise<LlmCompactResult> {
  const preparedContext = await prepareNativeCompactContentsMultimodal(request.contents, options);
  const normalizedContext = assertCanonicalProviderToolContext(preparedContext);
  const settings = await resolveCompactProviderSettings(request, methodConfig, normalizedContext, options);

  if (settings.provider !== 'openai-responses') {
    throw new Error('OpenAI 原生压缩仅支持 openai-responses 渠道格式。');
  }
  if (!settings.apiKey) {
    throw new Error('缺少 LLM API Key。请在全局设置的“渠道”页签里填写并保存。');
  }

  const unified = await importUnifiedLlmProvider();
  const registry = unified.createBootstrapExtensionRegistry();
  const proxy = normalizeOptionalString(await resolveMaybe(options.proxy));
  const proxyFetch = proxy ? createProxyFetch(proxy) : undefined;
  const providerFetch = createTerminalValidatedFetch(proxyFetch ?? fetch, settings.provider);
  const headers = mergeHeaders(await resolveMaybe(options.headers), settings.headers);
  const requestBody = requestBodyWithOpenAIPromptCacheKey(settings, request.conversationId);
  logCompressionDebug('provider.compact.openaiResponses.settings', {
    ...compactRequestDebugInfo(request),
    providerConfigId: settings.id,
    providerConfigName: settings.name,
    provider: settings.provider,
    baseUrl: settings.baseUrl,
    model: settings.model,
    methodConfigId: methodConfig.id,
    methodConfigKind: methodConfig.kind,
    hasProxy: !!proxy,
    headerKeys: headers ? Object.keys(headers) : [],
    hasRequestBody: !!requestBody
  });
  const provider = unified.createLLMFromConfig({
    provider: settings.provider,
    model: settings.model,
    apiKey: settings.apiKey,
    baseUrl: settings.baseUrl,
    ...(settings.contextWindowTokens ? { contextWindow: settings.contextWindowTokens } : {}),
    ...(headers ? { headers } : {}),
    ...(requestBody ? { requestBody } : {}),
    ...unifiedPromptCacheConfigEntry(settings, requestBody),
    ...(proxy ? { proxy } : {}),
    fetch: providerFetch
  }, registry.llmProviders) as unknown as { compact?: (request: unknown, options?: unknown) => Promise<UnifiedLLMCompactResponse> };

  if (typeof provider.compact !== 'function') {
    throw new Error('当前 unified-llm-provider 不支持 provider.compact。');
  }

  let compacted: UnifiedLLMCompactResponse;
  try {
    logCompressionDebug('provider.compact.openaiResponses.request', {
      ...compactRequestDebugInfo(request),
      normalizedContentCount: normalizedContext.length,
      signalAborted: signal?.aborted === true
    });
    compacted = await provider.compact(
      { contents: normalizedContext.flatMap((content) => toUnifiedContents(content, 'openai-responses')) },
      {
        inputFormat: 'unified',
        outputFormat: 'unified',
        signal,
        ...(requestBody ? { requestBody } : {})
      }
    );
    if (hasUnifiedError(compacted)) {
      throw new LlmAttemptFailureError(failureFromProviderError(compacted.error, { rawResponse: compacted.rawResponse ?? compacted }));
    }
    logCompressionDebug('provider.compact.openaiResponses.response', {
      ...compactRequestDebugInfo(request),
      responseId: compacted.id,
      object: compacted.object,
      contentCount: compacted.contents?.length ?? 0,
      hasUsage: compacted.usageMetadata !== undefined
    });
  } catch (error) {
    logCompressionDebug('provider.compact.openaiResponses.throw', {
      ...compactRequestDebugInfo(request),
      error: errorDebugInfo(error),
      signalAborted: signal?.aborted === true
    });
    // 压缩方法是用户明确选择的策略。OpenAI 原生压缩失败时必须保持该策略失败，
    // 交给外层按同一方法重试，不能在单次尝试内偷偷切换为分段总结。
    throw error;
  }

  return {
    id: compacted.id,
    object: compacted.object,
    createdAt: compacted.createdAt,
    // The compact endpoint returns the canonical next context window. Keep every unified item and
    // its top-level providerContext metadata intact; rebuilding known part variants here used to
    // discard raw retained-message ids/status and made the supposedly opaque result lossy.
    contents: (compacted.contents ?? []) as unknown as MessageContent[],
    usageMetadata: usageMetadataFromCompact(compacted.usageMetadata),
    settingsSnapshot: snapshotFromSettings(settings, methodConfig),
    rawResponse: compacted.rawResponse,
    methodConfig
  };
}

function isContextLengthExceededError(error: unknown): boolean {
  const text = errorSearchText(error).toLowerCase();
  return text.includes('context_length_exceeded')
    || text.includes('context window')
    || text.includes('exceeds the context')
    || text.includes('maximum context length')
    || text.includes('too many tokens');
}

function errorSearchText(error: unknown): string {
  const parts: string[] = [];
  if (typeof error === 'string') parts.push(error);
  if (error instanceof Error) {
    parts.push(error.name, error.message);
    const cause = (error as { cause?: unknown }).cause;
    if (cause !== undefined) parts.push(stringifyJson(toPlainJsonLike(cause)));
  }
  if (isRecord(error)) {
    parts.push(stringifyJson(toPlainJsonLike(error)));
    for (const key of ['message', 'bodyText', 'data', 'rawBody', 'rawResponse', 'response', 'error']) {
      const value = error[key];
      if (value !== undefined) parts.push(typeof value === 'string' ? value : stringifyJson(toPlainJsonLike(value)));
    }
  } else {
    parts.push(stringifyJson(toPlainJsonLike(error)));
  }
  return parts.filter(Boolean).join('\n');
}

interface PreparedCompressionMediaSemantics {
  request: LlmCompactRequest;
  profileSha256?: string;
  requirements: LlmAttachmentObservationRequirement[];
  observations: LlmAttachmentObservation[];
  provider?: ResolvedSummaryProvider;
}

interface CompressionAttachmentObservationContract {
  profileSha256?: string;
  requirements: LlmAttachmentObservationRequirement[];
}

interface SemanticContentsProjection {
  contents: MessageContent[];
  representedRefs: Set<string>;
}

const ATTACHMENT_OBSERVATION_TARGET_TOKENS = 1_024;
// Attachment observation runs one Provider call per media body and must survive
// reasoning-heavy models that spend part of the output budget on thoughts.
// Kept local on purpose: SUMMARY_PROVIDER_MIN_OUTPUT_TOKENS is shared with the
// generic summary floor and the truncation retry step, so widening it there
// would change unrelated compression paths.
const ATTACHMENT_OBSERVATION_MAX_OUTPUT_TOKENS = 8_192;
const ATTACHMENT_OBSERVATION_MAX_SUMMARY_CHARS = 8_000;
const ATTACHMENT_OBSERVATION_MAX_ITEM_CHARS = 2_000;
const ATTACHMENT_OBSERVATION_MAX_SALIENT_FACTS = 32;
const ATTACHMENT_OBSERVATION_MAX_UNCERTAINTIES = 16;
// Structured-output wobble is random, so one reformulated attempt recovers most
// failures that previously aborted the entire compression turn.
const ATTACHMENT_OBSERVATION_MAX_ATTEMPTS = 2;

export class LlmMediaSemanticsUnavailableError extends Error {
  public readonly code = 'media_semantics_unavailable';
  public readonly attachmentRef?: string;
  public readonly cause?: unknown;

  public constructor(reason: string, attachmentRef?: string, cause?: unknown) {
    super(`media_semantics_unavailable: ${attachmentRef ? `${attachmentRef}: ` : ''}${reason}`);
    this.name = 'LlmMediaSemanticsUnavailableError';
    this.attachmentRef = attachmentRef;
    this.cause = cause;
  }
}

const attachmentMediaSemanticsCache = new WeakMap<
  LlmCompactRequest,
  { cacheKey: string; promise: Promise<PreparedCompressionMediaSemantics> }
>();

async function prepareCompressionMediaSemantics(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  initialProvider?: ResolvedSummaryProvider
): Promise<PreparedCompressionMediaSemantics> {
  const cacheKey = [
    methodConfig.id,
    methodConfig.kind,
    request.attachmentObservationProfileSha256 ?? 'no-observation-profile'
  ].join('\0');
  const cached = attachmentMediaSemanticsCache.get(request);
  if (cached?.cacheKey === cacheKey) return cached.promise;
  const promise = prepareCompressionMediaSemanticsUncached(
    request,
    methodConfig,
    options,
    signal,
    initialProvider
  );
  const entry = { cacheKey, promise };
  attachmentMediaSemanticsCache.set(request, entry);
  try {
    return await promise;
  } catch (error) {
    if (attachmentMediaSemanticsCache.get(request) === entry) {
      attachmentMediaSemanticsCache.delete(request);
    }
    throw error;
  }
}

async function prepareCompressionMediaSemanticsUncached(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  initialProvider?: ResolvedSummaryProvider
): Promise<PreparedCompressionMediaSemantics> {
  const contract = normalizeCompressionAttachmentObservationContract(request);
  if (contract.requirements.length === 0) {
    if (compressionRequestContainsInlineMedia(request)) {
      throw new LlmMediaSemanticsUnavailableError(
        'text compression contains media without a frozen F-reference observation contract.'
      );
    }
    return {
      request,
      requirements: [],
      observations: [],
      ...(initialProvider ? { provider: initialProvider } : {})
    };
  }

  const requirementsById = new Map(contract.requirements.map((requirement) => [
    requirement.attachmentId,
    requirement
  ]));
  const bodies = collectCompressionMediaBodies(request, requirementsById);
  const observationsByRef = new Map<string, LlmAttachmentObservation>();
  const missing = contract.requirements.filter((requirement) => {
    if (!requirement.cachedObservation) return true;
    observationsByRef.set(requirement.attachmentRef, cloneAttachmentObservation(requirement.cachedObservation));
    return false;
  });

  let provider = initialProvider;
  if (missing.length > 0) {
    if (methodConfig.kind === 'deterministic_summary' || methodConfig.kind === 'manual_summary') {
      throw new LlmMediaSemanticsUnavailableError(
        `${methodConfig.kind} cannot inspect uncached media; choose a Provider-backed summary method.`
      );
    }
    provider ??= await resolveSummaryProvider(request, methodConfig, options);
    if (!provider.provider) {
      throw new LlmMediaSemanticsUnavailableError(
        'the frozen summary Provider is unavailable, so uncached media cannot be inspected.'
      );
    }
    const preparation = createMultimodalPreparationContext();
    const analyzed = await mapWithBoundedConcurrency(
      missing,
      isOpenAIResponsesWebSocketMode(provider.settings) ? 1 : SEGMENTED_SUMMARY_CONCURRENCY,
      async (requirement, _index, siblingSignal) => {
        const body = bodies.get(requirement.attachmentId);
        if (!body) {
          throw new LlmMediaSemanticsUnavailableError(
            'the frozen compression source does not contain a resolvable media body.',
            requirement.attachmentRef
          );
        }
        return analyzeCompressionAttachment(
          requirement,
          body,
          provider!,
          options,
          preparation,
          siblingSignal
        );
      },
      signal
    );
    analyzed.forEach((observation) => observationsByRef.set(observation.attachmentRef, observation));
  }

  const observations = contract.requirements.map((requirement) => {
    const observation = observationsByRef.get(requirement.attachmentRef);
    if (!observation) {
      throw new LlmMediaSemanticsUnavailableError(
        'no complete structured observation was produced.',
        requirement.attachmentRef
      );
    }
    return cloneAttachmentObservation(observation);
  });
  return {
    request: projectCompressionRequestWithObservations(request, contract.requirements, observations),
    profileSha256: contract.profileSha256,
    requirements: contract.requirements,
    observations,
    ...(provider ? { provider } : {})
  };
}

interface PreparedCompressionMediaSemanticsDryRun extends PreparedCompressionMediaSemantics {
  observationCalls: SummaryProviderCall[];
}

async function prepareCompressionMediaSemanticsDryRun(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  provider: ResolvedSummaryProvider
): Promise<PreparedCompressionMediaSemanticsDryRun> {
  const contract = normalizeCompressionAttachmentObservationContract(request);
  if (contract.requirements.length === 0) {
    if (compressionRequestContainsInlineMedia(request)) {
      throw new LlmMediaSemanticsUnavailableError(
        'text compression contains media without a frozen F-reference observation contract.'
      );
    }
    return {
      request,
      requirements: [],
      observations: [],
      provider,
      observationCalls: []
    };
  }
  const requirementsById = new Map(contract.requirements.map((requirement) => [
    requirement.attachmentId,
    requirement
  ]));
  const bodies = collectCompressionMediaBodies(request, requirementsById);
  const preparation = createMultimodalPreparationContext();
  const observations: LlmAttachmentObservation[] = [];
  const observationCalls: SummaryProviderCall[] = [];
  for (const requirement of contract.requirements) {
    if (requirement.cachedObservation) {
      observations.push(cloneAttachmentObservation(requirement.cachedObservation));
      continue;
    }
    if (!provider.provider) {
      throw new LlmMediaSemanticsUnavailableError(
        'the dry-run summary Provider is unavailable, so uncached media cannot be inspected.',
        requirement.attachmentRef
      );
    }
    const body = bodies.get(requirement.attachmentId);
    if (!body) {
      throw new LlmMediaSemanticsUnavailableError(
        'the frozen compression source does not contain a resolvable media body.',
        requirement.attachmentRef
      );
    }
    const media = await prepareAttachmentObservationMedia(
      requirement,
      body,
      options,
      preparation
    );
    observationCalls.push(buildAttachmentObservationProviderCall(
      requirement,
      media
    ));
    observations.push({
      attachmentRef: requirement.attachmentRef,
      summary: `[dry-run placeholder: runtime observation output for ${requirement.attachmentRef}]`,
      salientFacts: [],
      uncertainties: ['Dry-run cannot know the Provider observation response.']
    });
  }
  return {
    request: projectCompressionRequestWithObservations(request, contract.requirements, observations),
    profileSha256: contract.profileSha256,
    requirements: contract.requirements,
    observations,
    provider,
    observationCalls
  };
}

function normalizeCompressionAttachmentObservationContract(
  request: LlmCompactRequest
): CompressionAttachmentObservationContract {
  const profile = request.attachmentObservationProfileSha256;
  const rawRequirements = request.attachmentObservationRequirements;
  if (profile === undefined && rawRequirements === undefined) return { requirements: [] };
  if (typeof profile !== 'string' || !/^[0-9a-f]{64}$/i.test(profile)) {
    throw new TypeError('Compression Attachment observation profile must be a SHA-256 hex digest.');
  }
  if (!Array.isArray(rawRequirements) || rawRequirements.length === 0) {
    throw new TypeError('Compression Attachment observation requirements must be a non-empty array.');
  }
  const refs = new Set<string>();
  const attachmentIds = new Set<string>();
  const requirements = rawRequirements.map((value, index) => {
    const requirement = normalizeAttachmentObservationRequirement(
      value,
      `attachmentObservationRequirements[${index}]`
    );
    if (refs.has(requirement.attachmentRef) || attachmentIds.has(requirement.attachmentId)) {
      throw new Error('Compression Attachment observation requirements contain duplicate identities.');
    }
    refs.add(requirement.attachmentRef);
    attachmentIds.add(requirement.attachmentId);
    return requirement;
  });
  return { profileSha256: profile.toLowerCase(), requirements };
}

function compressionRequestContainsInlineMedia(request: LlmCompactRequest): boolean {
  const collections = [
    ...(request.priorSummaryContents ? [request.priorSummaryContents] : []),
    request.contents,
    ...(request.segments ?? [])
  ];
  return collections.some((contents) => contents.some((content) => content.parts.some((part) =>
    isInlineDataPart(part)
      || (isFunctionResponsePart(part) && (part.functionResponse.parts?.length ?? 0) > 0)
  )));
}

function collectCompressionMediaBodies(
  request: LlmCompactRequest,
  requirementsById: ReadonlyMap<string, LlmAttachmentObservationRequirement>
): Map<string, InlineDataPart> {
  const bodies = new Map<string, InlineDataPart>();
  const collections = [
    ...(request.priorSummaryContents ? [request.priorSummaryContents] : []),
    request.contents,
    ...(request.segments ?? [])
  ];
  for (const contents of collections) {
    for (const content of contents) {
      for (const part of content.parts) {
        if (isInlineDataPart(part)) collectCompressionMediaBody(part, requirementsById, bodies);
        if (isFunctionResponsePart(part)) {
          for (const media of part.functionResponse.parts ?? []) {
            collectCompressionMediaBody(media, requirementsById, bodies);
          }
        }
      }
    }
  }
  return bodies;
}

function collectCompressionMediaBody(
  part: InlineDataPart,
  requirementsById: ReadonlyMap<string, LlmAttachmentObservationRequirement>,
  bodies: Map<string, InlineDataPart>
): void {
  const attachmentId = part.inlineData.attachmentId?.trim();
  const requirement = attachmentId ? requirementsById.get(attachmentId) : undefined;
  if (!attachmentId || !requirement) {
    throw new LlmMediaSemanticsUnavailableError(
      'a summary media body has no matching frozen F-reference observation requirement.'
    );
  }
  if (part.inlineData.mimeType !== requirement.mimeType
    || (part.inlineData.name !== undefined && part.inlineData.name !== requirement.name)
    || (part.inlineData.sizeBytes !== undefined && part.inlineData.sizeBytes !== requirement.sizeBytes)) {
    throw new LlmMediaSemanticsUnavailableError(
      'media metadata conflicts with the frozen Attachment catalog.',
      requirement.attachmentRef
    );
  }
  if (!bodies.has(attachmentId)) bodies.set(attachmentId, cloneInlineDataPart(part));
}

async function analyzeCompressionAttachment(
  requirement: LlmAttachmentObservationRequirement,
  body: InlineDataPart,
  provider: ResolvedSummaryProvider,
  options: LlmProviderOptions,
  preparation: MultimodalPreparationContext,
  signal?: AbortSignal
): Promise<LlmAttachmentObservation> {
  const providerMedia = await prepareAttachmentObservationMedia(
    requirement,
    body,
    options,
    preparation,
    signal
  );
  let lastError: unknown;
  for (let attempt = 0; attempt < ATTACHMENT_OBSERVATION_MAX_ATTEMPTS; attempt += 1) {
    const correction = lastError === undefined
      ? undefined
      : (lastError instanceof Error ? lastError.message : String(lastError)).slice(0, 300);
    const call = buildAttachmentObservationProviderCall(
      requirement,
      providerMedia,
      correction
    );
    // Compatibility retry stays enabled so a truncated reply can grow its output
    // budget instead of failing the turn outright.
    const response = await executeSummaryProviderCall(
      provider,
      call.request,
      signal
    );
    try {
      return parseAttachmentObservationResponse(response, requirement.attachmentRef);
    } catch (error) {
      lastError = error;
      logCompressionDebug('provider.compact.observation.parseRetry', {
        attachmentRef: requirement.attachmentRef,
        attempt: attempt + 1,
        maxAttempts: ATTACHMENT_OBSERVATION_MAX_ATTEMPTS,
        reason: error instanceof Error ? error.message : String(error)
      });
    }
  }
  throw new LlmMediaSemanticsUnavailableError(
    'the analysis Provider did not return the required structured observation.',
    requirement.attachmentRef,
    lastError
  );
}

async function prepareAttachmentObservationMedia(
  requirement: LlmAttachmentObservationRequirement,
  body: InlineDataPart,
  options: LlmProviderOptions,
  preparation: MultimodalPreparationContext,
  signal?: AbortSignal
): Promise<InlineDataPart> {
  if (!isModelToolResponseMultimodalMimeType(requirement.mimeType)) {
    throw new LlmMediaSemanticsUnavailableError(
      `MIME type ${requirement.mimeType} is outside the stable multimodal analysis policy.`,
      requirement.attachmentRef
    );
  }
  let prepared: ContentPart;
  try {
    prepared = await prepareInlineDataForLlm(body, options, false, 'native_compact', preparation);
  } catch (error) {
    if (isRequestAbort(signal)) throw error;
    throw new LlmMediaSemanticsUnavailableError(
      'exact media bytes could not be resolved.',
      requirement.attachmentRef,
      error
    );
  }
  if (!isInlineDataPart(prepared) || !prepared.inlineData.data) {
    throw new LlmMediaSemanticsUnavailableError(
      'the Attachment resolver did not return an inline media body.',
      requirement.attachmentRef
    );
  }
  const resolvedBytes = requireCanonicalInlineDataSize(prepared, 'Attachment observation media');
  if (resolvedBytes !== requirement.sizeBytes || prepared.inlineData.mimeType !== requirement.mimeType) {
    throw new LlmMediaSemanticsUnavailableError(
      'resolved media bytes conflict with frozen Attachment metadata.',
      requirement.attachmentRef
    );
  }
  return {
    inlineData: {
      mimeType: requirement.mimeType,
      data: prepared.inlineData.data,
      name: requirement.name
    }
  };
}

function buildAttachmentObservationProviderCall(
  requirement: LlmAttachmentObservationRequirement,
  media: InlineDataPart,
  correction?: string
): SummaryProviderCall {
  const systemPrompt = [
    `Attachment observation contract revision: ${ATTACHMENT_OBSERVATION_PROMPT_REVISION}.`,
    'Inspect exactly the attached media body. Return only one JSON object, without Markdown fences or prose.',
    'Use exactly these keys: attachmentRef, summary, salientFacts, uncertainties.',
    'attachmentRef must equal the supplied F reference. summary must be concise but semantically complete.',
    'salientFacts and uncertainties must be JSON string arrays. Do not infer facts that are not visible.',
    `Hard limits: summary at most ${ATTACHMENT_OBSERVATION_MAX_SUMMARY_CHARS} characters;`
      + ` salientFacts at most ${ATTACHMENT_OBSERVATION_MAX_SALIENT_FACTS} items;`
      + ` uncertainties at most ${ATTACHMENT_OBSERVATION_MAX_UNCERTAINTIES} items;`
      + ` every array item at most ${ATTACHMENT_OBSERVATION_MAX_ITEM_CHARS} characters.`,
    'Emit the JSON object as the very first and only visible output. Never place it inside reasoning.',
    ...(correction ? [`Your previous reply was rejected: ${correction} Return only the corrected JSON object.`] : [])
  ].join('\n');
  const sourceContent: MessageContent = {
    role: 'user',
    parts: [{
      text: [
        `attachmentRef: ${requirement.attachmentRef}`,
        `name: ${requirement.name}`,
        `mimeType: ${requirement.mimeType}`,
        `sizeBytes: ${requirement.sizeBytes}`,
        'Analyze this body now and return the strict JSON observation.'
      ].join('\n')
    }, media]
  };
  return {
    label: `Attachment ${requirement.attachmentRef} observation`,
    sourceContents: [sourceContent],
    targetTokens: ATTACHMENT_OBSERVATION_TARGET_TOKENS,
    request: {
      contents: [sourceContent],
      systemInstruction: { parts: [{ text: systemPrompt }] },
      generationConfig: {
        temperature: 0,
        maxOutputTokens: ATTACHMENT_OBSERVATION_MAX_OUTPUT_TOKENS,
        thinkingConfig: { thinkingLevel: 'low' }
      }
    }
  };
}

// Scans for the first balanced top-level JSON object, ignoring braces inside
// strings. Lets a Provider that wraps its JSON in prose still succeed.
function extractFirstJsonObject(text: string): string | undefined {
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') { inString = true; continue; }
    if (char === '{') {
      if (depth === 0) start = index;
      depth += 1;
      continue;
    }
    if (char === '}' && depth > 0) {
      depth -= 1;
      if (depth === 0 && start >= 0) return text.slice(start, index + 1);
    }
  }
  return undefined;
}

// Clamps instead of rejecting: an over-long list is a formatting wobble, not a
// reason to abort the whole compression turn.
function coerceObservationTextList(value: unknown, maxItems: number): string[] {
  if (!Array.isArray(value)) return [];
  const items: string[] = [];
  for (const entry of value) {
    if (items.length >= maxItems) break;
    const text = typeof entry === 'string'
      ? entry
      : typeof entry === 'number' || typeof entry === 'boolean'
        ? String(entry)
        : undefined;
    if (text === undefined) continue;
    const trimmed = text.trim();
    if (trimmed.length === 0) continue;
    items.push(trimmed.slice(0, ATTACHMENT_OBSERVATION_MAX_ITEM_CHARS));
  }
  return items;
}

function parseAttachmentObservationResponse(
  value: string,
  expectedRef: string
): LlmAttachmentObservation {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new Error('Attachment observation response was empty.');
  }
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed);
  const candidate = fenced ? fenced[1]!.trim() : trimmed;
  let parsed: unknown;
  try {
    parsed = JSON.parse(candidate) as unknown;
  } catch (error) {
    const extracted = extractFirstJsonObject(candidate) ?? extractFirstJsonObject(trimmed);
    if (!extracted) throw error;
    parsed = JSON.parse(extracted) as unknown;
  }
  if (!isRecord(parsed)) throw new TypeError('Attachment observation response must be an object.');
  const summaryText = typeof parsed.summary === 'string' ? parsed.summary.trim() : '';
  if (summaryText.length === 0) {
    throw new TypeError('Attachment observation response is missing a summary.');
  }
  if (parsed.attachmentRef !== undefined && parsed.attachmentRef !== expectedRef) {
    logCompressionDebug('provider.compact.observation.refMismatch', {
      expectedRef,
      reportedRef: parsed.attachmentRef
    });
  }
  // Unknown keys are dropped and the F reference is taken from the frozen
  // requirement: each call carries exactly one media body, so the caller is the
  // authoritative source for the reference.
  return normalizeLlmAttachmentObservation({
    attachmentRef: expectedRef,
    summary: summaryText.slice(0, ATTACHMENT_OBSERVATION_MAX_SUMMARY_CHARS),
    salientFacts: coerceObservationTextList(
      parsed.salientFacts,
      ATTACHMENT_OBSERVATION_MAX_SALIENT_FACTS
    ),
    uncertainties: coerceObservationTextList(
      parsed.uncertainties,
      ATTACHMENT_OBSERVATION_MAX_UNCERTAINTIES
    )
  }, 'Attachment observation response');
}

function projectCompressionRequestWithObservations(
  request: LlmCompactRequest,
  requirements: readonly LlmAttachmentObservationRequirement[],
  observations: readonly LlmAttachmentObservation[]
): LlmCompactRequest {
  const requirementById = new Map(requirements.map((requirement) => [requirement.attachmentId, requirement]));
  const observationByRef = new Map(observations.map((observation) => [observation.attachmentRef, observation]));
  const prior = projectSemanticContents(request.priorSummaryContents ?? [], requirementById, observationByRef);
  const current = projectSemanticContents(request.contents, requirementById, observationByRef);
  const currentMissing = missingRepresentedObservations(requirements, prior.representedRefs, current.representedRefs);
  const contents = currentMissing.length > 0
    ? [attachmentObservationStateContent(currentMissing, observationByRef), ...current.contents]
    : current.contents;

  let segments: MessageContent[][] | undefined;
  if (request.segments !== undefined) {
    const represented = new Set(prior.representedRefs);
    segments = request.segments.map((segment) => {
      const projected = projectSemanticContents(segment, requirementById, observationByRef);
      projected.representedRefs.forEach((ref) => represented.add(ref));
      return projected.contents;
    });
    const missing = requirements.filter((requirement) => !represented.has(requirement.attachmentRef));
    if (missing.length > 0) {
      const state = attachmentObservationStateContent(missing, observationByRef);
      if (segments.length === 0) segments.push([state]);
      else segments[0] = [state, ...segments[0]];
    }
  }
  return {
    ...request,
    contents,
    ...(request.priorSummaryContents !== undefined ? { priorSummaryContents: prior.contents } : {}),
    ...(segments !== undefined ? { segments } : {})
  };
}

function projectSemanticContents(
  contents: readonly MessageContent[],
  requirementById: ReadonlyMap<string, LlmAttachmentObservationRequirement>,
  observationByRef: ReadonlyMap<string, LlmAttachmentObservation>
): SemanticContentsProjection {
  const representedRefs = new Set<string>();
  const projected = contents.map((content): MessageContent => ({
    role: content.role,
    parts: content.parts.flatMap((part): ContentPart[] => {
      if (isInlineDataPart(part)) {
        const requirement = requireObservationForMedia(part, requirementById);
        const observation = requireObservationByRef(requirement.attachmentRef, observationByRef);
        representedRefs.add(requirement.attachmentRef);
        return [{ text: attachmentObservationDescriptor(requirement, observation) }];
      }
      if (isFunctionResponsePart(part) && part.functionResponse.parts?.length) {
        const retained: InlineDataPart[] = [];
        const descriptors: ContentPart[] = [];
        for (const media of part.functionResponse.parts) {
          const requirement = requireObservationForMedia(media, requirementById);
          const observation = requireObservationByRef(requirement.attachmentRef, observationByRef);
          representedRefs.add(requirement.attachmentRef);
          descriptors.push({ text: attachmentObservationDescriptor(requirement, observation) });
        }
        const cloned = cloneJsonValue(part);
        if (retained.length > 0) cloned.functionResponse.parts = retained;
        else delete cloned.functionResponse.parts;
        return [cloned, ...descriptors];
      }
      return [cloneJsonValue(part)];
    })
  }));
  return { contents: projected, representedRefs };
}

function requireObservationForMedia(
  media: InlineDataPart,
  requirementById: ReadonlyMap<string, LlmAttachmentObservationRequirement>
): LlmAttachmentObservationRequirement {
  const attachmentId = media.inlineData.attachmentId?.trim();
  const requirement = attachmentId ? requirementById.get(attachmentId) : undefined;
  if (!requirement) {
    throw new LlmMediaSemanticsUnavailableError(
      'a summary media body has no matching frozen F-reference observation requirement.'
    );
  }
  return requirement;
}

function requireObservationByRef(
  attachmentRef: string,
  observations: ReadonlyMap<string, LlmAttachmentObservation>
): LlmAttachmentObservation {
  const observation = observations.get(attachmentRef);
  if (!observation) {
    throw new LlmMediaSemanticsUnavailableError('a required observation is missing.', attachmentRef);
  }
  return observation;
}

function missingRepresentedObservations(
  requirements: readonly LlmAttachmentObservationRequirement[],
  ...representedSets: ReadonlySet<string>[]
): LlmAttachmentObservationRequirement[] {
  return requirements.filter((requirement) =>
    representedSets.every((represented) => !represented.has(requirement.attachmentRef))
  );
}

function attachmentObservationStateContent(
  requirements: readonly LlmAttachmentObservationRequirement[],
  observations: ReadonlyMap<string, LlmAttachmentObservation>
): MessageContent {
  return renderAttachmentObservationStateContent(
    requirements,
    requirements.map((requirement) =>
      requireObservationByRef(requirement.attachmentRef, observations))
  );
}

function attachmentObservationDescriptor(
  requirement: LlmAttachmentObservationRequirement,
  observation: LlmAttachmentObservation
): string {
  return JSON.stringify({
    kind: 'attachment_observation',
    promptRevision: ATTACHMENT_OBSERVATION_PROMPT_REVISION,
    ...attachmentObservationModelRecord(requirement, observation)
  });
}

function attachmentObservationModelRecord(
  requirement: LlmAttachmentObservationRequirement,
  observation: LlmAttachmentObservation
): Record<string, unknown> {
  return {
    attachmentRef: requirement.attachmentRef,
    name: requirement.name,
    mimeType: requirement.mimeType,
    sizeBytes: requirement.sizeBytes,
    summary: observation.summary,
    salientFacts: [...observation.salientFacts],
    uncertainties: [...observation.uncertainties]
  };
}

function compressionSummaryContents(
  summary: string,
  targetTokens: number,
  requirements: readonly LlmAttachmentObservationRequirement[],
  observations: readonly LlmAttachmentObservation[]
): MessageContent[] {
  const contents = summaryContents(summary, targetTokens);
  if (requirements.length === 0) return contents;
  const byRef = new Map(observations.map((observation) => [observation.attachmentRef, observation]));
  return [...contents, attachmentObservationStateContent(requirements, byRef)];
}

function attachmentObservationResultFields(
  prepared: PreparedCompressionMediaSemantics
): Pick<LlmCompactResult, 'attachmentObservationProfileSha256' | 'attachmentObservations'> {
  if (!prepared.profileSha256) return {};
  return {
    attachmentObservationProfileSha256: prepared.profileSha256,
    attachmentObservations: prepared.observations.map(cloneAttachmentObservation)
  };
}

function cloneAttachmentObservation(observation: LlmAttachmentObservation): LlmAttachmentObservation {
  return {
    attachmentRef: observation.attachmentRef,
    summary: observation.summary,
    salientFacts: [...observation.salientFacts],
    uncertainties: [...observation.uncertainties]
  };
}

async function compactWithSummary(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal
): Promise<LlmCompactResult> {
  const mediaSemantics = await prepareCompressionMediaSemantics(request, methodConfig, options, signal);
  const summary = await generateSummaryText(
    mediaSemantics.request,
    methodConfig,
    options,
    signal,
    mediaSemantics.provider
  );
  const contents = compressionSummaryContents(
    summary.text,
    effectiveSummaryTargetTokens(methodConfig),
    mediaSemantics.requirements,
    mediaSemantics.observations
  );
  return {
    id: `summary-${request.blockId}`,
    object: 'limcode.context_summary',
    createdAt: Date.now(),
    contents,
    ...(summary.settings ? { settingsSnapshot: snapshotFromSettings(summary.settings, methodConfig) } : {}),
    methodConfig,
    ...attachmentObservationResultFields(mediaSemantics)
  };
}

/** Generates bounded deltas, then replaces prior+delta state with one structured rolling summary. */
async function compactWithSegmentedSummary(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal
): Promise<LlmCompactResult> {
  const initialProvider = await resolveSummaryProvider(request, methodConfig, options);
  const mediaSemantics = await prepareCompressionMediaSemantics(
    request,
    methodConfig,
    options,
    signal,
    initialProvider
  );
  const provider = mediaSemantics.provider ?? initialProvider;
  const semanticRequest = mediaSemantics.request;
  const targetTokens = effectiveSummaryTargetTokens(methodConfig);
  const priorSummaryText = semanticRequest.priorSummaryContents?.length
    ? plainTextOfContents(semanticRequest.priorSummaryContents)
    : '';
  const sourceContents = summaryDeltaContents(semanticRequest);
  const deterministic = deterministicReplacementSummary(priorSummaryText, sourceContents, targetTokens);
  let finalSummary = deterministic;

  if (sourceContents.length > 0 && provider.provider) {
    const calls = buildSegmentedSummaryProviderCalls(semanticRequest, methodConfig, provider.settings);
    const deltaSummaries = await mapWithBoundedConcurrency(
      calls,
      isOpenAIResponsesWebSocketMode(provider.settings) ? 1 : SEGMENTED_SUMMARY_CONCURRENCY,
      (call, _index, siblingSignal) => summarizeSingleRound(provider, call, siblingSignal),
      signal
    );
    const merged = await mergeSegmentedSummaryHierarchy(
      provider,
      calls.map((call, index) => ({
        summary: deltaSummaries[index] ?? '',
        sourceContents: call.sourceContents
      })),
      priorSummaryText,
      methodConfig,
      targetTokens,
      signal
    );
    if (merged) finalSummary = finalizeStructuredSummary(merged, deterministic, targetTokens);
  }

  const contents = compressionSummaryContents(
    finalSummary,
    targetTokens,
    mediaSemantics.requirements,
    mediaSemantics.observations
  );
  return {
    id: `summary-${request.blockId}`,
    object: 'limcode.context_summary',
    createdAt: Date.now(),
    contents,
    ...(provider.settings ? { settingsSnapshot: snapshotFromSettings(provider.settings, methodConfig) } : {}),
    methodConfig,
    ...attachmentObservationResultFields(mediaSemantics)
  };
}

/** Extracts a prior summary as merge input; it is never mechanically prefixed to the replacement. */
function plainTextOfContents(contents: MessageContent[]): string {
  const text = contents
    .flatMap((content) => content.parts.filter(isVisibleTextPart).map((part) => part.text))
    .join('\n')
    .trim();
  return text.replace(/^\[Context Summary\]\s*/, '').trim();
}

function summaryDeltaContents(request: LlmCompactRequest): MessageContent[] {
  if (request.segments && request.segments.length > 0) return request.segments.flatMap((segment) => segment);
  return request.contents;
}

/** 取一个回合中最后一条“正式回答”(model + 可见文本) 的可见文本，用作下一回合前情。 */
function finalAnswerTextOf(segment: MessageContent[]): string {
  for (let index = segment.length - 1; index >= 0; index -= 1) {
    const content = segment[index];
    if (content.role !== 'model') continue;
    const text = content.parts.filter(isVisibleTextPart).map((part) => part.text).join('\n').trim();
    if (text) return text;
  }
  return '';
}

const SUMMARY_TAG_PATTERN = /<summary>([\s\S]*?)<\/summary>/i;
function extractSummaryTag(text: string): string {
  const match = SUMMARY_TAG_PATTERN.exec(text);
  return (match ? match[1] : text).trim();
}

interface ResolvedSummaryProvider {
  provider: ReturnType<UnifiedModule['createLLMFromConfig']> | undefined;
  settings: LlmProviderConfigRecord;
  stream: boolean;
  apiKeyAvailable: boolean;
  unified?: UnifiedModule;
  proxy?: string;
  webSocketSessionKey?: string;
  omitUnsupportedMaxOutputTokens: boolean;
}

/** 组装总结用 provider（复用运行时渠道解析 + 代理/头合并）；无 API Key 时 provider 为 undefined 表示回退确定性摘要。 */
async function resolveSummaryProvider(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  behavior: { allowPlaceholderApiKey?: boolean } = {}
): Promise<ResolvedSummaryProvider> {
  const summarySettings = methodConfig.llmSummary;
  const providerConfigId = summarySettings?.providerConfigId?.trim();
  const model = summarySettings?.model?.trim();
  const settings = await resolveRuntimeSettings({
    id: request.id,
    contents: request.contents,
    tools: [],
    conversationId: request.conversationId,
    ...(request.settingsSnapshot ? { settingsSnapshot: request.settingsSnapshot } : {}),
    ...(providerConfigId || model ? { model: { ...(providerConfigId ? { providerConfigId } : {}), model: model || '' } } : {})
  }, options);

  const apiKeyAvailable = !!settings.apiKey;
  if (!apiKeyAvailable && behavior.allowPlaceholderApiKey !== true) {
    return {
      provider: undefined,
      settings,
      stream: false,
      apiKeyAvailable: false,
      omitUnsupportedMaxOutputTokens: false
    };
  }
  const runtimeSettings = apiKeyAvailable ? settings : { ...settings, apiKey: 'limcode-dry-run-placeholder-key' };
  const unified = await importUnifiedLlmProvider();
  const registry = unified.createBootstrapExtensionRegistry();
  const proxy = normalizeOptionalString(await resolveMaybe(options.proxy));
  const proxyFetch = proxy ? createProxyFetch(proxy) : undefined;
  const providerFetch = createTerminalValidatedFetch(proxyFetch ?? fetch, runtimeSettings.provider);
  const headers = mergeHeaders(await resolveMaybe(options.headers), settings.headers);
  const requestBody = requestBodyWithOpenAIPromptCacheKey(runtimeSettings, request.conversationId);
  const provider = installGeminiProviderCompatibility(unified.createLLMFromConfig({
    provider: runtimeSettings.provider,
    model: runtimeSettings.model,
    apiKey: runtimeSettings.apiKey,
    baseUrl: runtimeSettings.baseUrl,
    ...(runtimeSettings.contextWindowTokens ? { contextWindow: runtimeSettings.contextWindowTokens } : {}),
    ...(headers ? { headers } : {}),
    ...(requestBody ? { requestBody } : {}),
    ...unifiedPromptCacheConfigEntry(runtimeSettings, requestBody),
    ...openAIResponsesWebSocketConfigEntry(runtimeSettings, request.conversationId),
    ...(proxy ? { proxy } : {}),
    fetch: providerFetch
  }, registry.llmProviders), runtimeSettings.provider, runtimeSettings.model);
  return {
    provider,
    settings,
    stream: settings.stream !== false,
    apiKeyAvailable,
    unified,
    ...(proxy ? { proxy } : {}),
    ...(isOpenAIResponsesWebSocketMode(settings)
      ? {
          webSocketSessionKey: createOpenAIResponsesWebSocketSessionKey(
            settings,
            `${requireOpenAIResponsesWebSocketConversationId(request.conversationId)}\ncompression-summary\n${request.id}`
          )
        }
      : {}),
    omitUnsupportedMaxOutputTokens: false
  };
}

interface SummaryProviderCall {
  label: string;
  sourceContents: MessageContent[];
  targetTokens: number;
  request: {
    contents: MessageContent[];
    systemInstruction: { parts: Array<{ text: string }> };
    generationConfig?: LlmGenerationConfigRecord;
  };
}

const ROLLING_SUMMARY_STRUCTURE_INSTRUCTION = [
  '输出必须是一份替代全部旧摘要与新增记录的最新摘要，不要逐字拼接旧摘要。',
  '删除已被新事实替代的旧决定；同一事实只保留一次。',
  '必须严格使用以下标题，缺少内容时写“无”：',
  '目标',
  '重要约束、决定和准确标识',
  '工作状态',
  '  - 已完成',
  '  - 正在做',
  '  - 受阻',
  '下一步',
  '相关文件',
  '必须保留准确的路径、符号名、命令、报错、URL、版本号和业务 ID。'
].join('\n');

function buildSummaryProviderCall(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord
): SummaryProviderCall {
  const summarySettings = methodConfig.llmSummary;
  const targetTokens = effectiveSummaryTargetTokens(methodConfig);
  const systemPrompt = withSummaryTargetInstruction(
    `${summarySettings?.systemPrompt?.trim() || DEFAULT_LLM_COMPRESSION_SUMMARY_SYSTEM_PROMPT}\n\n${ROLLING_SUMMARY_STRUCTURE_INSTRUCTION}`,
    targetTokens
  );
  const userPrompt = summarySettings?.userPrompt?.trim() || DEFAULT_LLM_COMPRESSION_SUMMARY_USER_PROMPT;
  const transcript = renderContentsForSummary(request.contents);
  const priorSummary = request.priorSummaryContents?.length
    ? plainTextOfContents(request.priorSummaryContents)
    : '';
  return {
    label: 'Context Summary',
    sourceContents: request.contents,
    targetTokens,
    request: {
      contents: [{ role: 'user', parts: [{ text: [
        userPrompt,
        '',
        '【旧摘要（只作为待更新的前情，不要原样附加）】',
        priorSummary || '无',
        '',
        '【新增历史】',
        transcript || '无'
      ].join('\n') }] }],
      systemInstruction: { parts: [{ text: systemPrompt }] },
      generationConfig: summaryGenerationConfig(methodConfig, settings, targetTokens)
    }
  };
}

interface SegmentedSummaryChunk {
  requestContents: MessageContent[];
  sourceContents: MessageContent[];
}

interface SegmentedSummaryUnit extends SegmentedSummaryChunk {
  kind: 'message' | 'tool_exchange' | 'tool_results';
  estimatedTokens: number;
  functionCallCount: number;
  functionResponseCount: number;
}

function buildSegmentedSummaryProviderCalls(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord
): SummaryProviderCall[] {
  const totalTargetTokens = effectiveSummaryTargetTokens(methodConfig);
  const sourceSegments = (request.segments && request.segments.length > 0
    ? request.segments
    : request.contents.length > 0 ? [request.contents] : [])
    .filter((segment) => segment.length > 0);
  if (sourceSegments.length === 0) return [];
  const priorSummaryText = request.priorSummaryContents?.length ? plainTextOfContents(request.priorSummaryContents) : '';
  const units: SegmentedSummaryUnit[] = sourceSegments.flatMap((segment) =>
    groupAtomicMessageContents(segment).map((group) => ({
      kind: group.kind,
      estimatedTokens: group.estimatedTokens,
      functionCallCount: group.functionCallCount,
      functionResponseCount: group.functionResponseCount,
      requestContents: group.items,
      sourceContents: group.items
    }))
  );
  const groups: SegmentedSummaryChunk[] = [];
  let current: SegmentedSummaryChunk = { requestContents: [], sourceContents: [] };

  const priorFor = (index: number): string => index === 0
    ? priorSummaryText
    : finalAnswerTextOf(groups[index - 1]?.requestContents ?? []);
  const fits = (contents: MessageContent[], index: number): boolean => isSummaryProviderCallWithinWindow(
    buildSegmentDeltaCall(contents, index, priorFor(index), methodConfig, settings, totalTargetTokens),
    settings
  );
  const pushCurrent = (): void => {
    if (current.requestContents.length === 0) return;
    if (groups.length >= MAX_SEGMENTED_SUMMARY_LEAF_CALLS) {
      throw new Error(
        `compression_source_too_large: segmented summary exceeds the ${MAX_SEGMENTED_SUMMARY_LEAF_CALLS}-leaf call budget.`
      );
    }
    groups.push(current);
    current = { requestContents: [], sourceContents: [] };
  };

  for (const unit of units) {
    const candidate = [...current.requestContents, ...unit.requestContents];
    if (fits(candidate, groups.length)) {
      current.requestContents = candidate;
      current.sourceContents.push(...unit.sourceContents);
      continue;
    }
    pushCurrent();
    const safeUnits = splitOversizedSummaryUnit(
      unit,
      groups.length,
      priorFor(groups.length),
      methodConfig,
      settings,
      totalTargetTokens,
      MAX_SEGMENTED_SUMMARY_LEAF_CALLS - groups.length
    );
    for (const safeUnit of safeUnits) {
      const next = [...current.requestContents, ...safeUnit.requestContents];
      if (!fits(next, groups.length)) pushCurrent();
      if (!fits(safeUnit.requestContents, groups.length)) {
        throw new Error(
          `compression_request_too_large: fixed summary prompt cannot fit chunk ${groups.length + 1} in the frozen Provider window.`
        );
      }
      current.requestContents.push(...safeUnit.requestContents);
      current.sourceContents.push(...safeUnit.sourceContents);
    }
  }
  pushCurrent();
  if (groups.length > MAX_SEGMENTED_SUMMARY_LEAF_CALLS) {
    throw new Error(
      `compression_source_too_large: segmented summary requires ${groups.length} leaf calls; limit is ${MAX_SEGMENTED_SUMMARY_LEAF_CALLS}.`
    );
  }

  const targetTokensPerCall = Math.max(128, Math.ceil(totalTargetTokens / groups.length));
  return groups.map((group, index) => {
    const priorContext = index === 0 ? priorSummaryText : finalAnswerTextOf(groups[index - 1]?.requestContents ?? []);
    const call = buildSegmentDeltaCall(
      group.requestContents,
      index,
      priorContext,
      methodConfig,
      settings,
      targetTokensPerCall,
      group.sourceContents
    );
    if (!isSummaryProviderCallWithinWindow(call, settings)) {
      throw new Error(`compression_request_too_large: summary chunk ${index + 1} exceeds the frozen Provider input limit.`);
    }
    return call;
  });
}

function splitOversizedSummaryUnit(
  unit: SegmentedSummaryUnit,
  index: number,
  priorContext: string,
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokens: number,
  maxChunks: number
): SegmentedSummaryChunk[] {
  if (maxChunks <= 0) {
    throw new Error('compression_source_too_large: segmented summary exhausted the leaf call budget.');
  }
  const direct = buildSegmentDeltaCall(
    unit.requestContents,
    index,
    priorContext,
    methodConfig,
    settings,
    targetTokens
  );
  if (isSummaryProviderCallWithinWindow(direct, settings)) return [unit];

  const message = unit.kind === 'message' && unit.requestContents.length === 1
    ? unit.requestContents[0]
    : undefined;
  const textPart = message?.parts.length === 1 && isVisibleTextPart(message.parts[0])
    ? message.parts[0]
    : undefined;
  if (message && textPart) {
    const chunks: SegmentedSummaryChunk[] = [];
    const conservativePrior = sliceByTokens(
      'previous-context '.repeat(SEGMENTED_PRIOR_CONTEXT_TOKENS * 2),
      0,
      SEGMENTED_PRIOR_CONTEXT_TOKENS
    );
    let remaining = textPart.text;
    while (remaining.length > 0) {
      if (chunks.length >= maxChunks) {
        throw new Error('compression_source_too_large: oversized message exceeds the leaf call budget.');
      }
      const fitting = largestFittingSummaryTextPrefix(
        remaining,
        message.role,
        index + chunks.length,
        conservativePrior,
        methodConfig,
        settings,
        targetTokens
      );
      if (!fitting) break;
      const chunk: MessageContent = { role: message.role, parts: [{ ...textPart, text: fitting }] };
      chunks.push({ requestContents: [chunk], sourceContents: [chunk] });
      remaining = remaining.slice(fitting.length);
    }
    if (remaining.length === 0 && chunks.length > 0) return chunks;
  }

  const transcript = renderContentsForSummary(unit.requestContents);
  const chunks: SegmentedSummaryChunk[] = [];
  const conservativePrior = sliceByTokens(
    'previous-context '.repeat(SEGMENTED_PRIOR_CONTEXT_TOKENS * 2),
    0,
    SEGMENTED_PRIOR_CONTEXT_TOKENS
  );
  let remaining = transcript;
  while (remaining.length > 0) {
    if (chunks.length >= maxChunks) {
      throw new Error(`compression_source_too_large: oversized ${unit.kind} exceeds the leaf call budget.`);
    }
    const fitting = largestFittingSummaryTextPrefix(
      remaining,
      'user',
      index + chunks.length,
      conservativePrior,
      methodConfig,
      settings,
      targetTokens
    );
    if (!fitting) break;
    const chunk: MessageContent = { role: 'user', parts: [{ text: fitting }] };
    chunks.push({ requestContents: [chunk], sourceContents: [chunk] });
    remaining = remaining.slice(fitting.length);
  }
  if (remaining.length === 0 && chunks.length > 0) return chunks;
  throw new Error(
    `compression_request_too_large: ${unit.kind} summary unit cannot be losslessly split for the frozen Provider window.`
  );
}

function largestFittingSummaryTextPrefix(
  text: string,
  role: MessageContent['role'],
  index: number,
  priorContext: string,
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokens: number
): string {
  let low = 1;
  let high = Math.max(1, estimateTokenCount(text));
  let best = '';
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidateText = sliceByTokens(text, 0, middle);
    if (!candidateText) {
      low = middle + 1;
      continue;
    }
    const call = buildSegmentDeltaCall(
      [{ role, parts: [{ text: candidateText }] }],
      index,
      priorContext,
      methodConfig,
      settings,
      targetTokens
    );
    if (isSummaryProviderCallWithinWindow(call, settings)) {
      best = candidateText;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return best;
}

function buildSegmentDeltaCall(
  segment: MessageContent[],
  index: number,
  priorContext: string,
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokens: number,
  sourceContents: MessageContent[] = segment
): SummaryProviderCall {
  const boundedPrior = headTailTextByTokens(
    priorContext,
    Math.min(SEGMENTED_PRIOR_CONTEXT_TOKENS, targetTokens)
  );
  const transcript = renderContentsForSummary(segment);
  const userText = `${DEFAULT_SEGMENTED_SUMMARY_USER_PROMPT}\n\n【前情(只读，不要重新总结)】\n${boundedPrior || '无'}\n\n【本回合记录】\n${transcript}`;
  return {
    label: `Segment ${index + 1}`,
    sourceContents,
    targetTokens,
    request: {
      contents: [{ role: 'user', parts: [{ text: userText }] }],
      systemInstruction: {
        parts: [{ text: withSummaryTargetInstruction(
          `${DEFAULT_SEGMENTED_SUMMARY_SYSTEM_PROMPT}\n\n${ROLLING_SUMMARY_STRUCTURE_INSTRUCTION}`,
          targetTokens
        ) }]
      },
      generationConfig: summaryGenerationConfig(methodConfig, settings, targetTokens)
    }
  };
}

function buildSummaryReplacementMergeCall(
  priorSummaryText: string,
  deltaSummaries: readonly string[],
  sourceContents: MessageContent[],
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokens: number
): SummaryProviderCall {
  return {
    label: 'Summary replacement merge',
    sourceContents,
    targetTokens,
    request: {
      contents: [{ role: 'user', parts: [{ text: [
        '【旧摘要】',
        priorSummaryText || '无',
        '',
        '【新增分段摘要】',
        ...deltaSummaries.map((summary, index) => `--- delta ${index + 1} ---\n${extractSummaryTag(summary) || '无'}`)
      ].join('\n') }] }],
      systemInstruction: { parts: [{ text: withSummaryTargetInstruction(
        `把旧摘要与全部 delta 合并为一份新的 replacement summary。\n\n${ROLLING_SUMMARY_STRUCTURE_INSTRUCTION}`,
        targetTokens
      ) }] },
      generationConfig: summaryGenerationConfig(methodConfig, settings, targetTokens)
    }
  };
}

interface SegmentedSummaryNode {
  summary: string;
  sourceContents: MessageContent[];
}

async function mergeSegmentedSummaryHierarchy(
  provider: ResolvedSummaryProvider,
  initialNodes: readonly SegmentedSummaryNode[],
  priorSummaryText: string,
  methodConfig: LlmCompressionConfigRecord,
  targetTokens: number,
  signal?: AbortSignal
): Promise<string> {
  let nodes = [...initialNodes];
  for (let level = 0; nodes.length > 1; level += 1) {
    if (level >= MAX_SEGMENTED_SUMMARY_HIERARCHY_LEVELS) {
      throw new Error('compression_source_too_large: segmented summary exceeded the hierarchy depth limit.');
    }
    const groups = packSegmentedSummaryNodes(nodes, methodConfig, provider.settings, targetTokens);
    if (groups.length >= nodes.length) {
      throw new Error('compression_request_too_large: summary deltas cannot be merged inside the frozen Provider window.');
    }
    nodes = await mapWithBoundedConcurrency(
      groups,
      isOpenAIResponsesWebSocketMode(provider.settings) ? 1 : SEGMENTED_SUMMARY_CONCURRENCY,
      async (group, _index, siblingSignal) => {
        if (group.length === 1) return group[0]!;
        const call = buildSummaryReplacementMergeCall(
          '',
          group.map((node) => node.summary),
          group.flatMap((node) => node.sourceContents),
          methodConfig,
          provider.settings,
          targetTokens
        );
        return {
          summary: await summarizeSingleRound(provider, call, siblingSignal),
          sourceContents: call.sourceContents
        };
      },
      signal
    );
  }
  if (nodes.length === 0) return '';
  if (!priorSummaryText) return nodes[0]!.summary;
  const call = buildSummaryReplacementMergeCall(
    priorSummaryText,
    [nodes[0]!.summary],
    nodes[0]!.sourceContents,
    methodConfig,
    provider.settings,
    targetTokens
  );
  if (!isSummaryProviderCallWithinWindow(call, provider.settings)) {
    throw new Error('compression_request_too_large: prior summary and segmented delta cannot fit a merge request.');
  }
  return summarizeSingleRound(provider, call, signal);
}

function packSegmentedSummaryNodes(
  nodes: readonly SegmentedSummaryNode[],
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokens: number
): SegmentedSummaryNode[][] {
  const groups: SegmentedSummaryNode[][] = [];
  let current: SegmentedSummaryNode[] = [];
  for (const node of nodes) {
    const candidate = [...current, node];
    const fits = current.length === 0 || isSummaryProviderCallWithinWindow(
      buildSummaryReplacementMergeCall(
        '',
        candidate.map((entry) => entry.summary),
        [],
        methodConfig,
        settings,
        targetTokens
      ),
      settings
    );
    if (fits) {
      current = candidate;
      continue;
    }
    groups.push(current);
    current = [node];
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

function withSummaryTargetInstruction(prompt: string, targetTokens: number | undefined): string {
  if (typeof targetTokens !== 'number' || !Number.isFinite(targetTokens) || targetTokens <= 0) return prompt;
  return `${prompt}\n\n将可见摘要正文控制在约 ${Math.floor(targetTokens)} tokens；优先保留标识符、数字、文件名、依赖关系、决定和未完成事项。`;
}

const MAX_SEGMENTED_SUMMARY_LEAF_CALLS = 32;
const MAX_SEGMENTED_SUMMARY_HIERARCHY_LEVELS = 6;
const SEGMENTED_SUMMARY_CONCURRENCY = 3;
const SEGMENTED_PRIOR_CONTEXT_TOKENS = 1_024;
const SUMMARY_PROVIDER_MIN_OUTPUT_TOKENS = 2_048;
const SUMMARY_PROVIDER_REASONING_HEADROOM_MULTIPLIER = 2;
const SUMMARY_PROVIDER_ESTIMATOR_SLACK_TOKENS = 8_000;

/**
 * `targetTokens` is the desired visible summary length, while Provider output accounting also
 * includes hidden reasoning tokens. Keep those two budgets separate: use the target in the prompt,
 * default summary reasoning to low, and reserve a bounded hard-output ceiling. An explicit method
 * `maxOutputTokens`/`thinkingConfig` remains authoritative.
 */
function summaryGenerationConfig(
  methodConfig: LlmCompressionConfigRecord,
  settings: LlmProviderConfigRecord,
  targetTokensOverride?: number
): LlmGenerationConfigRecord | undefined {
  const targetTokens = targetTokensOverride ?? methodConfig.llmSummary?.targetTokens;
  const inherited = settings.generationConfig ?? {};
  const method = methodConfig.llmSummary?.generationConfig ?? {};
  const {
    maxOutputTokens: inheritedMaxOutputTokens,
    thinkingConfig: inheritedThinkingConfig,
    ...inheritedRest
  } = inherited;
  const {
    maxOutputTokens: methodMaxOutputTokens,
    thinkingConfig: methodThinkingConfig,
    ...methodRest
  } = method;
  const derivedMaxOutputTokens = typeof targetTokens === 'number'
    && Number.isFinite(targetTokens)
    && targetTokens > 0
    ? Math.max(
        SUMMARY_PROVIDER_MIN_OUTPUT_TOKENS,
        Math.min(
          DEFAULT_LLM_COMPRESSION_OUTPUT_RESERVE_TOKENS,
          Math.ceil(targetTokens * SUMMARY_PROVIDER_REASONING_HEADROOM_MULTIPLIER)
        )
      )
    : inheritedMaxOutputTokens;
  const generationConfig = {
    ...inheritedRest,
    ...methodRest,
    ...((methodMaxOutputTokens ?? derivedMaxOutputTokens) !== undefined
      ? { maxOutputTokens: methodMaxOutputTokens ?? derivedMaxOutputTokens }
      : {}),
    thinkingConfig: methodThinkingConfig ?? {
      ...(inheritedThinkingConfig ?? {}),
      thinkingLevel: 'low' as const
    }
  };
  return Object.keys(generationConfig).length > 0 ? generationConfig : undefined;
}

async function summarizeSingleRound(
  resolved: ResolvedSummaryProvider,
  call: SummaryProviderCall,
  signal?: AbortSignal
): Promise<string> {
  const fallback = deterministicReplacementSummary('', call.sourceContents, call.targetTokens);
  if (!resolved.provider) return fallback;

  try {
    const trimmed = (await executeSummaryProviderCall(
      resolved,
      call.request,
      signal,
      { allowCompatibilityRetry: false }
    )).trim();
    return finalizeStructuredSummary(extractSummaryTag(trimmed), fallback, call.targetTokens);
  } catch (error) {
    if (isRequestAbort(signal)) throw error;
    const contextLength = isContextLengthExceededError(error);
    logCompressionDebug('provider.compact.segmentedSummary.segmentFallback', {
      error: errorDebugInfo(error),
      segmentContents: call.sourceContents.length,
      contextLength
    });
    if (!contextLength) throw error;
    return fallback;
  }
}

async function executeSummaryProviderCall(
  resolved: ResolvedSummaryProvider,
  request: SummaryProviderCall['request'],
  signal?: AbortSignal,
  options: { allowCompatibilityRetry?: boolean } = {}
): Promise<string> {
  if (!resolved.provider) return '';
  const execute = async (activeRequest: SummaryProviderCall['request']): Promise<string> => {
    if (resolved.stream || isOpenAIResponsesWebSocketMode(resolved.settings)) {
      let text = '';
      const stream = isOpenAIResponsesWebSocketMode(resolved.settings)
        ? createSummaryWebSocketStream(resolved, activeRequest, signal)
        : resolved.provider!.chatStream<UnifiedLLMStreamChunk>(activeRequest, {
            inputFormat: 'unified',
            outputFormat: 'unified',
            signal
          });
      for await (const chunk of stream) {
        if (hasUnifiedError(chunk)) {
          throw new LlmAttemptFailureError(failureFromProviderError(chunk.error, {
            rawChunk: (chunk as { rawChunk?: unknown }).rawChunk ?? chunk
          }));
        }
        text += chunk.textDelta ?? visibleTextFromParts(chunk.partsDelta ?? []);
      }
      return text;
    }

    const response = await resolved.provider!.chat<UnifiedLLMResponse>(activeRequest, {
      inputFormat: 'unified',
      outputFormat: 'unified',
      signal
    });
    if (hasUnifiedError(response)) {
      throw new LlmAttemptFailureError(failureFromProviderError(response.error, {
        rawResponse: response.rawResponse ?? response
      }));
    }
    return visibleTextFromParts(response.content?.parts ?? []);
  };

  const initialRequest = resolved.omitUnsupportedMaxOutputTokens
    ? withoutMaxOutputTokens(request)
    : request;
  try {
    return await execute(initialRequest);
  } catch (error) {
    if (options.allowCompatibilityRetry === false) throw error;
    if (hasMaxOutputTokens(initialRequest) && isUnsupportedMaxOutputTokensError(error)) {
      resolved.omitUnsupportedMaxOutputTokens = true;
      logCompressionDebug('provider.compact.summary.compatibilityRetry', {
        providerConfigId: resolved.settings.id,
        provider: resolved.settings.provider,
        transport: resolved.settings.openaiResponsesTransport,
        removedParameter: 'max_output_tokens'
      });
      return execute(withoutMaxOutputTokens(initialRequest));
    }
    if (hasMaxOutputTokens(initialRequest) && isMaxOutputTokensIncompleteError(error)) {
      const previousMaxOutputTokens = initialRequest.generationConfig!.maxOutputTokens!;
      const nextMaxOutputTokens = Math.min(DEFAULT_LLM_COMPRESSION_OUTPUT_RESERVE_TOKENS, Math.max(
        previousMaxOutputTokens + SUMMARY_PROVIDER_MIN_OUTPUT_TOKENS,
        previousMaxOutputTokens * 2
      ));
      if (nextMaxOutputTokens > previousMaxOutputTokens) {
        logCompressionDebug('provider.compact.summary.outputBudgetRetry', {
          providerConfigId: resolved.settings.id,
          provider: resolved.settings.provider,
          previousMaxOutputTokens,
          nextMaxOutputTokens
        });
        return execute(withMaxOutputTokens(initialRequest, nextMaxOutputTokens));
      }
    }
    throw error;
  }
}

async function* createSummaryWebSocketStream(
  resolved: ResolvedSummaryProvider,
  request: SummaryProviderCall['request'],
  signal?: AbortSignal
): AsyncGenerator<UnifiedLLMStreamChunk> {
  const providerDryRun = (resolved.provider as unknown as Partial<UnifiedDryRunCapable> | undefined)?.dryRun;
  if (!resolved.provider || typeof providerDryRun !== 'function' || !resolved.unified || !resolved.webSocketSessionKey) {
    throw new Error('OpenAI Responses WebSocket 摘要缺少已解析的 Provider 传输信息。');
  }
  const dryRun = await providerDryRun.call(resolved.provider, request, {
    inputFormat: 'unified',
    outputFormat: 'unified',
    stream: true
  });
  const format = new resolved.unified.OpenAIResponsesFormat(resolved.settings.model) as OpenAIResponsesFormatAdapter;
  const { streamOpenAIResponsesWebSocketSession } = await openAIResponsesWebSocketSession();
  yield* streamOpenAIResponsesWebSocketSession({
    sessionKey: resolved.webSocketSessionKey,
    url: dryRun.url,
    headers: dryRun.headers,
    body: dryRun.body,
    format,
    signal,
    proxy: resolved.proxy
  });
}

function hasMaxOutputTokens(request: SummaryProviderCall['request']): boolean {
  return typeof request.generationConfig?.maxOutputTokens === 'number';
}

function withoutMaxOutputTokens(request: SummaryProviderCall['request']): SummaryProviderCall['request'] {
  if (!hasMaxOutputTokens(request)) return request;
  const generationConfig = { ...request.generationConfig };
  delete generationConfig.maxOutputTokens;
  const next = { ...request };
  if (Object.keys(generationConfig).length > 0) next.generationConfig = generationConfig;
  else delete next.generationConfig;
  return next;
}

function withMaxOutputTokens(
  request: SummaryProviderCall['request'],
  maxOutputTokens: number
): SummaryProviderCall['request'] {
  return {
    ...request,
    generationConfig: { ...(request.generationConfig ?? {}), maxOutputTokens }
  };
}

function isUnsupportedMaxOutputTokensError(error: unknown): boolean {
  const text = summaryErrorSearchText(error);
  return text.includes('max_output_tokens')
    && (text.includes('unsupported parameter') || text.includes('unknown parameter') || text.includes('not supported'));
}

function isMaxOutputTokensIncompleteError(error: unknown): boolean {
  const text = summaryErrorSearchText(error);
  return text.includes('max_output_tokens')
    && (text.includes('incomplete') || text.includes('exhaust') || text.includes('limit'));
}

function summaryErrorSearchText(error: unknown): string {
  const failure = error instanceof LlmAttemptFailureError
    ? stringifyJson(toPlainJsonLike(error.failure))
    : '';
  return `${errorSearchText(error)}\n${failure}`.toLowerCase();
}

interface GeneratedSummaryTextResult { text: string; settings?: LlmProviderConfigRecord }

async function generateSummaryText(
  request: LlmCompactRequest,
  methodConfig: LlmCompressionConfigRecord,
  options: LlmProviderOptions,
  signal?: AbortSignal,
  resolvedProvider?: ResolvedSummaryProvider
): Promise<GeneratedSummaryTextResult> {
  const targetTokens = effectiveSummaryTargetTokens(methodConfig);
  const priorSummaryText = request.priorSummaryContents?.length
    ? plainTextOfContents(request.priorSummaryContents)
    : '';
  const fallback = deterministicReplacementSummary(priorSummaryText, request.contents, targetTokens);
  if (methodConfig.kind === 'deterministic_summary' || methodConfig.kind === 'manual_summary') {
    return { text: fallback };
  }

  if (request.contents.length === 0) return { text: fallback };

  const resolved = resolvedProvider ?? await resolveSummaryProvider(request, methodConfig, options);
  if (!resolved.provider) return { text: fallback, settings: resolved.settings };

  const call = buildSummaryProviderCall(request, methodConfig, resolved.settings);
  if (!isSummaryProviderCallWithinWindow(call, resolved.settings)) {
    throw new Error('compression_request_too_large: summary input exceeds the frozen Provider input limit.');
  }
  const text = extractSummaryTag((await executeSummaryProviderCall(resolved, call.request, signal)).trim());
  return {
    text: finalizeStructuredSummary(text, fallback, targetTokens),
    settings: resolved.settings
  };
}

function normalizeCompressionConfig(input: LlmCompressionConfigRecord | undefined, fallbackKind?: LlmCompressionConfigRecord['kind']): LlmCompressionConfigRecord {
  const now = Date.now();
  const kind = input?.kind ?? fallbackKind ?? 'llm_summary';
  return {
    id: input?.id ?? 'inline-compression-config',
    name: input?.name ?? '临时压缩方法',
    kind,
    trigger: input?.trigger ?? { mode: 'manual' },
    ...(input?.openaiResponsesCompact ? { openaiResponsesCompact: input.openaiResponsesCompact } : {}),
    ...(input?.llmSummary ? { llmSummary: input.llmSummary } : {}),
    createdAt: input?.createdAt ?? now,
    updatedAt: input?.updatedAt ?? now
  };
}

function usageMetadataFromCompact(value: unknown): LlmUsageMetadataRecord | undefined {
  const cleaned = stripUndefined(value);
  return isRecord(cleaned) && Object.keys(cleaned).length > 0 ? cleaned as LlmUsageMetadataRecord : undefined;
}

function renderContentsForSummary(contents: MessageContent[]): string {
  return contents.map((content, index) => `${index + 1}. ${content.role}: ${content.parts.map(renderSummaryPart).filter(Boolean).join('\n') || '[empty]'}`).join('\n\n');
}

function renderSummaryPart(part: ContentPart): string {
  if (isTextPart(part)) return part.thought === true ? '' : part.text;
  if (isFunctionCallPart(part)) return `[tool call] ${part.functionCall.name}: ${stringifyJson(part.functionCall.args)}`;
  if (isFunctionResponsePart(part)) return `[tool result] ${part.functionResponse.name}: ${stringifyJson(part.functionResponse.response)}`;
  if (isInlineDataPart(part)) return `[inline data] ${part.inlineData.mimeType}`;
  if (isFileDataPart(part)) return `[file] ${part.fileData.uri}`;
  if (isProviderContextPart(part)) return `[provider context] ${part.providerContext.format}:${part.providerContext.itemType ?? 'context'}`;
  return '';
}

type StructuredSummaryField = 'goals' | 'constraints' | 'completed' | 'active' | 'blocked' | 'next' | 'files';

interface StructuredSummary {
  goals: string[];
  constraints: string[];
  completed: string[];
  active: string[];
  blocked: string[];
  next: string[];
  files: string[];
}

function emptyStructuredSummary(): StructuredSummary {
  return { goals: [], constraints: [], completed: [], active: [], blocked: [], next: [], files: [] };
}

function effectiveSummaryTargetTokens(methodConfig: LlmCompressionConfigRecord): number {
  const configured = methodConfig.llmSummary?.targetTokens;
  if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
    return DEFAULT_LLM_COMPRESSION_SUMMARY_TARGET_TOKENS;
  }
  return Math.max(1, Math.min(DEFAULT_LLM_COMPRESSION_SUMMARY_TARGET_TOKENS, Math.floor(configured)));
}

function deterministicReplacementSummary(
  priorSummaryText: string,
  contents: MessageContent[],
  targetTokens: number
): string {
  const prior = parseStructuredSummary(priorSummaryText)
    ?? structuredSummaryFromLooseText(priorSummaryText, 'active');
  const delta = structuredSummaryFromContents(contents);
  return fitStructuredSummary(mergeStructuredSummaries(prior, delta), targetTokens);
}

function structuredSummaryFromContents(contents: MessageContent[]): StructuredSummary {
  const summary = emptyStructuredSummary();
  for (const content of contents) {
    for (const part of content.parts) {
      const rendered = renderSummaryPart(part).trim();
      if (!rendered) continue;
      const facts = splitSummaryFacts(rendered);
      const textField: StructuredSummaryField = content.role === 'user' ? 'goals' : 'completed';
      for (const fact of facts) {
        appendSummaryFact(summary, textField, fact);
        if (looksLikeConstraint(fact)) appendSummaryFact(summary, 'constraints', fact);
        if (looksBlocked(fact)) appendSummaryFact(summary, 'blocked', fact);
        if (looksLikeNextStep(fact)) appendSummaryFact(summary, 'next', fact);
        for (const file of extractFileReferences(fact)) appendSummaryFact(summary, 'files', file);
      }
    }
  }
  if (contents.length > 0 && summary.active.length === 0) {
    const latest = [...contents].reverse().find((content) => content.role === 'model');
    const latestText = latest?.parts.map(renderSummaryPart).filter(Boolean).join('\n').trim();
    if (latestText) appendSummaryFact(summary, 'active', `最近状态：${headTailTextByTokens(latestText, 256)}`);
  }
  return summary;
}

function structuredSummaryFromLooseText(text: string, fallbackField: StructuredSummaryField): StructuredSummary {
  const summary = emptyStructuredSummary();
  for (const fact of splitSummaryFacts(stripSummaryEnvelope(text))) {
    let field = fallbackField;
    if (looksLikeConstraint(fact)) field = 'constraints';
    else if (looksBlocked(fact)) field = 'blocked';
    else if (looksLikeNextStep(fact)) field = 'next';
    appendSummaryFact(summary, field, fact);
    for (const file of extractFileReferences(fact)) appendSummaryFact(summary, 'files', file);
  }
  return summary;
}

function parseStructuredSummary(text: string): StructuredSummary | undefined {
  const source = stripSummaryEnvelope(text);
  if (!isStructuredSummaryText(source)) return undefined;
  const summary = emptyStructuredSummary();
  let field: StructuredSummaryField | undefined;
  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim().replace(/^#{1,6}\s*/, '');
    if (!line) continue;
    const heading = summaryHeading(line);
    if (heading) {
      field = heading.field;
      if (heading.rest && heading.rest !== '无') appendSummaryFact(summary, field, heading.rest);
      continue;
    }
    if (line === '工作状态' || line === '工作状态：') {
      field = undefined;
      continue;
    }
    if (!field) continue;
    const fact = line.replace(/^[-*•]\s*/, '').trim();
    if (fact && fact !== '无') appendSummaryFact(summary, field, fact);
  }
  return summary;
}

function summaryHeading(line: string): { field: StructuredSummaryField; rest: string } | undefined {
  const normalized = line.replace(/[：:]\s*/, ':');
  const headings: Array<[string, StructuredSummaryField]> = [
    ['重要约束、决定和准确标识', 'constraints'],
    ['重要约束、决定和标识', 'constraints'],
    ['- 已完成', 'completed'],
    ['已完成', 'completed'],
    ['- 正在做', 'active'],
    ['正在做', 'active'],
    ['- 受阻', 'blocked'],
    ['受阻', 'blocked'],
    ['下一步', 'next'],
    ['相关文件', 'files'],
    ['目标', 'goals']
  ];
  for (const [heading, field] of headings) {
    if (normalized === heading || normalized === `${heading}:`) return { field, rest: '' };
    if (normalized.startsWith(`${heading}:`)) return { field, rest: normalized.slice(heading.length + 1).trim() };
  }
  return undefined;
}

function isStructuredSummaryText(text: string): boolean {
  const source = stripSummaryEnvelope(text);
  return ['目标', '重要约束、决定', '工作状态', '已完成', '正在做', '受阻', '下一步', '相关文件']
    .every((heading) => source.includes(heading));
}

function finalizeStructuredSummary(candidate: string, fallback: string, targetTokens: number): string {
  const fallbackSummary = parseStructuredSummary(fallback)
    ?? structuredSummaryFromLooseText(fallback, 'active');
  const parsed = parseStructuredSummary(candidate);
  if (!parsed || structuredSummaryFactCount(parsed) === 0) {
    return fitStructuredSummary(fallbackSummary, targetTokens);
  }
  return fitStructuredSummary(mergeStructuredSummaries(fallbackSummary, parsed), targetTokens);
}

function structuredSummaryFactCount(summary: StructuredSummary): number {
  return Object.values(summary).reduce((count, facts) => count + facts.length, 0);
}

function mergeStructuredSummaries(prior: StructuredSummary, delta: StructuredSummary): StructuredSummary {
  const merged = emptyStructuredSummary();
  for (const field of Object.keys(merged) as StructuredSummaryField[]) {
    merged[field] = replacementMergeFacts(prior[field], delta[field]);
  }
  return merged;
}

const MAX_STRUCTURED_SUMMARY_FACTS_PER_FIELD = 80;

function replacementMergeFacts(prior: readonly string[], delta: readonly string[]): string[] {
  const facts = new Map<string, string>();
  for (const fact of [...prior, ...delta]) {
    const normalized = normalizeSummaryFact(fact);
    if (!normalized || normalized === '无') continue;
    const key = summaryReplacementKey(normalized);
    facts.delete(key);
    facts.set(key, normalized);
  }
  const values = [...facts.values()];
  if (values.length <= MAX_STRUCTURED_SUMMARY_FACTS_PER_FIELD) return values;
  return [
    values[0]!,
    ...values.slice(-(MAX_STRUCTURED_SUMMARY_FACTS_PER_FIELD - 1))
  ];
}

function summaryReplacementKey(fact: string): string {
  const keyValue = /^(.{1,96}?)[：:=]\s*/.exec(fact)?.[1]
    ?.trim()
    .replace(/^(?:必须|不得|不要|只能|需要|require|must|never|only)\s*/i, '')
    .toLowerCase();
  if (keyValue) return `key:${keyValue}`;
  const file = extractFileReferences(fact)[0];
  if (file && fact.length < 180) return `file:${file.toLowerCase()}`;
  return `fact:${fact.toLowerCase()}`;
}

function appendSummaryFact(summary: StructuredSummary, field: StructuredSummaryField, fact: string): void {
  const normalized = normalizeSummaryFact(fact);
  if (!normalized || normalized === '无') return;
  summary[field] = replacementMergeFacts(summary[field], [headTailTextByTokens(normalized, 512)]);
}

function normalizeSummaryFact(value: string): string {
  return value.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').replace(/\s+/g, ' ').trim();
}

function splitSummaryFacts(text: string): string[] {
  const lines = text.split(/\r?\n+/).map(normalizeSummaryFact).filter(Boolean);
  if (lines.length > 1) return lines;
  return text.split(/(?<=[。！？.!?])\s+/).map(normalizeSummaryFact).filter(Boolean);
}

function looksLikeConstraint(text: string): boolean {
  return /(?:必须|不得|不要|只能|需要|限制|约束|require|must|never|only)/i.test(text);
}

function looksBlocked(text: string): boolean {
  return /(?:受阻|失败|错误|报错|无法|缺少|blocked|failed|error|cannot|missing)/i.test(text);
}

function looksLikeNextStep(text: string): boolean {
  return /(?:下一步|待办|随后|接下来|尚未|todo|next|remaining)/i.test(text);
}

function extractFileReferences(text: string): string[] {
  const matches = text.match(/(?:[A-Za-z]:\\[^\s"'`]+|\/(?:[^\s"'`]+\/)*[^\s"'`]+|(?:[\w.@-]+\/)+[\w.@-]+\.[A-Za-z0-9_-]+)/g) ?? [];
  return [...new Set(matches.map((entry) => entry.replace(/[),.;，。；]+$/, '')))];
}

function formatStructuredSummary(summary: StructuredSummary): string {
  const list = (facts: readonly string[]) => facts.length > 0
    ? facts.map((fact) => `- ${fact}`).join('\n')
    : '- 无';
  return [
    '目标',
    list(summary.goals),
    '',
    '重要约束、决定和准确标识',
    list(summary.constraints),
    '',
    '工作状态',
    '  - 已完成',
    indentSummaryFacts(summary.completed),
    '  - 正在做',
    indentSummaryFacts(summary.active),
    '  - 受阻',
    indentSummaryFacts(summary.blocked),
    '',
    '下一步',
    list(summary.next),
    '',
    '相关文件',
    list(summary.files)
  ].join('\n');
}

function indentSummaryFacts(facts: readonly string[]): string {
  return facts.length > 0
    ? facts.map((fact) => `    - ${fact}`).join('\n')
    : '    - 无';
}

function fitStructuredSummary(input: StructuredSummary, targetTokens: number): string {
  const summary = Object.fromEntries((Object.keys(input) as StructuredSummaryField[]).map((field) => [
    field,
    replacementMergeFacts([], input[field])
  ])) as unknown as StructuredSummary;
  let rendered = formatStructuredSummary(summary);
  if (estimateTokenCount(rendered) <= targetTokens) return rendered;

  const dropOrder: StructuredSummaryField[] = ['completed', 'files', 'goals', 'constraints', 'next', 'blocked', 'active'];
  let changed = true;
  while (estimateTokenCount(rendered) > targetTokens && changed) {
    changed = false;
    for (const field of dropOrder) {
      if (summary[field].length <= 2) continue;
      summary[field].splice(1, 1);
      changed = true;
      rendered = formatStructuredSummary(summary);
      if (estimateTokenCount(rendered) <= targetTokens) return rendered;
    }
  }

  for (const perFactLimit of [256, 128, 64, 32, 16, 8]) {
    for (const field of Object.keys(summary) as StructuredSummaryField[]) {
      summary[field] = summary[field].map((fact) => headTailTextByTokens(fact, perFactLimit));
    }
    rendered = formatStructuredSummary(summary);
    if (estimateTokenCount(rendered) <= targetTokens) return rendered;
  }
  return fitTextToTokenLimit(rendered, targetTokens);
}

function summaryContents(summary: string, targetTokens: number): MessageContent[] {
  const prefix = '[Context Summary]\n\n';
  const prefixTokens = estimateTokenCount(prefix);
  const bodyBudget = Math.max(1, targetTokens - prefixTokens);
  const structured = parseStructuredSummary(summary);
  const boundedBody = structured
    ? fitStructuredSummary(structured, bodyBudget)
    : fitTextToTokenLimit(summary, bodyBudget);
  const text = targetTokens > prefixTokens
    ? `${prefix}${boundedBody}`
    : fitTextToTokenLimit(boundedBody, targetTokens);
  return [{ role: 'user', parts: [{ text: fitTextToTokenLimit(text, targetTokens) }] }];
}

function stripSummaryEnvelope(text: string): string {
  return extractSummaryTag(text.replace(/^\s*\[Context Summary\]\s*/i, '')).trim();
}

function headTailTextByTokens(text: string, limit: number): string {
  const normalizedLimit = Math.max(1, Math.floor(limit));
  if (estimateTokenCount(text) <= normalizedLimit) return text;
  const marker = ' … [缩短] … ';
  const markerTokens = estimateTokenCount(marker);
  if (normalizedLimit <= markerTokens + 1) return fitTextToTokenLimit(text, normalizedLimit);
  const available = normalizedLimit - markerTokens;
  const headTokens = Math.max(1, Math.floor(available * 0.6));
  const tailTokens = Math.max(1, available - headTokens);
  return fitTextToTokenLimit(
    `${sliceByTokens(text, 0, headTokens)}${marker}${sliceByTokens(text, -tailTokens)}`,
    normalizedLimit
  );
}

function fitTextToTokenLimit(text: string, limit: number): string {
  const normalizedLimit = Math.max(1, Math.floor(limit));
  if (estimateTokenCount(text) <= normalizedLimit) return text;
  let end = normalizedLimit;
  let sliced = sliceByTokens(text, 0, end);
  while (end > 0 && estimateTokenCount(sliced) > normalizedLimit) {
    end -= 1;
    sliced = sliceByTokens(text, 0, end);
  }
  return sliced;
}

function isSummaryProviderCallWithinWindow(
  call: SummaryProviderCall,
  settings: LlmProviderConfigRecord
): boolean {
  const contextWindowTokens = settings.contextWindowTokens ?? DEFAULT_LLM_CONTEXT_WINDOW_TOKENS;
  const outputTokens = call.request.generationConfig?.maxOutputTokens
    ?? DEFAULT_LLM_COMPRESSION_OUTPUT_RESERVE_TOKENS;
  const inputLimit = contextWindowTokens
    - Math.max(DEFAULT_LLM_COMPRESSION_OUTPUT_RESERVE_TOKENS, outputTokens)
    - SUMMARY_PROVIDER_ESTIMATOR_SLACK_TOKENS;
  if (inputLimit <= 0) return false;
  return estimateTokenCount(JSON.stringify({
    contents: call.request.contents,
    systemInstruction: call.request.systemInstruction
  })) <= inputLimit;
}

function modelCatalogEntryToRecord(model: UnifiedModelCatalogEntry): LlmProviderModelRecord {
  return {
    id: model.id,
    name: model.displayName || model.label || model.name || model.id,
    ...(model.createdAt ? { createdAt: model.createdAt } : {})
  };
}

function normalizeSettings(settings: LlmProviderConfigRecord | undefined): LlmProviderConfigRecord {
  const headers = normalizeHeaders(settings?.headers);
  const generationConfig = settings?.generationConfig;
  const requestBody = settings?.requestBody;
  const contextWindowTokens = normalizeContextWindowTokens(settings?.contextWindowTokens);
  const retryMaxAttempts = normalizeRetryMaxAttempts(settings?.retryMaxAttempts) ?? DEFAULT_LLM_RETRY_MAX_ATTEMPTS;
  return {
    id: settings?.id?.trim() || 'llm-provider-config-default',
    name: settings?.name?.trim() || '默认渠道',
    provider: normalizeProvider(settings?.provider),
    baseUrl: settings?.baseUrl?.trim() || DEFAULT_LLM_BASE_URL,
    model: settings?.model?.trim() ?? '',
    models: settings?.models ?? [],
    apiKey: settings?.apiKey?.trim() ?? '',
    toolCallFormat: normalizeToolCallFormat(settings?.toolCallFormat),
    openaiResponsesTransport: normalizeOpenAIResponsesTransport(settings?.openaiResponsesTransport),
    stream: settings?.stream !== false,
    retryOnError: settings?.retryOnError !== false ? DEFAULT_LLM_RETRY_ON_ERROR : false,
    retryMaxAttempts,
    enableMultimodalTools: settings?.enableMultimodalTools !== false,
    ...(contextWindowTokens ? { contextWindowTokens } : {}),
    systemPromptPrefix: typeof settings?.systemPromptPrefix === 'string' ? settings.systemPromptPrefix : '',
    ...(headers ? { headers } : {}),
    ...(nonEmptyRecord(generationConfig) ? { generationConfig } : {}),
    ...(nonEmptyRecord(requestBody) ? { requestBody } : {}),
    promptCache: normalizePromptCache(settings?.promptCache, normalizeProvider(settings?.provider)),
    modelConfigs: settings?.modelConfigs ?? [],
    createdAt: settings?.createdAt ?? 0,
    updatedAt: settings?.updatedAt ?? 0
  };
}

async function resolveRuntimeSettings(
  request: LlmStartRequest | LlmCompactRequest,
  options: LlmProviderOptions,
  resolvedRuntimeSettingsByInvocationId?: Map<string, LlmProviderConfigRecord>
): Promise<LlmProviderConfigRecord> {
  const cached = request.invocationId ? resolvedRuntimeSettingsByInvocationId?.get(request.invocationId) : undefined;
  if (cached) return normalizeSettings(cached);
  return normalizeSettings(await resolveMaybe(options.settings, request));
}

function snapshotFromSettings(settings: LlmProviderConfigRecord, compressionConfig?: LlmCompressionConfigRecord): LlmInvocationSettingsSnapshotRecord {
  const modelId = settings.model.trim();
  const modelName = modelId ? settings.models.find((model) => model.id === modelId)?.name.trim() || modelId : undefined;
  return {
    providerConfigId: settings.id,
    providerConfigName: settings.name,
    provider: settings.provider,
    baseUrl: settings.baseUrl,
    ...(modelId ? { modelId } : {}),
    ...(modelName ? { modelName, displayModelName: modelName } : {}),
    toolCallFormat: settings.toolCallFormat,
    openaiResponsesTransport: settings.openaiResponsesTransport,
    stream: settings.stream !== false,
    retryOnError: settings.retryOnError !== false,
    retryMaxAttempts: normalizeRetryMaxAttempts(settings.retryMaxAttempts) ?? DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
    enableMultimodalTools: settings.enableMultimodalTools !== false,
    ...(settings.contextWindowTokens ? { contextWindowTokens: settings.contextWindowTokens } : {}),
    ...(settings.systemPromptPrefix.trim() ? { systemPromptPrefix: settings.systemPromptPrefix } : {}),
    ...(settings.generationConfig ? { generationConfig: settings.generationConfig } : {}),
    ...(settings.requestBody ? { requestBody: settings.requestBody } : {}),
    ...(settings.promptCache ? { promptCache: settings.promptCache } : {}),
    ...(compressionConfig?.id ? { compressionConfigId: compressionConfig.id } : {}),
    ...(compressionConfig?.kind ? { compressionMethodKind: compressionConfig.kind } : {}),
    ...(compressionConfig?.trigger ? { compressionTrigger: compressionConfig.trigger } : {}),
    ...(compressionConfig ? { compressionConfigSnapshot: cloneJsonValue(compressionConfig) } : {}),
    ...(settings.headers ? { headers: maskSensitiveHeaders(settings.headers) } : {})
  };
}

function cloneJsonValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function resolveModelDisplayName(settings: LlmProviderConfigRecord): string | undefined {
  const modelId = settings.model.trim();
  if (!modelId) return undefined;
  const catalogName = settings.models.find((model) => model.id === modelId)?.name.trim();
  return catalogName || modelId;
}

function maskSensitiveHeaders(headers: LlmProviderHeadersRecord): LlmProviderHeadersRecord {
  const masked: LlmProviderHeadersRecord = {};
  for (const [key, value] of Object.entries(headers)) {
    masked[key] = isSensitiveHeaderName(key) ? maskSecretValue(value) : value;
  }
  return masked;
}

function isSensitiveHeaderName(name: string): boolean {
  const normalized = name.trim().toLowerCase();
  return normalized === 'authorization' || normalized === 'x-api-key' || normalized === 'x-goog-api-key' || normalized === 'api-key' || normalized === 'openai-key' || normalized.includes('token') || normalized.includes('secret') || normalized.includes('key');
}

function maskSecretValue(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return '';
  return trimmed.length <= 8 ? '••••••••' : `${trimmed.slice(0, 4)}••••${trimmed.slice(-4)}`;
}

function normalizeProvider(provider: LlmProviderKind | undefined): LlmProviderKind {
  return provider === 'gemini' || provider === 'claude' || provider === 'openai-compatible' || provider === 'openai-responses' || provider === 'deepseek'
    ? provider
    : 'openai-compatible';
}

function normalizeToolCallFormat(format: LlmToolCallFormat | undefined): LlmToolCallFormat {
  return format === 'function-call' ? format : 'function-call';
}

function normalizeOpenAIResponsesTransport(value: unknown): LlmOpenAIResponsesTransport {
  return value === 'websocket' ? 'websocket' : 'http';
}

function normalizePromptCache(input: LlmPromptCacheConfigRecord | undefined, provider: LlmProviderKind): LlmPromptCacheConfigRecord {
  if (!input || typeof input !== 'object') return createDefaultLlmPromptCacheConfig(provider);
  return {
    enabled: typeof input.enabled === 'boolean' ? input.enabled : true,
    mode: normalizePromptCacheMode(input.mode, provider),
    ttl: normalizePromptCacheTtl(input.ttl, provider)
  };
}

function normalizePromptCacheMode(input: unknown, provider: LlmProviderKind): LlmPromptCacheMode {
  if (provider === 'openai-responses' && input === 'explicit') return 'explicit';
  return defaultLlmPromptCacheModeForProvider(provider);
}

function normalizePromptCacheTtl(input: unknown, provider: LlmProviderKind): LlmPromptCacheTtl {
  if (provider === 'openai-responses') return '30m';
  if (provider === 'claude') return input === '5m' || input === '1h' ? input : defaultLlmPromptCacheTtlForProvider(provider);
  return defaultLlmPromptCacheTtlForProvider(provider);
}

function unifiedPromptCacheConfigEntry(settings: LlmProviderConfigRecord, requestBody?: LlmRequestBodyRecord): { promptCache: Record<string, unknown> } | Record<string, never> {
  const promptCache = unifiedPromptCacheFromSettings(settings, requestBody);
  return promptCache ? { promptCache } : {};
}

function unifiedPromptCacheFromSettings(settings: LlmProviderConfigRecord, requestBody?: LlmRequestBodyRecord): Record<string, unknown> | undefined {
  if (!isPromptCacheSupportedProvider(settings.provider)) return undefined;
  const promptCache = normalizePromptCache(settings.promptCache, settings.provider);
  if (!promptCache.enabled) return undefined;
  if (settings.provider === 'openai-responses') {
    const effectiveRequestBody = requestBody ?? settings.requestBody;
    const key = typeof effectiveRequestBody?.prompt_cache_key === 'string' && effectiveRequestBody.prompt_cache_key.trim()
      ? effectiveRequestBody.prompt_cache_key.trim()
      : undefined;
    if (promptCache.mode === 'key') return key ? { enabled: true, mode: 'key', key } : undefined;
    return {
      enabled: true,
      mode: 'explicit',
      ttl: promptCache.ttl,
      breakpoints: { messages: true },
      ...(key ? { key } : {})
    };
  }
  return {
    enabled: true,
    ttl: promptCache.ttl,
    mode: 'explicit',
    breakpoints: { system: true, tools: true, messages: true }
  };
}

function requestBodyWithOpenAIPromptCacheKey(settings: LlmProviderConfigRecord, conversationId?: string): LlmRequestBodyRecord | undefined {
  const requestBody = settings.requestBody;
  if (settings.provider !== 'openai-responses') return requestBody;
  if (typeof requestBody?.prompt_cache_key === 'string' && requestBody.prompt_cache_key.trim()) return requestBody;
  const promptCache = normalizePromptCache(settings.promptCache, settings.provider);
  if (!promptCache.enabled || !conversationId?.trim()) return requestBody;
  return {
    ...(requestBody ?? {}),
    prompt_cache_key: createOpenAIPromptCacheKey(settings, conversationId)
  };
}

function createOpenAIPromptCacheKey(settings: LlmProviderConfigRecord, conversationId: string): string {
  return createHash('sha256')
    .update([
      settings.id,
      settings.model,
      conversationId
    ].join('\n'))
    .digest('hex')
    .slice(0, 32);
}

function openAIResponsesWebSocketConfigEntry(settings: LlmProviderConfigRecord, conversationId?: string): Record<string, unknown> {
  if (!isOpenAIResponsesWebSocketMode(settings)) return {};
  return {
    transport: 'websocket',
    webSocketSessionKey: createOpenAIResponsesWebSocketSessionKey(
      settings,
      requireOpenAIResponsesWebSocketConversationId(conversationId)
    )
  };
}

function openAIResponsesWebSocketDryRunResult(result: UnifiedDryRunResult, includeApiKey: boolean): UnifiedDryRunResult & { maskedCurl: string } {
  const url = toWebSocketUrl(result.url);
  const body = openAIResponsesWebSocketDryRunPayload(result.body);
  const headers = result.headers;
  return {
    ...result,
    providerName: `${result.providerName} WebSocket`,
    url,
    body,
    bodyText: stringifyJsonPretty(body),
    curl: formatWebSocketDryRun(url, includeApiKey ? headers : maskSensitiveHeaders(headers), body),
    maskedCurl: formatWebSocketDryRun(url, maskSensitiveHeaders(headers), body)
  };
}

function openAIResponsesWebSocketDryRunPayload(body: unknown): Record<string, unknown> {
  const record = isRecord(body) ? stripOpenAIResponsesWebSocketUnsupportedFields(body) as Record<string, unknown> : {};
  delete record.type;
  delete record.stream;
  delete record.background;
  delete record.previous_response_id;
  delete record.prompt_cache_options;
  return { type: 'response.create', ...record, store: false };
}

function stripOpenAIResponsesWebSocketUnsupportedFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripOpenAIResponsesWebSocketUnsupportedFields);
  if (!isRecord(value)) return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => key !== 'prompt_cache_breakpoint')
    .map(([key, nested]) => [key, stripOpenAIResponsesWebSocketUnsupportedFields(nested)]));
}

function formatWebSocketDryRun(url: string, headers: Record<string, string>, body: unknown): string {
  return [
    '# WebSocket mode：先建立连接，再发送 response.create JSON 事件。',
    `CONNECT ${url}`,
    '',
    '# Headers',
    stringifyJsonPretty(headers),
    '',
    '# Send',
    stringifyJsonPretty(body)
  ].join('\n');
}

function toWebSocketUrl(url: string): string {
  const parsed = new URL(url);
  if (parsed.protocol === 'https:') parsed.protocol = 'wss:';
  else if (parsed.protocol === 'http:') parsed.protocol = 'ws:';
  return parsed.toString();
}

function stringifyJsonPretty(value: unknown): string {
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

function isOpenAIResponsesWebSocketMode(settings: LlmProviderConfigRecord): boolean {
  return settings.provider === 'openai-responses' && settings.openaiResponsesTransport === 'websocket';
}

export function createOpenAIResponsesWebSocketSessionKey(
  settings: LlmProviderConfigRecord,
  conversationId: string
): string {
  return createHash('sha256')
    .update([
      'openai-responses-websocket',
      settings.id,
      settings.baseUrl,
      settings.model,
      requireOpenAIResponsesWebSocketConversationId(conversationId)
    ].join('\n'))
    .digest('hex')
    .slice(0, 32);
}

function requireOpenAIResponsesWebSocketConversationId(conversationId: string | undefined): string {
  const normalized = conversationId?.trim();
  if (!normalized) {
    throw new TypeError('OpenAI Responses WebSocket requests require a non-empty conversationId.');
  }
  return normalized;
}

function normalizeHeaders(headers: unknown): LlmProviderHeadersRecord | undefined {
  if (!isRecord(headers)) return undefined;
  const result: LlmProviderHeadersRecord = {};
  for (const [rawKey, rawValue] of Object.entries(headers)) {
    const key = rawKey.trim();
    if (!key) continue;
    if (typeof rawValue !== 'string' && typeof rawValue !== 'number' && typeof rawValue !== 'boolean') continue;
    result[key] = String(rawValue).trim();
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function mergeHeaders(...records: Array<Record<string, string> | undefined>): LlmProviderHeadersRecord | undefined {
  const result: LlmProviderHeadersRecord = {};
  for (const record of records) {
    if (!record) continue;
    for (const [rawKey, rawValue] of Object.entries(record)) {
      const key = rawKey.trim();
      if (!key) continue;
      const existingKey = Object.keys(result).find((candidate) => candidate.toLowerCase() === key.toLowerCase());
      if (existingKey && existingKey !== key) delete result[existingKey];
      result[key] = rawValue;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function normalizeContextWindowTokens(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : undefined;
}

function normalizeRetryMaxAttempts(value: unknown): number | undefined {
  const number = Number(value);
  if (!Number.isFinite(number)) return undefined;
  const attempts = Math.floor(number);
  return attempts < -1 ? -1 : attempts;
}

function nonEmptyRecord(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && Object.keys(value).length > 0;
}

// 产品能力边界：附件存储/预览不受此集合限制；送模只使用各 Provider 的共同稳定类型。
// 其他 MIME 会转为显式文本占位，避免静默丢失，也不伪装模型已经读取过该附件。
const TOOL_RESPONSE_CONTEXT_FALLBACK_MESSAGE = '工具调用在本次 LLM 请求上下文中没有对应响应，已自动补充兜底响应。原工具执行结果不可用；如仍需要结果，请重新执行相关操作。';

interface ToolCallContextNormalizationResult {
  contents: MessageContent[];
  orphanResponseCount: number;
  fallbackResponseCount: number;
}

interface TrackedFunctionCall {
  part: FunctionCallPart;
  contentIndex: number;
  closed: boolean;
}

async function prepareLlmStartRequestMultimodal(request: LlmStartRequest, options: LlmProviderOptions): Promise<LlmStartRequest> {
  const preparation = createMultimodalPreparationContext();
  const [contents, systemInstruction] = await Promise.all([
    Promise.all(request.contents.map((content) =>
      prepareLlmContentMultimodal(content, options, false, 'ordinary', preparation))),
    request.systemInstruction
      ? prepareLlmContentMultimodal(request.systemInstruction, options, false, 'ordinary', preparation)
      : Promise.resolve(undefined)
  ]);
  const normalized = assertCanonicalProviderToolContext(contents);
  return {
    ...request,
    contents: normalized,
    ...(systemInstruction ? { systemInstruction } : {})
  };
}

export async function prepareNativeCompactContentsMultimodal(
  contents: MessageContent[],
  options: LlmProviderOptions
): Promise<MessageContent[]> {
  const preparation = createMultimodalPreparationContext();
  return Promise.all(contents.map((content) =>
    prepareLlmContentMultimodal(content, options, false, 'native_compact', preparation)));
}

function assertCanonicalProviderToolContext(contents: MessageContent[]): MessageContent[] {
  const normalized = normalizeToolCallResponseContext(contents);
  if (normalized.orphanResponseCount > 0 || normalized.fallbackResponseCount > 0) {
    throw new Error(
      `Provider boundary rejected non-canonical tool context: ${normalized.orphanResponseCount} orphan response(s), ${normalized.fallbackResponseCount} unresolved call(s).`
    );
  }
  return contents;
}

function normalizeToolCallResponseContext(contents: MessageContent[]): ToolCallContextNormalizationResult {
  const pendingById = new Map<string, TrackedFunctionCall>();
  const pendingByName = new Map<string, TrackedFunctionCall[]>();
  const calls: TrackedFunctionCall[] = [];
  let orphanResponseCount = 0;

  const normalized = contents.map((content, contentIndex) => {
    let changed = false;
    const parts = content.parts.map((part) => {
      if (isFunctionCallPart(part)) {
        const tracked: TrackedFunctionCall = { part, contentIndex, closed: false };
        calls.push(tracked);
        const id = normalizeToolCallId(part.id);
        if (id) {
          pendingById.set(id, tracked);
        } else {
          const list = pendingByName.get(part.functionCall.name) ?? [];
          list.push(tracked);
          pendingByName.set(part.functionCall.name, list);
        }
        return part;
      }

      if (!isFunctionResponsePart(part)) return part;

      const matched = consumeMatchingFunctionCall(part, pendingById, pendingByName);
      if (matched) return part;

      orphanResponseCount += 1;
      changed = true;
      return orphanFunctionResponseTextPart(part);
    });
    return changed ? { ...content, parts } : content;
  });

  const fallbackResponsesByContentIndex = new Map<number, FunctionResponsePart[]>();
  for (const call of calls) {
    if (call.closed) continue;
    const list = fallbackResponsesByContentIndex.get(call.contentIndex) ?? [];
    list.push(fallbackFunctionResponsePart(call.part));
    fallbackResponsesByContentIndex.set(call.contentIndex, list);
  }

  if (fallbackResponsesByContentIndex.size === 0) {
    return { contents: normalized, orphanResponseCount, fallbackResponseCount: 0 };
  }

  const repaired: MessageContent[] = [];
  let fallbackResponseCount = 0;
  normalized.forEach((content, index) => {
    repaired.push(content);
    const fallbackResponses = fallbackResponsesByContentIndex.get(index);
    if (!fallbackResponses?.length) return;
    fallbackResponseCount += fallbackResponses.length;
    repaired.push({ role: 'user', parts: fallbackResponses });
  });

  return { contents: repaired, orphanResponseCount, fallbackResponseCount };
}

function consumeMatchingFunctionCall(
  response: FunctionResponsePart,
  pendingById: Map<string, TrackedFunctionCall>,
  pendingByName: Map<string, TrackedFunctionCall[]>
): TrackedFunctionCall | undefined {
  const responseId = normalizeToolCallId(response.id);
  if (responseId) {
    const matched = pendingById.get(responseId);
    if (matched) {
      matched.closed = true;
      pendingById.delete(responseId);
      return matched;
    }
  }

  const queue = pendingByName.get(response.functionResponse.name);
  const matched = queue?.shift();
  if (!matched) return undefined;
  matched.closed = true;
  if (queue && queue.length === 0) pendingByName.delete(response.functionResponse.name);
  return matched;
}

function orphanFunctionResponseTextPart(part: FunctionResponsePart): ContentPart {
  return {
    text: [
      '[工具响应上下文兜底]',
      '原因: 当前 LLM 请求上下文中没有找到这条工具响应对应的工具调用，已转为普通文本，避免 provider 拒绝请求。',
      `name: ${part.functionResponse.name}`,
      ...(part.id ? [`callId: ${part.id}`] : []),
      `response: ${stringifyJson(part.functionResponse.response)}`
    ].join('\n')
  };
}

function fallbackFunctionResponsePart(call: FunctionCallPart): FunctionResponsePart {
  return {
    ...(call.id ? { id: call.id } : {}),
    functionResponse: {
      name: call.functionCall.name,
      response: {
        ok: false,
        status: 'error',
        recovered: true,
        interrupted: true,
        message: TOOL_RESPONSE_CONTEXT_FALLBACK_MESSAGE,
        ...(call.id ? { toolCallId: call.id } : {})
      }
    }
  };
}

function normalizeToolCallId(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed || undefined;
}

type MultimodalPreparationMode = 'ordinary' | 'native_compact';

interface AttachmentResolutionCacheEntry {
  mimeType?: string;
  name?: string;
  promise: Promise<InlineDataPart | undefined>;
}

interface MultimodalPreparationContext {
  attachmentResolutions: Map<string, AttachmentResolutionCacheEntry>;
}

function createMultimodalPreparationContext(): MultimodalPreparationContext {
  return { attachmentResolutions: new Map<string, AttachmentResolutionCacheEntry>() };
}

async function prepareLlmContentMultimodal(
  content: MessageContent,
  options: LlmProviderOptions,
  toolResponse: boolean,
  mode: MultimodalPreparationMode = 'ordinary',
  preparation: MultimodalPreparationContext = createMultimodalPreparationContext()
): Promise<MessageContent> {
  const parts = await Promise.all(content.parts.map((part) =>
    prepareLlmPartMultimodal(part, options, toolResponse, mode, preparation)));
  return { ...content, parts: parts.flat() };
}

async function prepareLlmPartMultimodal(
  part: ContentPart,
  options: LlmProviderOptions,
  toolResponse: boolean,
  mode: MultimodalPreparationMode,
  preparation: MultimodalPreparationContext
): Promise<ContentPart[]> {
  if (isInlineDataPart(part)) {
    return [await prepareInlineDataForLlm(part, options, toolResponse, mode, preparation)];
  }
  if (isFunctionResponsePart(part) && part.functionResponse.parts?.length) {
    const prepared = await Promise.all(part.functionResponse.parts.map((inlinePart) =>
      prepareInlineDataForLlm(inlinePart, options, true, mode, preparation)));
    const inlineParts = prepared.filter(isInlineDataPart).filter((inlinePart) => isSupportedToolResponseInlineData(inlinePart));
    const placeholders = prepared.filter(isTextPart).map((textPart) => textPart.text).filter(Boolean);
    return [{
      ...part,
      functionResponse: {
        ...part.functionResponse,
        response: placeholders.length > 0 ? withAttachmentPlaceholders(part.functionResponse.response, placeholders) : part.functionResponse.response,
        ...(inlineParts.length > 0 ? { parts: inlineParts } : {})
      }
    }];
  }
  return [part];
}

async function prepareInlineDataForLlm(
  part: InlineDataPart,
  options: LlmProviderOptions,
  toolResponse: boolean,
  mode: MultimodalPreparationMode,
  preparation: MultimodalPreparationContext
): Promise<ContentPart> {
  if (mode === 'ordinary' && toolResponse && !isSupportedToolResponseInlineData(part)) {
    return attachmentPlaceholderPart(part, '附件类型不在工具响应白名单中');
  }
  if (part.inlineData.data) {
    if (mode === 'native_compact') requireCanonicalInlineDataSize(part, 'Native Compact media');
    return toolResponse && !isSupportedToolResponseInlineData(part)
      ? attachmentPlaceholderPart(part, '附件类型不在工具响应白名单中')
      : part;
  }

  let resolved: InlineDataPart | undefined;
  try {
    resolved = await resolveAttachmentOnce(part, options, preparation);
  } catch (error) {
    if (error instanceof AttachmentResolutionMetadataConflictError) throw error;
    if (mode === 'native_compact') {
      throw new LlmNativeCompactMediaError(mediaReferenceLabel(part), errorSearchText(error));
    }
    resolved = undefined;
  }
  if (resolved?.inlineData.data) {
    if (mode === 'native_compact') requireCanonicalInlineDataSize(resolved, 'Resolved Native Compact media');
    return toolResponse && !isSupportedToolResponseInlineData(resolved)
      ? attachmentPlaceholderPart(resolved, '附件类型不在工具响应白名单中')
      : resolved;
  }
  if (mode === 'native_compact') {
    throw new LlmNativeCompactMediaError(
      mediaReferenceLabel(part),
      resolved?.inlineData.error ?? (options.resolveAttachment ? 'attachment resolver returned no bytes' : 'attachment resolver is unavailable')
    );
  }
  return attachmentPlaceholderPart(part, resolved?.inlineData.error ?? '附件读取失败');
}

class AttachmentResolutionMetadataConflictError extends Error {
  public constructor(key: string) {
    super(`Attachment resolver metadata conflicts for ${key}.`);
    this.name = 'AttachmentResolutionMetadataConflictError';
  }
}

async function resolveAttachmentOnce(
  part: InlineDataPart,
  options: LlmProviderOptions,
  preparation: MultimodalPreparationContext
): Promise<InlineDataPart | undefined> {
  if (!options.resolveAttachment) return undefined;
  const attachmentId = part.inlineData.attachmentId?.trim();
  const sourcePath = part.inlineData.sourcePath?.trim();
  const mimeType = part.inlineData.mimeType?.trim() || undefined;
  const name = part.inlineData.name?.trim() || undefined;
  const key = attachmentId
    ? `attachment:${attachmentId}`
    : sourcePath
      ? `source:${sourcePath}`
      : `descriptor:${mimeType ?? ''}\0${name ?? ''}`;
  const existing = preparation.attachmentResolutions.get(key);
  if (existing) {
    if ((existing.mimeType && mimeType && existing.mimeType !== mimeType)
      || (existing.name && name && existing.name !== name)) {
      throw new AttachmentResolutionMetadataConflictError(key);
    }
    existing.mimeType ??= mimeType;
    existing.name ??= name;
    const resolved = await existing.promise;
    return resolved ? cloneInlineDataPart(resolved) : undefined;
  }
  const input = {
    attachmentId: part.inlineData.attachmentId,
    sourcePath: part.inlineData.sourcePath,
    mimeType: part.inlineData.mimeType,
    name: part.inlineData.name
  };
  const promise = Promise.resolve().then(() => options.resolveAttachment!(input));
  preparation.attachmentResolutions.set(key, { mimeType, name, promise });
  const resolved = await promise;
  return resolved ? cloneInlineDataPart(resolved) : undefined;
}

function cloneInlineDataPart(part: InlineDataPart): InlineDataPart {
  return { ...part, inlineData: { ...part.inlineData } };
}

export class LlmNativeCompactMediaError extends Error {
  public readonly code = 'media_size_unknown';

  public constructor(reference: string, reason: string) {
    super(`media_size_unknown: Native Compact cannot resolve exact media bytes for ${reference}: ${reason}`);
    this.name = 'LlmNativeCompactMediaError';
  }
}

function requireCanonicalInlineDataSize(part: InlineDataPart, label: string): number {
  const data = part.inlineData.data;
  if (!data) {
    throw new LlmNativeCompactMediaError(mediaReferenceLabel(part), `${label} is not canonical base64`);
  }
  let bytes: Buffer;
  try {
    bytes = decodeCanonicalBase64(data);
  } catch {
    throw new LlmNativeCompactMediaError(mediaReferenceLabel(part), `${label} is not canonical base64`);
  }
  if (part.inlineData.sizeBytes !== undefined && part.inlineData.sizeBytes !== bytes.byteLength) {
    throw new LlmNativeCompactMediaError(
      mediaReferenceLabel(part),
      `${label} declared ${part.inlineData.sizeBytes} bytes but resolved ${bytes.byteLength}`
    );
  }
  return bytes.byteLength;
}

function mediaReferenceLabel(part: InlineDataPart): string {
  return part.inlineData.attachmentId
    ?? part.inlineData.sourcePath
    ?? part.inlineData.name
    ?? part.inlineData.mimeType;
}

function isSupportedToolResponseInlineData(part: InlineDataPart): boolean {
  return isModelToolResponseMultimodalMimeType(part.inlineData.mimeType);
}

function withAttachmentPlaceholders(response: unknown, placeholders: string[]): unknown {
  const key = 'multimodalAttachmentPlaceholders';
  if (isRecord(response)) {
    const previous = Array.isArray(response[key]) ? response[key].filter((item): item is string => typeof item === 'string') : [];
    return { ...response, [key]: [...previous, ...placeholders] };
  }
  return { response, [key]: placeholders };
}

function attachmentPlaceholderPart(part: InlineDataPart, reason: string): ContentPart {
  const name = part.inlineData.name || part.inlineData.sourcePath || part.inlineData.attachmentId || '未命名附件';
  return {
    text: `[附件不可用: ${name}; mimeType=${part.inlineData.mimeType}; reason=${reason}]`
  };
}

function toUnifiedRequest(
  request: LlmStartRequest,
  generationConfig?: LlmGenerationConfigRecord,
  providerKind?: LlmProviderKind
): UnifiedLLMRequest {
  const contents = providerKind === 'gemini'
    ? mergeGeminiFunctionResponseTurns(request.contents)
    : request.contents;
  return {
    contents: contents.flatMap((content) => toUnifiedContents(content, providerKind)),
    ...(request.systemInstruction ? { systemInstruction: { parts: request.systemInstruction.parts.map(toUnifiedPart) } } : {}),
    ...(request.tools.length === 0 ? {} : {
      tools: [{
        functionDeclarations: request.tools.map((tool) => toUnifiedFunctionDeclaration(tool, providerKind))
      }]
    }),
    ...(nonEmptyRecord(generationConfig) ? { generationConfig } : {})
  };
}

function mergeGeminiFunctionResponseTurns(contents: readonly MessageContent[]): MessageContent[] {
  const merged: MessageContent[] = [];
  for (let index = 0; index < contents.length; index += 1) {
    const content = contents[index];
    if (content.role !== 'user' || content.parts.length === 0 || !content.parts.every(isFunctionResponsePart)) {
      merged.push(content);
      continue;
    }
    const parts: ContentPart[] = [...content.parts];
    while (
      index + 1 < contents.length
      && contents[index + 1].role === 'user'
      && contents[index + 1].parts.length > 0
      && contents[index + 1].parts.every(isFunctionResponsePart)
    ) {
      parts.push(...contents[index + 1].parts);
      index += 1;
    }
    merged.push(parts.length === content.parts.length ? content : { ...content, parts });
  }
  return merged;
}

function toUnifiedContents(
  content: MessageContent,
  providerKind?: LlmProviderKind
): UnifiedContent[] {
  if (providerKind !== 'openai-responses' || content.role !== 'model'
    || (content as MessageContent & { providerContext?: unknown }).providerContext) {
    return [toUnifiedContent(content)];
  }

  const groups: ContentPart[][] = [];
  for (const part of content.parts) {
    const current = groups[groups.length - 1];
    const currentIdentity = current?.[0]?.outputItem?.id;
    const nextIdentity = part.outputItem?.id;
    if (current && currentIdentity === nextIdentity) current.push(part);
    else groups.push([part]);
  }
  return groups.map((parts) => {
    const outputItem = parts[0]?.outputItem;
    if (outputItem && parts.every((part) => isVisibleTextPart(part))) {
      const rawItem = {
        type: 'message',
        role: 'assistant',
        ...(outputItem.phase ? { phase: outputItem.phase } : {}),
        content: parts.map((part) => ({
          type: 'output_text',
          text: isTextPart(part) ? part.text : ''
        }))
      };
      return {
        role: 'model',
        parts: [],
        providerContext: {
          provider: 'openai',
          format: 'openai-responses',
          endpoint: 'responses',
          itemType: 'message',
          rawItem
        }
      } as UnifiedContent;
    }
    return toUnifiedContent({ role: content.role, parts });
  });
}

function toUnifiedContent(content: MessageContent): UnifiedContent {
  const providerContext = (content as MessageContent & { providerContext?: unknown }).providerContext;
  return {
    role: content.role === 'model' ? 'model' : 'user',
    parts: content.parts.map(toUnifiedPart),
    ...(providerContext ? { providerContext } : {})
  } as UnifiedContent;
}

function toUnifiedPart(part: ContentPart): UnifiedPart {
  if (isTextPart(part)) {
    const thoughtSignatures = thoughtSignaturesFromPortableSignature(part.thoughtSignature);
    return {
      text: part.text,
      ...(part.thought !== undefined ? { thought: part.thought } : {}),
      ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {}),
      ...(thoughtSignatures ? { thoughtSignatures } : {}),
      ...(part.thoughtElapsedMs !== undefined ? { thoughtElapsedMs: part.thoughtElapsedMs } : {})
    };
  }
  if (isFunctionCallPart(part)) {
    const thoughtSignatures = thoughtSignaturesFromPortableSignature(part.thoughtSignature);
    return {
      functionCall: { name: part.functionCall.name, args: asRecord(part.functionCall.args), ...(part.id ? { callId: part.id } : {}) },
      // Gemini 会校验带工具调用的 thoughtSignature；作为 part 同层级字段透传给 provider。
      ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {}),
      ...(thoughtSignatures ? { thoughtSignatures } : {})
    };
  }
  if (isFunctionResponsePart(part)) {
    const functionResponse: Record<string, unknown> = {
      name: part.functionResponse.name,
      response: asRecord(part.functionResponse.response),
      ...(part.id ? { callId: part.id } : {})
    };
    const inlineParts = (part.functionResponse.parts ?? [])
      .filter((inlinePart) => inlinePart.inlineData.data)
      .map((inlinePart) => ({
        inlineData: {
          mimeType: inlinePart.inlineData.mimeType,
          data: inlinePart.inlineData.data!,
          ...(inlinePart.inlineData.name ? { name: inlinePart.inlineData.name } : {})
        }
      }));
    if (inlineParts.length > 0) functionResponse.parts = inlineParts;
    return {
      functionResponse
    } as unknown as UnifiedPart;
  }
  if (isInlineDataPart(part)) return part.inlineData.data
    ? { inlineData: { mimeType: part.inlineData.mimeType, data: part.inlineData.data, ...(part.inlineData.name ? { name: part.inlineData.name } : {}) } }
    : { text: `[inlineData unavailable: ${part.inlineData.name ?? part.inlineData.attachmentId ?? part.inlineData.sourcePath ?? part.inlineData.mimeType}]` };
  if (isFileDataPart(part)) {
    // unified-llm-provider 当前统一 Part 没有 fileData；先作为文本占位保留语义。
    return { text: `[fileData:${part.fileData.mimeType ?? 'unknown'}:${part.fileData.uri}]` };
  }
  if (isProviderContextPart(part)) return { providerContext: part.providerContext } as unknown as UnifiedPart;
  return assertNever(part);
}

function toUnifiedFunctionDeclaration(
  tool: ToolSchema,
  providerKind?: LlmProviderKind
): UnifiedFunctionDeclaration {
  const parameters = isFunctionParameters(tool.parameters)
    ? providerCompatibleFunctionParameters(tool.name, tool.parameters, providerKind)
    : { type: 'object' as const, properties: {} };
  return {
    name: tool.name,
    description: tool.description,
    parameters
  };
}

function providerCompatibleFunctionParameters(
  toolName: string,
  parameters: UnifiedFunctionDeclaration['parameters'],
  providerKind?: LlmProviderKind
): UnifiedFunctionDeclaration['parameters'] {
  if (
    toolName !== 'edit'
    || (providerKind !== 'claude' && providerKind !== 'gemini')
    || !isRecord(parameters)
  ) return parameters;
  // Claude rejects top-level unions and Gemini drops them. Keep the full branch properties for
  // those formats; the shared runtime validator remains the authoritative fail-closed boundary.
  const parameterRecord = parameters as unknown as Record<string, unknown>;
  const { oneOf: _unsupportedUnion, ...compatible } = parameterRecord;
  return compatible as UnifiedFunctionDeclaration['parameters'];
}

function installGeminiProviderCompatibility<T>(
  provider: T,
  providerKind: LlmProviderKind,
  modelId: string
): T {
  return installGeminiOpenAICompatibleThoughtSignatures(
    installGeminiSchemaEncoder(provider, providerKind, modelId),
    providerKind,
    modelId
  );
}

function installGeminiSchemaEncoder<T>(
  provider: T,
  providerKind: LlmProviderKind,
  modelId: string
): T {
  if (providerKind !== 'gemini') return provider;
  const runtimeProvider = provider as T & {
    format?: {
      encodeRequest?: (request: unknown, stream: boolean) => unknown;
      __limcodeGeminiSchemaEncoder?: true;
    };
  };
  const format = runtimeProvider.format;
  if (!format || typeof format.encodeRequest !== 'function' || format.__limcodeGeminiSchemaEncoder) return provider;
  const originalEncodeRequest = format.encodeRequest.bind(format);
  format.encodeRequest = (request, stream) => {
    const normalizedRequest = normalizeGeminiThinkingRequest(request, modelId);
    const encoded = originalEncodeRequest(normalizedRequest, stream);
    restoreGeminiToolPropertyNames(encoded, normalizedRequest);
    return encoded;
  };
  format.__limcodeGeminiSchemaEncoder = true;
  return provider;
}

const GEMINI_THOUGHT_SIGNATURE_SKIP_VALIDATOR = 'skip_thought_signature_validator';

export function installGeminiOpenAICompatibleThoughtSignatures<T>(
  provider: T,
  providerKind: LlmProviderKind,
  modelId: string
): T {
  if (
    providerKind !== 'openai-compatible'
    || geminiThinkingCapabilityForModel(modelId).kind !== 'thinkingLevel'
  ) return provider;
  const runtimeProvider = provider as T & {
    format?: {
      encodeRequest?: (request: unknown, stream: boolean) => unknown;
      decodeResponse?: (raw: unknown) => unknown;
      decodeStreamChunk?: (raw: unknown, state: unknown) => unknown;
      __limcodeGeminiOpenAIThoughtSignatures?: true;
    };
  };
  const format = runtimeProvider.format;
  if (!format || format.__limcodeGeminiOpenAIThoughtSignatures) return provider;

  if (typeof format.encodeRequest === 'function') {
    const encodeRequest = format.encodeRequest.bind(format);
    format.encodeRequest = (request, stream) => {
      const encoded = encodeRequest(request, stream);
      attachGeminiOpenAIThoughtSignaturesToRequest(encoded, request);
      return encoded;
    };
  }
  if (typeof format.decodeResponse === 'function') {
    const decodeResponse = format.decodeResponse.bind(format);
    format.decodeResponse = (raw) => {
      const signatures = readGeminiOpenAIToolCallSignatures(raw, false);
      const decoded = decodeResponse(raw);
      attachGeminiSignaturesToUnifiedCalls(decoded, signatures);
      return decoded;
    };
  }
  if (typeof format.decodeStreamChunk === 'function') {
    const decodeStreamChunk = format.decodeStreamChunk.bind(format);
    const streamSignatures = new WeakMap<object, GeminiOpenAIToolCallSignatures>();
    format.decodeStreamChunk = (raw, state) => {
      const stateKey = isRecord(state) ? state : format;
      const signatures = streamSignatures.get(stateKey) ?? emptyGeminiOpenAIToolCallSignatures();
      mergeGeminiOpenAIToolCallSignatures(signatures, readGeminiOpenAIToolCallSignatures(raw, true));
      streamSignatures.set(stateKey, signatures);
      const decoded = decodeStreamChunk(raw, state);
      attachGeminiSignaturesToUnifiedCalls(decoded, signatures);
      return decoded;
    };
  }
  format.__limcodeGeminiOpenAIThoughtSignatures = true;
  return provider;
}

interface GeminiOpenAIToolCallSignatures {
  byId: Map<string, string>;
  byIndex: Map<number, string>;
}

function attachGeminiOpenAIThoughtSignaturesToRequest(encoded: unknown, source: unknown): void {
  if (!isRecord(encoded) || !isRecord(source)) return;
  const messages = Array.isArray(encoded.messages) ? encoded.messages.filter(isRecord) : [];
  const encodedCallGroups = messages.flatMap((message) =>
    message.role === 'assistant' && Array.isArray(message.tool_calls)
      ? [message.tool_calls.filter(isRecord)]
      : []);
  const sourceContents = Array.isArray(source.contents) ? source.contents.filter(isRecord) : [];
  const sourceCallGroups = sourceContents.flatMap((content) => {
    if (content.role !== 'model' || !Array.isArray(content.parts)) return [];
    const calls = content.parts.filter((part) => isRecord(part) && isRecord(part.functionCall));
    return calls.length > 0 ? [calls] : [];
  });

  for (let groupIndex = 0; groupIndex < Math.min(encodedCallGroups.length, sourceCallGroups.length); groupIndex += 1) {
    const encodedCalls = encodedCallGroups[groupIndex];
    const sourceCalls = sourceCallGroups[groupIndex];
    const sourceSignatures = sourceCalls.map(geminiSignatureFromUnifiedPart);
    const transferredGroup = sourceSignatures.every((signature) => !signature);
    for (let callIndex = 0; callIndex < Math.min(encodedCalls.length, sourceCalls.length); callIndex += 1) {
      const signature = sourceSignatures[callIndex]
        ?? (transferredGroup ? GEMINI_THOUGHT_SIGNATURE_SKIP_VALIDATOR : undefined);
      if (!signature) continue;
      const toolCall = encodedCalls[callIndex];
      const extraContent = isRecord(toolCall.extra_content) ? toolCall.extra_content : {};
      const google = isRecord(extraContent.google) ? extraContent.google : {};
      const attachedSignature = normalizedSignatureString(google.thought_signature)
        ?? normalizedSignatureString(google.thoughtSignature)
        ?? signature;
      toolCall.extra_content = {
        ...extraContent,
        google: {
          ...google,
          thought_signature: attachedSignature,
          thoughtSignature: attachedSignature
        }
      };
    }
  }
}

function readGeminiOpenAIToolCallSignatures(raw: unknown, stream: boolean): GeminiOpenAIToolCallSignatures {
  const signatures = emptyGeminiOpenAIToolCallSignatures();
  if (!isRecord(raw) || !Array.isArray(raw.choices)) return signatures;
  const choice = raw.choices.find(isRecord);
  if (!choice) return signatures;
  const rawMessage = choice[stream ? 'delta' : 'message'];
  if (!isRecord(rawMessage) || !Array.isArray(rawMessage.tool_calls)) return signatures;
  rawMessage.tool_calls
    .filter((toolCall): toolCall is Record<string, unknown> => isRecord(toolCall))
    .forEach((toolCall, ordinal) => {
      const signature = geminiOpenAIToolCallSignature(toolCall);
      if (!signature) return;
      const callId = normalizedSignatureString(toolCall.id);
      const index = typeof toolCall.index === 'number' && Number.isSafeInteger(toolCall.index)
        ? toolCall.index
        : ordinal;
      if (callId) signatures.byId.set(callId, signature);
      signatures.byIndex.set(index, signature);
    });
  return signatures;
}

function geminiOpenAIToolCallSignature(toolCall: Record<string, unknown>): string | undefined {
  const extraContent = isRecord(toolCall.extra_content) ? toolCall.extra_content : undefined;
  const google = isRecord(extraContent?.google) ? extraContent.google : undefined;
  const vertex = isRecord(extraContent?.vertex) ? extraContent.vertex : undefined;
  return normalizedSignatureString(google?.thought_signature)
    ?? normalizedSignatureString(google?.thoughtSignature)
    ?? normalizedSignatureString(vertex?.thought_signature)
    ?? normalizedSignatureString(vertex?.thoughtSignature);
}

function attachGeminiSignaturesToUnifiedCalls(
  decoded: unknown,
  signatures: GeminiOpenAIToolCallSignatures
): void {
  if (!isRecord(decoded)) return;
  const candidates = [
    ...(Array.isArray(decoded.functionCalls) ? decoded.functionCalls : []),
    ...(Array.isArray(decoded.partsDelta) ? decoded.partsDelta : []),
    ...(isRecord(decoded.content) && Array.isArray(decoded.content.parts) ? decoded.content.parts : [])
  ];
  const seen = new Set<object>();
  let ordinal = 0;
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.functionCall) || seen.has(candidate)) continue;
    seen.add(candidate);
    const callId = normalizedSignatureString(candidate.functionCall.callId);
    const signature = (callId ? signatures.byId.get(callId) : undefined) ?? signatures.byIndex.get(ordinal);
    ordinal += 1;
    if (!signature) continue;
    const existing = isRecord(candidate.thoughtSignatures) ? candidate.thoughtSignatures : {};
    candidate.thoughtSignatures = { ...existing, gemini: signature };
  }
}

function geminiSignatureFromUnifiedPart(part: Record<string, unknown>): string | undefined {
  const signatures = isRecord(part.thoughtSignatures) ? part.thoughtSignatures : undefined;
  const mapped = normalizedSignatureString(signatures?.gemini);
  if (mapped) return mapped;
  const portable = normalizedSignatureString(part.thoughtSignature);
  if (!portable) return undefined;
  return portable.startsWith('gemini:') ? portable.slice('gemini:'.length) : portable;
}

function emptyGeminiOpenAIToolCallSignatures(): GeminiOpenAIToolCallSignatures {
  return { byId: new Map(), byIndex: new Map() };
}

function mergeGeminiOpenAIToolCallSignatures(
  target: GeminiOpenAIToolCallSignatures,
  source: GeminiOpenAIToolCallSignatures
): void {
  for (const [id, signature] of source.byId) target.byId.set(id, signature);
  for (const [index, signature] of source.byIndex) target.byIndex.set(index, signature);
}

function normalizeGeminiThinkingRequest(request: unknown, modelId: string): unknown {
  if (!isRecord(request)) return request;
  const capability = geminiThinkingCapabilityForModel(modelId);
  if (capability.kind === 'unknown') return request;

  const generationConfig = isRecord(request.generationConfig) ? request.generationConfig : {};
  const sourceThinkingConfig = isRecord(generationConfig.thinkingConfig)
    ? generationConfig.thinkingConfig
    : {};
  const thinkingConfig: Record<string, unknown> = { ...sourceThinkingConfig };

  if (capability.kind === 'thinkingLevel') {
    const configuredLevel = sourceThinkingConfig.thinkingLevel;
    thinkingConfig.thinkingLevel = isGeminiThinkingLevelSupported(capability, configuredLevel)
      ? configuredLevel
      : capability.defaultLevel;
    delete thinkingConfig.thinkingBudget;
    if (thinkingConfig.includeThoughts === undefined) thinkingConfig.includeThoughts = true;
  } else {
    delete thinkingConfig.thinkingLevel;
    if (capability.kind === 'unsupported') delete thinkingConfig.thinkingBudget;
  }

  const nextGenerationConfig: Record<string, unknown> = { ...generationConfig };
  if (Object.keys(thinkingConfig).length > 0) nextGenerationConfig.thinkingConfig = thinkingConfig;
  else delete nextGenerationConfig.thinkingConfig;
  const { generationConfig: _sourceGenerationConfig, ...requestWithoutGenerationConfig } = request;
  return Object.keys(nextGenerationConfig).length > 0
    ? { ...requestWithoutGenerationConfig, generationConfig: nextGenerationConfig }
    : requestWithoutGenerationConfig;
}

function restoreGeminiToolPropertyNames(encodedRequest: unknown, sourceRequest: unknown): void {
  if (!isRecord(encodedRequest) || !isRecord(sourceRequest)) return;
  const encodedGroups = Array.isArray(encodedRequest.tools) ? encodedRequest.tools : [];
  const sourceGroups = Array.isArray(sourceRequest.tools) ? sourceRequest.tools : [];
  const sourceDeclarations = sourceGroups.flatMap((group) => {
    if (!isRecord(group) || !Array.isArray(group.functionDeclarations)) return [];
    return group.functionDeclarations.filter(isRecord);
  });
  const sourceByName = new Map(sourceDeclarations
    .filter((declaration) => typeof declaration.name === 'string')
    .map((declaration) => [declaration.name as string, declaration]));
  for (const group of encodedGroups) {
    if (!isRecord(group) || !Array.isArray(group.functionDeclarations)) continue;
    for (const declaration of group.functionDeclarations) {
      if (!isRecord(declaration) || typeof declaration.name !== 'string') continue;
      const source = sourceByName.get(declaration.name);
      if (!source?.parameters) continue;
      declaration.parameters = sanitizeGeminiFunctionSchema(source.parameters);
    }
  }
}

const GEMINI_UNSUPPORTED_SCHEMA_KEYS = new Set([
  'title',
  'default',
  'const',
  '$defs',
  'definitions',
  '$schema',
  'not',
  'if',
  'then',
  'else',
  'prefixItems',
  'additionalProperties',
  'multipleOf'
]);

function sanitizeGeminiFunctionSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizeGeminiFunctionSchema);
  if (!isRecord(value)) return value;

  const result: Record<string, unknown> = {};
  let stringifiedEnum = false;
  for (const [key, child] of Object.entries(value)) {
    if (GEMINI_UNSUPPORTED_SCHEMA_KEYS.has(key)) continue;
    if (key === 'properties' && isRecord(child)) {
      result.properties = Object.fromEntries(
        Object.entries(child).map(([propertyName, propertySchema]) => [
          propertyName,
          sanitizeGeminiFunctionSchema(propertySchema)
        ])
      );
      continue;
    }
    if ((key === 'anyOf' || key === 'oneOf' || key === 'allOf') && Array.isArray(child)) {
      const otherKeys = Object.keys(value).filter((candidate) =>
        candidate !== key && !GEMINI_UNSUPPORTED_SCHEMA_KEYS.has(candidate)
      );
      if (otherKeys.length === 0 && child.length > 0) {
        const first = sanitizeGeminiFunctionSchema(child[0]);
        if (isRecord(first)) Object.assign(result, first);
      }
      continue;
    }
    if (key === 'enum' && Array.isArray(child)) {
      result.enum = child.map((item) => String(item));
      stringifiedEnum = true;
      continue;
    }
    result[key] = sanitizeGeminiFunctionSchema(child);
  }

  if (stringifiedEnum && (result.type === 'integer' || result.type === 'number')) result.type = 'string';
  if (Array.isArray(result.required) && isRecord(result.properties)) {
    const required = result.required.filter((propertyName): propertyName is string =>
      typeof propertyName === 'string' && Object.prototype.hasOwnProperty.call(result.properties, propertyName)
    );
    if (required.length > 0) result.required = required;
    else delete result.required;
  }
  return result;
}

function fromUnifiedCompletedContent(content: UnifiedContent): MessageContent {
  const parts: ContentPart[] = [];
  for (const part of content.parts ?? []) {
    const outputItem = modelOutputItemFromValue(part);
    if (isUnifiedThoughtTextPart(part)) {
      const signature = thoughtSignatureFromPart(part);
      parts.push({
        text: part.text ?? '',
        thought: true,
        ...(signature ? { thoughtSignature: signature } : {}),
        ...(outputItem ? { outputItem } : {})
      });
      continue;
    }
    if ('text' in part && typeof part.text === 'string') {
      parts.push({ text: part.text, ...(outputItem ? { outputItem } : {}) });
      continue;
    }
    if (isUnifiedFunctionCallPart(part)) {
      const signature = thoughtSignatureFromPart(part);
      parts.push({
        ...(part.functionCall.callId ? { id: part.functionCall.callId } : {}),
        functionCall: {
          name: part.functionCall.name,
          args: part.functionCall.args ?? {}
        },
        ...(signature ? { thoughtSignature: signature } : {}),
        ...(outputItem ? { outputItem } : {})
      });
    }
  }
  return { role: 'model', parts };
}

export function emitUnifiedChunk(requestId: string, chunk: UnifiedLLMStreamChunk, emit: Emit): void {
  const outputItem = modelOutputItemFromValue(chunk);
  const text = chunk.textDelta ?? visibleTextFromParts(chunk.partsDelta ?? []);
  if (text) emit({
    type: LlmEventType.Delta,
    payload: { requestId, text, ...(outputItem ? { outputItem } : {}) }
  });

  const argumentDeltas = (chunk as LimCodeOpenAIResponsesStreamChunk).toolCallArgumentDeltas ?? [];
  if (argumentDeltas.length > 0) {
    emit({
      type: LlmEventType.ToolCallDelta,
      payload: {
        requestId,
        ...(outputItem ? { outputItem } : {}),
        calls: argumentDeltas.map((delta) => ({
          id: delta.callId,
          ...(delta.name ? { name: delta.name } : {}),
          argumentsDelta: delta.argumentsDelta,
          ...(delta.replace ? { replace: true } : {}),
          ...(delta.streamIndex ? { streamIndex: delta.streamIndex } : {})
        }))
      }
    });
  }

  const callParts = [
    ...(chunk.functionCalls ?? []),
    ...(chunk.partsDelta ?? []).filter(isUnifiedFunctionCallPart)
  ];
  const stableCallIndexes = new Map<string, number>();
  const calls: Array<{ id: string; name: string; argsJson: string; thoughtSignature?: string }> = [];
  callParts.forEach((part, index) => {
    const stableCallId = part.functionCall.callId;
    const thoughtSignature = thoughtSignatureFromPart(part);
    const candidate = {
      id: stableCallId ?? `tool_call_${index}`,
      name: part.functionCall.name,
      argsJson: stringifyJson(part.functionCall.args ?? {}),
      ...(thoughtSignature ? { thoughtSignature } : {})
    };
    const existingIndex = stableCallId ? stableCallIndexes.get(stableCallId) : undefined;
    if (existingIndex === undefined) {
      if (stableCallId) stableCallIndexes.set(stableCallId, calls.length);
      calls.push(candidate);
      return;
    }

    const existing = calls[existingIndex];
    calls[existingIndex] = {
      id: existing.id,
      name: existing.name.trim() ? existing.name : candidate.name,
      argsJson: candidate.argsJson.length > existing.argsJson.length ? candidate.argsJson : existing.argsJson,
      ...(existing.thoughtSignature || candidate.thoughtSignature
        ? { thoughtSignature: existing.thoughtSignature ?? candidate.thoughtSignature }
        : {})
    };
  });

  if (calls.length > 0) {
    emit({
      type: LlmEventType.ToolCallPreviewDone,
      payload: { requestId, callIds: calls.map((call) => call.id).filter((id): id is string => !!id) }
    });
    emit({ type: LlmEventType.ToolCall, payload: {
      requestId,
      ...(outputItem ? { outputItem } : {}),
      calls
    } });
  }

  const outputItemDone = modelOutputItemDoneFromChunk(chunk);
  if (outputItemDone) {
    emit({
      type: LlmEventType.OutputItemDone,
      payload: { requestId, outputItem: outputItemDone }
    });
  }
}

function emitUnifiedResponse(requestId: string, response: UnifiedLLMResponse, emit: Emit): void {
  const parts = response.content?.parts ?? [];
  const visibleText = visibleTextFromParts(parts);
  if (visibleText) emit({ type: LlmEventType.Delta, payload: { requestId, text: visibleText } });

  const thoughtParts = parts.filter(isUnifiedThoughtTextPart);
  for (const part of thoughtParts) {
    const text = typeof (part as { text?: unknown }).text === 'string' ? (part as { text: string }).text : '';
    const signature = thoughtSignatureFromPart(part);
    const thoughtStartedAt = Date.now();
    if (text) emit({ type: LlmEventType.ThoughtDelta, payload: { requestId, text, thoughtStartedAt, thoughtElapsedMs: 0, ...(signature ? { thoughtSignature: signature } : {}) } });
    if (text || signature) emit({ type: LlmEventType.ThoughtDone, payload: { requestId, thoughtStartedAt, thoughtDurationMs: 0, ...(signature ? { thoughtSignature: signature } : {}) } });
  }

  const calls = parts.filter(isUnifiedFunctionCallPart).map((part, index) => {
    const thoughtSignature = thoughtSignatureFromPart(part);
    return {
      id: part.functionCall.callId ?? `tool_call_${index}`,
      name: part.functionCall.name,
      argsJson: stringifyJson(part.functionCall.args ?? {}),
      ...(thoughtSignature ? { thoughtSignature } : {})
    };
  });
  if (calls.length > 0) emit({ type: LlmEventType.ToolCall, payload: { requestId, calls } });
}

interface LlmDoneTiming {
  createdAt: number;
  streamOutputDurationMs?: number;
}

function createDoneTiming(
  firstChunkAt: number | undefined,
  finishedAt = Date.now(),
  firstChunkMark?: number,
  finishedMark?: number,
  _streamChunkCount = 0
): LlmDoneTiming {
  const rawDurationMs = firstChunkAt === undefined
    ? undefined
    : firstChunkMark !== undefined && finishedMark !== undefined
      ? finishedMark - firstChunkMark
      : finishedAt - firstChunkAt;

  const streamOutputDurationMs = rawDurationMs !== undefined
    ? Math.max(0, Math.round(rawDurationMs))
    : undefined;

  return {
    createdAt: firstChunkAt ?? finishedAt,
    ...(streamOutputDurationMs !== undefined ? { streamOutputDurationMs } : {})
  };
}

function nowMonotonicMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function usageMetadataFromChunk(chunk: UnifiedLLMStreamChunk): LlmUsageMetadataRecord | undefined {
  const cleaned = stripUndefined(chunk.usageMetadata);
  return isRecord(cleaned) && Object.keys(cleaned).length > 0
    ? cleaned as LlmUsageMetadataRecord
    : undefined;
}

function mergeUsageMetadata(
  previous: LlmUsageMetadataRecord | undefined,
  next: LlmUsageMetadataRecord
): LlmUsageMetadataRecord {
  if (!previous) return next;
  return { ...previous, ...next };
}

function stripUndefined(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripUndefined);
  if (!isRecord(value)) return value;

  const result: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (child === undefined) continue;
    result[key] = stripUndefined(child);
  }
  return result;
}

function modelOutputItemFromValue(value: unknown): ModelOutputItemReference | undefined {
  const source = isRecord(value) ? isRecord(value.outputItem) ? value.outputItem : undefined : undefined;
  if (!source || typeof source.id !== 'string' || !source.id.trim()
    || typeof source.ordinal !== 'number' || !Number.isSafeInteger(source.ordinal) || source.ordinal < 0) {
    return undefined;
  }
  const phase = source.phase === 'commentary' || source.phase === 'final_answer'
    ? source.phase
    : undefined;
  return {
    id: source.id,
    ordinal: source.ordinal,
    ...(phase ? { phase } : {})
  };
}

function modelOutputItemDoneFromChunk(chunk: UnifiedLLMStreamChunk): ModelOutputItemReference | undefined {
  const value = (chunk as LimCodeOpenAIResponsesStreamChunk).outputItemDone;
  return value ? modelOutputItemFromValue({ outputItem: value }) : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function hasStreamTimingChunk(chunk: UnifiedLLMStreamChunk): boolean {
  return hasStreamOutput(chunk)
    || hasThoughtOutput(chunk)
    || ((chunk as LimCodeOpenAIResponsesStreamChunk).toolCallArgumentDeltas?.length ?? 0) > 0;
}

function hasThoughtOutput(chunk: UnifiedLLMStreamChunk): boolean {
  return !!thoughtSignatureFromChunk(chunk) || (chunk.partsDelta ?? []).some((part) => isUnifiedThoughtTextPart(part) && (!!part.text || !!thoughtSignatureFromPart(part)));
}

function hasStreamOutput(chunk: UnifiedLLMStreamChunk): boolean {
  if (chunk.textDelta || visibleTextFromParts(chunk.partsDelta ?? [])) return true;
  if ((chunk.functionCalls?.length ?? 0) > 0) return true;
  return (chunk.partsDelta ?? []).some(isUnifiedFunctionCallPart);
}

function visibleTextFromParts(parts: UnifiedPart[]): string {
  return parts.map((part) => 'text' in part && (part as { thought?: unknown }).thought !== true ? part.text ?? '' : '').join('');
}

interface ActiveThoughtBlock {
  startedAt: number;
  progressTimer?: ReturnType<typeof setInterval>;
  thoughtSignature?: string;
  outputItem?: ModelOutputItemReference;
}

function emitThoughtDeltas(requestId: string, current: ActiveThoughtBlock | undefined, chunk: UnifiedLLMStreamChunk, at: number, emit: Emit): ActiveThoughtBlock | undefined {
  let block = current;
  const outputItem = modelOutputItemFromValue(chunk);
  const chunkSignature = thoughtSignatureFromChunk(chunk);
  if (chunkSignature) {
    block ??= createActiveThoughtBlock(requestId, at, emit, outputItem);
    block.outputItem ??= outputItem;
    block.thoughtSignature = chunkSignature;
  }
  for (const part of chunk.partsDelta ?? []) {
    if (!isUnifiedThoughtTextPart(part)) continue;
    const text = part.text ?? '';
    block ??= createActiveThoughtBlock(requestId, at, emit, outputItem);
    block.outputItem ??= outputItem;
    const signature = thoughtSignatureFromPart(part);
    if (signature) block.thoughtSignature = signature;
    if (!text) continue;
    emit({
      type: LlmEventType.ThoughtDelta,
      payload: {
        requestId,
        text,
        thoughtStartedAt: block.startedAt,
        thoughtElapsedMs: Math.max(0, at - block.startedAt),
        ...(block.outputItem ? { outputItem: block.outputItem } : {}),
        ...(signature ? { thoughtSignature: signature } : {})
      }
    });
  }
  return block;
}

function createActiveThoughtBlock(
  requestId: string,
  startedAt: number,
  emit: Emit,
  outputItem?: ModelOutputItemReference
): ActiveThoughtBlock {
  const block: ActiveThoughtBlock = { startedAt, ...(outputItem ? { outputItem } : {}) };
  block.progressTimer = setInterval(() => {
    emit({
      type: LlmEventType.ThoughtProgress,
      payload: {
        requestId,
        thoughtStartedAt: block.startedAt,
        thoughtElapsedMs: Math.max(0, Date.now() - block.startedAt),
        ...(block.outputItem ? { outputItem: block.outputItem } : {}),
        ...(block.thoughtSignature ? { thoughtSignature: block.thoughtSignature } : {})
      }
    });
  }, THOUGHT_PROGRESS_INTERVAL_MS);
  return block;
}

function disposeThoughtBlock(block: ActiveThoughtBlock): undefined {
  if (block.progressTimer) clearInterval(block.progressTimer);
  return undefined;
}

function shouldCloseThoughtBlock(chunk: UnifiedLLMStreamChunk): boolean {
  return (chunk as LimCodeOpenAIResponsesStreamChunk).reasoningItemDone === true
    || !!chunk.finishReason
    || hasStreamOutput(chunk)
    || hasThoughtSignatureOnlyOutput(chunk);
}

function finishThoughtBlock(requestId: string, block: ActiveThoughtBlock, finishedAt: number, emit: Emit): undefined {
  disposeThoughtBlock(block);
  emit({
    type: LlmEventType.ThoughtDone,
    payload: {
      requestId,
      thoughtStartedAt: block.startedAt,
      thoughtDurationMs: Math.max(0, finishedAt - block.startedAt),
      ...(block.outputItem ? { outputItem: block.outputItem } : {}),
      ...(block.thoughtSignature ? { thoughtSignature: block.thoughtSignature } : {})
    }
  });
  return undefined;
}

function isUnifiedThoughtTextPart(part: UnifiedPart): part is UnifiedPart & { text?: string; thought?: unknown } {
  return (part as { thought?: unknown }).thought === true;
}

function isUnifiedFunctionCallPart(part: UnifiedPart): part is Extract<UnifiedPart, { functionCall: unknown }> {
  return 'functionCall' in part;
}

function thoughtSignatureFromPart(part: UnifiedPart): string | undefined {
  const record = part as { thoughtSignature?: unknown; thoughtSignatures?: unknown };
  return normalizedSignatureString(record.thoughtSignature) ?? portableThoughtSignatureFromMap(record.thoughtSignatures);
}

function thoughtSignatureFromChunk(chunk: UnifiedLLMStreamChunk): string | undefined {
  const record = chunk as { thoughtSignature?: unknown; thoughtSignatures?: unknown };
  return normalizedSignatureString(record.thoughtSignature) ?? portableThoughtSignatureFromMap(record.thoughtSignatures);
}

function hasThoughtSignatureOnlyOutput(chunk: UnifiedLLMStreamChunk): boolean {
  const parts = chunk.partsDelta ?? [];
  const hasSignature = !!thoughtSignatureFromChunk(chunk) || parts.some((part) => isUnifiedThoughtTextPart(part) && !!thoughtSignatureFromPart(part));
  if (!hasSignature) return false;
  return !parts.some((part) => isUnifiedThoughtTextPart(part) && !!part.text);
}

function normalizedSignatureString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

const THOUGHT_SIGNATURE_PROVIDER_ORDER = ['gemini', 'claude', 'openai-compatible', 'openai-responses'] as const;

function portableThoughtSignatureFromMap(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  for (const provider of THOUGHT_SIGNATURE_PROVIDER_ORDER) {
    const signature = portableThoughtSignatureFromEntry(provider, value[provider]);
    if (signature) return signature;
  }
  for (const [provider, raw] of Object.entries(value)) {
    const signature = portableThoughtSignatureFromEntry(provider, raw);
    if (signature) return signature;
  }
  return undefined;
}

function portableThoughtSignatureFromEntry(provider: string, raw: unknown): string | undefined {
  const signature = normalizedSignatureString(raw);
  if (!signature) return undefined;
  const parsedSignature = parsePortableThoughtSignature(signature);
  if (parsedSignature) return `${parsedSignature.provider}:${parsedSignature.value}`;
  const normalizedProvider = normalizedSignatureProvider(provider);
  return normalizedProvider ? `${normalizedProvider}:${signature}` : undefined;
}

function thoughtSignaturesFromPortableSignature(signature: string | undefined): Record<string, string> | undefined {
  const normalized = normalizedSignatureString(signature);
  if (!normalized) return undefined;
  const parsed = parsePortableThoughtSignature(normalized);
  return parsed ? { [parsed.provider]: parsed.value } : undefined;
}

function parsePortableThoughtSignature(signature: string): { provider: string; value: string } | undefined {
  const colonIndex = signature.indexOf(':');
  if (colonIndex <= 0) return undefined;
  const provider = normalizedSignatureProvider(signature.slice(0, colonIndex));
  const value = signature.slice(colonIndex + 1).trim();
  if (!provider || !value) return undefined;
  return { provider, value };
}

function normalizedSignatureProvider(provider: string): string | undefined {
  const normalized = provider.trim().toLowerCase();
  if (!normalized || normalized === 'openai' || !/^[a-z0-9_-]+$/.test(normalized)) return undefined;
  return normalized;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>;
  return { value };
}

function isFunctionParameters(value: unknown): value is UnifiedFunctionDeclaration['parameters'] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  return (value as { type?: unknown }).type === 'object';
}

function stringifyJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return String(value);
  }
}

function normalizeOptionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed || undefined;
}

async function resolveMaybe<T, TArg = void>(value: MaybeProvider<T, TArg>, arg?: TArg): Promise<T | undefined> {
  if (typeof value === 'function') return (value as (input: TArg | undefined) => T | undefined | Promise<T | undefined>)(arg);
  return value;
}

async function importUnifiedLlmProvider(): Promise<UnifiedModule> {
  const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<UnifiedModule>;
  return dynamicImport('unified-llm-provider');
}
function createAbortError(message: string): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function isRequestAbort(signal?: AbortSignal): boolean {
  // 只在本请求自己的 AbortController 被触发时静默取消。
  // 某些网络层会把 ECONNRESET / socket hang up / 超时包装成 AbortError；
  // 如果不校验 signal.aborted，这类真实失败会被误判为用户取消，导致压缩块一直停在 running。
  return signal?.aborted === true;
}

function compactRequestDebugInfo(request: LlmCompactRequest): Record<string, unknown> {
  return {
    requestId: request.id,
    blockId: request.blockId,
    conversationId: request.conversationId,
    invocationId: request.invocationId,
    methodKind: request.methodKind,
    methodConfigId: request.methodConfigId,
    sourceHash: request.sourceHash,
    contentCount: request.contents.length,
    segmentCount: request.segments?.length ?? 0,
    priorSummaryCount: request.priorSummaryContents?.length ?? 0
  };
}

function errorDebugInfo(error: unknown): unknown {
  return toPlainJsonLike(error);
}

function abortReasonText(reason: unknown): string | undefined {
  if (reason === undefined || reason === null) return undefined;
  if (reason instanceof Error) return `${reason.name}: ${reason.message}`;
  return String(reason);
}

function logCompressionDebug(stage: string, payload: Record<string, unknown>): void {
  const log = /throw|error|cancel|abort/i.test(stage) ? console.warn : console.info;
  log('[LimCode][Compression][Provider]', stage, payload);
}

function emitLlmStarted(emit: Emit, requestId: string, invocationId: string | undefined, model: string | undefined): void {
  emit({ type: LlmEventType.Started, payload: { requestId, ...(invocationId ? { invocationId } : {}), ...(model ? { model } : {}), startedAt: Date.now() } });
}

function emitLlmError(
  emit: Emit,
  requestId: string,
  message: string,
  rawError?: LlmRawErrorInfoRecord,
  extra: { retryAttempt?: number; retryMaxAttempts?: number; createdAt?: number; streamOutputDurationMs?: number } = {}
): void {
  emit({
    type: LlmEventType.Error,
    payload: {
      requestId,
      message,
      ...(rawError ? { rawError } : {}),
      ...(extra.retryAttempt !== undefined ? { retryAttempt: extra.retryAttempt } : {}),
      ...(extra.retryMaxAttempts !== undefined ? { retryMaxAttempts: extra.retryMaxAttempts } : {}),
      ...(extra.createdAt !== undefined ? { createdAt: extra.createdAt } : {}),
      ...(extra.streamOutputDurationMs !== undefined ? { streamOutputDurationMs: extra.streamOutputDurationMs } : {})
    }
  });
}

function emitLlmRetryScheduled(emit: Emit, requestId: string, message: string, rawError: LlmRawErrorInfoRecord | undefined, retryAttempt: number, retryMaxAttempts: number, retryDelayMs: number): void {
  emit({ type: LlmEventType.RetryScheduled, payload: { requestId, message, retryAttempt, retryMaxAttempts, retryDelayMs, createdAt: Date.now(), ...(rawError ? { rawError } : {}) } });
}

function emitLlmRetryStarted(emit: Emit, requestId: string, message: string, rawError: LlmRawErrorInfoRecord | undefined, retryAttempt: number, retryMaxAttempts: number): void {
  emit({ type: LlmEventType.RetryStarted, payload: { requestId, message, retryAttempt, retryMaxAttempts, createdAt: Date.now(), ...(rawError ? { rawError } : {}) } });
}

function emitLlmRetryCancelled(emit: Emit, requestId: string, message: string, retryAttempt: number, retryMaxAttempts: number, rawError?: LlmRawErrorInfoRecord): void {
  emit({ type: LlmEventType.RetryCancelled, payload: { requestId, message, retryAttempt, retryMaxAttempts, createdAt: Date.now(), ...(rawError ? { rawError } : {}) } });
}

function emitLlmRetryRecovered(emit: Emit, requestId: string, message: string, retryAttempt: number, retryMaxAttempts: number): void {
  emit({ type: LlmEventType.RetryRecovered, payload: { requestId, message, retryAttempt, retryMaxAttempts, createdAt: Date.now() } });
}

function assertNever(value: never): never {
  throw new Error(`Unexpected content part: ${String(value)}`);
}
