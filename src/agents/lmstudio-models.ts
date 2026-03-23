import type { ModelDefinitionConfig } from "../config/types.models.js";
import { fetchWithSsrFGuard } from "../infra/net/fetch-guard.js";
import type { SsrFPolicy } from "../infra/net/ssrf.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import {
  LMSTUDIO_DEFAULT_BASE_URL,
  LMSTUDIO_DEFAULT_LOAD_CONTEXT_LENGTH,
} from "./lmstudio-defaults.js";
import { buildLmstudioAuthHeaders } from "./lmstudio-runtime.js";
import {
  SELF_HOSTED_DEFAULT_CONTEXT_WINDOW,
  SELF_HOSTED_DEFAULT_COST,
  SELF_HOSTED_DEFAULT_MAX_TOKENS,
} from "./self-hosted-provider-defaults.js";

const log = createSubsystemLogger("agents/lmstudio");

export type LmstudioModelWire = {
  type?: "llm" | "embedding";
  key?: string;
  display_name?: string;
  max_context_length?: number;
  format?: "gguf" | "mlx" | null;
  capabilities?: {
    vision?: boolean;
    trained_for_tool_use?: boolean;
    reasoning?: LmstudioReasoningCapabilityWire;
  };
  loaded_instances?: Array<{
    id?: string;
    config?: {
      context_length?: number;
    } | null;
  } | null>;
};

type LmstudioReasoningCapabilityWire = {
  allowed_options?: unknown;
  default?: unknown;
};

type LmstudioModelsResponseWire = {
  models?: LmstudioModelWire[];
};

type LmstudioLoadResponse = {
  status?: string;
};

type FetchLmstudioModelsResult = {
  reachable: boolean;
  status?: number;
  models: LmstudioModelWire[];
  error?: unknown;
};

function normalizeReasoningOption(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return normalized.length > 0 ? normalized : null;
}

function isReasoningEnabledOption(value: unknown): boolean {
  const normalized = normalizeReasoningOption(value);
  if (!normalized) {
    return false;
  }
  return normalized !== "off";
}

/**
 * Resolves LM Studio reasoning support from capabilities payloads.
 * Defaults to false when the server omits reasoning metadata.
 */
export function resolveLmstudioReasoningCapability(
  entry: Pick<LmstudioModelWire, "capabilities">,
): boolean {
  const reasoning = entry.capabilities?.reasoning;
  if (reasoning === undefined || reasoning === null) {
    return false;
  }
  const allowedOptionsRaw = reasoning.allowed_options;
  const allowedOptions = Array.isArray(allowedOptionsRaw)
    ? allowedOptionsRaw
        .map((option) => normalizeReasoningOption(option))
        .filter((option): option is string => option !== null)
    : [];
  if (allowedOptions.length > 0) {
    return allowedOptions.some((option) => isReasoningEnabledOption(option));
  }
  return isReasoningEnabledOption(reasoning.default);
}

/**
 * Reads loaded LM Studio instances and returns the largest valid context window.
 * Returns null when no usable loaded context is present.
 */
export function resolveLoadedContextWindow(
  entry: Pick<LmstudioModelWire, "loaded_instances">,
): number | null {
  const loadedInstances = Array.isArray(entry.loaded_instances) ? entry.loaded_instances : [];
  let contextWindow: number | null = null;
  for (const instance of loadedInstances) {
    // Discovery payload is external JSON, so tolerate malformed entries.
    const length = instance?.config?.context_length;
    if (length === undefined || !Number.isFinite(length) || length <= 0) {
      continue;
    }
    const normalized = Math.floor(length);
    contextWindow = contextWindow === null ? normalized : Math.max(contextWindow, normalized);
  }
  return contextWindow;
}

/**
 * Normalizes a server path by stripping trailing slash and inference suffixes.
 *
 * LM Studio users often copy their inference URL (e.g. "http://localhost:1234/v1") instead
 * of the server root. This function strips a trailing "/v1" or "/api/v1" so the caller always
 * receives a clean root base URL. The expected input is the server root without any API version
 * path (e.g. "http://localhost:1234").
 */
function normalizeUrlPath(pathname: string): string {
  const trimmed = pathname.replace(/\/+$/, "");
  if (!trimmed) {
    return "";
  }
  return trimmed.replace(/\/api\/v1$/i, "").replace(/\/v1$/i, "");
}

/** Resolves LM Studio server base URL (without /v1 or /api/v1). */
export function resolveLmstudioServerBase(configuredBaseUrl?: string): string {
  // Use configured value when present; otherwise target local LM Studio default.
  const configured = configuredBaseUrl?.trim();
  const resolved = configured && configured.length > 0 ? configured : LMSTUDIO_DEFAULT_BASE_URL;
  try {
    const parsed = new URL(resolved);
    const pathname = normalizeUrlPath(parsed.pathname);
    parsed.pathname = pathname.length > 0 ? pathname : "/";
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString().replace(/\/$/, "");
  } catch {
    const trimmed = resolved.replace(/\/+$/, "");
    const normalized = normalizeUrlPath(trimmed);
    return normalized.length > 0 ? normalized : LMSTUDIO_DEFAULT_BASE_URL;
  }
}

/** Resolves LM Studio inference base URL and always appends /v1. */
export function resolveLmstudioInferenceBase(configuredBaseUrl?: string): string {
  const serverBase = resolveLmstudioServerBase(configuredBaseUrl);
  return `${serverBase}/v1`;
}

async function fetchLmstudioEndpoint(params: {
  url: string;
  init?: RequestInit;
  timeoutMs: number;
  fetchImpl?: typeof fetch;
  ssrfPolicy?: SsrFPolicy;
  auditContext: string;
}): Promise<{ response: Response; release: () => Promise<void> }> {
  if (params.ssrfPolicy) {
    return await fetchWithSsrFGuard({
      url: params.url,
      init: params.init,
      timeoutMs: params.timeoutMs,
      fetchImpl: params.fetchImpl,
      policy: params.ssrfPolicy,
      auditContext: params.auditContext,
    });
  }
  const fetchFn = params.fetchImpl ?? fetch;
  return {
    response: await fetchFn(params.url, {
      ...params.init,
      signal: AbortSignal.timeout(params.timeoutMs),
    }),
    release: async () => {},
  };
}

/** Fetches /api/v1/models and reports transport reachability separately from HTTP status. */
export async function fetchLmstudioModels(params: {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  timeoutMs?: number;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
}): Promise<FetchLmstudioModelsResult> {
  const baseUrl = resolveLmstudioServerBase(params.baseUrl);
  const timeoutMs = params.timeoutMs ?? 5000;
  try {
    const { response, release } = await fetchLmstudioEndpoint({
      url: `${baseUrl}/api/v1/models`,
      init: {
        headers: buildLmstudioAuthHeaders({
          apiKey: params.apiKey,
          headers: params.headers,
        }),
      },
      timeoutMs,
      fetchImpl: params.fetchImpl,
      ssrfPolicy: params.ssrfPolicy,
      auditContext: "lmstudio-model-discovery",
    });
    try {
      if (!response.ok) {
        return {
          reachable: true,
          status: response.status,
          models: [],
        };
      }
      // External service payload is untrusted JSON; parse with a permissive wire type.
      const payload = (await response.json()) as LmstudioModelsResponseWire;
      return {
        reachable: true,
        status: response.status,
        models: Array.isArray(payload.models) ? payload.models : [],
      };
    } finally {
      await release();
    }
  } catch (error) {
    return {
      reachable: false,
      models: [],
      error,
    };
  }
}

function buildLmstudioModelName(model: {
  displayName: string;
  format: "gguf" | "mlx" | null;
  vision: boolean;
  trainedForToolUse: boolean;
  loaded: boolean;
}): string {
  const tags: string[] = [];
  if (model.format === "mlx") {
    tags.push("MLX");
  } else if (model.format === "gguf") {
    tags.push("GGUF");
  }
  if (model.vision) {
    tags.push("vision");
  }
  if (model.trainedForToolUse) {
    tags.push("tool-use");
  }
  if (model.loaded) {
    tags.push("loaded");
  }
  if (tags.length === 0) {
    return model.displayName;
  }
  return `${model.displayName} (${tags.join(", ")})`;
}

/**
 * Base model fields extracted from a single LM Studio wire entry.
 * Shared by the setup layer (persists simple names to config) and the runtime
 * discovery path (which enriches the name with format/state tags).
 */
export type LmstudioModelBase = {
  id: string;
  displayName: string;
  format: "gguf" | "mlx" | null;
  vision: boolean;
  trainedForToolUse: boolean;
  loaded: boolean;
  reasoning: boolean;
  input: ModelDefinitionConfig["input"];
  cost: ModelDefinitionConfig["cost"];
  contextWindow: number;
  maxTokens: number;
};

/**
 * Maps a single LM Studio wire entry to its base model fields.
 * Returns null for non-LLM entries or entries with no usable key.
 *
 * Shared by both the setup layer (persists simple names to config) and the
 * runtime discovery path (which enriches the name with format/state tags via
 * buildLmstudioModelName).
 */
export function mapLmstudioWireEntry(entry: LmstudioModelWire): LmstudioModelBase | null {
  if (entry.type !== "llm") {
    return null;
  }
  const id = entry.key?.trim() ?? "";
  if (!id) {
    return null;
  }
  const loadedContextWindow = resolveLoadedContextWindow(entry);
  const contextWindow =
    loadedContextWindow ??
    (entry.max_context_length !== undefined &&
    Number.isFinite(entry.max_context_length) &&
    entry.max_context_length > 0
      ? entry.max_context_length
      : SELF_HOSTED_DEFAULT_CONTEXT_WINDOW);
  const rawDisplayName = entry.display_name?.trim();
  return {
    id,
    displayName: rawDisplayName && rawDisplayName.length > 0 ? rawDisplayName : id,
    format: entry.format ?? null,
    vision: entry.capabilities?.vision === true,
    trainedForToolUse: entry.capabilities?.trained_for_tool_use === true,
    // Use the same validity check as resolveLoadedContextWindow so malformed entries
    // like [null, {}] don't produce a false positive "loaded" tag.
    loaded: loadedContextWindow !== null,
    reasoning: resolveLmstudioReasoningCapability(entry),
    input: entry.capabilities?.vision ? ["text", "image"] : ["text"],
    cost: SELF_HOSTED_DEFAULT_COST,
    contextWindow,
    maxTokens: Math.max(1, Math.min(contextWindow, SELF_HOSTED_DEFAULT_MAX_TOKENS)),
  };
}

type DiscoverLmstudioModelsParams = {
  baseUrl: string;
  apiKey: string;
  headers?: Record<string, string>;
  quiet: boolean;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
};

/** Discovers LLM models from LM Studio and maps them to OpenClaw model definitions. */
export async function discoverLmstudioModels(
  params: DiscoverLmstudioModelsParams,
): Promise<ModelDefinitionConfig[]> {
  const fetched = await fetchLmstudioModels({
    baseUrl: params.baseUrl,
    apiKey: params.apiKey,
    headers: params.headers,
    fetchImpl: params.fetchImpl,
  });
  const quiet = params.quiet;
  if (!fetched.reachable) {
    if (!quiet) {
      log.debug(`Failed to discover LM Studio models: ${String(fetched.error)}`);
    }
    return [];
  }
  if (fetched.status !== undefined && fetched.status >= 400) {
    if (!quiet) {
      log.debug(`Failed to discover LM Studio models: ${fetched.status}`);
    }
    return [];
  }
  const models = fetched.models;
  if (models.length === 0) {
    if (!quiet) {
      log.debug("No LM Studio models found on local instance");
    }
    return [];
  }

  return models
    .map((entry): ModelDefinitionConfig | null => {
      const base = mapLmstudioWireEntry(entry);
      if (!base) {
        return null;
      }
      return {
        id: base.id,
        // Runtime display: include format/vision/tool-use/loaded tags in the name.
        name: buildLmstudioModelName(base),
        reasoning: base.reasoning,
        input: base.input,
        cost: base.cost,
        contextWindow: base.contextWindow,
        maxTokens: base.maxTokens,
      };
    })
    .filter((entry): entry is ModelDefinitionConfig => entry !== null);
}

/** Ensures a model is loaded in LM Studio before first real inference/embedding call. */
export async function ensureLmstudioModelLoaded(params: {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  modelKey: string;
  timeoutMs?: number;
  /** Injectable fetch implementation; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
}): Promise<void> {
  const modelKey = params.modelKey.trim();
  if (!modelKey) {
    throw new Error("LM Studio model key is required");
  }

  const timeoutMs = params.timeoutMs ?? 30_000;
  const baseUrl = resolveLmstudioServerBase(params.baseUrl);
  const preflight = await fetchLmstudioModels({
    baseUrl,
    apiKey: params.apiKey,
    headers: params.headers,
    ssrfPolicy: params.ssrfPolicy,
    timeoutMs,
    fetchImpl: params.fetchImpl,
  });
  if (!preflight.reachable) {
    throw new Error(`LM Studio model discovery failed: ${String(preflight.error)}`);
  }
  if (preflight.status !== undefined && preflight.status >= 400) {
    throw new Error(`LM Studio model discovery failed (${preflight.status})`);
  }
  const matchingModel = preflight.models.find((entry) => entry.key?.trim() === modelKey);
  if (matchingModel && resolveLoadedContextWindow(matchingModel) !== null) {
    return;
  }
  const advertisedContextLimit =
    matchingModel?.max_context_length !== undefined &&
    Number.isFinite(matchingModel.max_context_length) &&
    matchingModel.max_context_length > 0
      ? Math.floor(matchingModel.max_context_length)
      : null;
  const requestedContextLength =
    advertisedContextLimit === null
      ? LMSTUDIO_DEFAULT_LOAD_CONTEXT_LENGTH
      : Math.min(LMSTUDIO_DEFAULT_LOAD_CONTEXT_LENGTH, advertisedContextLimit);

  const { response, release } = await fetchLmstudioEndpoint({
    url: `${baseUrl}/api/v1/models/load`,
    init: {
      method: "POST",
      headers: buildLmstudioAuthHeaders({
        apiKey: params.apiKey,
        headers: params.headers,
        json: true,
      }),
      body: JSON.stringify({
        model: modelKey,
        // Ask LM Studio to load with our default target, capped to the model's own limit.
        context_length: requestedContextLength,
      }),
    },
    timeoutMs,
    fetchImpl: params.fetchImpl,
    ssrfPolicy: params.ssrfPolicy,
    auditContext: "lmstudio-model-load",
  });
  try {
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`LM Studio model load failed (${response.status})${body ? `: ${body}` : ""}`);
    }
    const payload = (await response.json()) as LmstudioLoadResponse;
    if (typeof payload.status === "string" && payload.status.toLowerCase() !== "loaded") {
      throw new Error(`LM Studio model load returned unexpected status: ${payload.status}`);
    }
  } finally {
    await release();
  }
}
