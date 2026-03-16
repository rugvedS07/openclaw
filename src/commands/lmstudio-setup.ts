import {
  LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
  LMSTUDIO_DEFAULT_INFERENCE_BASE_URL,
  LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER,
  LMSTUDIO_MODEL_PLACEHOLDER,
  LMSTUDIO_DEFAULT_BASE_URL,
  LMSTUDIO_PROVIDER_LABEL,
  LMSTUDIO_DEFAULT_MODEL_ID,
  LMSTUDIO_PROVIDER_ID as PROVIDER_ID,
} from "../agents/lmstudio-defaults.js";
import {
  fetchLmstudioModels,
  mapLmstudioWireEntry,
  type LmstudioModelWire,
  resolveLmstudioInferenceBase,
} from "../agents/lmstudio-models.js";
import { resolveLmstudioProviderHeaders } from "../agents/lmstudio-runtime.js";
import { resolveUsableCustomProviderApiKey } from "../agents/model-auth.js";
import { buildLmstudioProvider } from "../agents/models-config.providers.discovery.js";
import { projectConfigOntoRuntimeSourceSnapshot, type OpenClawConfig } from "../config/config.js";
import { createConfigRuntimeEnv } from "../config/env-vars.js";
import type { ModelDefinitionConfig, ModelProviderConfig } from "../config/types.models.js";
import { resolveSecretInputRef, type SecretInput } from "../config/types.secrets.js";
import type {
  ProviderAuthMethodNonInteractiveContext,
  ProviderAuthResult,
  ProviderDiscoveryContext,
} from "../plugins/types.js";
import { resolveSecretInputString } from "../secrets/resolve-secret-input-string.js";
import { normalizeOptionalSecretInput } from "../utils/normalize-secret-input.js";
import { WizardCancelledError } from "../wizard/prompts.js";
import type { WizardPrompter } from "../wizard/prompts.js";
import { ensureApiKeyFromEnvOrPrompt } from "./auth-choice.apply-helpers.js";
import { buildApiKeyCredential } from "./onboard-auth.credentials.js";
import type { SecretInputMode } from "./onboard-types.js";
import { configureOpenAICompatibleSelfHostedProviderNonInteractive } from "./self-hosted-provider-setup.js";

type ProviderPromptText = (params: {
  message: string;
  initialValue?: string;
  placeholder?: string;
  validate?: (value: string | undefined) => string | undefined;
}) => Promise<string | undefined>;

type ProviderPromptNote = (message: string, title?: string) => Promise<void> | void;

function shouldUseLmstudioApiKeyPlaceholder(params: {
  hasModels: boolean;
  resolvedApiKey: ModelProviderConfig["apiKey"] | undefined;
}): boolean {
  return params.hasModels && !params.resolvedApiKey;
}

function resolveLmstudioProviderAuthMode(
  apiKey: ModelProviderConfig["apiKey"] | undefined,
): ModelProviderConfig["auth"] | undefined {
  const normalized = normalizeOptionalSecretInput(apiKey);
  if (normalized !== undefined) {
    return normalized && normalized !== LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER ? "api-key" : undefined;
  }
  return resolveSecretInputRef({ value: apiKey }).ref ? "api-key" : undefined;
}

/** Keeps explicit model entries first and appends unique discovered entries. */
function mergeDiscoveredModels(params: {
  explicitModels?: ModelDefinitionConfig[];
  discoveredModels?: ModelDefinitionConfig[];
}): ModelDefinitionConfig[] {
  const explicitModels = Array.isArray(params.explicitModels) ? params.explicitModels : [];
  const discoveredModels = Array.isArray(params.discoveredModels) ? params.discoveredModels : [];
  if (explicitModels.length === 0) {
    return discoveredModels;
  }
  if (discoveredModels.length === 0) {
    return explicitModels;
  }

  const merged = [...explicitModels];
  const seen = new Set(explicitModels.map((model) => model.id.trim()).filter(Boolean));
  for (const model of discoveredModels) {
    const id = model.id.trim();
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    merged.push(model);
  }
  return merged;
}

/**
 * Maps LM Studio discovery payload to provider catalog entries.
 * Uses simple display names (no runtime tags) since these entries are persisted to config.
 */
function mapFetchedLmstudioModelsToCatalog(models: LmstudioModelWire[]): ModelDefinitionConfig[] {
  return models
    .map((entry) => {
      const base = mapLmstudioWireEntry(entry);
      if (!base) {
        return null;
      }
      return {
        id: base.id,
        name: base.displayName,
        reasoning: base.reasoning,
        input: base.input,
        cost: base.cost,
        contextWindow: base.contextWindow,
        maxTokens: base.maxTokens,
      };
    })
    .filter((entry): entry is ModelDefinitionConfig => entry !== null);
}

/** Builds `agents.defaults.models` allowlist entries from discovered model IDs. */
function mapDiscoveredLmstudioModelsToAllowlistEntries(models: ModelDefinitionConfig[]) {
  const entries: Record<string, Record<string, never>> = {};
  for (const model of models) {
    const id = model.id.trim();
    if (!id) {
      continue;
    }
    entries[`${PROVIDER_ID}/${id}`] = {};
  }
  return entries;
}

function selectDiscoveredLmstudioDefaultModel(
  discoveredModels: ModelDefinitionConfig[],
): string | undefined {
  const discoveredIds = discoveredModels.map((model) => model.id.trim()).filter(Boolean);
  if (discoveredIds.length === 0) {
    return undefined;
  }
  const preferredModelId = discoveredIds.includes(LMSTUDIO_DEFAULT_MODEL_ID)
    ? LMSTUDIO_DEFAULT_MODEL_ID
    : discoveredIds[0];
  return `${PROVIDER_ID}/${preferredModelId}`;
}

async function resolveExplicitLmstudioSecretRefDiscoveryApiKey(params: {
  config: OpenClawConfig;
  env: NodeJS.ProcessEnv;
  apiKey: ModelProviderConfig["apiKey"] | undefined;
}): Promise<string | undefined> {
  const sourceConfig = projectConfigOntoRuntimeSourceSnapshot(params.config);
  const { ref } = resolveSecretInputRef({
    value: params.apiKey,
    defaults: sourceConfig.secrets?.defaults,
  });
  if (!ref) {
    return undefined;
  }

  try {
    return await resolveSecretInputString({
      config: sourceConfig,
      env: createConfigRuntimeEnv(sourceConfig, params.env),
      value: params.apiKey,
      normalize: normalizeOptionalSecretInput,
    });
  } catch {
    return undefined;
  }
}

/** Interactive LM Studio setup with connectivity and model-availability checks. */
export async function promptAndConfigureLmstudioInteractive(params: {
  config: OpenClawConfig;
  prompter?: WizardPrompter;
  secretInputMode?: SecretInputMode;
  allowSecretRefPrompt?: boolean;
  promptText?: ProviderPromptText;
  note?: ProviderPromptNote;
}): Promise<ProviderAuthResult> {
  const promptText = params.prompter?.text ?? params.promptText;
  if (!promptText) {
    throw new Error("LM Studio interactive setup requires a text prompter.");
  }
  const note = params.prompter?.note ?? params.note;
  const baseUrlRaw = await promptText({
    message: `${LMSTUDIO_PROVIDER_LABEL} base URL`,
    initialValue: LMSTUDIO_DEFAULT_BASE_URL,
    placeholder: LMSTUDIO_DEFAULT_BASE_URL,
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const baseUrl = resolveLmstudioInferenceBase(String(baseUrlRaw ?? ""));
  let credentialInput: SecretInput | undefined;
  let credentialMode: SecretInputMode | undefined;
  const implicitRefMode = params.allowSecretRefPrompt === false && !params.secretInputMode;
  const autoRefEnvKey = process.env[LMSTUDIO_DEFAULT_API_KEY_ENV_VAR]?.trim();
  const apiKey =
    params.prompter && implicitRefMode && autoRefEnvKey
      ? autoRefEnvKey
      : params.prompter
        ? await ensureApiKeyFromEnvOrPrompt({
            config: params.config,
            provider: PROVIDER_ID,
            envLabel: LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
            promptMessage: `${LMSTUDIO_PROVIDER_LABEL} API key`,
            normalize: (value) => value.trim(),
            validate: (value) => (value.trim() ? undefined : "Required"),
            prompter: params.prompter,
            secretInputMode:
              params.allowSecretRefPrompt === false
                ? (params.secretInputMode ?? "plaintext")
                : params.secretInputMode,
            setCredential: async (apiKeyValue, mode) => {
              credentialInput = apiKeyValue;
              credentialMode = mode;
            },
          })
        : String(
            await promptText({
              message: `${LMSTUDIO_PROVIDER_LABEL} API key`,
              placeholder: "sk-... (or any non-empty string)",
              validate: (value) => (value?.trim() ? undefined : "Required"),
            }),
          ).trim();
  const credential = params.prompter
    ? buildApiKeyCredential(
        PROVIDER_ID,
        credentialInput ??
          (implicitRefMode && autoRefEnvKey ? `\${${LMSTUDIO_DEFAULT_API_KEY_ENV_VAR}}` : apiKey),
        undefined,
        credentialMode
          ? { secretInputMode: credentialMode }
          : implicitRefMode && autoRefEnvKey
            ? { secretInputMode: "ref" }
            : undefined,
      )
    : {
        type: "api_key" as const,
        provider: PROVIDER_ID,
        key: apiKey,
      };
  const existingProvider = params.config.models?.providers?.[PROVIDER_ID];
  const resolvedHeaders = await resolveLmstudioProviderHeaders({
    config: params.config,
    env: process.env,
    headers: existingProvider?.headers,
  });
  const discovery = await fetchLmstudioModels({
    baseUrl,
    apiKey,
    ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
    timeoutMs: 5000,
  });
  if (!discovery.reachable) {
    await note?.(
      [
        `LM Studio could not be reached at ${baseUrl}.`,
        "Start LM Studio (or run lms server start) and re-run setup.",
      ].join("\n"),
      "LM Studio",
    );
    throw new WizardCancelledError("LM Studio not reachable");
  }
  if (discovery.status !== undefined && discovery.status >= 400) {
    await note?.(
      [
        `LM Studio returned HTTP ${discovery.status} while listing models at ${baseUrl}.`,
        "Check the base URL and API key, then re-run setup.",
      ].join("\n"),
      "LM Studio",
    );
    throw new WizardCancelledError(`LM Studio discovery failed (${discovery.status})`);
  }
  const hasUsableModel = discovery.models.some(
    (model) => model.type === "llm" && Boolean(model.key?.trim()),
  );
  if (!hasUsableModel) {
    await note?.(
      [
        `No LM Studio LLM models were found at ${baseUrl}.`,
        "Load at least one model in LM Studio (or run lms load), then re-run setup.",
      ].join("\n"),
      "LM Studio",
    );
    throw new WizardCancelledError("No LM Studio models found");
  }
  const discoveredModels = mapFetchedLmstudioModelsToCatalog(discovery.models);
  const allowlistEntries = mapDiscoveredLmstudioModelsToAllowlistEntries(discoveredModels);
  const defaultModel = selectDiscoveredLmstudioDefaultModel(discoveredModels);

  return {
    profiles: [
      {
        profileId: `${PROVIDER_ID}:default`,
        credential,
      },
    ],
    configPatch: {
      agents: {
        defaults: {
          models: allowlistEntries,
        },
      },
      models: {
        // Respect existing global mode; self-hosted provider setup should merge by default.
        mode: params.config.models?.mode ?? "merge",
        providers: {
          [PROVIDER_ID]: {
            ...existingProvider,
            baseUrl,
            api: existingProvider?.api ?? "openai-completions",
            auth: "api-key",
            apiKey: existingProvider?.apiKey ?? LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
            models: discoveredModels,
          },
        },
      },
    },
    defaultModel,
  };
}

/** Non-interactive setup path backed by the shared self-hosted helper. */
export async function configureLmstudioNonInteractive(
  ctx: ProviderAuthMethodNonInteractiveContext,
): Promise<OpenClawConfig | null> {
  const customBaseUrl = ctx.opts.customBaseUrl?.trim();
  const baseUrl = resolveLmstudioInferenceBase(
    customBaseUrl || LMSTUDIO_DEFAULT_INFERENCE_BASE_URL,
  );
  const normalizedCtx = customBaseUrl
    ? {
        ...ctx,
        opts: {
          ...ctx.opts,
          customBaseUrl: baseUrl,
        },
      }
    : ctx;
  const modelId = normalizedCtx.opts.customModelId?.trim();
  if (!modelId) {
    const configured = await configureOpenAICompatibleSelfHostedProviderNonInteractive({
      ctx: normalizedCtx,
      providerId: PROVIDER_ID,
      providerLabel: LMSTUDIO_PROVIDER_LABEL,
      defaultBaseUrl: LMSTUDIO_DEFAULT_INFERENCE_BASE_URL,
      defaultApiKeyEnvVar: LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
      modelPlaceholder: LMSTUDIO_MODEL_PLACEHOLDER,
    });
    if (!configured) {
      return null;
    }
    const configuredProvider = configured.models?.providers?.[PROVIDER_ID];
    if (!configuredProvider) {
      return configured;
    }
    return {
      ...configured,
      models: {
        ...configured.models,
        providers: {
          ...configured.models?.providers,
          [PROVIDER_ID]: {
            ...configuredProvider,
            auth: "api-key",
          },
        },
      },
    };
  }

  const resolved = await normalizedCtx.resolveApiKey({
    provider: PROVIDER_ID,
    flagValue: normalizedCtx.opts.customApiKey,
    flagName: "--custom-api-key",
    envVar: LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
    envVarName: LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
  });
  if (!resolved) {
    return null;
  }

  const existingProvider = normalizedCtx.config.models?.providers?.[PROVIDER_ID];
  const resolvedHeaders = await resolveLmstudioProviderHeaders({
    config: normalizedCtx.config,
    env: process.env,
    headers: existingProvider?.headers,
  });
  const discovery = await fetchLmstudioModels({
    baseUrl,
    apiKey: resolved.key,
    ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
    timeoutMs: 5000,
  });
  if (!discovery.reachable) {
    normalizedCtx.runtime.error(
      [
        `LM Studio could not be reached at ${baseUrl}.`,
        "Start LM Studio (or run lms server start) and re-run setup.",
      ].join("\n"),
    );
    normalizedCtx.runtime.exit(1);
    return null;
  }
  if (discovery.status !== undefined && discovery.status >= 400) {
    normalizedCtx.runtime.error(
      [
        `LM Studio returned HTTP ${discovery.status} while listing models at ${baseUrl}.`,
        "Check the base URL and API key, then re-run setup.",
      ].join("\n"),
    );
    normalizedCtx.runtime.exit(1);
    return null;
  }

  const discoveredModels = mapFetchedLmstudioModelsToCatalog(discovery.models);
  if (discoveredModels.length === 0) {
    normalizedCtx.runtime.error(
      [
        `No LM Studio LLM models were found at ${baseUrl}.`,
        "Load at least one model in LM Studio (or run lms load), then re-run setup.",
      ].join("\n"),
    );
    normalizedCtx.runtime.exit(1);
    return null;
  }
  if (!discoveredModels.some((model) => model.id === modelId)) {
    normalizedCtx.runtime.error(
      [
        `LM Studio model ${modelId} was not found at ${baseUrl}.`,
        `Available models: ${discoveredModels.map((model) => model.id).join(", ")}`,
      ].join("\n"),
    );
    normalizedCtx.runtime.exit(1);
    return null;
  }

  // Delegate to the shared helper even when modelId is set so that onboarding
  // state and credential storage are handled consistently. The pre-resolved key
  // is injected via resolveApiKey to skip a second prompt. The returned config
  // is then post-patched below to add the discovered model list and base URL.
  const configured = await configureOpenAICompatibleSelfHostedProviderNonInteractive({
    ctx: {
      ...normalizedCtx,
      resolveApiKey: async () => resolved,
    },
    providerId: PROVIDER_ID,
    providerLabel: LMSTUDIO_PROVIDER_LABEL,
    defaultBaseUrl: LMSTUDIO_DEFAULT_INFERENCE_BASE_URL,
    defaultApiKeyEnvVar: LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
    modelPlaceholder: LMSTUDIO_MODEL_PLACEHOLDER,
  });
  if (!configured) {
    return null;
  }

  return {
    ...configured,
    models: {
      ...configured.models,
      providers: {
        ...configured.models?.providers,
        // Keep existing auth marker fields while refreshing LM Studio transport + model catalog.
        [PROVIDER_ID]: {
          ...existingProvider,
          ...configured.models?.providers?.[PROVIDER_ID],
          ...(existingProvider?.apiKey ? { apiKey: existingProvider.apiKey } : {}),
          ...(existingProvider?.headers ? { headers: existingProvider.headers } : {}),
          baseUrl,
          api: configured.models?.providers?.[PROVIDER_ID]?.api ?? "openai-completions",
          auth: "api-key",
          models: discoveredModels,
        },
      },
    },
  };
}

/** Discovers provider settings, merging explicit config with live model discovery. */
export async function discoverLmstudioProvider(ctx: ProviderDiscoveryContext): Promise<{
  provider: ModelProviderConfig;
} | null> {
  const explicit = ctx.config.models?.providers?.[PROVIDER_ID];
  const explicitWithoutHeaders = explicit
    ? (() => {
        const { headers: _headers, ...rest } = explicit;
        return rest;
      })()
    : undefined;
  const hasExplicitModels = Array.isArray(explicit?.models) && explicit.models.length > 0;
  const { apiKey, discoveryApiKey } = ctx.resolveProviderApiKey(PROVIDER_ID);
  const explicitDiscoveryApiKey = resolveUsableCustomProviderApiKey({
    cfg: ctx.config,
    provider: PROVIDER_ID,
    env: ctx.env,
  })?.apiKey;
  const explicitSecretRefDiscoveryApiKey = await resolveExplicitLmstudioSecretRefDiscoveryApiKey({
    config: ctx.config,
    env: ctx.env,
    apiKey: explicit?.apiKey,
  });
  const resolvedHeaders = await resolveLmstudioProviderHeaders({
    config: ctx.config,
    env: ctx.env,
    headers: explicit?.headers,
  });
  // CLI/runtime-resolved key takes precedence over static provider config key.
  const resolvedApiKey = apiKey ?? explicit?.apiKey;
  if (hasExplicitModels && explicitWithoutHeaders) {
    const persistedApiKey =
      resolvedApiKey ??
      (shouldUseLmstudioApiKeyPlaceholder({
        hasModels: hasExplicitModels,
        resolvedApiKey,
      })
        ? LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER
        : undefined);
    return {
      provider: {
        ...explicitWithoutHeaders,
        ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
        baseUrl: resolveLmstudioInferenceBase(explicitWithoutHeaders.baseUrl),
        // Keep explicit API unless absent, then fall back to provider default.
        api: explicitWithoutHeaders.api ?? "openai-completions",
        ...(persistedApiKey ? { apiKey: persistedApiKey } : {}),
        ...(resolveLmstudioProviderAuthMode(persistedApiKey) ? { auth: "api-key" } : {}),
        models: explicitWithoutHeaders.models,
      },
    };
  }
  const provider = await buildLmstudioProvider({
    baseUrl: explicit?.baseUrl,
    // Prefer dedicated discovery key, then explicit env-backed custom-provider key,
    // then explicit SecretRef-backed provider keys.
    apiKey: discoveryApiKey ?? explicitDiscoveryApiKey ?? explicitSecretRefDiscoveryApiKey,
    headers: resolvedHeaders,
    quiet: !apiKey && !explicit && !explicitDiscoveryApiKey && !explicitSecretRefDiscoveryApiKey,
  });
  const models = mergeDiscoveredModels({
    explicitModels: explicit?.models,
    discoveredModels: provider.models,
  });
  if (models.length === 0 && !apiKey && !explicit?.apiKey) {
    return null;
  }
  const persistedApiKey =
    resolvedApiKey ??
    (shouldUseLmstudioApiKeyPlaceholder({
      hasModels: models.length > 0,
      resolvedApiKey,
    })
      ? LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER
      : undefined);
  return {
    provider: {
      ...provider,
      ...explicitWithoutHeaders,
      ...(resolvedHeaders ? { headers: resolvedHeaders } : {}),
      baseUrl: resolveLmstudioInferenceBase(explicit?.baseUrl ?? provider.baseUrl),
      ...(persistedApiKey ? { apiKey: persistedApiKey } : {}),
      ...(resolveLmstudioProviderAuthMode(persistedApiKey) ? { auth: "api-key" } : {}),
      models,
    },
  };
}
