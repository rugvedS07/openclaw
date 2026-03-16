import { projectConfigOntoRuntimeSourceSnapshot, type OpenClawConfig } from "../config/config.js";
import { createConfigRuntimeEnv } from "../config/env-vars.js";
import { formatErrorMessage } from "../infra/errors.js";
import { resolveSecretInputString } from "../secrets/resolve-secret-input-string.js";
import { normalizeOptionalSecretInput } from "../utils/normalize-secret-input.js";
import { LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER, LMSTUDIO_PROVIDER_ID } from "./lmstudio-defaults.js";
import { isKnownEnvApiKeyMarker, isNonSecretApiKeyMarker } from "./model-auth-markers.js";
import { resolveApiKeyForProvider } from "./model-auth.js";

type LmstudioAuthHeadersParams = {
  apiKey?: string;
  json?: boolean;
  headers?: Record<string, string>;
};

export function buildLmstudioAuthHeaders(
  params: LmstudioAuthHeadersParams,
): Record<string, string> | undefined {
  const headers: Record<string, string> = { ...params.headers };
  // Keep auth optional because LM Studio can run without a token.
  const apiKey = params.apiKey?.trim();
  if (apiKey && !isNonSecretApiKeyMarker(apiKey)) {
    for (const headerName of Object.keys(headers)) {
      if (headerName.toLowerCase() === "authorization") {
        delete headers[headerName];
      }
    }
    headers.Authorization = `Bearer ${apiKey}`;
  }
  if (params.json) {
    headers["Content-Type"] = "application/json";
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function sanitizeStringHeaders(headers: unknown): Record<string, string> | undefined {
  if (!headers || typeof headers !== "object" || Array.isArray(headers)) {
    return undefined;
  }
  const next: Record<string, string> = {};
  for (const [headerName, headerValue] of Object.entries(headers)) {
    if (typeof headerValue !== "string") {
      continue;
    }
    const normalized = headerValue.trim();
    if (!normalized) {
      continue;
    }
    next[headerName] = normalized;
  }
  return Object.keys(next).length > 0 ? next : undefined;
}

export async function resolveLmstudioConfiguredApiKey(params: {
  config?: OpenClawConfig;
  env?: NodeJS.ProcessEnv;
  allowLocalFallback?: boolean;
  path?: string;
}): Promise<string | undefined> {
  const providerConfig = params.config?.models?.providers?.[LMSTUDIO_PROVIDER_ID];
  const apiKeyInput = providerConfig?.apiKey;
  const allowLocalFallback = params.allowLocalFallback === true;
  const allowFallbackForProvider = allowLocalFallback && providerConfig?.auth !== "api-key";
  if (apiKeyInput === undefined || apiKeyInput === null) {
    return allowFallbackForProvider ? LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER : undefined;
  }

  const directApiKey = normalizeOptionalSecretInput(apiKeyInput);
  if (directApiKey !== undefined) {
    const trimmed = directApiKey.trim();
    if (!trimmed) {
      return allowFallbackForProvider ? LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER : undefined;
    }
    if (trimmed === LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER) {
      return allowFallbackForProvider ? trimmed : undefined;
    }
    if (isKnownEnvApiKeyMarker(trimmed)) {
      return normalizeOptionalSecretInput((params.env ?? process.env)[trimmed]);
    }
    return isNonSecretApiKeyMarker(trimmed) ? undefined : trimmed;
  }

  const sourceConfig = params.config
    ? projectConfigOntoRuntimeSourceSnapshot(params.config)
    : undefined;
  if (!sourceConfig) {
    return undefined;
  }
  const env = createConfigRuntimeEnv(sourceConfig, params.env ?? process.env);
  const path = params.path ?? "models.providers.lmstudio.apiKey";
  return await resolveSecretInputString({
    config: sourceConfig,
    env,
    value: apiKeyInput,
    onResolveRefError: (error, ref) => {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${path}: could not resolve SecretRef "${ref.source}:${ref.provider}:${ref.id}": ${detail}`,
      );
    },
  });
}

export async function resolveLmstudioProviderHeaders(params: {
  config?: OpenClawConfig;
  env?: NodeJS.ProcessEnv;
  headers?: unknown;
  path?: string;
}): Promise<Record<string, string> | undefined> {
  const headerInputs = params.headers;
  if (!headerInputs || typeof headerInputs !== "object" || Array.isArray(headerInputs)) {
    return undefined;
  }

  const sourceConfig = params.config
    ? projectConfigOntoRuntimeSourceSnapshot(params.config)
    : undefined;
  if (!sourceConfig) {
    return sanitizeStringHeaders(headerInputs);
  }

  const env = createConfigRuntimeEnv(sourceConfig, params.env ?? process.env);
  const pathPrefix = params.path ?? "models.providers.lmstudio.headers";
  const resolved: Record<string, string> = {};
  for (const [headerName, headerValue] of Object.entries(headerInputs)) {
    const resolvedValue = await resolveSecretInputString({
      config: sourceConfig,
      env,
      value: headerValue,
      onResolveRefError: (error, ref) => {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(
          `${pathPrefix}.${headerName}: could not resolve SecretRef "${ref.source}:${ref.provider}:${ref.id}": ${detail}`,
        );
      },
    });
    if (!resolvedValue) {
      continue;
    }
    resolved[headerName] = resolvedValue;
  }
  return Object.keys(resolved).length > 0 ? resolved : undefined;
}

/**
 * Resolves LM Studio runtime API key from config.
 */
export async function resolveLmstudioRuntimeApiKey(params: {
  config?: OpenClawConfig;
  agentDir?: string;
  allowMissingAuth?: boolean;
  env?: NodeJS.ProcessEnv;
}): Promise<string | undefined> {
  const config = params.config;
  if (!config) {
    return undefined;
  }
  let configuredApiKeyPromise: Promise<string | undefined> | undefined;
  const getConfiguredApiKey = async () => {
    configuredApiKeyPromise ??= resolveLmstudioConfiguredApiKey({
      config,
      env: params.env,
      allowLocalFallback: true,
    });
    return await configuredApiKeyPromise;
  };
  let resolved: Awaited<ReturnType<typeof resolveApiKeyForProvider>>;
  try {
    resolved = await resolveApiKeyForProvider({
      provider: LMSTUDIO_PROVIDER_ID,
      cfg: config,
      agentDir: params.agentDir,
    });
  } catch (error) {
    const configuredApiKey = await getConfiguredApiKey();
    if (configuredApiKey) {
      return configuredApiKey;
    }
    if (
      params.allowMissingAuth &&
      formatErrorMessage(error).includes(`No API key found for provider "${LMSTUDIO_PROVIDER_ID}"`)
    ) {
      return undefined;
    }
    throw error;
  }
  // Normalize empty/whitespace keys to undefined for callers.
  const resolvedApiKey = resolved.apiKey?.trim();
  return resolvedApiKey && resolvedApiKey.length > 0 ? resolvedApiKey : await getConfiguredApiKey();
}
