import {
  LMSTUDIO_DEFAULT_EMBEDDING_MODEL,
  LMSTUDIO_PROVIDER_ID,
} from "../agents/lmstudio-defaults.js";
import {
  ensureLmstudioModelLoaded,
  resolveLmstudioInferenceBase,
} from "../agents/lmstudio-models.js";
import {
  buildLmstudioAuthHeaders,
  resolveLmstudioProviderHeaders,
  resolveLmstudioRuntimeApiKey,
} from "../agents/lmstudio-runtime.js";
import { formatErrorMessage } from "../infra/errors.js";
import type { SsrFPolicy } from "../infra/net/ssrf.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { normalizeEmbeddingModelWithPrefixes } from "./embeddings-model-normalize.js";
import { createRemoteEmbeddingProvider } from "./embeddings-remote-provider.js";
import type { EmbeddingProvider, EmbeddingProviderOptions } from "./embeddings.js";
import { buildRemoteBaseUrlPolicy } from "./remote-http.js";

const log = createSubsystemLogger("memory/embeddings");

export type LmstudioEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
};
export const DEFAULT_LMSTUDIO_EMBEDDING_MODEL = LMSTUDIO_DEFAULT_EMBEDDING_MODEL;

/** Normalizes LM Studio embedding model refs and accepts `lmstudio/` prefix. */
function normalizeLmstudioModel(model: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_LMSTUDIO_EMBEDDING_MODEL,
    prefixes: ["lmstudio/"],
  });
}

/** Resolves API key from runtime/provider auth config.
 *  Returns undefined when no key is configured, and throws on auth lookup failures. */
async function resolveLmstudioApiKey(
  options: EmbeddingProviderOptions,
): Promise<string | undefined> {
  const authMode = options.config.models?.providers?.lmstudio?.auth;
  return await resolveLmstudioRuntimeApiKey({
    config: options.config,
    agentDir: options.agentDir,
    // Explicit api-key auth should fail fast when credentials are missing.
    allowMissingAuth: authMode !== "api-key",
  });
}

/** Creates the LM Studio embedding provider client and best-effort warms the target model. */
export async function createLmstudioEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<{ provider: EmbeddingProvider; client: LmstudioEmbeddingClient }> {
  const providerConfig = options.config.models?.providers?.lmstudio;
  const providerBaseUrl = providerConfig?.baseUrl?.trim();
  const configuredBaseUrl =
    providerBaseUrl && providerBaseUrl.length > 0 ? providerBaseUrl : undefined;
  const baseUrl = resolveLmstudioInferenceBase(configuredBaseUrl);
  const model = normalizeLmstudioModel(options.model);
  const apiKey = await resolveLmstudioApiKey(options);
  const providerHeaders = await resolveLmstudioProviderHeaders({
    config: options.config,
    env: process.env,
    headers: providerConfig?.headers,
  });
  const headerOverrides = Object.assign({}, providerHeaders);
  const headers =
    buildLmstudioAuthHeaders({
      apiKey,
      json: true,
      headers: headerOverrides,
    }) ?? {};
  const ssrfPolicy = buildRemoteBaseUrlPolicy(baseUrl);
  const client: LmstudioEmbeddingClient = {
    baseUrl,
    model,
    headers,
    ssrfPolicy,
  };

  // Warmup is best-effort only; do not block provider construction on cold model load.
  void ensureLmstudioModelLoaded({
    baseUrl,
    apiKey,
    headers: headerOverrides,
    ssrfPolicy,
    modelKey: model,
    timeoutMs: 120_000,
  }).catch((error) => {
    log.warn("lmstudio embeddings warmup failed; continuing without preload", {
      baseUrl,
      model,
      error: formatErrorMessage(error),
    });
  });

  return {
    provider: createRemoteEmbeddingProvider({
      id: LMSTUDIO_PROVIDER_ID,
      client,
      errorPrefix: "lmstudio embeddings failed",
    }),
    client,
  };
}
