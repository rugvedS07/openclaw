import { beforeEach, describe, expect, it, vi } from "vitest";
import { LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER } from "../agents/lmstudio-defaults.js";
import type { OpenClawConfig } from "../config/config.js";
import { resolveAgentModelPrimaryValue } from "../config/model-input.js";
import type { ModelDefinitionConfig } from "../config/types.models.js";
import type {
  ProviderAuthMethodNonInteractiveContext,
  ProviderDiscoveryContext,
} from "../plugins/types.js";
import {
  configureLmstudioNonInteractive,
  discoverLmstudioProvider,
  promptAndConfigureLmstudioInteractive,
} from "./lmstudio-setup.js";
import { mergeConfigPatch } from "./provider-auth-helpers.js";

const fetchLmstudioModelsMock = vi.hoisted(() => vi.fn());
const buildLmstudioProviderMock = vi.hoisted(() => vi.fn());
const configureSelfHostedNonInteractiveMock = vi.hoisted(() => vi.fn());

vi.mock("../agents/lmstudio-models.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../agents/lmstudio-models.js")>();
  return {
    ...actual,
    fetchLmstudioModels: (...args: unknown[]) => fetchLmstudioModelsMock(...args),
  };
});

vi.mock("../agents/models-config.providers.discovery.js", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../agents/models-config.providers.discovery.js")>();
  return {
    ...actual,
    buildLmstudioProvider: (...args: unknown[]) => buildLmstudioProviderMock(...args),
  };
});

vi.mock("./self-hosted-provider-setup.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./self-hosted-provider-setup.js")>();
  return {
    ...actual,
    configureOpenAICompatibleSelfHostedProviderNonInteractive: (...args: unknown[]) =>
      configureSelfHostedNonInteractiveMock(...args),
  };
});

function createModel(id: string, name = id): ModelDefinitionConfig {
  return {
    id,
    name,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 8192,
    maxTokens: 8192,
  };
}

function buildConfig(): OpenClawConfig {
  return {
    models: {
      providers: {
        lmstudio: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "LM_API_TOKEN",
          api: "openai-completions",
          models: [],
        },
      },
    },
  };
}

function buildDiscoveryContext(params?: {
  config?: OpenClawConfig;
  apiKey?: string;
  discoveryApiKey?: string;
  env?: NodeJS.ProcessEnv;
}): ProviderDiscoveryContext {
  return {
    config: params?.config ?? ({} as OpenClawConfig),
    env: params?.env ?? {},
    resolveProviderApiKey: () => ({
      apiKey: params?.apiKey,
      discoveryApiKey: params?.discoveryApiKey,
    }),
    resolveProviderAuth: () => ({
      apiKey: params?.apiKey,
      discoveryApiKey: params?.discoveryApiKey,
      mode: "none" as const,
      source: "none" as const,
    }),
  };
}

function buildNonInteractiveContext(params?: {
  config?: OpenClawConfig;
  customBaseUrl?: string;
  customApiKey?: string;
  customModelId?: string;
  resolvedApiKey?: string | null;
}): ProviderAuthMethodNonInteractiveContext & {
  runtime: {
    error: ReturnType<typeof vi.fn>;
    exit: ReturnType<typeof vi.fn>;
    log: ReturnType<typeof vi.fn>;
  };
  resolveApiKey: ReturnType<typeof vi.fn>;
  toApiKeyCredential: ReturnType<typeof vi.fn>;
} {
  const error = vi.fn<(...args: unknown[]) => void>();
  const exit = vi.fn<(code: number) => void>();
  const log = vi.fn<(...args: unknown[]) => void>();
  const resolveApiKey = vi.fn(async () =>
    params?.resolvedApiKey === null
      ? null
      : {
          key: params?.resolvedApiKey ?? "lmstudio-test-key",
          source: "flag" as const,
        },
  );
  const toApiKeyCredential = vi.fn();
  return {
    authChoice: "lmstudio",
    config: params?.config ?? buildConfig(),
    baseConfig: params?.config ?? buildConfig(),
    opts: {
      customBaseUrl: params?.customBaseUrl,
      customApiKey: params?.customApiKey ?? "lmstudio-test-key",
      customModelId: params?.customModelId,
    } as ProviderAuthMethodNonInteractiveContext["opts"],
    runtime: { error, exit, log },
    resolveApiKey,
    toApiKeyCredential,
  };
}

describe("lmstudio setup", () => {
  beforeEach(() => {
    fetchLmstudioModelsMock.mockReset();
    buildLmstudioProviderMock.mockReset();
    configureSelfHostedNonInteractiveMock.mockReset();

    fetchLmstudioModelsMock.mockResolvedValue({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "qwen3-8b-instruct",
        },
      ],
    });
    buildLmstudioProviderMock.mockResolvedValue({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });
    configureSelfHostedNonInteractiveMock.mockImplementation(async (args: unknown) => {
      const params = args as {
        providerId: string;
        ctx: ProviderAuthMethodNonInteractiveContext;
      };
      const providerId = params.providerId;
      const customModelId =
        typeof params.ctx.opts.customModelId === "string"
          ? params.ctx.opts.customModelId.trim()
          : "";
      const modelId = customModelId || "qwen3-8b-instruct";
      const customBaseUrl =
        typeof params.ctx.opts.customBaseUrl === "string"
          ? params.ctx.opts.customBaseUrl
          : undefined;
      return {
        agents: {
          defaults: {
            model: {
              primary: `${providerId}/${modelId}`,
            },
          },
        },
        models: {
          providers: {
            [providerId]: {
              baseUrl: customBaseUrl ?? "http://localhost:1234/v1",
              api: "openai-completions",
              auth: "api-key",
              apiKey: "LM_API_TOKEN",
              models: [createModel(modelId, "Qwen3 8B")],
            },
          },
        },
      };
    });
  });

  it("non-interactive setup discovers catalog and writes LM Studio provider config", async () => {
    const ctx = buildNonInteractiveContext({
      customBaseUrl: "http://localhost:1234/api/v1/",
      customModelId: "qwen3-8b-instruct",
    });
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "qwen3-8b-instruct",
          display_name: "Qwen3 8B",
          loaded_instances: [{ id: "inst-1", config: { context_length: 64000 } }],
        },
        {
          type: "embedding",
          key: "text-embedding-nomic-embed-text-v1.5",
        },
      ],
    });

    const result = await configureLmstudioNonInteractive(ctx);

    expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "lmstudio-test-key",
      timeoutMs: 5000,
    });
    expect(result?.models?.providers?.lmstudio).toMatchObject({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      auth: "api-key",
      apiKey: "LM_API_TOKEN",
      models: [
        {
          id: "qwen3-8b-instruct",
          contextWindow: 64000,
        },
      ],
    });
    expect(resolveAgentModelPrimaryValue(result?.agents?.defaults?.model)).toBe(
      "lmstudio/qwen3-8b-instruct",
    );
  });

  it("non-interactive setup fails when requested model is missing", async () => {
    const ctx = buildNonInteractiveContext({
      customModelId: "missing-model",
    });

    await expect(configureLmstudioNonInteractive(ctx)).resolves.toBeNull();

    expect(ctx.runtime.error).toHaveBeenCalledWith(
      expect.stringContaining("LM Studio model missing-model was not found"),
    );
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
    expect(configureSelfHostedNonInteractiveMock).not.toHaveBeenCalled();
  });

  it("interactive setup canonicalizes base URL and persists provider/default model", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/api/v1/")
      .mockResolvedValueOnce("lmstudio-test-key");

    const result = await promptAndConfigureLmstudioInteractive({
      config: buildConfig(),
      promptText,
    });
    const mergedConfig = mergeConfigPatch(buildConfig(), result.configPatch);

    expect(result.configPatch?.models?.mode).toBe("merge");
    expect(mergedConfig).toMatchObject({
      models: {
        providers: {
          lmstudio: {
            baseUrl: "http://localhost:1234/v1",
            api: "openai-completions",
            auth: "api-key",
            apiKey: "LM_API_TOKEN",
          },
        },
      },
    });
    expect(result.defaultModel).toBe("lmstudio/qwen3-8b-instruct");
    expect(result.profiles[0]).toMatchObject({
      profileId: "lmstudio:default",
      credential: {
        type: "api_key",
        provider: "lmstudio",
        key: "lmstudio-test-key",
      },
    });
  });

  it("interactive setup returns clear errors for unreachable/http-empty results", async () => {
    const cases = [
      {
        name: "unreachable",
        discovery: { reachable: false, models: [] },
        expectedError: "LM Studio not reachable",
      },
      {
        name: "http error",
        discovery: { reachable: true, status: 401, models: [] },
        expectedError: "LM Studio discovery failed (401)",
      },
      {
        name: "no llm models",
        discovery: {
          reachable: true,
          status: 200,
          models: [{ type: "embedding", key: "text-embedding-nomic-embed-text-v1.5" }],
        },
        expectedError: "No LM Studio models found",
      },
    ];

    for (const testCase of cases) {
      const promptText = vi
        .fn()
        .mockResolvedValueOnce("http://localhost:1234/v1")
        .mockResolvedValueOnce("lmstudio-test-key");
      fetchLmstudioModelsMock.mockResolvedValueOnce(testCase.discovery);
      await expect(
        promptAndConfigureLmstudioInteractive({
          config: buildConfig(),
          promptText,
        }),
        testCase.name,
      ).rejects.toThrow(testCase.expectedError);
    }
  });

  it("discoverLmstudioProvider short-circuits when explicit models are configured", async () => {
    const explicitModels = [createModel("qwen3-8b-instruct", "Qwen3 8B")];
    const result = await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/api/v1/",
                models: explicitModels,
              },
            },
          },
        } as OpenClawConfig,
      }),
    );

    expect(buildLmstudioProviderMock).not.toHaveBeenCalled();
    expect(result).toEqual({
      provider: {
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER,
        models: explicitModels,
      },
    });
  });

  it("discoverLmstudioProvider uses resolved key/headers and non-quiet discovery", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    const result = await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: {
                  source: "env",
                  provider: "default",
                  id: "LMSTUDIO_DISCOVERY_TOKEN",
                },
                headers: {
                  "X-Proxy-Auth": {
                    source: "env",
                    provider: "default",
                    id: "LMSTUDIO_PROXY_TOKEN",
                  },
                },
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        env: {
          LMSTUDIO_DISCOVERY_TOKEN: "secretref-lmstudio-key",
          LMSTUDIO_PROXY_TOKEN: "proxy-token-from-env",
        },
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "secretref-lmstudio-key",
      headers: {
        "X-Proxy-Auth": "proxy-token-from-env",
      },
      quiet: false,
    });
    expect(result?.provider.models?.map((model) => model.id)).toEqual(["qwen3-8b-instruct"]);
  });

  it("discoverLmstudioProvider uses quiet mode and returns null when unconfigured", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [],
    });

    const result = await discoverLmstudioProvider(buildDiscoveryContext());

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: undefined,
      apiKey: undefined,
      quiet: true,
    });
    expect(result).toBeNull();
  });
});
