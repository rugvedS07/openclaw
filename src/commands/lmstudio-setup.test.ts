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
  const runtime: ProviderAuthMethodNonInteractiveContext["runtime"] & {
    error: typeof error;
    exit: typeof exit;
    log: typeof log;
  } = {
    error,
    exit,
    log,
  };
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
    runtime,
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
      const modelId = params.ctx.opts.customModelId?.trim() || "qwen3-8b-instruct";
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
              baseUrl: params.ctx.opts.customBaseUrl ?? "http://localhost:1234/v1",
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

  it("non-interactive setup fetches LM Studio models and persists the discovered catalog", async () => {
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
          type: "llm",
          key: "phi-4",
          max_context_length: 32768,
        },
        {
          type: "embedding",
          key: "text-embedding-nomic-embed-text-v1.5",
        },
      ],
    });

    const result = await configureLmstudioNonInteractive(ctx);

    expect(ctx.resolveApiKey).toHaveBeenCalledWith({
      provider: "lmstudio",
      flagValue: "lmstudio-test-key",
      flagName: "--custom-api-key",
      envVar: "LM_API_TOKEN",
      envVarName: "LM_API_TOKEN",
    });
    expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "lmstudio-test-key",
      timeoutMs: 5000,
    });
    expect(configureSelfHostedNonInteractiveMock).toHaveBeenCalledWith(
      expect.objectContaining({
        ctx: expect.objectContaining({
          opts: expect.objectContaining({
            customBaseUrl: "http://localhost:1234/v1",
          }),
        }),
        providerId: "lmstudio",
        providerLabel: "LM Studio",
        defaultBaseUrl: "http://localhost:1234/v1",
        defaultApiKeyEnvVar: "LM_API_TOKEN",
      }),
    );
    expect(result?.models?.providers?.lmstudio).toMatchObject({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      auth: "api-key",
      apiKey: "LM_API_TOKEN",
      models: [
        {
          id: "qwen3-8b-instruct",
          name: "Qwen3 8B",
          contextWindow: 64000,
          maxTokens: 8192,
        },
        {
          id: "phi-4",
          name: "phi-4",
          contextWindow: 32768,
          maxTokens: 8192,
        },
      ],
    });
    expect(resolveAgentModelPrimaryValue(result?.agents?.defaults?.model)).toBe(
      "lmstudio/qwen3-8b-instruct",
    );
  });

  it("non-interactive setup resolves SecretRef-backed headers during discovery", async () => {
    const prevEnv = process.env.LMSTUDIO_PROXY_TOKEN;
    process.env.LMSTUDIO_PROXY_TOKEN = "proxy-token-from-env";
    try {
      const ctx = buildNonInteractiveContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: "LM_API_TOKEN",
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
        customBaseUrl: "http://localhost:1234/v1",
        customModelId: "qwen3-8b-instruct",
      });

      await configureLmstudioNonInteractive(ctx);

      expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lmstudio-test-key",
        headers: {
          "X-Proxy-Auth": "proxy-token-from-env",
        },
        timeoutMs: 5000,
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LMSTUDIO_PROXY_TOKEN;
      } else {
        process.env.LMSTUDIO_PROXY_TOKEN = prevEnv;
      }
    }
  });

  it("non-interactive setup preserves existing LM Studio auth fields when refreshing models", async () => {
    const prevEnv = process.env.LMSTUDIO_PROXY_TOKEN;
    process.env.LMSTUDIO_PROXY_TOKEN = "proxy-token-from-env";
    const existingApiKey = {
      source: "env" as const,
      provider: "default" as const,
      id: "CUSTOM_LMSTUDIO_KEY",
    };
    const existingHeaders = {
      "X-Proxy-Auth": {
        source: "env" as const,
        provider: "default" as const,
        id: "LMSTUDIO_PROXY_TOKEN",
      },
    };
    const ctx = buildNonInteractiveContext({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              api: "openai-completions",
              apiKey: existingApiKey,
              headers: existingHeaders,
              models: [createModel("old-model", "Old Model")],
            },
          },
        },
      } as OpenClawConfig,
      customBaseUrl: "http://localhost:1234/v1",
      customModelId: "qwen3-8b-instruct",
    });

    try {
      const result = await configureLmstudioNonInteractive(ctx);

      expect(result?.models?.providers?.lmstudio).toMatchObject({
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        apiKey: existingApiKey,
        headers: existingHeaders,
        models: [expect.objectContaining({ id: "qwen3-8b-instruct" })],
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LMSTUDIO_PROXY_TOKEN;
      } else {
        process.env.LMSTUDIO_PROXY_TOKEN = prevEnv;
      }
    }
  });

  it("non-interactive setup rewrites generic helper output to LM Studio responses transport", async () => {
    configureSelfHostedNonInteractiveMock.mockResolvedValueOnce({
      agents: {
        defaults: {
          model: {
            primary: "lmstudio/qwen3-8b-instruct",
          },
        },
      },
      models: {
        providers: {
          lmstudio: {
            baseUrl: "http://localhost:1234/v1",
            api: "openai-completions",
            apiKey: "LM_API_TOKEN",
            models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
          },
        },
      },
    });

    const result = await configureLmstudioNonInteractive(buildNonInteractiveContext());

    expect(configureSelfHostedNonInteractiveMock).toHaveBeenCalledTimes(1);
    expect(result?.models?.providers?.lmstudio?.api).toBe("openai-completions");
  });

  it("non-interactive setup fails when the requested model is not in the LM Studio catalog", async () => {
    const ctx = buildNonInteractiveContext({
      customModelId: "missing-model",
    });
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "qwen3-8b-instruct",
        },
      ],
    });

    await expect(configureLmstudioNonInteractive(ctx)).resolves.toBeNull();

    expect(ctx.runtime.error).toHaveBeenCalledWith(
      expect.stringContaining("LM Studio model missing-model was not found"),
    );
    expect(ctx.runtime.error).toHaveBeenCalledWith(
      expect.stringContaining("Available models: qwen3-8b-instruct"),
    );
    expect(ctx.runtime.exit).toHaveBeenCalledWith(1);
    expect(configureSelfHostedNonInteractiveMock).not.toHaveBeenCalled();
  });

  it("interactive setup requires API key and canonicalizes LM Studio base URL", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/api/v1/")
      .mockResolvedValueOnce("lmstudio-test-key");
    const note = vi.fn();

    const result = await promptAndConfigureLmstudioInteractive({
      config: buildConfig(),
      promptText,
      note,
    });
    const mergedConfig = mergeConfigPatch(buildConfig(), result.configPatch);

    expect(result).toMatchObject({
      profiles: [
        {
          profileId: "lmstudio:default",
          credential: {
            type: "api_key",
            provider: "lmstudio",
            key: "lmstudio-test-key",
          },
        },
      ],
    });
    expect(mergedConfig).toMatchObject({
      agents: {
        defaults: {
          models: {
            "lmstudio/qwen3-8b-instruct": {},
          },
        },
      },
      models: {
        providers: {
          lmstudio: {
            baseUrl: "http://localhost:1234/v1",
            api: "openai-completions",
            auth: "api-key",
            apiKey: "LM_API_TOKEN",
            models: [
              {
                id: "qwen3-8b-instruct",
                name: "qwen3-8b-instruct",
                reasoning: false,
                input: ["text"],
              },
            ],
          },
        },
      },
    });
    expect(result.configPatch?.models?.mode).toBe("merge");
    const apiKeyPrompt = promptText.mock.calls[1]?.[0] as
      | { validate?: (value: string | undefined) => string | undefined }
      | undefined;
    expect(apiKeyPrompt?.validate?.("")).toBe("Required");
    expect(apiKeyPrompt?.validate?.("lmstudio-test-key")).toBeUndefined();
    expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "lmstudio-test-key",
      timeoutMs: 5000,
    });
    expect(note).not.toHaveBeenCalled();
    expect(result.defaultModel).toBe("lmstudio/qwen3-8b-instruct");
  });

  it("interactive setup resolves SecretRef-backed headers during discovery", async () => {
    const prevEnv = process.env.LMSTUDIO_PROXY_TOKEN;
    process.env.LMSTUDIO_PROXY_TOKEN = "proxy-token-from-env";
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    try {
      await promptAndConfigureLmstudioInteractive({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: "LM_API_TOKEN",
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
        promptText,
      });

      expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lmstudio-test-key",
        headers: {
          "X-Proxy-Auth": "proxy-token-from-env",
        },
        timeoutMs: 5000,
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LMSTUDIO_PROXY_TOKEN;
      } else {
        process.env.LMSTUDIO_PROXY_TOKEN = prevEnv;
      }
    }
  });

  it("interactive setup preserves existing LM Studio auth fields in config patch", async () => {
    const prevEnv = process.env.LMSTUDIO_PROXY_TOKEN;
    process.env.LMSTUDIO_PROXY_TOKEN = "proxy-token-from-env";
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    const existingApiKey = {
      source: "env" as const,
      provider: "default" as const,
      id: "CUSTOM_LMSTUDIO_KEY",
    };
    const existingHeaders = {
      "X-Proxy-Auth": {
        source: "env" as const,
        provider: "default" as const,
        id: "LMSTUDIO_PROXY_TOKEN",
      },
    };
    try {
      const result = await promptAndConfigureLmstudioInteractive({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: existingApiKey,
                headers: existingHeaders,
                models: [createModel("old-model", "Old Model")],
              },
            },
          },
        } as OpenClawConfig,
        promptText,
      });

      expect(result.configPatch?.models?.providers?.lmstudio).toMatchObject({
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        apiKey: existingApiKey,
        headers: existingHeaders,
        models: [expect.objectContaining({ id: "qwen3-8b-instruct" })],
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LMSTUDIO_PROXY_TOKEN;
      } else {
        process.env.LMSTUDIO_PROXY_TOKEN = prevEnv;
      }
    }
  });

  it("interactive setup stores an env keyRef when secret input mode is ref", async () => {
    const prevEnv = process.env.LM_API_TOKEN;
    process.env.LM_API_TOKEN = "lmstudio-env-test-key";
    const text = vi.fn().mockResolvedValueOnce("http://localhost:1234/v1");
    const select = vi.fn().mockResolvedValueOnce("ref").mockResolvedValueOnce("env");
    const note = vi.fn();
    try {
      const result = await promptAndConfigureLmstudioInteractive({
        config: buildConfig(),
        prompter: {
          text,
          select,
          note,
        } as unknown as Parameters<typeof promptAndConfigureLmstudioInteractive>[0]["prompter"],
      });

      expect(result.profiles[0]).toMatchObject({
        profileId: "lmstudio:default",
        credential: {
          type: "api_key",
          provider: "lmstudio",
          keyRef: {
            source: "env",
            provider: "default",
            id: "LM_API_TOKEN",
          },
        },
      });
      expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lmstudio-env-test-key",
        timeoutMs: 5000,
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LM_API_TOKEN;
      } else {
        process.env.LM_API_TOKEN = prevEnv;
      }
    }
  });

  it("interactive setup auto-stores LM_API_TOKEN as keyRef when secret mode is implicit and env is set", async () => {
    const prevEnv = process.env.LM_API_TOKEN;
    process.env.LM_API_TOKEN = "lmstudio-auto-ref-key";
    const text = vi.fn().mockResolvedValueOnce("http://localhost:1234/v1");
    const note = vi.fn();
    try {
      const result = await promptAndConfigureLmstudioInteractive({
        config: buildConfig(),
        prompter: {
          text,
          note,
        } as unknown as Parameters<typeof promptAndConfigureLmstudioInteractive>[0]["prompter"],
        allowSecretRefPrompt: false,
      });

      expect(result.profiles[0]).toMatchObject({
        profileId: "lmstudio:default",
        credential: {
          type: "api_key",
          provider: "lmstudio",
          keyRef: {
            source: "env",
            provider: "default",
            id: "LM_API_TOKEN",
          },
        },
      });
      expect(fetchLmstudioModelsMock).toHaveBeenCalledWith({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lmstudio-auto-ref-key",
        timeoutMs: 5000,
      });
    } finally {
      if (prevEnv === undefined) {
        delete process.env.LM_API_TOKEN;
      } else {
        process.env.LM_API_TOKEN = prevEnv;
      }
    }
  });

  it("interactive setup preserves an existing models mode", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/api/v1/")
      .mockResolvedValueOnce("lmstudio-test-key");

    const result = await promptAndConfigureLmstudioInteractive({
      config: {
        models: {
          mode: "replace",
          providers: {},
        },
      } as OpenClawConfig,
      promptText,
    });

    expect(result.configPatch?.models?.mode).toBe("replace");
  });

  it("interactive setup refreshes provider models and preserves existing agent default entries", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "fresh-model",
          display_name: "Fresh Model",
        },
      ],
    });

    const config = {
      agents: {
        defaults: {
          models: {
            "openai/gpt-5.4": { alias: "gpt" },
            "lmstudio/stale-model": { alias: "stale" },
            "lmstudio/fresh-model": { alias: "fresh" },
          },
        },
      },
      models: {
        providers: {
          lmstudio: {
            baseUrl: "http://localhost:1234/v1",
            api: "openai-completions",
            apiKey: "LM_API_TOKEN",
            models: [createModel("stale-model", "Stale Model")],
          },
        },
      },
    } satisfies OpenClawConfig;

    const result = await promptAndConfigureLmstudioInteractive({
      config,
      promptText,
    });
    const mergedConfig = mergeConfigPatch(config, result.configPatch);

    expect(result.configPatch?.models?.providers?.lmstudio?.models).toMatchObject([
      {
        id: "fresh-model",
        name: "Fresh Model",
      },
    ]);
    expect(mergedConfig.agents?.defaults?.models).toEqual({
      "openai/gpt-5.4": { alias: "gpt" },
      "lmstudio/stale-model": { alias: "stale" },
      "lmstudio/fresh-model": { alias: "fresh" },
    });
    expect(result.configPatch?.models?.providers?.lmstudio?.models).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "stale-model" })]),
    );
    expect(result.defaultModel).toBe("lmstudio/fresh-model");
  });

  it("interactive setup prefers the canonical LM Studio default when it is discovered", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "other-model",
        },
        {
          type: "llm",
          key: "qwen/qwen3.5-9b",
        },
      ],
    });

    const result = await promptAndConfigureLmstudioInteractive({
      config: buildConfig(),
      promptText,
    });

    expect(result.defaultModel).toBe("lmstudio/qwen/qwen3.5-9b");
  });

  it("interactive setup prefers loaded instance context length over max_context_length", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [
        {
          type: "llm",
          key: "qwen3-8b-instruct",
          max_context_length: 262144,
          loaded_instances: [{ id: "inst-1", config: { context_length: 64000 } }],
        },
      ],
    });

    const result = await promptAndConfigureLmstudioInteractive({
      config: buildConfig(),
      promptText,
    });

    expect(result.configPatch?.models?.providers?.lmstudio?.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "qwen3-8b-instruct",
          contextWindow: 64000,
          maxTokens: 8192,
        }),
      ]),
    );
  });

  it("skips discovery fetch when explicit models are configured", async () => {
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
        apiKey: "LM_API_TOKEN",
        discoveryApiKey: "env-lmstudio-key",
      }),
    );

    expect(buildLmstudioProviderMock).not.toHaveBeenCalled();
    expect(result).toEqual({
      provider: {
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        auth: "api-key",
        apiKey: "LM_API_TOKEN",
        models: explicitModels,
      },
    });
  });

  it("keeps resolved SecretRef headers when explicit models short-circuit discovery", async () => {
    const explicitModels = [createModel("qwen3-8b-instruct", "Qwen3 8B")];
    const result = await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                headers: {
                  "X-Proxy-Auth": {
                    source: "env",
                    provider: "default",
                    id: "LMSTUDIO_PROXY_TOKEN",
                  },
                },
                models: explicitModels,
              },
            },
          },
        } as OpenClawConfig,
        apiKey: "LM_API_TOKEN",
        env: {
          LMSTUDIO_PROXY_TOKEN: "proxy-token-from-env",
        },
      }),
    );

    expect(buildLmstudioProviderMock).not.toHaveBeenCalled();
    expect(result?.provider.headers).toEqual({
      "X-Proxy-Auth": "proxy-token-from-env",
    });
  });

  it("adds a local auth marker when explicit local LM Studio models have no apiKey", async () => {
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
    expect(result?.provider).not.toHaveProperty("auth");
  });

  it("cancels interactive setup when LM Studio is unreachable", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: false,
      models: [],
    });
    const note = vi.fn();

    await expect(
      promptAndConfigureLmstudioInteractive({
        config: buildConfig(),
        promptText,
        note,
      }),
    ).rejects.toThrow("LM Studio not reachable");
    expect(note).toHaveBeenCalledWith(
      expect.stringContaining("LM Studio could not be reached at http://localhost:1234/v1."),
      "LM Studio",
    );
  });

  it("cancels interactive setup when LM Studio returns an HTTP error", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 401,
      models: [],
    });
    const note = vi.fn();

    await expect(
      promptAndConfigureLmstudioInteractive({
        config: buildConfig(),
        promptText,
        note,
      }),
    ).rejects.toThrow("LM Studio discovery failed (401)");
    expect(note).toHaveBeenCalledWith(
      expect.stringContaining(
        "LM Studio returned HTTP 401 while listing models at http://localhost:1234/v1.",
      ),
      "LM Studio",
    );
    expect(note.mock.calls[0]?.[0]).toContain("Check the base URL and API key");
  });

  it("cancels interactive setup when no LM Studio models are available", async () => {
    const promptText = vi
      .fn()
      .mockResolvedValueOnce("http://localhost:1234/v1")
      .mockResolvedValueOnce("lmstudio-test-key");
    fetchLmstudioModelsMock.mockResolvedValueOnce({
      reachable: true,
      status: 200,
      models: [{ type: "embedding", key: "text-embedding-nomic-embed-text-v1.5" }],
    });
    const note = vi.fn();

    await expect(
      promptAndConfigureLmstudioInteractive({
        config: buildConfig(),
        promptText,
        note,
      }),
    ).rejects.toThrow("No LM Studio models found");
    expect(note).toHaveBeenCalledWith(
      expect.stringContaining("No LM Studio LLM models were found at http://localhost:1234/v1."),
      "LM Studio",
    );
  });

  it("uses quiet discovery when LM Studio is not explicitly configured and no key is available", async () => {
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

  it("uses non-quiet discovery when runtime LM Studio key is available", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [],
    });

    const result = await discoverLmstudioProvider(
      buildDiscoveryContext({
        apiKey: "LM_API_TOKEN",
        discoveryApiKey: "env-lmstudio-key",
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: undefined,
      apiKey: "env-lmstudio-key",
      quiet: false,
    });
    expect(result).toEqual({
      provider: {
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        auth: "api-key",
        apiKey: "LM_API_TOKEN",
        models: [],
      },
    });
  });

  it("uses explicit usable LM Studio key for discovery when available", async () => {
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
                apiKey: "LM_API_TOKEN",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        apiKey: undefined,
        env: {
          LM_API_TOKEN: "env-lmstudio-key",
        },
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "env-lmstudio-key",
      headers: undefined,
      quiet: false,
    });
    expect(result?.provider.models?.map((model) => model.id)).toEqual(["qwen3-8b-instruct"]);
  });

  it("resolves SecretRef-backed LM Studio apiKey during discovery", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    await discoverLmstudioProvider(
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
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        env: {
          LMSTUDIO_DISCOVERY_TOKEN: "secretref-lmstudio-key",
        },
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "secretref-lmstudio-key",
      headers: undefined,
      quiet: false,
    });
  });

  it("falls back to keyless discovery when SecretRef-backed LM Studio apiKey cannot be resolved", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    await discoverLmstudioProvider(
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
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        env: {},
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: undefined,
      headers: undefined,
      quiet: false,
    });
  });

  it("adds a local auth marker for discovered authless local LM Studio configs", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    const result = await discoverLmstudioProvider(buildDiscoveryContext());

    expect(result).toEqual({
      provider: {
        baseUrl: "http://localhost:1234/v1",
        api: "openai-completions",
        apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER,
        models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
      },
    });
    expect(result?.provider).not.toHaveProperty("auth");
  });

  it("adds a local auth marker for discovered authless non-local LM Studio configs", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://lmstudio.internal:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    const result = await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://lmstudio.internal:1234/v1",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
      }),
    );

    expect(result).toEqual({
      provider: {
        baseUrl: "http://lmstudio.internal:1234/v1",
        api: "openai-completions",
        apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER,
        models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
      },
    });
    expect(result?.provider).not.toHaveProperty("auth");
  });

  it("forwards configured LM Studio headers during discovery", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                headers: {
                  "X-Proxy-Auth": "proxy-token",
                },
                models: [],
              },
            },
          },
        } as OpenClawConfig,
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: undefined,
      headers: {
        "X-Proxy-Auth": "proxy-token",
      },
      quiet: false,
    });
  });

  it("resolves SecretRef-backed LM Studio headers during discovery", async () => {
    buildLmstudioProviderMock.mockResolvedValueOnce({
      baseUrl: "http://localhost:1234/v1",
      api: "openai-completions",
      models: [createModel("qwen3-8b-instruct", "Qwen3 8B")],
    });

    await discoverLmstudioProvider(
      buildDiscoveryContext({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
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
          LMSTUDIO_PROXY_TOKEN: "proxy-token-from-env",
        },
      }),
    );

    expect(buildLmstudioProviderMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: undefined,
      headers: {
        "X-Proxy-Auth": "proxy-token-from-env",
      },
      quiet: false,
    });
  });

  it("keeps resolved SecretRef headers in discovered provider output", async () => {
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
          LMSTUDIO_PROXY_TOKEN: "proxy-token-from-env",
        },
      }),
    );

    expect(result?.provider.headers).toEqual({
      "X-Proxy-Auth": "proxy-token-from-env",
    });
  });
});
