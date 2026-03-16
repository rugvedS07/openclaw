import { beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER } from "./lmstudio-defaults.js";
import {
  buildLmstudioAuthHeaders,
  resolveLmstudioConfiguredApiKey,
  resolveLmstudioProviderHeaders,
  resolveLmstudioRuntimeApiKey,
} from "./lmstudio-runtime.js";

const resolveApiKeyForProviderMock = vi.hoisted(() => vi.fn());

vi.mock("./model-auth.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./model-auth.js")>();
  return {
    ...actual,
    resolveApiKeyForProvider: (...args: unknown[]) => resolveApiKeyForProviderMock(...args),
  };
});

describe("lmstudio-runtime", () => {
  beforeEach(() => {
    resolveApiKeyForProviderMock.mockReset();
  });

  it("uses shared provider auth precedence for runtime API key resolution", async () => {
    resolveApiKeyForProviderMock.mockResolvedValue({
      apiKey: "profile-lmstudio-key",
      profileId: "lmstudio:default",
      source: "profile:lmstudio:default",
      mode: "api-key",
    });

    const config = {
      models: {
        providers: {
          lmstudio: {
            apiKey: "LM_API_TOKEN",
            api: "openai-completions",
            baseUrl: "http://localhost:1234/v1",
            models: [],
          },
        },
      },
    } as OpenClawConfig;

    await expect(
      resolveLmstudioRuntimeApiKey({
        config,
        agentDir: "/tmp/lmstudio-agent",
      }),
    ).resolves.toBe("profile-lmstudio-key");

    expect(resolveApiKeyForProviderMock).toHaveBeenCalledWith({
      provider: "lmstudio",
      cfg: config,
      agentDir: "/tmp/lmstudio-agent",
    });
  });

  it("does not resolve SecretRef config apiKey when profile auth already resolves", async () => {
    resolveApiKeyForProviderMock.mockResolvedValue({
      apiKey: "profile-lmstudio-key",
      profileId: "lmstudio:default",
      source: "profile:lmstudio:default",
      mode: "api-key",
    });

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: {
                  source: "env",
                  provider: "default",
                  id: "LM_API_TOKEN",
                },
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        env: {},
      }),
    ).resolves.toBe("profile-lmstudio-key");
  });

  it("normalizes blank runtime API keys to the keyless auth marker", async () => {
    resolveApiKeyForProviderMock.mockResolvedValue({
      apiKey: "   ",
      source: "profile:lmstudio:default",
      mode: "api-key",
    });

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {} as OpenClawConfig,
      }),
    ).resolves.toBe(LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("falls back to the keyless auth marker when auth resolution fails", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(new Error("missing auth"));

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {} as OpenClawConfig,
      }),
    ).resolves.toBe(LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("returns the local no-auth marker when missing auth is allowed for default local LM Studio", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(
      new Error('No API key found for provider "lmstudio". Auth store: /tmp/auth-profiles.json.'),
    );

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {} as OpenClawConfig,
        allowMissingAuth: true,
      }),
    ).resolves.toBe(LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("returns the keyless auth marker when missing auth is allowed", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(
      new Error('No API key found for provider "lmstudio". Auth store: /tmp/auth-profiles.json.'),
    );

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {} as OpenClawConfig,
        allowMissingAuth: true,
      }),
    ).resolves.toBe(LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("still falls back to the keyless auth marker when auth store access fails", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(new Error("auth profile store unreadable"));

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {} as OpenClawConfig,
        allowMissingAuth: true,
      }),
    ).resolves.toBe(LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER);
  });

  it("resolves SecretRef-backed provider apiKey from config", async () => {
    await expect(
      resolveLmstudioConfiguredApiKey({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                apiKey: {
                  source: "env",
                  provider: "default",
                  id: "LM_API_TOKEN",
                },
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        env: {
          LM_API_TOKEN: "secretref-lmstudio-key",
        },
      }),
    ).resolves.toBe("secretref-lmstudio-key");
  });

  it("does not use local fallback marker when auth is explicitly api-key and apiKey is missing", async () => {
    await expect(
      resolveLmstudioConfiguredApiKey({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                auth: "api-key",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        allowLocalFallback: true,
      }),
    ).resolves.toBeUndefined();
  });

  it("returns undefined for explicit api-key auth when auth is missing and allowMissingAuth is true", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(
      new Error('No API key found for provider "lmstudio". Auth store: /tmp/auth-profiles.json.'),
    );

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                api: "openai-completions",
                auth: "api-key",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        allowMissingAuth: true,
      }),
    ).resolves.toBeUndefined();
  });

  describe("resolveLmstudioProviderHeaders", () => {
    it("resolves SecretRef-backed provider headers from config", async () => {
      const resolved = await resolveLmstudioProviderHeaders({
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
          LMSTUDIO_PROXY_TOKEN: "proxy-token",
        },
        headers: {
          "X-Proxy-Auth": {
            source: "env",
            provider: "default",
            id: "LMSTUDIO_PROXY_TOKEN",
          },
        },
      });

      expect(resolved).toEqual({
        "X-Proxy-Auth": "proxy-token",
      });
    });

    it("throws a path-specific error when a SecretRef header cannot be resolved", async () => {
      await expect(
        resolveLmstudioProviderHeaders({
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
          env: {},
          headers: {
            "X-Proxy-Auth": {
              source: "env",
              provider: "default",
              id: "LMSTUDIO_PROXY_TOKEN",
            },
          },
        }),
      ).rejects.toThrow(/models\.providers\.lmstudio\.headers\.X-Proxy-Auth/i);
    });
  });

  describe("buildLmstudioAuthHeaders", () => {
    it("returns undefined when no params produce headers", () => {
      expect(buildLmstudioAuthHeaders({})).toBeUndefined();
    });

    it("adds Authorization header when apiKey is provided", () => {
      expect(buildLmstudioAuthHeaders({ apiKey: "sk-test" })).toEqual({
        Authorization: "Bearer sk-test",
      });
    });

    it("trims whitespace from apiKey", () => {
      expect(buildLmstudioAuthHeaders({ apiKey: "  sk-test  " })).toEqual({
        Authorization: "Bearer sk-test",
      });
    });

    it("ignores blank apiKey", () => {
      expect(buildLmstudioAuthHeaders({ apiKey: "   " })).toBeUndefined();
    });

    it("does not synthesize Authorization for the local no-auth marker", () => {
      expect(
        buildLmstudioAuthHeaders({ apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER }),
      ).toBeUndefined();
    });

    it("adds Content-Type when json flag is set", () => {
      expect(buildLmstudioAuthHeaders({ json: true })).toEqual({
        "Content-Type": "application/json",
      });
    });

    it("combines apiKey and json headers", () => {
      expect(buildLmstudioAuthHeaders({ apiKey: "sk-test", json: true })).toEqual({
        Authorization: "Bearer sk-test",
        "Content-Type": "application/json",
      });
    });

    it("merges custom headers with auth headers", () => {
      expect(
        buildLmstudioAuthHeaders({
          apiKey: "sk-test",
          headers: { "X-Proxy": "proxy-token" },
        }),
      ).toEqual({
        "X-Proxy": "proxy-token",
        Authorization: "Bearer sk-test",
      });
    });

    it("apiKey overrides Authorization in custom headers", () => {
      expect(
        buildLmstudioAuthHeaders({
          apiKey: "sk-new",
          headers: { Authorization: "Bearer sk-old" },
        }),
      ).toEqual({
        Authorization: "Bearer sk-new",
      });
    });

    it("apiKey overrides authorization case-insensitively in custom headers", () => {
      expect(
        buildLmstudioAuthHeaders({
          apiKey: "sk-new",
          headers: { authorization: "Bearer sk-old" },
        }),
      ).toEqual({
        Authorization: "Bearer sk-new",
      });
    });

    it("preserves explicit Authorization when apiKey is the local no-auth marker", () => {
      expect(
        buildLmstudioAuthHeaders({
          apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER,
          headers: { Authorization: "Bearer proxy-token" },
        }),
      ).toEqual({
        Authorization: "Bearer proxy-token",
      });
    });

    it("returns custom headers alone when no apiKey or json", () => {
      expect(
        buildLmstudioAuthHeaders({
          headers: { "X-Custom": "value" },
        }),
      ).toEqual({
        "X-Custom": "value",
      });
    });
  });
});
