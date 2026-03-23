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

function buildLmstudioConfig(overrides?: {
  apiKey?: unknown;
  headers?: unknown;
  auth?: "api-key";
}): OpenClawConfig {
  return {
    models: {
      providers: {
        lmstudio: {
          baseUrl: "http://localhost:1234/v1",
          api: "openai-completions",
          ...(overrides?.auth ? { auth: overrides.auth } : {}),
          ...(overrides?.apiKey !== undefined ? { apiKey: overrides.apiKey } : {}),
          ...(overrides?.headers !== undefined ? { headers: overrides.headers } : {}),
          models: [],
        },
      },
    },
  } as OpenClawConfig;
}

describe("lmstudio-runtime", () => {
  beforeEach(() => {
    resolveApiKeyForProviderMock.mockReset();
  });

  it("falls back to keyless marker for blank runtime auth", async () => {
    resolveApiKeyForProviderMock.mockResolvedValueOnce({
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

  it("does not synthesize local marker for explicit api-key auth", async () => {
    resolveApiKeyForProviderMock.mockRejectedValue(
      new Error('No API key found for provider "lmstudio". Auth store: /tmp/auth-profiles.json.'),
    );

    await expect(
      resolveLmstudioRuntimeApiKey({
        config: buildLmstudioConfig({ auth: "api-key" }),
        allowMissingAuth: true,
      }),
    ).resolves.toBeUndefined();

    await expect(
      resolveLmstudioConfiguredApiKey({
        config: buildLmstudioConfig({ auth: "api-key" }),
        allowLocalFallback: true,
      }),
    ).resolves.toBeUndefined();
  });

  it("resolves SecretRef api key and headers", async () => {
    const headerRef = {
      "X-Proxy-Auth": {
        source: "env" as const,
        provider: "default" as const,
        id: "LMSTUDIO_PROXY_TOKEN",
      },
    };
    await expect(
      resolveLmstudioConfiguredApiKey({
        config: buildLmstudioConfig({
          apiKey: {
            source: "env",
            provider: "default",
            id: "LM_API_TOKEN",
          },
        }),
        env: {
          LM_API_TOKEN: "secretref-lmstudio-key",
        },
      }),
    ).resolves.toBe("secretref-lmstudio-key");

    await expect(
      resolveLmstudioProviderHeaders({
        config: buildLmstudioConfig({ headers: headerRef }),
        env: {
          LMSTUDIO_PROXY_TOKEN: "proxy-token",
        },
        headers: headerRef,
      }),
    ).resolves.toEqual({
      "X-Proxy-Auth": "proxy-token",
    });
  });

  it("throws a path-specific error when a SecretRef header cannot be resolved", async () => {
    const headerRef = {
      "X-Proxy-Auth": {
        source: "env" as const,
        provider: "default" as const,
        id: "LMSTUDIO_PROXY_TOKEN",
      },
    };
    await expect(
      resolveLmstudioProviderHeaders({
        config: buildLmstudioConfig({ headers: headerRef }),
        env: {},
        headers: headerRef,
      }),
    ).rejects.toThrow(/models\.providers\.lmstudio\.headers\.X-Proxy-Auth/i);
  });

  it("builds auth headers with key precedence and json support", () => {
    expect(buildLmstudioAuthHeaders({})).toBeUndefined();
    expect(buildLmstudioAuthHeaders({ apiKey: "  sk-test  " })).toEqual({
      Authorization: "Bearer sk-test",
    });
    expect(buildLmstudioAuthHeaders({ apiKey: "   " })).toBeUndefined();
    expect(
      buildLmstudioAuthHeaders({ apiKey: LMSTUDIO_LOCAL_API_KEY_PLACEHOLDER }),
    ).toBeUndefined();
    expect(
      buildLmstudioAuthHeaders({
        apiKey: "sk-new",
        json: true,
        headers: {
          authorization: "Bearer sk-old",
          "X-Proxy": "proxy-token",
        },
      }),
    ).toEqual({
      "Content-Type": "application/json",
      "X-Proxy": "proxy-token",
      Authorization: "Bearer sk-new",
    });
  });
});
