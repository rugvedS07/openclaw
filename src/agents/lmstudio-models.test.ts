import { afterEach, describe, expect, it, vi } from "vitest";
import { LMSTUDIO_DEFAULT_LOAD_CONTEXT_LENGTH } from "./lmstudio-defaults.js";
import {
  clearLmstudioWarmupCache,
  discoverLmstudioModels,
  ensureLmstudioModelLoaded,
  resolveLmstudioReasoningCapability,
  resolveLmstudioInferenceBase,
  resolveLmstudioServerBase,
  warmupLmstudioModelBestEffort,
} from "./lmstudio-models.js";
import {
  SELF_HOSTED_DEFAULT_CONTEXT_WINDOW,
  SELF_HOSTED_DEFAULT_MAX_TOKENS,
} from "./self-hosted-provider-defaults.js";

const fetchWithSsrFGuardMock = vi.hoisted(() => vi.fn());

vi.mock("../infra/net/fetch-guard.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../infra/net/fetch-guard.js")>();
  return {
    ...actual,
    fetchWithSsrFGuard: (...args: unknown[]) => fetchWithSsrFGuardMock(...args),
  };
});

describe("lmstudio-models", () => {
  afterEach(() => {
    clearLmstudioWarmupCache();
    fetchWithSsrFGuardMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("normalizes server and inference base URLs", () => {
    expect(resolveLmstudioServerBase()).toBe("http://localhost:1234");
    expect(resolveLmstudioInferenceBase()).toBe("http://localhost:1234/v1");
    expect(resolveLmstudioServerBase("http://localhost:1234/v1/")).toBe("http://localhost:1234");
    expect(resolveLmstudioServerBase("http://localhost:1234/api/v1")).toBe("http://localhost:1234");
    expect(resolveLmstudioInferenceBase("http://localhost:1234/api/v1")).toBe(
      "http://localhost:1234/v1",
    );
  });

  it("parses reasoning capability variants and defaults missing reasoning to false", () => {
    expect(resolveLmstudioReasoningCapability({ capabilities: undefined })).toBe(false);
    expect(
      resolveLmstudioReasoningCapability({
        capabilities: {
          reasoning: {
            allowed_options: ["low", "medium", "high"],
            default: "low",
          },
        },
      }),
    ).toBe(true);
    expect(
      resolveLmstudioReasoningCapability({
        capabilities: {
          reasoning: {
            allowed_options: ["off", "on"],
            default: "on",
          },
        },
      }),
    ).toBe(true);
    expect(
      resolveLmstudioReasoningCapability({
        capabilities: {
          reasoning: {
            allowed_options: ["off", "low", "on"],
            default: "on",
          },
        },
      }),
    ).toBe(true);
    expect(
      resolveLmstudioReasoningCapability({
        capabilities: {
          reasoning: {
            allowed_options: ["off"],
            default: "off",
          },
        },
      }),
    ).toBe(false);
  });

  it("discovers llm models from /api/v1/models and maps metadata", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        models: [
          {
            type: "llm",
            key: "qwen3-8b-instruct",
            display_name: "Qwen3 8B",
            max_context_length: 262144,
            format: "mlx",
            capabilities: {
              vision: true,
              trained_for_tool_use: true,
              reasoning: {
                allowed_options: ["off", "on"],
                default: "on",
              },
            },
            loaded_instances: [{ id: "inst-1", config: { context_length: 64000 } }],
          },
          {
            type: "llm",
            key: "deepseek-r1",
          },
          {
            type: "embedding",
            key: "text-embedding-nomic-embed-text-v1.5",
          },
          {
            type: "llm",
            key: "   ",
          },
        ],
      }),
    }));

    const models = await discoverLmstudioModels({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "lm-token",
      quiet: false,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/api/v1/models",
      expect.objectContaining({
        headers: {
          Authorization: "Bearer lm-token",
        },
      }),
    );

    expect(models).toHaveLength(2);
    expect(models[0]).toEqual({
      id: "qwen3-8b-instruct",
      name: "Qwen3 8B (MLX, vision, tool-use, loaded)",
      reasoning: true,
      input: ["text", "image"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 64000,
      maxTokens: SELF_HOSTED_DEFAULT_MAX_TOKENS,
    });
    expect(models[1]).toEqual({
      id: "deepseek-r1",
      name: "deepseek-r1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: SELF_HOSTED_DEFAULT_CONTEXT_WINDOW,
      maxTokens: SELF_HOSTED_DEFAULT_MAX_TOKENS,
    });
  });

  it("ignores malformed loaded_instances entries during discovery", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        models: [
          {
            type: "llm",
            key: "qwen3-8b-instruct",
            max_context_length: 32768,
            loaded_instances: [
              null,
              {},
              { id: "bad-null", config: null },
              { id: "bad-negative", config: { context_length: -1 } },
              { id: "ok", config: { context_length: 16384 } },
            ],
          },
        ],
      }),
    }));

    const models = await discoverLmstudioModels({
      baseUrl: resolveLmstudioInferenceBase(),
      apiKey: "",
      quiet: false,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    expect(models).toHaveLength(1);
    expect(models[0]?.contextWindow).toBe(16384);
  });

  it("does not call /models/load when model is already loaded", async () => {
    const fetchMock = vi.fn(async (_url: string | URL, _init?: RequestInit) => ({
      ok: true,
      json: async () => ({
        models: [
          {
            type: "llm",
            key: "qwen3-8b-instruct",
            loaded_instances: [{ id: "inst-1", config: { context_length: 64000 } }],
          },
        ],
      }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        modelKey: "qwen3-8b-instruct",
      }),
    ).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstCall = fetchMock.mock.calls.at(0);
    expect(firstCall).toBeDefined();
    expect(String(firstCall?.[0])).toBe("http://localhost:1234/api/v1/models");
    expect(
      fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/api/v1/models/load")),
    ).toBe(false);
  });

  it("loads model when loaded_instances payload is malformed", async () => {
    const fetchMock = vi.fn(async (url: string | URL, _init?: RequestInit) => {
      if (String(url).endsWith("/api/v1/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [
              {
                type: "llm",
                key: "qwen3-8b-instruct",
                loaded_instances: [null, {}, { id: "missing-config" }, { id: "bad", config: {} }],
              },
            ],
          }),
        };
      }
      if (String(url).endsWith("/api/v1/models/load")) {
        return {
          ok: true,
          json: async () => ({ status: "loaded" }),
        };
      }
      throw new Error(`Unexpected fetch URL: ${String(url)}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        modelKey: "qwen3-8b-instruct",
      }),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(
      fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/api/v1/models/load")),
    ).toBe(true);
  });

  it("does not call /models/load when model discovery is unreachable", async () => {
    const fetchMock = vi.fn(async (_url: string | URL, _init?: RequestInit) => {
      throw new Error("connect ECONNREFUSED localhost:1234");
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        modelKey: "qwen3-8b-instruct",
      }),
    ).rejects.toThrow("LM Studio model discovery failed");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstCall = fetchMock.mock.calls.at(0);
    expect(firstCall).toBeDefined();
    expect(String(firstCall?.[0])).toBe("http://localhost:1234/api/v1/models");
    expect(
      fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/api/v1/models/load")),
    ).toBe(false);
  });

  it("does not call /models/load when model discovery returns an HTTP error", async () => {
    const fetchMock = vi.fn(async (_url: string | URL, _init?: RequestInit) => ({
      ok: false,
      status: 401,
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        modelKey: "qwen3-8b-instruct",
      }),
    ).rejects.toThrow("LM Studio model discovery failed (401)");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstCall = fetchMock.mock.calls.at(0);
    expect(firstCall).toBeDefined();
    expect(String(firstCall?.[0])).toBe("http://localhost:1234/api/v1/models");
    expect(
      fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/api/v1/models/load")),
    ).toBe(false);
  });

  it("loads model when not currently loaded and sends default context length", async () => {
    const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
      if (String(url).endsWith("/api/v1/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [
              {
                type: "llm",
                key: "qwen3-8b-instruct",
                loaded_instances: [],
              },
            ],
          }),
        };
      }
      if (String(url).endsWith("/api/v1/models/load")) {
        return {
          ok: true,
          json: async () => ({ status: "loaded" }),
          requestInit: init,
        };
      }
      throw new Error(`Unexpected fetch URL: ${String(url)}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lm-token",
        modelKey: " qwen3-8b-instruct ",
      }),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const loadCall = fetchMock.mock.calls.find((call) => String(call[0]).endsWith("/models/load"));
    expect(loadCall).toBeDefined();
    const loadInit = loadCall?.[1] ?? {};
    expect(loadInit.method).toBe("POST");
    expect(loadInit.headers).toEqual({
      Authorization: "Bearer lm-token",
      "Content-Type": "application/json",
    });
    expect(loadInit.body).toBe(
      JSON.stringify({
        model: "qwen3-8b-instruct",
        context_length: LMSTUDIO_DEFAULT_LOAD_CONTEXT_LENGTH,
      }),
    );
  });

  it("clamps model load context length to the advertised model limit", async () => {
    const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
      if (String(url).endsWith("/api/v1/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [
              {
                type: "llm",
                key: "qwen3-8b-instruct",
                max_context_length: 32768,
                loaded_instances: [],
              },
            ],
          }),
        };
      }
      if (String(url).endsWith("/api/v1/models/load")) {
        return {
          ok: true,
          json: async () => ({ status: "loaded" }),
          requestInit: init,
        };
      }
      throw new Error(`Unexpected fetch URL: ${String(url)}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        modelKey: "qwen3-8b-instruct",
      }),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const loadCall = fetchMock.mock.calls.find((call) => String(call[0]).endsWith("/models/load"));
    expect(loadCall).toBeDefined();
    const loadInit = loadCall?.[1] ?? {};
    expect(loadInit.body).toBe(
      JSON.stringify({
        model: "qwen3-8b-instruct",
        context_length: 32768,
      }),
    );
  });

  it("includes configured headers in discovery and load requests", async () => {
    const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
      if (String(url).endsWith("/api/v1/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [
              {
                type: "llm",
                key: "qwen3-8b-instruct",
                loaded_instances: [],
              },
            ],
          }),
          requestInit: init,
        };
      }
      if (String(url).endsWith("/api/v1/models/load")) {
        return {
          ok: true,
          json: async () => ({ status: "loaded" }),
          requestInit: init,
        };
      }
      throw new Error(`Unexpected fetch URL: ${String(url)}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lm-token",
        headers: {
          "X-Proxy-Auth": "required",
          Authorization: "Bearer override",
        },
        modelKey: "qwen3-8b-instruct",
      }),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const preflightCall = fetchMock.mock.calls.at(0);
    expect(preflightCall).toBeDefined();
    const preflightInit = preflightCall?.[1];
    expect(preflightInit?.headers).toEqual({
      "X-Proxy-Auth": "required",
      Authorization: "Bearer lm-token",
    });
    const loadCall = fetchMock.mock.calls.at(1);
    expect(loadCall).toBeDefined();
    const loadInit = loadCall?.[1];
    expect(loadInit?.headers).toEqual({
      "X-Proxy-Auth": "required",
      Authorization: "Bearer lm-token",
      "Content-Type": "application/json",
    });
  });

  it("routes discovery and load through the SSRF guard when a policy is provided", async () => {
    const releaseDiscovery = vi.fn(async () => {});
    const releaseLoad = vi.fn(async () => {});
    fetchWithSsrFGuardMock
      .mockResolvedValueOnce({
        response: new Response(
          JSON.stringify({
            models: [{ type: "llm", key: "qwen3-8b-instruct", loaded_instances: [] }],
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          },
        ),
        finalUrl: "http://localhost:1234/api/v1/models",
        release: releaseDiscovery,
      })
      .mockResolvedValueOnce({
        response: new Response(JSON.stringify({ status: "loaded" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
        finalUrl: "http://localhost:1234/api/v1/models/load",
        release: releaseLoad,
      });
    const directFetchSpy = vi.fn(() => {
      throw new Error("raw fetch should not be used when ssrfPolicy is set");
    });
    vi.stubGlobal("fetch", directFetchSpy as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        baseUrl: "http://localhost:1234/v1",
        apiKey: "lm-token",
        headers: {
          "X-Proxy-Auth": "required",
        },
        ssrfPolicy: { allowedHostnames: ["localhost"] },
        modelKey: "qwen3-8b-instruct",
      }),
    ).resolves.toBeUndefined();

    expect(directFetchSpy).not.toHaveBeenCalled();
    expect(fetchWithSsrFGuardMock).toHaveBeenCalledTimes(2);
    expect(fetchWithSsrFGuardMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        url: "http://localhost:1234/api/v1/models",
        policy: { allowedHostnames: ["localhost"] },
        auditContext: "lmstudio-model-discovery",
      }),
    );
    expect(fetchWithSsrFGuardMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        url: "http://localhost:1234/api/v1/models/load",
        policy: { allowedHostnames: ["localhost"] },
        auditContext: "lmstudio-model-load",
      }),
    );
    expect(releaseDiscovery).toHaveBeenCalledOnce();
    expect(releaseLoad).toHaveBeenCalledOnce();
  });

  it("throws when model load returns non-success status", async () => {
    const fetchMock = vi.fn(async (url: string | URL) => {
      if (String(url).endsWith("/api/v1/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [{ type: "llm", key: "qwen3-8b-instruct", loaded_instances: [] }],
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ status: "queued" }),
      };
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      ensureLmstudioModelLoaded({
        modelKey: "qwen3-8b-instruct",
      }),
    ).rejects.toThrow("unexpected status");
  });

  it("allows lifecycle hooks to clear in-flight warmups", async () => {
    const resolvers: Array<() => void> = [];
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolvers.push(() =>
            resolve({
              ok: true,
              json: async () => ({
                models: [
                  {
                    type: "llm",
                    key: "qwen3-8b-instruct",
                    loaded_instances: [{ id: "inst-1", config: { context_length: 64000 } }],
                  },
                ],
              }),
            } as Response),
          );
        }),
    );
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    warmupLmstudioModelBestEffort({
      baseUrl: "http://localhost:1234/v1",
      modelKey: "qwen3-8b-instruct",
    });
    warmupLmstudioModelBestEffort({
      baseUrl: "http://localhost:1234/v1",
      modelKey: "qwen3-8b-instruct",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    clearLmstudioWarmupCache();

    warmupLmstudioModelBestEffort({
      baseUrl: "http://localhost:1234/v1",
      modelKey: "qwen3-8b-instruct",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    for (const resolve of resolvers) {
      resolve();
    }
    await vi.waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  });
});
