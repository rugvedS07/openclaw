import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";

const ensureLmstudioModelLoadedMock = vi.hoisted(() => vi.fn());
const resolveLmstudioRuntimeApiKeyMock = vi.hoisted(() => vi.fn());

vi.mock("../agents/lmstudio-models.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../agents/lmstudio-models.js")>();
  return {
    ...actual,
    ensureLmstudioModelLoaded: (...args: unknown[]) => ensureLmstudioModelLoadedMock(...args),
  };
});

vi.mock("../agents/lmstudio-runtime.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../agents/lmstudio-runtime.js")>();
  return {
    ...actual,
    resolveLmstudioRuntimeApiKey: (...args: unknown[]) => resolveLmstudioRuntimeApiKeyMock(...args),
  };
});

let createLmstudioEmbeddingProvider: typeof import("./embeddings-lmstudio.js").createLmstudioEmbeddingProvider;

describe("embeddings-lmstudio", () => {
  const originalFetch = globalThis.fetch;
  const jsonResponse = (embedding: number[]) =>
    new Response(
      JSON.stringify({
        data: [{ embedding }],
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    );

  function mockEmbeddingFetch(embedding: number[]) {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(jsonResponse(embedding));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    return fetchMock;
  }

  beforeEach(async () => {
    vi.resetModules();
    ({ createLmstudioEmbeddingProvider } = await import("./embeddings-lmstudio.js"));
    ensureLmstudioModelLoadedMock.mockReset();
    resolveLmstudioRuntimeApiKeyMock.mockReset();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("embeds against inference base and warms model with resolved key", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    const fetchMock = mockEmbeddingFetch([0.1, 0.2]);

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/api/v1/",
              headers: { "X-Provider": "provider" },
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "lmstudio/text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          Authorization: "Bearer profile-lmstudio-key",
          "X-Provider": "provider",
        }),
      }),
    );
    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "profile-lmstudio-key",
      headers: {
        "X-Provider": "provider",
      },
      ssrfPolicy: { allowedHostnames: ["localhost"] },
      modelKey: "text-embedding-nomic-embed-text-v1.5",
      timeoutMs: 120_000,
    });
  });

  it("ignores memorySearch remote overrides and uses lmstudio provider config", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-key");

    const fetchMock = mockEmbeddingFetch([1, 2, 3]);

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234",
              headers: {
                "X-Provider": "provider",
                "X-Config-Only": "from-provider",
              },
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "",
      fallback: "none",
      remote: {
        baseUrl: "http://localhost:9999",
        apiKey: "remote-lmstudio-key",
        headers: {
          "X-Provider": "remote",
          "X-Remote-Only": "from-remote",
        },
      },
    });

    await provider.embedBatch(["one", "two"]);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer profile-key",
          "X-Provider": "provider",
          "X-Config-Only": "from-provider",
        }),
      }),
    );
    const callHeaders = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(callHeaders["X-Remote-Only"]).toBeUndefined();
    expect(resolveLmstudioRuntimeApiKeyMock).toHaveBeenCalled();
    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "profile-key",
      headers: {
        "X-Provider": "provider",
        "X-Config-Only": "from-provider",
      },
      ssrfPolicy: { allowedHostnames: ["localhost"] },
      modelKey: "text-embedding-nomic-embed-text-v1.5",
      timeoutMs: 120_000,
    });
  });

  it("allows keyless local LM Studio", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue(undefined);

    const fetchMock = mockEmbeddingFetch([1, 2, 3]);

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    const callHeaders = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(callHeaders.Authorization).toBeUndefined();
    expect(callHeaders["Content-Type"]).toBe("application/json");
  });

  it("fails fast when explicit api-key auth cannot be resolved", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockRejectedValue(
      new Error('No API key found for provider "lmstudio".'),
    );

    await expect(
      createLmstudioEmbeddingProvider({
        config: {
          models: {
            providers: {
              lmstudio: {
                auth: "api-key",
                baseUrl: "http://localhost:1234/v1",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        provider: "lmstudio",
        model: "text-embedding-nomic-embed-text-v1.5",
        fallback: "none",
      }),
    ).rejects.toThrow('No API key found for provider "lmstudio".');

    expect(resolveLmstudioRuntimeApiKeyMock).toHaveBeenCalledWith(
      expect.objectContaining({ allowMissingAuth: false }),
    );
  });

  it("continues embedding when warmup fails", async () => {
    ensureLmstudioModelLoadedMock.mockRejectedValue(new Error("warmup failed"));
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    const fetchMock = mockEmbeddingFetch([1, 2, 3]);

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        method: "POST",
      }),
    );
  });

  it("awaits warmup before returning provider", async () => {
    let resolveWarmup: (() => void) | undefined;
    ensureLmstudioModelLoadedMock.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveWarmup = resolve;
        }),
    );
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    let settled = false;
    const providerPromise = createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
    }).then(() => {
      settled = true;
    });

    await Promise.resolve();
    expect(settled).toBe(false);

    resolveWarmup?.();
    await providerPromise;
    expect(settled).toBe(true);
  });
});
