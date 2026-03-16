import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { createLmstudioEmbeddingProvider } from "./embeddings-lmstudio.js";

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

describe("embeddings-lmstudio", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    ensureLmstudioModelLoadedMock.mockReset();
    resolveLmstudioRuntimeApiKeyMock.mockReset();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("calls /embeddings on LM Studio inference base and ensures model load", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

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
      remote: {
        headers: { "X-Remote": "remote" },
      },
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
          "X-Remote": "remote",
        }),
      }),
    );
    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "profile-lmstudio-key",
      headers: {
        "X-Provider": "provider",
        "X-Remote": "remote",
      },
      ssrfPolicy: { allowedHostnames: ["localhost"] },
      modelKey: "text-embedding-nomic-embed-text-v1.5",
      timeoutMs: 120_000,
    });
    expect(resolveLmstudioRuntimeApiKeyMock).toHaveBeenCalledTimes(1);
    expect(resolveLmstudioRuntimeApiKeyMock).toHaveBeenCalledWith(
      expect.objectContaining({ allowMissingAuth: true }),
    );
  });

  it("resolves SecretRef-backed LM Studio provider headers for embeddings", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");
    vi.stubEnv("LMSTUDIO_PROXY_TOKEN", "proxy-token");

    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              headers: {
                "X-Provider": {
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
      provider: "lmstudio",
      model: "text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
      remote: {
        headers: { "X-Remote": "remote" },
      },
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Provider": "proxy-token",
          "X-Remote": "remote",
        }),
      }),
    );
    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledWith({
      baseUrl: "http://localhost:1234/v1",
      apiKey: "profile-lmstudio-key",
      headers: {
        "X-Provider": "proxy-token",
        "X-Remote": "remote",
      },
      ssrfPolicy: { allowedHostnames: ["localhost"] },
      modelKey: "text-embedding-nomic-embed-text-v1.5",
      timeoutMs: 120_000,
    });
  });

  it("uses remote apiKey and remote baseUrl when provided", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-key");
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [1, 2, 3] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "lmstudio",
      model: "",
      fallback: "none",
      remote: {
        baseUrl: "http://localhost:1234",
        apiKey: "remote-lmstudio-key",
      },
    });

    await provider.embedBatch(["one", "two"]);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer remote-lmstudio-key",
        }),
      }),
    );
    expect(resolveLmstudioRuntimeApiKeyMock).not.toHaveBeenCalled();
  });

  it("keeps resolved api key when header overrides include Authorization", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [1, 2, 3] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createLmstudioEmbeddingProvider({
      config: {
        models: {
          providers: {
            lmstudio: {
              baseUrl: "http://localhost:1234/v1",
              headers: { Authorization: "Bearer provider-override" },
              models: [],
            },
          },
        },
      } as OpenClawConfig,
      provider: "lmstudio",
      model: "text-embedding-nomic-embed-text-v1.5",
      fallback: "none",
      remote: {
        headers: { Authorization: "Bearer remote-override" },
      },
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:1234/v1/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer profile-lmstudio-key",
        }),
      }),
    );
  });

  it("works without an API key for keyless local LM Studio", async () => {
    ensureLmstudioModelLoadedMock.mockResolvedValue(undefined);
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue(undefined);

    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [1, 2, 3] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

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

    // No Authorization header when key is absent.
    const callHeaders = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(callHeaders.Authorization).toBeUndefined();
    expect(callHeaders["Content-Type"]).toBe("application/json");
  });

  it("fails fast when LM Studio auth is api-key and runtime key is missing", async () => {
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

  it("continues when LM Studio warmup fails", async () => {
    ensureLmstudioModelLoadedMock.mockRejectedValue(new Error("warmup failed"));
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [1, 2, 3] }],
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

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

  it("does not block provider construction while LM Studio warmup is in flight", async () => {
    ensureLmstudioModelLoadedMock.mockImplementation(() => new Promise(() => {}));
    resolveLmstudioRuntimeApiKeyMock.mockResolvedValue("profile-lmstudio-key");

    const createPromise = createLmstudioEmbeddingProvider({
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

    const raced = await Promise.race([
      createPromise.then(() => "resolved"),
      new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 200)),
    ]);

    expect(raced).toBe("resolved");
    expect(ensureLmstudioModelLoadedMock).toHaveBeenCalledTimes(1);
  });

  it("surfaces runtime auth resolution failures", async () => {
    resolveLmstudioRuntimeApiKeyMock.mockRejectedValue(new Error("missing auth profile"));

    await expect(
      createLmstudioEmbeddingProvider({
        config: {
          models: {
            providers: {
              lmstudio: {
                baseUrl: "http://localhost:1234/v1",
                apiKey: "LM_API_TOKEN",
                models: [],
              },
            },
          },
        } as OpenClawConfig,
        provider: "lmstudio",
        model: "text-embedding-nomic-embed-text-v1.5",
        fallback: "none",
      }),
    ).rejects.toThrow("missing auth profile");
    expect(ensureLmstudioModelLoadedMock).not.toHaveBeenCalled();
  });
});
