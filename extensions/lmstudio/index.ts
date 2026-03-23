import {
  LMSTUDIO_DEFAULT_API_KEY_ENV_VAR,
  LMSTUDIO_PROVIDER_LABEL,
} from "openclaw/plugin-sdk/lmstudio-defaults";
import {
  definePluginEntry,
  type OpenClawPluginApi,
  type ProviderAuthContext,
  type ProviderAuthMethodNonInteractiveContext,
  type ProviderAuthResult,
  type ProviderDiscoveryContext,
} from "openclaw/plugin-sdk/plugin-entry";

const PROVIDER_ID = "lmstudio";

/** Lazily loads setup helpers so provider wiring stays lightweight at startup. */
async function loadProviderSetup() {
  return await import("openclaw/plugin-sdk/lmstudio-setup");
}

export default definePluginEntry({
  id: PROVIDER_ID,
  name: "LM Studio Provider",
  description: "Bundled LM Studio provider plugin",
  register(api: OpenClawPluginApi) {
    api.registerProvider({
      id: PROVIDER_ID,
      label: "LM Studio",
      docsPath: "/providers/lmstudio",
      envVars: [LMSTUDIO_DEFAULT_API_KEY_ENV_VAR],
      auth: [
        {
          id: "custom",
          label: LMSTUDIO_PROVIDER_LABEL,
          hint: "Local/self-hosted LM Studio server",
          kind: "custom",
          run: async (ctx: ProviderAuthContext): Promise<ProviderAuthResult> => {
            const providerSetup = await loadProviderSetup();
            return await providerSetup.promptAndConfigureLmstudioInteractive({
              config: ctx.config,
              prompter: ctx.prompter,
              secretInputMode: ctx.secretInputMode,
              allowSecretRefPrompt: ctx.allowSecretRefPrompt,
            });
          },
          runNonInteractive: async (ctx: ProviderAuthMethodNonInteractiveContext) => {
            const providerSetup = await loadProviderSetup();
            return await providerSetup.configureLmstudioNonInteractive(ctx);
          },
        },
      ],
      discovery: {
        // Run after early providers so local LM Studio detection does not dominate resolution.
        order: "late",
        run: async (ctx: ProviderDiscoveryContext) => {
          const providerSetup = await loadProviderSetup();
          return await providerSetup.discoverLmstudioProvider(ctx);
        },
      },
      wizard: {
        setup: {
          choiceId: PROVIDER_ID,
          choiceLabel: "LM Studio",
          choiceHint: "Local/self-hosted LM Studio server",
          groupId: PROVIDER_ID,
          groupLabel: "LM Studio",
          groupHint: "Self-hosted open-weight models",
          methodId: "custom",
        },
        modelPicker: {
          label: "LM Studio (custom)",
          hint: "Detect models from LM Studio /api/v1/models",
          methodId: "custom",
        },
      },
    });
  },
});
