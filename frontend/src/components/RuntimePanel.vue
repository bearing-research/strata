<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import EnvVarsEditor from './EnvVarsEditor.vue'
import TimeoutConfigEditor from './TimeoutConfigEditor.vue'

const {
  notebook,
  connected,
  updateNotebookTimeoutAction,
  updateNotebookEnvAction,
  refreshSecretManagerAction,
  updateNotebookSecretManagerConfigAction,
} = useNotebook()

const showPanel = ref(false)
const refreshing = ref(false)
const savingSecretsConfig = ref(false)
const secretManagerConfigError = ref<string | null>(null)

const envCount = computed(() => Object.keys(notebook.env).length)
const timeoutLabel = computed(() => (notebook.timeout == null ? 'default' : `${notebook.timeout}s`))

// Local copy of notebook.secretManagerConfig until Save; a new backend config
// resets it.
const form = ref<{
  provider: string
  project_id: string
  environment: string
  path: string
  base_url: string
}>({
  provider: '',
  project_id: '',
  environment: '',
  path: '',
  base_url: '',
})

function resetForm() {
  form.value.provider = notebook.secretManagerConfig.provider ?? ''
  form.value.project_id = notebook.secretManagerConfig.project_id ?? ''
  form.value.environment = notebook.secretManagerConfig.environment ?? ''
  form.value.path = notebook.secretManagerConfig.path ?? ''
  form.value.base_url = notebook.secretManagerConfig.base_url ?? ''
}

watch(
  () => notebook.secretManagerConfig,
  () => {
    resetForm()
  },
  { immediate: true, deep: true },
)

const secretManagerConfigured = computed(() => Object.keys(notebook.secretManagerConfig).length > 0)

const secretManagerName = computed(() => {
  if (notebook.secretManagerConfig.provider) return notebook.secretManagerConfig.provider
  for (const src of Object.values(notebook.envSources)) {
    if (src && src !== 'manual') return src
  }
  return null
})

const formDirty = computed(
  () =>
    form.value.provider !== (notebook.secretManagerConfig.provider ?? '') ||
    form.value.project_id !== (notebook.secretManagerConfig.project_id ?? '') ||
    form.value.environment !== (notebook.secretManagerConfig.environment ?? '') ||
    form.value.path !== (notebook.secretManagerConfig.path ?? '') ||
    form.value.base_url !== (notebook.secretManagerConfig.base_url ?? ''),
)

async function handleRefresh() {
  if (refreshing.value || !connected) return
  refreshing.value = true
  try {
    await refreshSecretManagerAction()
  } catch {
    // Fetch errors arrive in notebook.envFetchError; a throw means the POST
    // itself failed.
  } finally {
    refreshing.value = false
  }
}

async function saveSecretsConfig() {
  if (savingSecretsConfig.value || !connected) return
  secretManagerConfigError.value = null
  savingSecretsConfig.value = true
  try {
    await updateNotebookSecretManagerConfigAction({
      provider: form.value.provider || null,
      project_id: form.value.project_id || null,
      environment: form.value.environment || null,
      path: form.value.path || null,
      base_url: form.value.base_url || null,
    })
  } catch (e: any) {
    secretManagerConfigError.value = e?.message || 'Failed to save secret-manager config'
  } finally {
    savingSecretsConfig.value = false
  }
}

async function disconnectSecretManager() {
  if (savingSecretsConfig.value || !connected) return
  secretManagerConfigError.value = null
  savingSecretsConfig.value = true
  try {
    await updateNotebookSecretManagerConfigAction({})
  } catch (e: any) {
    secretManagerConfigError.value = e?.message || 'Failed to disconnect'
  } finally {
    savingSecretsConfig.value = false
  }
}
</script>

<template>
  <div class="runtime-panel">
    <button class="runtime-toggle" @click="showPanel = !showPanel">
      Runtime
      <span class="runtime-label">{{ timeoutLabel }}</span>
      <span class="runtime-label">{{ envCount }} env</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="runtime-content">
      <p class="runtime-copy">
        Notebook defaults. Cells can override these, and source annotations still win.
      </p>
      <TimeoutConfigEditor
        :timeout="notebook.timeout"
        title="Notebook Default Timeout"
        :read-only="!connected"
        @save="updateNotebookTimeoutAction"
      />
      <EnvVarsEditor
        :env="notebook.env"
        :env-sources="notebook.envSources"
        title="Notebook Default Env"
        :read-only="!connected"
        @save="updateNotebookEnvAction"
      />
      <div class="secrets-panel">
        <div class="secrets-header">
          <span class="secrets-title">
            Secret manager
            <span v-if="secretManagerName" class="secrets-provider">{{ secretManagerName }}</span>
          </span>
          <button
            v-if="secretManagerConfigured"
            class="secrets-refresh"
            :disabled="refreshing || !connected"
            @click="handleRefresh"
          >
            {{ refreshing ? 'Refreshing…' : 'Refresh' }}
          </button>
        </div>
        <p v-if="notebook.envFetchError" class="secrets-error">
          {{ notebook.envFetchError }}
        </p>
        <p v-else-if="notebook.envFetchedAt && secretManagerConfigured" class="secrets-meta">
          Last refreshed {{ notebook.envFetchedAt }}
        </p>
        <div class="secrets-form">
          <label class="secrets-field">
            <span class="secrets-field-label">Provider</span>
            <select
              v-model="form.provider"
              class="secrets-select"
              :disabled="!connected || savingSecretsConfig"
            >
              <option value="">(none)</option>
              <option value="infisical">Infisical</option>
            </select>
          </label>
          <template v-if="form.provider">
            <label class="secrets-field">
              <span class="secrets-field-label">Project ID</span>
              <input
                v-model="form.project_id"
                class="secrets-input"
                placeholder="your-project-id"
                :disabled="!connected || savingSecretsConfig"
              />
            </label>
            <div class="secrets-row">
              <label class="secrets-field">
                <span class="secrets-field-label">Environment</span>
                <input
                  v-model="form.environment"
                  class="secrets-input"
                  placeholder="dev"
                  :disabled="!connected || savingSecretsConfig"
                />
              </label>
              <label class="secrets-field">
                <span class="secrets-field-label">Path</span>
                <input
                  v-model="form.path"
                  class="secrets-input"
                  placeholder="/"
                  :disabled="!connected || savingSecretsConfig"
                />
              </label>
            </div>
            <label class="secrets-field">
              <span class="secrets-field-label">Base URL (self-hosted only)</span>
              <input
                v-model="form.base_url"
                class="secrets-input"
                placeholder="https://app.infisical.com"
                :disabled="!connected || savingSecretsConfig"
              />
            </label>
            <p class="secrets-hint">
              Authenticate in the shell that launched Strata. Either set
              <code>INFISICAL_CLIENT_ID</code> + <code>INFISICAL_CLIENT_SECRET</code> (Machine
              Identity, recommended) or <code>INFISICAL_TOKEN</code>
              (service token, legacy). Credentials are never written to disk.
            </p>
          </template>
          <p v-if="secretManagerConfigError" class="secrets-error">
            {{ secretManagerConfigError }}
          </p>
          <div class="secrets-actions">
            <button
              v-if="secretManagerConfigured"
              class="secrets-disconnect"
              :disabled="savingSecretsConfig || !connected"
              @click="disconnectSecretManager"
            >
              Disconnect
            </button>
            <button
              class="secrets-save"
              :disabled="!formDirty || savingSecretsConfig || !connected"
              @click="saveSecretsConfig"
            >
              {{ savingSecretsConfig ? 'Saving…' : 'Save' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.runtime-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.runtime-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  padding: 4px 0;
}

.runtime-toggle:hover {
  color: var(--text-primary);
}

.runtime-label {
  background: var(--bg-input);
  color: var(--accent-warning);
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
}

.toggle-icon {
  margin-left: auto;
  font-size: 10px;
}

.runtime-content {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.runtime-copy {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
}

.secrets-panel {
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.secrets-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.secrets-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 6px;
}
.secrets-provider {
  background: var(--bg-input);
  color: var(--text-secondary);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
  text-transform: lowercase;
}
.secrets-refresh {
  background: none;
  border: 1px solid var(--bg-input);
  border-radius: 4px;
  color: var(--text-secondary);
  font-size: 11px;
  padding: 2px 8px;
  cursor: pointer;
}
.secrets-refresh:hover:not(:disabled) {
  color: var(--accent-primary);
  border-color: var(--accent-primary);
}
.secrets-refresh:disabled {
  opacity: 0.5;
  cursor: default;
}
.secrets-error {
  font-size: 11px;
  color: var(--accent-danger);
  margin: 0;
  line-height: 1.4;
}
.secrets-meta {
  font-size: 11px;
  color: var(--text-muted);
  margin: 0;
  line-height: 1.4;
}
.secrets-form {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 4px;
}
.secrets-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}
.secrets-field-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.secrets-row {
  display: flex;
  gap: 6px;
}
.secrets-input,
.secrets-select {
  background: var(--bg-base);
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 12px;
  padding: 6px 8px;
  width: 100%;
  min-width: 0;
}
.secrets-input:disabled,
.secrets-select:disabled {
  opacity: 0.5;
}
.secrets-hint {
  margin: 0;
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.4;
}
.secrets-hint code {
  background: var(--bg-base);
  padding: 0 4px;
  border-radius: 3px;
  font-size: 10px;
}
.secrets-actions {
  display: flex;
  gap: 6px;
  justify-content: flex-end;
}
.secrets-save,
.secrets-disconnect {
  background: none;
  border: 1px solid var(--bg-input);
  border-radius: 4px;
  color: var(--text-secondary);
  font-size: 11px;
  padding: 3px 10px;
  cursor: pointer;
}
.secrets-save:hover:not(:disabled) {
  color: var(--accent-success);
  border-color: var(--accent-success);
}
.secrets-disconnect:hover:not(:disabled) {
  color: var(--accent-danger);
  border-color: var(--accent-danger);
}
.secrets-save:disabled,
.secrets-disconnect:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
