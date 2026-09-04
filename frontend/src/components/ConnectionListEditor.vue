<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { ConnectionSpec } from '../types/notebook'

const props = withDefaults(
  defineProps<{
    connections: ConnectionSpec[]
    readOnly?: boolean
  }>(),
  { readOnly: false },
)

const emit = defineEmits<{
  save: [connections: ConnectionSpec[]]
}>()

interface DraftConnection {
  _key: string
  name: string
  driver: string
  // SQLite
  path: string
  // Postgres / Snowflake / generic URI drivers
  uri: string
  // Postgres / Snowflake
  authUser: string
  authPassword: string
  role: string
  searchPath: string
  // Snowflake
  account: string
  warehouse: string
  database: string
  schema: string
  writeRole: string
  // BigQuery
  projectId: string
  datasetId: string
  credentialsPath: string
  writeCredentialsPath: string
  // Round-trip for fields the form doesn't editorialize. Two slots:
  //  - extras: top-level keys outside the known set (``options``,
  //    plus driver-specific extras a future driver may add).
  //  - extraAuth: auth-map keys other than ``user``/``password``,
  //    so a driver-specific credential (e.g. ``api_token``) survives
  //    a save unchanged.
  extras: Record<string, unknown>
  extraAuth: Record<string, string>
}

// Field names handled explicitly in the per-driver forms. Anything
// else found on a ConnectionSpec is preserved verbatim via
// ``extras`` and re-emitted by ``toSpec``.
const KNOWN_TOP_LEVEL_KEYS = new Set([
  'name',
  'driver',
  'path',
  'uri',
  'auth',
  'role',
  'search_path',
  'account',
  'warehouse',
  'database',
  'schema',
  'write_role',
  'project_id',
  'dataset_id',
  'credentials_path',
  'write_credentials_path',
])

const DRIVER_OPTIONS = [
  { value: 'sqlite', label: 'SQLite' },
  { value: 'postgresql', label: 'PostgreSQL' },
  { value: 'snowflake', label: 'Snowflake' },
  { value: 'bigquery', label: 'BigQuery' },
] as const

const KNOWN_DRIVERS = new Set(DRIVER_OPTIONS.map((o) => o.value))

const draft = ref<DraftConnection[]>([])
let nextKey = 0
const saving = ref(false)
const saveError = ref<string | null>(null)
const validationErrors = ref<Record<string, string>>({})

function toDraft(spec?: ConnectionSpec): DraftConnection {
  const auth = (spec?.auth ?? {}) as Record<string, string>
  const { user: authUser = '', password: authPassword = '', ...extraAuth } = auth
  const extras: Record<string, unknown> = {}
  if (spec) {
    for (const [key, value] of Object.entries(spec)) {
      if (KNOWN_TOP_LEVEL_KEYS.has(key)) continue
      extras[key] = value
    }
  }
  const specAny = (spec ?? {}) as Record<string, unknown>
  return {
    _key: `conn-${nextKey++}`,
    name: spec?.name ?? '',
    driver: spec?.driver ?? 'sqlite',
    path: typeof spec?.path === 'string' ? spec.path : '',
    uri: typeof spec?.uri === 'string' ? spec.uri : '',
    authUser,
    authPassword,
    role: typeof spec?.role === 'string' ? spec.role : '',
    searchPath: typeof spec?.search_path === 'string' ? spec.search_path : '',
    account: typeof specAny.account === 'string' ? (specAny.account as string) : '',
    warehouse: typeof specAny.warehouse === 'string' ? (specAny.warehouse as string) : '',
    database: typeof specAny.database === 'string' ? (specAny.database as string) : '',
    schema: typeof specAny.schema === 'string' ? (specAny.schema as string) : '',
    writeRole: typeof specAny.write_role === 'string' ? (specAny.write_role as string) : '',
    projectId: typeof specAny.project_id === 'string' ? (specAny.project_id as string) : '',
    datasetId: typeof specAny.dataset_id === 'string' ? (specAny.dataset_id as string) : '',
    credentialsPath:
      typeof specAny.credentials_path === 'string' ? (specAny.credentials_path as string) : '',
    writeCredentialsPath:
      typeof specAny.write_credentials_path === 'string'
        ? (specAny.write_credentials_path as string)
        : '',
    extras,
    extraAuth,
  }
}

watch(
  () => props.connections,
  (specs) => {
    draft.value = specs.map((s) => toDraft(s))
  },
  { immediate: true, deep: true },
)

const dirty = computed(() => {
  // Cheap deep-compare via JSON; the lists are small.
  const original = JSON.stringify(props.connections.map((s) => toDraft(s)).map(stripKey))
  const current = JSON.stringify(draft.value.map(stripKey))
  return original !== current
})

function stripKey(d: DraftConnection): Omit<DraftConnection, '_key'> {
  const { _key: _ignored, ...rest } = d
  return rest
}

function addConnection() {
  draft.value.push(toDraft())
}

function removeConnection(index: number) {
  draft.value.splice(index, 1)
}

function validate(): boolean {
  const errors: Record<string, string> = {}
  const seenNames = new Set<string>()
  for (const d of draft.value) {
    if (!d.name.trim()) {
      errors[d._key] = 'Connection name is required'
      continue
    }
    if (seenNames.has(d.name)) {
      errors[d._key] = `Duplicate name: ${d.name}`
      continue
    }
    seenNames.add(d.name)
    if (d.driver === 'sqlite') {
      if (!d.path.trim() && !d.uri.trim()) {
        errors[d._key] = 'SQLite needs either a path or a URI'
      }
    } else if (d.driver === 'postgresql') {
      if (!d.uri.trim()) {
        errors[d._key] = 'PostgreSQL needs a connection URI'
      }
    } else if (d.driver === 'snowflake') {
      if (!d.uri.trim() && !d.account.trim()) {
        errors[d._key] = 'Snowflake needs either an account identifier or a full URI'
      }
    } else if (d.driver === 'bigquery') {
      if (!d.projectId.trim()) {
        errors[d._key] = 'BigQuery needs a project ID'
      }
    }
  }
  validationErrors.value = errors
  return Object.keys(errors).length === 0
}

function toSpec(d: DraftConnection): ConnectionSpec {
  // Start from preserved extras so unknown driver-specific fields
  // (``options``, future-driver-keys) survive a save unchanged.
  // The known fields below overwrite, never drop.
  const spec: ConnectionSpec = {
    ...d.extras,
    name: d.name.trim(),
    driver: d.driver,
  }
  if (d.driver === 'sqlite') {
    if (d.path.trim()) spec.path = d.path.trim()
    else delete spec.path
    if (d.uri.trim()) spec.uri = d.uri.trim()
    else delete spec.uri
  } else if (d.driver === 'postgresql') {
    if (d.uri.trim()) spec.uri = d.uri.trim()
    else delete spec.uri
    if (d.path.trim()) spec.path = d.path.trim()
    else delete spec.path

    const auth: Record<string, string> = { ...d.extraAuth }
    if (d.authUser.trim()) auth.user = d.authUser.trim()
    if (d.authPassword.trim()) auth.password = d.authPassword.trim()
    if (Object.keys(auth).length) spec.auth = auth
    else delete spec.auth

    if (d.role.trim()) spec.role = d.role.trim()
    else delete spec.role
    if (d.searchPath.trim()) spec.search_path = d.searchPath.trim()
    else delete spec.search_path
  } else if (d.driver === 'bigquery') {
    if (d.projectId.trim()) spec.project_id = d.projectId.trim()
    else delete spec.project_id
    if (d.datasetId.trim()) spec.dataset_id = d.datasetId.trim()
    else delete spec.dataset_id
    if (d.credentialsPath.trim()) spec.credentials_path = d.credentialsPath.trim()
    else delete spec.credentials_path
    if (d.writeCredentialsPath.trim()) spec.write_credentials_path = d.writeCredentialsPath.trim()
    else delete spec.write_credentials_path
  } else if (d.driver === 'snowflake') {
    if (d.uri.trim()) spec.uri = d.uri.trim()
    else delete spec.uri
    if (d.account.trim()) spec.account = d.account.trim()
    else delete spec.account
    if (d.warehouse.trim()) spec.warehouse = d.warehouse.trim()
    else delete spec.warehouse
    if (d.database.trim()) spec.database = d.database.trim()
    else delete spec.database
    if (d.schema.trim()) spec.schema = d.schema.trim()
    else delete spec.schema
    if (d.role.trim()) spec.role = d.role.trim()
    else delete spec.role
    if (d.writeRole.trim()) spec.write_role = d.writeRole.trim()
    else delete spec.write_role

    const auth: Record<string, string> = { ...d.extraAuth }
    if (d.authUser.trim()) auth.user = d.authUser.trim()
    if (d.authPassword.trim()) auth.password = d.authPassword.trim()
    if (Object.keys(auth).length) spec.auth = auth
    else delete spec.auth
  } else {
    // Unknown driver — preserve every editable text field but
    // don't impose Postgres-shaped auth structure.
    if (d.uri.trim()) spec.uri = d.uri.trim()
    else delete spec.uri
    if (d.path.trim()) spec.path = d.path.trim()
    else delete spec.path
    const auth: Record<string, string> = { ...d.extraAuth }
    if (d.authUser.trim()) auth.user = d.authUser.trim()
    if (d.authPassword.trim()) auth.password = d.authPassword.trim()
    if (Object.keys(auth).length) spec.auth = auth
    else delete spec.auth
  }
  return spec
}

async function save() {
  if (props.readOnly) return
  if (!validate()) return
  saving.value = true
  saveError.value = null
  try {
    emit(
      'save',
      draft.value.map((d) => toSpec(d)),
    )
  } catch (err) {
    saveError.value = err instanceof Error ? err.message : String(err)
  } finally {
    saving.value = false
  }
}

function reset() {
  draft.value = props.connections.map((s) => toDraft(s))
  validationErrors.value = {}
}

function isLiteralSecret(value: string): boolean {
  if (!value) return false
  return !/^\$\{[A-Za-z_][A-Za-z0-9_]*\}$/.test(value)
}

function isUnknownDriver(driver: string): boolean {
  return driver.length > 0 && !KNOWN_DRIVERS.has(driver as 'sqlite' | 'postgresql')
}

function preservedExtraSummary(d: DraftConnection): string {
  const keys: string[] = []
  for (const k of Object.keys(d.extras)) keys.push(k)
  for (const k of Object.keys(d.extraAuth)) keys.push(`auth.${k}`)
  return keys.join(', ')
}
</script>

<template>
  <div class="conn-editor">
    <div v-if="draft.length === 0" class="conn-empty">
      No connections yet. Click "Add connection" to declare one.
    </div>

    <div
      v-for="(conn, idx) in draft"
      :key="conn._key"
      class="conn-row"
      :class="{ 'has-error': validationErrors[conn._key] }"
    >
      <div class="conn-row-header">
        <input
          v-model="conn.name"
          class="conn-name-input"
          placeholder="connection-name"
          :disabled="readOnly"
        />
        <select v-model="conn.driver" class="conn-driver-select" :disabled="readOnly">
          <option v-for="opt in DRIVER_OPTIONS" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
          <!-- Preserve an unknown driver (e.g. one declared by hand
               in notebook.toml) instead of silently coercing to
               sqlite. The value still selects in the dropdown, just
               via the synthetic option below. -->
          <option v-if="isUnknownDriver(conn.driver)" :value="conn.driver">
            {{ conn.driver }} (custom)
          </option>
        </select>
        <button
          class="conn-remove-btn"
          :disabled="readOnly"
          title="Remove this connection"
          @click="removeConnection(idx)"
        >
          ×
        </button>
      </div>

      <p v-if="isUnknownDriver(conn.driver)" class="conn-driver-hint">
        Unknown driver — only the URI / path / auth fields are editable here. Everything else
        round-trips unchanged.
      </p>
      <p
        v-if="Object.keys(conn.extras).length > 0 || Object.keys(conn.extraAuth).length > 0"
        class="conn-extras-hint"
      >
        Preserved from notebook.toml: <code>{{ preservedExtraSummary(conn) }}</code>
      </p>

      <div class="conn-fields">
        <template v-if="conn.driver === 'sqlite'">
          <label class="conn-field">
            <span class="field-label">Path</span>
            <input
              v-model="conn.path"
              type="text"
              placeholder="analytics.db (relative to notebook dir)"
              :disabled="readOnly"
            />
          </label>
        </template>
        <template v-else-if="conn.driver === 'postgresql'">
          <label class="conn-field">
            <span class="field-label">URI</span>
            <input
              v-model="conn.uri"
              type="text"
              placeholder="postgresql://host:5432/dbname"
              :disabled="readOnly"
            />
          </label>
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">User</span>
              <input
                v-model="conn.authUser"
                type="text"
                placeholder="${PGUSER}"
                :disabled="readOnly"
                :class="{ 'literal-secret': isLiteralSecret(conn.authUser) }"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">Password</span>
              <input
                v-model="conn.authPassword"
                type="text"
                placeholder="${PGPASS}"
                :disabled="readOnly"
                :class="{ 'literal-secret': isLiteralSecret(conn.authPassword) }"
              />
            </label>
          </div>
          <p
            v-if="isLiteralSecret(conn.authUser) || isLiteralSecret(conn.authPassword)"
            class="conn-secret-hint"
          >
            Use <code>$&#123;VAR&#125;</code> indirection — literal credentials are blanked when
            notebook.toml is saved.
          </p>
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">Role</span>
              <input v-model="conn.role" type="text" placeholder="reader" :disabled="readOnly" />
            </label>
            <label class="conn-field">
              <span class="field-label">search_path</span>
              <input
                v-model="conn.searchPath"
                type="text"
                placeholder="public"
                :disabled="readOnly"
              />
            </label>
          </div>
        </template>
        <template v-else-if="conn.driver === 'snowflake'">
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">Account</span>
              <input
                v-model="conn.account"
                type="text"
                placeholder="ACME-PROD"
                :disabled="readOnly"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">Warehouse</span>
              <input
                v-model="conn.warehouse"
                type="text"
                placeholder="ANALYTICS_WH"
                :disabled="readOnly"
              />
            </label>
          </div>
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">Database</span>
              <input
                v-model="conn.database"
                type="text"
                placeholder="EVENTS"
                :disabled="readOnly"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">Schema</span>
              <input v-model="conn.schema" type="text" placeholder="PUBLIC" :disabled="readOnly" />
            </label>
          </div>
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">Role (read cells)</span>
              <input
                v-model="conn.role"
                type="text"
                placeholder="ANALYTICS_RO"
                :disabled="readOnly"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">write_role (write cells, optional)</span>
              <input
                v-model="conn.writeRole"
                type="text"
                placeholder="ANALYTICS_RW"
                :disabled="readOnly"
              />
            </label>
          </div>
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">User</span>
              <input
                v-model="conn.authUser"
                type="text"
                placeholder="${SF_USER}"
                :disabled="readOnly"
                :class="{ 'literal-secret': isLiteralSecret(conn.authUser) }"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">Password</span>
              <input
                v-model="conn.authPassword"
                type="text"
                placeholder="${SF_PASS}"
                :disabled="readOnly"
                :class="{ 'literal-secret': isLiteralSecret(conn.authPassword) }"
              />
            </label>
          </div>
          <p
            v-if="isLiteralSecret(conn.authUser) || isLiteralSecret(conn.authPassword)"
            class="conn-secret-hint"
          >
            Use <code>$&#123;VAR&#125;</code> indirection — literal credentials are blanked when
            notebook.toml is saved.
          </p>
          <p class="conn-secret-hint">
            Snowflake read-only enforcement is role-based: configure a role that has SELECT but not
            INSERT/UPDATE/DELETE for read cells.
          </p>
          <label class="conn-field">
            <span class="field-label">URI (overrides the fields above)</span>
            <input
              v-model="conn.uri"
              type="text"
              placeholder="snowflake://reader@ACME/EVENTS?authenticator=externalbrowser"
              :disabled="readOnly"
            />
          </label>
        </template>
        <template v-else-if="conn.driver === 'bigquery'">
          <div class="conn-field-row">
            <label class="conn-field">
              <span class="field-label">Project ID</span>
              <input
                v-model="conn.projectId"
                type="text"
                placeholder="acme-prod"
                :disabled="readOnly"
              />
            </label>
            <label class="conn-field">
              <span class="field-label">Dataset (default)</span>
              <input
                v-model="conn.datasetId"
                type="text"
                placeholder="events"
                :disabled="readOnly"
              />
            </label>
          </div>
          <label class="conn-field">
            <span class="field-label">Service account JSON path (read cells)</span>
            <input
              v-model="conn.credentialsPath"
              type="text"
              placeholder="/secrets/bq-reader.json"
              :disabled="readOnly"
            />
          </label>
          <label class="conn-field">
            <span class="field-label">Write credentials path (write cells, optional)</span>
            <input
              v-model="conn.writeCredentialsPath"
              type="text"
              placeholder="/secrets/bq-writer.json"
              :disabled="readOnly"
            />
          </label>
          <p class="conn-secret-hint">
            BigQuery read-only enforcement is credential-based: the service account's IAM grants
            decide what's allowed. Use a SA with
            <code>roles/bigquery.dataViewer</code> for read cells; pair with a write SA carrying
            <code>roles/bigquery.dataEditor</code> for <code># @sql write=true</code>.
          </p>
        </template>
        <template v-else-if="isUnknownDriver(conn.driver)">
          <!-- Unknown driver: surface the common URI + path slots
               so the user can adjust them without losing the rest
               of the on-disk shape. ``extras`` round-trips
               anything the form doesn't editorialize. -->
          <label class="conn-field">
            <span class="field-label">URI</span>
            <input v-model="conn.uri" type="text" placeholder="driver://…" :disabled="readOnly" />
          </label>
          <label class="conn-field">
            <span class="field-label">Path</span>
            <input v-model="conn.path" type="text" :disabled="readOnly" />
          </label>
        </template>
      </div>

      <p v-if="validationErrors[conn._key]" class="conn-error">
        {{ validationErrors[conn._key] }}
      </p>
    </div>

    <div class="conn-actions">
      <button class="conn-add-btn" :disabled="readOnly" @click="addConnection">
        + Add connection
      </button>
      <div class="conn-actions-right">
        <button v-if="dirty" class="conn-reset-btn" :disabled="readOnly" @click="reset">
          Reset
        </button>
        <button class="conn-save-btn" :disabled="readOnly || saving || !dirty" @click="save">
          {{ saving ? 'Saving…' : 'Save' }}
        </button>
      </div>
    </div>
    <p v-if="saveError" class="conn-error">{{ saveError }}</p>
  </div>
</template>

<style scoped>
.conn-editor {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.conn-empty {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
  padding: 8px;
}

.conn-row {
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 10px;
  background: var(--bg-elevated);
  color: var(--text-primary);
}

.conn-row.has-error {
  border-color: var(--accent-danger);
}

.conn-row-header {
  display: flex;
  gap: 8px;
  align-items: center;
}

.conn-name-input {
  flex: 1;
  padding: 4px 8px;
  font-size: 13px;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  background: var(--bg-input);
  color: var(--text-primary);
}

/* Set color on both the select and its options. Browsers (esp.
   Chrome on macOS dark mode) ignore the inherited color on
   <option> in some configurations, so an option list rendered
   on a system-default white background needs the option color
   set explicitly to stay readable. */
.conn-driver-select,
.conn-driver-select option {
  padding: 4px 6px;
  font-size: 12px;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  background: var(--bg-input);
  color: var(--text-primary);
}

.conn-remove-btn {
  width: 24px;
  height: 24px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

.conn-remove-btn:hover:not(:disabled) {
  color: var(--accent-danger);
}

.conn-fields {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.conn-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.conn-field-row {
  display: flex;
  gap: 8px;
}

.field-label {
  font-size: 11px;
  color: var(--text-secondary);
  font-weight: 500;
}

.conn-field input {
  padding: 4px 8px;
  font-size: 12px;
  font-family: var(--font-mono, monospace);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  background: var(--bg-input);
  color: var(--text-primary);
}

.conn-field input::placeholder {
  color: var(--text-muted);
}

.conn-field input.literal-secret {
  border-color: var(--accent-warning, #f39c12);
  background: var(--tint-warning);
}

.conn-secret-hint,
.conn-driver-hint,
.conn-extras-hint {
  font-size: 11px;
  color: var(--text-secondary);
  margin: 4px 0 0;
}

.conn-driver-hint {
  color: var(--accent-warning, #b58400);
}

.conn-secret-hint code,
.conn-extras-hint code {
  background: var(--bg-input);
  padding: 1px 4px;
  border-radius: 3px;
}

.conn-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.conn-actions-right {
  display: flex;
  gap: 6px;
}

.conn-add-btn,
.conn-reset-btn,
.conn-save-btn {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 4px;
  border: 1px solid var(--border-strong, var(--border-subtle));
  cursor: pointer;
  background: var(--bg-input);
  color: var(--text-primary);
}

.conn-add-btn:hover:not(:disabled),
.conn-reset-btn:hover:not(:disabled) {
  background: var(--bg-hover, rgba(255, 255, 255, 0.06));
}

.conn-save-btn {
  background: var(--accent-primary, #4a90e2);
  color: #fff;
  border-color: transparent;
}

.conn-save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.conn-error {
  font-size: 11px;
  color: var(--accent-danger);
  margin: 4px 0 0;
}
</style>
