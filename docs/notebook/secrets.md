# Secret Manager Integration

Strata can pull environment variables from an external secret manager so API keys, database URLs, and other sensitive config don't have to be re-entered every time a notebook is reopened. Values fetched from the manager flow into the same `notebook.env` map the Runtime panel uses, so cells read them with plain `os.environ`, no notebook code changes.

## Supported providers

| Provider               | Status              | Auth                                             |
| ---------------------- | ------------------- | ------------------------------------------------ |
| Infisical              | Supported           | Machine Identity (recommended) or service token  |
| HashiCorp Vault        | Not yet implemented | -                                                |
| AWS Secrets Manager    | Not yet implemented | -                                                |
| Doppler                | Not yet implemented | -                                                |

Today Strata ships one provider (Infisical). Vault / AWS Secrets Manager / Doppler are **not yet implemented** - the `SecretProvider` protocol in `strata.notebook.secret_manager` is structured so each is a one-file drop-in, but the code doesn't exist yet. File an issue if you need a specific provider sooner.

## Setting up Infisical

### 1. Authenticate the server

Strata reaches Infisical with credentials it reads from the **process environment** of the running server, not from the notebook UI, never from disk. Two auth paths:

**Machine Identity / Universal Auth (recommended).** Create a Machine Identity in your Infisical project, grant it read access to the secrets you want Strata to see, and export the resulting client id + secret:

```bash
export INFISICAL_CLIENT_ID="your-client-id"
export INFISICAL_CLIENT_SECRET="your-client-secret"
```

**Service token (legacy).** If you already have a service token configured, it still works:

```bash
export INFISICAL_TOKEN="st.xxxx..."
```

Service tokens are being deprecated upstream; new setups should use Machine Identity. If both are set, Machine Identity wins.

Self-hosted Infisical? Also export:

```bash
export INFISICAL_HOST="https://your-self-hosted.infisical.com"
```

### 2. Launch the server with those vars in scope

```bash
export INFISICAL_CLIENT_ID="..."
export INFISICAL_CLIENT_SECRET="..."
uv run strata-notebook
```

The credentials only live in the server process, never in `notebook.toml`, `.strata/`, logs, or any commit.

### 3. Wire up the notebook

Open the notebook and use the **Secret manager** section in the Runtime panel:

1. Pick the provider (`infisical`).
2. Fill in `project_id`, `environment` (`dev` / `staging` / `prod`), and `path` (defaults to `/`). Base URL is optional: the server accepts only its own `INFISICAL_HOST` there (or the public default when that is unset), so a self-hosted deployment is set with `INFISICAL_HOST` where the server starts.
3. Save.

The result lands in `notebook.toml` as:

```toml
[secret_manager]
provider = "infisical"
project_id = "your-project-id"
environment = "dev"
path = "/"
# base_url = "https://your-self-hosted.infisical.com"  # must match INFISICAL_HOST
```

All four fields are non-sensitive routing info and safe to commit. Save triggers a reload + immediate fetch, so the Runtime panel's env rows light up with `INFISICAL` badges the moment the save completes.

## How values flow

```
Runtime panel edits  ──►  notebook.toml [env]   (sensitive names blanked)
        │                        │
        ▼                        │ (read at session open)
      memory  ◄──────────────────┘
        ▲
        │ (merged at session open + on refresh; never written to disk)
Infisical (project_id, env, path)
```

- On **session open**, Strata pulls all secrets at the configured path and merges them into `notebook.env` where the key isn't already present (or where the existing value is a blanked sensitive placeholder from disk).
- On **Refresh** (button in the Runtime panel), Strata re-fetches without reopening. New/rotated values take effect for the next cell run.
- Saving a new secret-manager config fetches again. Other edits (adding a cell, changing the env, a worker or a timeout) reuse the values from the last fetch rather than calling the manager.
- Values typed **manually in the Runtime panel** override the manager's. Saving writes a manual value to the committed `notebook.toml` `[env]` unless its name looks sensitive (see [Security notes](#security-notes)), so a manual `DATABASE_URL` still wins after a reopen; a sensitive-looking one lasts for the current session. Remove the row to fall back to the manager's version.
- Saving the panel leaves a fetched row alone when you didn't type a new value: its value is not written to `notebook.toml` (a key the file already declares stays declared, blank), it keeps its source badge, and the next Refresh replaces it. This holds for a tab opened before a rotation too: its save keeps the rotated value.

Each env row in the Runtime panel shows a green source badge (`INFISICAL`) next to its name when the value came from the manager. Rows without a badge are manual overrides or local-only vars.

## Rotation

Rotate the secret in Infisical, then hit the **Refresh** button. Cells that run after the refresh see the new value immediately (the executor reads the cell's `env` each run). A cell whose source reads the variable by name (`os.environ["KEY"]`, `os.environ.get("KEY")`, `os.getenv("KEY")`, or `Sys.getenv("KEY")` in an R cell) or declares it with `# @env` has its value folded into its provenance, so it goes stale after a rotation and re-runs. A cell that reaches the secret indirectly (a client library reading it for you) does not, and its cached artifacts stay cached; rerun it (`↻`) to execute with the new value.

## Fetch errors

When a fetch fails, bad credentials, network error, wrong project, the notebook still opens. The error surfaces in the Runtime panel's Secret manager block. A manager that does not answer within 20 seconds counts as a failed fetch. Common messages:

> No Infisical credentials in the process environment. Set either `INFISICAL_CLIENT_ID` + `INFISICAL_CLIENT_SECRET` (Machine Identity / Universal Auth, recommended) or `INFISICAL_TOKEN` (service token, legacy) in the shell that launched Strata.

> Infisical authentication failed: …

> Infisical list_secrets failed: …

Fix the cause (rotate the credential, check `project_id` / `environment` / `path`, confirm the Machine Identity has read access), then hit Refresh. You don't have to reopen the notebook.

## Security notes

- Secret **values never leave the server**: no REST response or WebSocket frame carries a fetched value, or a value whose name looks sensitive (contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD` or `CREDENTIAL`), to anyone, the notebook's owner included. The Runtime panel shows such a row as set but hidden; leave it blank to keep the value, or type a new one to replace it. Other values, such as a manual `LOG_LEVEL`, are shown as typed.
- Fetched values are **not written to disk**: they live in memory and are re-fetched on each open. Manual edits are saved: `[env]` in `notebook.toml` keeps the names of sensitive-looking keys but blanks their values, and stores other values as typed. Don't type a secret into a row whose name doesn't look sensitive: it is committed and shown to anyone who can read the notebook.
- A cell on a `signed` worker receives the env in the manifest posted to the worker; the build and artifact rows the server keeps record only the names and a digest of the values.
- If a cell **prints** an env var, its value is captured in the cell's console output and persisted in `.strata/console/` alongside stdout/stderr. Don't `print(os.environ)` in production notebooks.
- Authenticating credentials (`INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET` or `INFISICAL_TOKEN`) live in the process environment, set by whoever launches the server. Distribute them the same way you'd distribute any deploy secret (systemd unit, k8s secret, `.envrc` with direnv-allow, etc.) **not** in a committed file.
- The Infisical host is the server's, in every mode: a notebook `base_url` other than `INFISICAL_HOST` (or the public default) is refused before any login, since the login would send the server's credentials there. A cloned notebook cannot point your personal server at its own host either. `project_id`, `environment` and `path` still come from the notebook, so every author can read whatever the server's machine identity can; scope it accordingly. See [Service Mode: A notebook's secret manager](../deployment/service-mode.md#a-notebooks-secret-manager).
- By default a cell subprocess inherits the server's whole environment, so a cell can read those credentials too. On a server other people run cells on, set `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST` and `STRATA_NOTEBOOK_HARNESS_USER`; see [Service Mode: What a cell can read](../deployment/service-mode.md#what-a-cell-can-read). The notebook's own `[env]`, including fetched secrets, still reaches its cells.

## Limits

This MVP is a read-only integration: Strata **reads** secrets from Infisical, it doesn't write to or rotate them. For updates, use the Infisical dashboard or CLI, then hit Refresh in the Runtime panel.
