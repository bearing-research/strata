# Secret Manager Integration

Strata can pull environment variables from an external secret manager so API keys, database URLs, and other sensitive config don't have to be re-entered every time a notebook is reopened. Values fetched from the manager flow into the same `notebook.env` map the Runtime panel uses, so cells read them with plain `os.environ`, no notebook code changes.

## Supported providers

| Provider    | Status    | Auth                                             |
| ----------- | --------- | ------------------------------------------------ |
| Infisical   | Supported | Machine Identity (recommended) or service token  |

Adding a new provider is a one-file drop-in behind the `SecretProvider` protocol in `strata.notebook.secret_manager`. Vault / AWS Secrets Manager / Doppler can plug in without changing the session or routing code.

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
uv run strata-server
```

The credentials only live in the server process, never in `notebook.toml`, `.strata/`, logs, or any commit.

### 3. Wire up the notebook

Open the notebook and use the **Secret manager** section in the Runtime panel:

1. Pick the provider (`infisical`).
2. Fill in `project_id`, `environment` (`dev` / `staging` / `prod`), and `path` (defaults to `/`). Base URL is only needed for self-hosted deployments.
3. Save.

The result lands in `notebook.toml` as:

```toml
[secret_manager]
provider = "infisical"
project_id = "your-project-id"
environment = "dev"
path = "/"
# base_url = "https://your-self-hosted.infisical.com"
```

All four fields are non-sensitive routing info and safe to commit. Save triggers a reload + immediate fetch, so the Runtime panel's env rows light up with `INFISICAL` badges the moment the save completes.

## How values flow

```
notebook.toml [env]  ──►  memory  ◄──  Runtime panel edits (session-only)
                            ▲
                            │ (merged at session open + on refresh)
                   Infisical (project_id, env, path)
```

- On **session open**, Strata pulls all secrets at the configured path and merges them into `notebook.env` where the key isn't already present (or where the existing value is a blanked sensitive placeholder from disk).
- On **Refresh** (button in the Runtime panel), Strata re-fetches without reopening. New/rotated values take effect for the next cell run.
- Values typed **manually in the Runtime panel** override anything from the manager for the current session. Useful for one-off overrides; remove the value to fall back to the manager's version.

Each env row in the Runtime panel shows a green source badge (`INFISICAL`) next to its name when the value came from the manager. Rows without a badge are manual overrides or local-only vars.

## Rotation

Rotate the secret in Infisical, then hit the **Refresh** button. Cells that run after the refresh see the new value immediately (the executor reads the cell's `env` each run). Cells with cached artifacts under the *old* value stay cached, rotating a secret doesn't invalidate history. If you need to re-execute with the new value, edit or force-run the downstream cell.

## Fetch errors

When a fetch fails, bad credentials, network error, wrong project, the notebook still opens. The error surfaces in the Runtime panel's Secret manager block. Common messages:

> No Infisical credentials in the process environment. Set either `INFISICAL_CLIENT_ID` + `INFISICAL_CLIENT_SECRET` (Machine Identity / Universal Auth, recommended) or `INFISICAL_TOKEN` (service token, legacy) in the shell that launched Strata.

> Infisical authentication failed: …

> Infisical list_secrets failed: …

Fix the cause (rotate the credential, check `project_id` / `environment` / `path`, confirm the Machine Identity has read access), then hit Refresh. You don't have to reopen the notebook.

## Security notes

- Secret **values** never ship to the frontend in cleartext from the env endpoints, values are only visible in the notebook venv's `os.environ`. The UI shows the key names + source, not the values.
- Secret values are **not written to disk**. `[env]` blocks on disk blank sensitive keys (`KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL` name patterns) before persisting; secrets fetched at open time are in-memory only.
- If a cell **prints** an env var, its value is captured in the cell's console output and persisted in `.strata/console/` alongside stdout/stderr. Don't `print(os.environ)` in production notebooks.
- Authenticating credentials (`INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET` or `INFISICAL_TOKEN`) live in the process environment, set by whoever launches the server. Distribute them the same way you'd distribute any deploy secret (systemd unit, k8s secret, `.envrc` with direnv-allow, etc.) **not** in a committed file.

## Limits

This MVP is a read-only integration: Strata **reads** secrets from Infisical, it doesn't write to or rotate them. For updates, use the Infisical dashboard or CLI, then hit Refresh in the Runtime panel.
