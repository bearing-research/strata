# Run cells over SSH

If you have a box you reach over SSH (a GPU machine, a bigger instance, one closer to the data), Strata can turn it into a worker for you in one step, without you deploying anything or hand-editing `notebook.toml`. Strata SSHes in, installs `strata-worker` if it's missing, launches it bound to the box's localhost, opens an `ssh -L` tunnel back to the notebook server, and registers it as a `direct`-transport worker. Cells then run on that box, cached by provenance like any other worker.

**Driving a coding agent (the common case).** When you're driving the notebook with a coding agent ([Drive with a Coding Agent](../../notebook/agent.md)), tell it the target and it calls the `connect_ssh_worker` MCP tool:

```
You: use ssh user@gpu-box for the training cells
Agent → connect_ssh_worker(session_id, "user@gpu-box")
```

The agent's working agreement already tells it to reach for this when you hand it an SSH target, and to keep light cells local with `# @worker local`. To wire the worker up as the session opens instead, pass it to the on-ramp:

```bash
strata agent ./my-notebook --worker-ssh user@gpu-box
```

**From the CLI.** Against a running server (the tunnel is owned by the server, since that's where cells dispatch from), with the session id the server assigned:

```bash
strata worker add-ssh user@gpu-box --server http://localhost:8765 --session <session-id>
# tear it down (add --stop-remote to also stop the worker process on the box):
strata worker rm-ssh gpu-box --server http://localhost:8765 --session <session-id>
```

`add-ssh` makes the new worker the notebook default (pass `--no-default` to skip); `--name` overrides the name derived from the host, and `--no-install` fails fast rather than installing `strata-worker` on the box.

**What to know:**

- **Key-based SSH only.** Strata runs `ssh` in batch mode and never handles passwords, so the target must authenticate non-interactively (an agent/key that works when you run `ssh user@gpu-box` yourself). A `user@host`, a bare `host`, or an `~/.ssh/config` alias all work.
- **The first connect can take a minute** while it installs `strata-worker` on the box (via `uv tool install`); a reconnect skips the install, but it starts a fresh worker rather than adopting one already running.
- **Security.** The worker binds the box's `127.0.0.1` (never a public port) and is reachable only through the authenticated SSH tunnel. A per-worker bearer token is generated for defense-in-depth; it's held in the notebook server's memory and **never written to `notebook.toml`**. Each launch also hands the worker a one-off `launch_id` (over stdin, like the token), and Strata connects only when `/health` through the tunnel reports it, so another process already listening on the box's port never receives the token.
- **A remote cell runs on the box's filesystem.** Absolute paths in the cell resolve there, not on your machine, and a `file://` mount is refused on any remote worker, and cloud mounts use the box's own credentials. A mount's `options` may set only the [non-secret keys a shared server allows](../../reference/notebook-toml.md#mount-options-on-a-shared-server), so an endpoint or a profile goes in the box's `STRATA_NOTEBOOK_CREDENTIALS`. The box keeps its read-only mount copies across jobs, in `STRATA_NOTEBOOK_MOUNT_CACHE_DIR` (default `~/.strata/mount_cache`), bounded by the same two settings as the server's. Results are cached under the remote environment's identity, so they don't collide with local runs.
