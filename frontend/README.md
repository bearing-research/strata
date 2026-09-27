# Strata Notebook frontend

The notebook web UI: Vue 3, TypeScript and Vite. It talks to a Strata
server at `http://localhost:8765`; set `VITE_STRATA_URL` to point it
elsewhere.

```bash
npm ci
npm run dev           # hot-reload dev server against a running Strata
npm run build         # type-check and build into dist/
npm test              # unit tests
npm run format:check  # prettier
```

`python -m strata` serves a built UI itself, so the dev server is only
needed for hot reload. It serves `src/strata/_frontend/` when that
exists and `frontend/dist/` otherwise, so a stale copy in `_frontend/`
hides a fresh build. Release wheels bundle the UI into `_frontend/`, and
the Docker image builds its own.
