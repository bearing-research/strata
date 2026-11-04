# Contributing

## Set up

You need [uv](https://docs.astral.sh/uv/) 0.8 or newer, and Node for the frontend.

```bash
uv sync --all-extras     # install dependencies (matches CI)
```

## Check your work

```bash
uv run pytest                        # run the tests
uv run pytest tests/test_smoke.py -v # run one file
uv run pre-commit run --all-files    # lint and format
uv run ty check                      # type check
```

## Run it

```bash
uv run python -m strata              # start the server on http://localhost:8765
```

In a second terminal, for the frontend with hot reload:

```bash
cd frontend
npm ci
npm run dev                          # talks to :8765; override with VITE_STRATA_URL
npm test                             # frontend tests
npm run build                        # type check and production build
```

## Pull requests

Open them against `main`. Add a line to `CHANGELOG.md` and `docs/changelog.md` for anything
a user would notice. Architecture notes are in [`CLAUDE.md`](CLAUDE.md).
