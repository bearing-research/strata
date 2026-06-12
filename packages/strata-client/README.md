# strata-client

A lightweight Python client for a [Strata](https://github.com/bearing-research/strata)
server. Depends only on `httpx` + `pyarrow` — none of the server's stack
(no pyiceberg / fastapi / duckdb / pydantic, no Rust extension), so it drops
into any analysis venv, training image, CI job, or notebook without dragging
the deployable service along.

```bash
pip install strata-client
```

```python
from strata_client import StrataClient

with StrataClient() as client:  # resolves the server URL from
                                # STRATA_SERVER_URL / STRATA_HOST / STRATA_PORT
    art = client.materialize(
        inputs=["file:///warehouse#db.events"],
        transform={"executor": "scan@v1", "params": {}},
    )
    table = client.fetch(art.uri)

    # Registry: names, aliases, tags, audit
    client.put(table, name="team/dataset/raw")
    client.set_alias("taxi/tip-model", "champion", art.artifact_id, art.version)
```

The server distribution (`strata-notebook`) depends on this package and
re-exports it as `strata.client` for backward compatibility.
