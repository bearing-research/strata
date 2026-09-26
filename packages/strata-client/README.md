# strata-client

A lightweight Python client for a [Strata](https://github.com/bearing-research/strata)
server. Depends only on `httpx` + `pyarrow`, none of the server's stack
(no pyiceberg / fastapi / duckdb / pydantic, no Rust extension), so it drops
into any analysis venv, training image, CI job, or notebook without dragging
the deployable service along.

```bash
pip install strata-client
pip install "strata-client[duckdb]"   # or pandas, polars, datafusion, all
```

```python
from strata_client import StrataClient

with StrataClient() as client:  # resolves the server URL from STRATA_SERVER_URL,
                                # STRATA_HOST / STRATA_PORT, or [tool.strata]
    art = client.materialize(
        inputs=["file:///warehouse#db.events"],
        transform={"executor": "scan@v1", "params": {}},
    )
    table = client.fetch(art.uri)

    # Persist a result computed locally, with its lineage
    clean = client.put(
        inputs=[art.uri],
        transform={"executor": "clean@v1", "params": {}},
        data=table,
        name="team/dataset/clean",
    )

    # Registry: names, aliases, tags, audit
    client.set_alias("team/dataset/clean", "champion", clean.artifact_id, clean.version)
```

The client and the server distribution (`strata-notebook`) are independent:
they share only the JSON wire protocol, and neither depends on the other. Code
written against the old `strata.client` module imports `strata_client` instead.
