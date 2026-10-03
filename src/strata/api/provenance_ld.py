"""RO-Crate / JSON-LD description of a published artifact and its chain.

One builder feeds both the deposited RO-Crate 1.1 bundle and the schema.org JSON-LD
inlined on the hosted page, so the machine-readable account cannot drift from the page.

Mapping:

- The published artifact is the payload: a ``File`` with its digest. A bundle's Parquet
  copy is a second ``File`` with its own digest.
- Every upstream step is a ``CreativeWork``, not a ``File``: its bytes are not in the
  crate, and listing it as a present file would be a lie a validator cannot catch.
- Each execution is a ``CreateAction`` (``instrument`` = cell source as
  ``SoftwareSourceCode``, ``object`` = inputs, ``result`` = output), the shape PROV-O
  and RO-Crate both expect.
"""

from __future__ import annotations

import datetime

from strata.url_safety import web_url_or_none

# The crate context plus one local term. ``sha256`` is not defined in RO-Crate
# 1.1, and JSON-LD expansion discards undefined terms, so without it the
# integrity digest would vanish from the RDF.
_CONTEXT = [
    "https://w3id.org/ro/crate/1.1/context",
    {"sha256": "http://pending.schema.org/sha256"},
]
_PROFILE = "https://w3id.org/ro/crate/1.1"


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).isoformat()


def _prune(entity: dict) -> dict:
    """Drop empty values.

    An absent field says "not recorded"; a null would falsely claim it was recorded as nothing.
    """
    return {k: v for k, v in entity.items() if v not in (None, "", [], {})}


def _fragment(value: str) -> str:
    """Percent-encode a value for use in a fragment identifier.

    Author names are free-form (``"F. Li"`` has a space, illegal in an IRI); a strict RDF
    processor would silently drop the node.
    """
    from urllib.parse import quote

    return quote(value, safe="")


def _source_id(node) -> str:
    return f"#source-{_fragment(f'{node.artifact_id}@v={node.version}')}"


def _action_id(node) -> str:
    return f"#action-{_fragment(f'{node.artifact_id}@v={node.version}')}"


def _agent_id(name: str) -> str:
    return f"#agent-{_fragment(name)}"


def build_crate(
    *,
    publication,
    artifact,
    lineage,
    content_type: str,
    payload_id: str,
    include_descriptor: bool,
    companions: list[dict[str, str]] | None = None,
) -> dict:
    """Build the RO-Crate graph.

    ``payload_id`` addresses the payload: a filename inside a bundle, or the absolute ``/data``
    URL for the hosted page. ``include_descriptor`` adds the ``ro-crate-metadata.json``
    self-description a deposited crate must carry. ``companions`` are other renderings of the
    payload shipped beside it (``{file, content_type, sha256}``, as in ``manifest.json``).
    """
    companions = companions or []
    graph: list[dict] = []

    if include_descriptor:
        graph.append(
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "conformsTo": {"@id": _PROFILE},
                "about": {"@id": "./"},
            }
        )

    root_id = f"{artifact.id}@v={artifact.version}"
    steps = [
        node for node in lineage.nodes if node.type == "artifact" and node.artifact_id is not None
    ]
    # Match (id, version), not id: one id can appear at two versions in a graph
    # (a stale re-run, a loop carry), and id-only matching would drop a real
    # ancestor yet still map it onto the payload.
    upstream = [
        node
        for node in steps
        if (node.artifact_id, node.version) != (artifact.id, artifact.version)
    ]

    graph.append(
        _prune(
            {
                "@id": "./",
                "@type": "Dataset",
                "name": publication.title or root_id,
                "description": (
                    "A Strata artifact published with the code, inputs and "
                    "environment recorded when it was produced."
                ),
                "datePublished": _iso(publication.published_at),
                # A repository indexes on the DOI, so the root carries it as a
                # resolvable URL.
                "identifier": _identifier_of(publication),
                # Author order carries meaning. ``published_by`` is the fallback
                # for publications made before authors existed.
                "author": _authors_of(publication),
                "hasPart": [{"@id": payload_id}, *({"@id": c["file"]} for c in companions)],
                "mainEntity": {"@id": payload_id},
                "mentions": [{"@id": _action_id(node)} for node in steps],
            }
        )
    )

    graph.append(
        _prune(
            {
                "@id": payload_id,
                "@type": "File",
                "name": root_id,
                "encodingFormat": content_type,
                "contentSize": str(artifact.byte_size) if artifact.byte_size else None,
                "sha256": publication.content_sha256,
                "dateCreated": _iso(artifact.created_at),
            }
        )
    )

    for companion in companions:
        graph.append(
            {
                "@id": companion["file"],
                "@type": "File",
                "name": f"{root_id} ({companion['file']})",
                "encodingFormat": companion["content_type"],
                "sha256": companion["sha256"],
                "description": f"The same rows as {payload_id}, in another format.",
            }
        )

    for node in upstream:
        graph.append(
            _prune(
                {
                    "@id": f"{node.artifact_id}@v={node.version}",
                    # Not a File: these bytes are described, never shipped.
                    "@type": "CreativeWork",
                    "name": f"{node.artifact_id}@v={node.version}",
                    "dateCreated": _iso(node.created_at),
                }
            )
        )

    # Table inputs and unresolved leaves are referenced by ``_inputs_for``, so
    # they need nodes or the @id dangles. Datasets, not Files: a table lives in
    # a lake, not in this crate.
    for node in lineage.nodes:
        if node.type == "fetch":
            # A web data entity is a File whose @id is its URL (RO-Crate 1.1
            # allows this outside the crate). The digest is what was read, not
            # what the URL serves now.
            graph.append(
                _prune(
                    {
                        "@id": node.uri,
                        "@type": "File",
                        "name": node.uri,
                        "sha256": node.content_sha256,
                        "description": "Bytes read from this URL when the step ran.",
                        # RO-Crate's term for when a web data entity was retrieved.
                        "sdDatePublished": _iso(node.created_at),
                    }
                )
            )
        elif node.type != "artifact":
            graph.append(
                _prune(
                    {
                        "@id": node.uri,
                        "@type": "Dataset",
                        "name": node.uri,
                        "description": "An input read from outside this store.",
                    }
                )
            )

    agents: dict[str, dict] = {}
    for node in steps:
        step_id = f"{node.artifact_id}@v={node.version}"
        produced = payload_id if node.artifact_id == artifact.id else step_id
        inputs = _inputs_for(node, lineage, artifact, payload_id)

        if node.source:
            graph.append(
                {
                    "@id": _source_id(node),
                    "@type": "SoftwareSourceCode",
                    "name": f"Source of {step_id}",
                    "programmingLanguage": "Python",
                    "text": node.source,
                }
            )

        if node.principal:
            agents.setdefault(
                node.principal,
                {
                    "@id": _agent_id(node.principal),
                    "@type": "Person",
                    "name": node.principal,
                },
            )

        graph.append(
            _prune(
                {
                    "@id": _action_id(node),
                    "@type": "CreateAction",
                    "name": f"Computation of {step_id}",
                    "endTime": _iso(node.created_at),
                    "instrument": ({"@id": _source_id(node)} if node.source else None),
                    "object": [{"@id": ref} for ref in inputs],
                    "result": {"@id": produced},
                    "agent": ({"@id": _agent_id(node.principal)} if node.principal else None),
                    # Where these bytes were made, not a claim they'd be made
                    # again there.
                    "description": (f"Ran under {node.build_env}" if node.build_env else None),
                }
            )
        )

    for author in publication.authors:
        # An ORCID ``@id`` lets two crates say they name the same researcher.
        node_id = (
            f"https://orcid.org/{author['orcid']}"
            if author.get("orcid")
            else _agent_id(author["name"])
        )
        agents.setdefault(
            node_id,
            _prune(
                {
                    "@id": node_id,
                    "@type": "Person",
                    "name": author["name"],
                    "affiliation": author.get("affiliation"),
                }
            ),
        )
    if publication.published_by and not publication.authors:
        agents.setdefault(
            publication.published_by,
            {
                "@id": _agent_id(publication.published_by),
                "@type": "Person",
                "name": publication.published_by,
            },
        )
    graph.extend(agents.values())

    return {"@context": _CONTEXT, "@graph": graph}


def _identifier_of(publication) -> str | None:
    """The publication's resolvable identifier, preferring a DOI.

    ``None`` when it has none, so ``_prune`` drops it rather than emitting an empty identifier.
    """
    by_scheme = {entry["scheme"]: entry["value"] for entry in publication.external_ids}
    for scheme, template in (
        ("doi", "https://doi.org/{}"),
        ("arxiv", "https://arxiv.org/abs/{}"),
        ("zenodo", "https://zenodo.org/record/{}"),
        ("url", "{}"),
    ):
        value = by_scheme.get(scheme)
        if value:
            # A stored identifier is whatever was written; only a web URL is
            # emitted.
            url = web_url_or_none(value) or web_url_or_none(template.format(value))
            if url:
                return url
    return None


def _authors_of(publication):
    """Root-dataset authors: the declared ones, else whoever made the grant."""
    if publication.authors:
        return [
            {
                "@id": (
                    f"https://orcid.org/{author['orcid']}"
                    if author.get("orcid")
                    else _agent_id(author["name"])
                )
            }
            for author in publication.authors
        ]
    if publication.published_by:
        return {"@id": _agent_id(publication.published_by)}
    return None


def _inputs_for(node, lineage, artifact, payload_id: str) -> list[str]:
    """The ids this step consumed, as they appear elsewhere in the graph."""
    consumed = [edge.from_uri for edge in lineage.edges if edge.to_uri == node.uri]
    by_uri = {other.uri: other for other in lineage.nodes}

    ids: list[str] = []
    for uri in consumed:
        source = by_uri.get(uri)
        if source is None or source.artifact_id is None:
            # A table or unresolved input: its URI is the only handle, and the
            # input must not drop out of the record.
            ids.append(uri)
            continue
        if source.artifact_id == artifact.id:
            ids.append(payload_id)
        else:
            ids.append(f"{source.artifact_id}@v={source.version}")
    return ids
