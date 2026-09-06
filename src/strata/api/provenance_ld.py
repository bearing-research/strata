"""RO-Crate / JSON-LD description of a published artifact and its chain.

The page says what produced a result to a person. This says the same thing to
a machine, in the vocabulary research infrastructure already reads: RO-Crate
1.1 for a deposited bundle, and the same graph inline on the hosted page as
schema.org JSON-LD, which is what indexers and link unfurlers look for.

One builder feeds both. A crate that described the result differently from the
page would be a second account of the same thing, and a reader reconciling two
accounts of a provenance record is exactly the position this feature exists to
keep them out of.

The mapping, and why:

- The published artifact is the crate's payload — a ``File`` that is really
  there, with its digest.
- Every upstream step is a ``CreativeWork``, **not** a ``File``. Their bytes
  are deliberately not in the crate (publishing shows which steps produced a
  result; it does not hand over the upstream datasets), and listing them as
  files present would be a lie a validator cannot catch.
- Each execution is a ``CreateAction`` whose ``instrument`` is the cell source
  as ``SoftwareSourceCode``, ``object`` the inputs and ``result`` the output.
  That is the shape PROV-O and RO-Crate both expect for "this code, on these
  inputs, made this".
"""

from __future__ import annotations

import datetime

# The crate context plus one local term. ``sha256`` is not defined in RO-Crate
# 1.1, and an undefined term is *discarded* on JSON-LD expansion — so the
# integrity digest this whole feature rests on would look present in the raw
# JSON and be invisible in RDF, which is the worst of both.
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
    """Drop empty values. An absent field says "not recorded"; a null says it
    was recorded as nothing, which is a different and false claim."""
    return {k: v for k, v in entity.items() if v not in (None, "", [], {})}


def _fragment(value: str) -> str:
    """Percent-encode a value for use in a fragment identifier.

    Author names and principals are free-form — ``--author "F. Li"`` produces a
    space, which is illegal in an IRI. A strict processor drops or errors on
    the node, and the authorship link simply vanishes from the RDF while
    looking fine in the JSON.
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
) -> dict:
    """Build the RO-Crate graph.

    ``payload_id`` is how the payload is addressed: a filename inside a bundle,
    or the absolute ``/data`` URL for the hosted page. ``include_descriptor``
    adds the ``ro-crate-metadata.json`` self-description a deposited crate must
    carry and an inline page has no file to describe.
    """
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
    # Compared with the version, not the id alone. One id can appear at two
    # versions in a single graph — a stale cell re-run, or a loop carry — and
    # matching on the id would drop a genuine ancestor while still mapping it
    # onto the payload, producing an action whose result is someone else's
    # output.
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
                "author": (
                    {"@id": _agent_id(publication.published_by)}
                    if publication.published_by
                    else None
                ),
                "hasPart": [{"@id": payload_id}],
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

    # Table inputs and unresolved leaves are referenced by ``_inputs_for`` and
    # would otherwise be named by nothing — a dangling @id, which is exactly
    # the invariant this module claims to hold. They are Datasets rather than
    # Files: a table lives in a lake, not in this crate.
    for node in lineage.nodes:
        if node.type != "artifact":
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
                    # Recorded rather than asserted: it says where these bytes
                    # were made, not that they would be made again there.
                    "description": (f"Ran under {node.build_env}" if node.build_env else None),
                }
            )
        )

    if publication.published_by:
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


def _inputs_for(node, lineage, artifact, payload_id: str) -> list[str]:
    """The ids this step consumed, as they appear elsewhere in the graph."""
    consumed = [edge.from_uri for edge in lineage.edges if edge.to_uri == node.uri]
    by_uri = {other.uri: other for other in lineage.nodes}

    ids: list[str] = []
    for uri in consumed:
        source = by_uri.get(uri)
        if source is None or source.artifact_id is None:
            # A table or an unresolved input: name it by its URI, which is the
            # only handle there is, rather than dropping an input from the
            # record because it is not an artifact in this store.
            ids.append(uri)
            continue
        if source.artifact_id == artifact.id:
            ids.append(payload_id)
        else:
            ids.append(f"{source.artifact_id}@v={source.version}")
    return ids
