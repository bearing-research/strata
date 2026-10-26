--------------------------- MODULE ArtifactLifecycle ---------------------------
(***************************************************************************)
(* Model of the artifact_versions state machine in vestigo/store.py,      *)
(* with the retention roots and required edges of ADR 002 section 7.4.    *)
(*                                                                         *)
(* Each SQL transaction is one atomic action: create_artifact,             *)
(* finalize_and_set_name (finalize_artifact), finalize_canonical_together, *)
(* fail_artifact, retain, release, and one row of a garbage_collect sweep. *)
(*                                                                         *)
(* Dedup: at most one ready row per provenance. A finalize that finds the  *)
(* provenance held ready under another id supersedes its own row with a    *)
(* pointer (superseded_by) at the holder and drops its own bytes; the      *)
(* row reads the holder's bytes through the pointer. A refresh of the same *)
(* id, and finalize_canonical_together, supersede the holder instead,      *)
(* which keeps its bytes. A sweep takes a canonical only once nothing      *)
(* points at it.                                                           *)
(*                                                                         *)
(* Abstractions:                                                           *)
(*   - Tenants are ignored (a single tenant '').                           *)
(*   - Names and aliases are ignored, so every version is "unnamed", and   *)
(*     every id is caller-named (not minted): its current value is a GC    *)
(*     root.                                                               *)
(*   - Time is abstracted away: garbage_collect may treat any row as idle  *)
(*     enough. This over-approximates; counterexamples need the affected   *)
(*     row to be old enough in practice.                                   *)
(*   - Blob bytes are a boolean "present" flag.                            *)
(*   - A run's several outputs are finalized one at a time here; no        *)
(*     invariant below tells one transaction from several.                 *)
(*   - With Retention, a version may be retained (retain/release, any      *)
(*     kind and holder collapse to one root) and may declare at creation   *)
(*     the stored versions it requires (an output folder's files). GC      *)
(*     protects the closure of its roots over required edges and pointers. *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Ids,        \* artifact ids, e.g. two notebook cells in two notebooks
    Provs,      \* provenance hashes
    MaxVer,     \* bound on versions per id (keeps the state space finite)
    EnableGC,   \* include garbage_collect in the model
    Retention   \* include retain/release and required edges

Versions == 1..MaxVer
States   == {"absent", "building", "ready", "superseded", "failed"}

VARIABLES
    st,         \* st[i][v]   : row state ("absent" = no row / deleted)
    prov,       \* prov[i][v] : provenance hash of the row
    blob,       \* blob[i][v] : blob bytes of its own present
    ptr,        \* superseded_by: set of <<<<i, v>>, <<j, w>>>>, i@v reads j@w's bytes
    everReady,  \* everReady[i] : id i has held a current value at some point
    roots,      \* retention_roots: set of <<i, v>> something retains
    req         \* retention_edges: set of <<<<i, v>>, <<j, w>>>>, i@v requires j@w

vars == <<st, prov, blob, ptr, everReady, roots, req>>

Refs == Ids \X Versions

TypeOK ==
    /\ st \in [Ids -> [Versions -> States]]
    /\ prov \in [Ids -> [Versions -> Provs]]
    /\ blob \in [Ids -> [Versions -> BOOLEAN]]
    /\ ptr \subseteq Refs \X Refs
    /\ everReady \in [Ids -> BOOLEAN]
    /\ roots \subseteq Refs
    /\ req \subseteq Refs \X Refs

Stored == {r \in Refs : st[r[1]][r[2]] # "absent"}

\* Highest version number with a row in any state (0 when none).
\* garbage_collect's gap rule uses MAX(version) over all states.
MaxRowVer(i) ==
    IF \E v \in Versions : st[i][v] # "absent"
    THEN CHOOSE v \in Versions : st[i][v] # "absent" /\
                                \A w \in Versions : st[i][w] # "absent" => w <= v
    ELSE 0

\* States get_latest_version(i) accepts: superseded counts, since an id's
\* newest result is superseded when another id claims the same provenance.
Current == {"ready", "superseded"}

\* get_latest_version(i): the highest current version, or 0 for None.
LatestReady(i) ==
    IF \E v \in Versions : st[i][v] \in Current
    THEN CHOOSE v \in Versions : st[i][v] \in Current /\
                                \A w \in Versions : st[i][w] \in Current => w <= v
    ELSE 0

ReadyWithProv(p) == {<<i, v>> \in Refs : st[i][v] = "ready" /\ prov[i][v] = p}

\* The versions pointing at r through superseded_by.
PointersAt(r) == {e[1] : e \in {e \in ptr : e[2] = r}}

\* garbage_collect's roots (_protected_reachable): retention roots, rows still
\* building, and the current value of every id; a held version holds whatever
\* it requires and whatever it reads through its pointer, so the closure over
\* required edges and pointers is protected.
RECURSIVE Reach(_)
Reach(S) ==
    LET next == S \union {e[2] : e \in {e \in (req \union ptr) : e[1] \in S}} IN
    IF next = S THEN S ELSE Reach(next)

Held == Reach(roots
              \union {r \in Refs : st[r[1]][r[2]] = "building"}
              \union {<<i, LatestReady(i)>> : i \in {j \in Ids : LatestReady(j) # 0}})

Init ==
    /\ st = [i \in Ids |-> [v \in Versions |-> "absent"]]
    /\ prov \in [Ids -> [Versions -> Provs]]   \* value irrelevant until created
    /\ blob = [i \in Ids |-> [v \in Versions |-> FALSE]]
    /\ ptr = {}
    /\ everReady = [i \in Ids |-> FALSE]
    /\ roots = {}
    /\ req = {}

-----------------------------------------------------------------------------
(* create_artifact: next version in 'building', requiring R (stored      *)
(* versions, written as edges in the row's transaction). A consumer may  *)
(* store a folder's files before the manifest that requires them.         *)
Create(i, p, R) ==
    LET v == MaxRowVer(i) + 1 IN
    /\ v \in Versions
    /\ R \subseteq Stored
    /\ Retention \/ R = {}
    /\ st' = [st EXCEPT ![i][v] = "building"]
    /\ prov' = [prov EXCEPT ![i][v] = p]
    /\ req' = req \union {<<<<i, v>>, r>> : r \in R}
    /\ UNCHANGED <<blob, ptr, everReady, roots>>

(* retain: a root on a stored version, idempotent per holder.             *)
Retain(i, v) ==
    /\ Retention
    /\ st[i][v] # "absent"
    /\ roots' = roots \union {<<i, v>>}
    /\ UNCHANGED <<st, prov, blob, ptr, everReady, req>>

(* release: the holder lets go.                                            *)
Release(i, v) ==
    /\ <<i, v>> \in roots
    /\ roots' = roots \ {<<i, v>>}
    /\ UNCHANGED <<st, prov, blob, ptr, everReady, req>>

(* blob_store.write_blob                                                  *)
WriteBlob(i, v) ==
    /\ st[i][v] = "building"
    /\ ~blob[i][v]
    /\ blob' = [blob EXCEPT ![i][v] = TRUE]
    /\ UNCHANGED <<st, prov, ptr, everReady, roots, req>>

(* Build error: fail_artifact.                                            *)
Fail(i, v) ==
    /\ st[i][v] = "building"
    /\ st' = [st EXCEPT ![i][v] = "failed"]
    /\ UNCHANGED <<prov, blob, ptr, everReady, roots, req>>

(* finalize_and_set_name, one of three outcomes:                          *)
(*  - another id already holds a ready row with this provenance: ours is  *)
(*    superseded with a pointer at it and its own bytes dropped            *)
(*    (_supersede_duplicate, in the lookup path or the integrity-error     *)
(*    path); the caller is handed the holder;                             *)
(*  - an older ready version of the same id has this provenance: supersede *)
(*    it (it keeps its bytes) and promote ours (refresh rebuild);          *)
(*  - otherwise promote ours.                                             *)
(* Every outcome leaves id i with a current value.                        *)
Finalize(i, v) ==
    LET p == prov[i][v]
        foreign == {x \in ReadyWithProv(p) : x[1] # i}
        own     == {x \in ReadyWithProv(p) : x[1] = i}
    IN
    /\ st[i][v] = "building"
    /\ blob[i][v]
    /\ IF foreign # {}
       THEN LET f == CHOOSE x \in foreign : TRUE IN
            /\ st' = [st EXCEPT ![i][v] = "superseded"]
            /\ ptr' = ptr \union {<<<<i, v>>, f>>}
            /\ blob' = [blob EXCEPT ![i][v] = FALSE]
       ELSE /\ st' = [x \in Ids |-> [w \in Versions |->
                        IF x = i /\ w = v THEN "ready"
                        ELSE IF <<x, w>> \in own THEN "superseded"
                        ELSE st[x][w]]]
            /\ UNCHANGED <<ptr, blob>>
    /\ everReady' = [everReady EXCEPT ![i] = TRUE]
    /\ UNCHANGED <<prov, roots, req>>

(* finalize_canonical_together, for one staged row: ours becomes the     *)
(* single ready row for its provenance under its own id, and every other  *)
(* ready holder is superseded, keeping its bytes (and the pointers at it). *)
FinalizeCanonical(i, v) ==
    LET p == prov[i][v] IN
    /\ st[i][v] = "building"
    /\ blob[i][v]
    /\ st' = [x \in Ids |-> [w \in Versions |->
                IF x = i /\ w = v THEN "ready"
                ELSE IF st[x][w] = "ready" /\ prov[x][w] = p THEN "superseded"
                ELSE st[x][w]]]
    /\ everReady' = [everReady EXCEPT ![i] = TRUE]
    /\ UNCHANGED <<prov, blob, ptr, roots, req>>

(* garbage_collect (collect_latest=False, no names/aliases), one row:     *)
(* delete a ready/superseded/failed row that is not the MAX(version) of   *)
(* its id (the gap rule: deleting the highest row would let create_artifact *)
(* reuse its number), not the current value of its id, not read through a  *)
(* pointer by any row (the claim's check; a pointer goes before or with    *)
(* its canonical), and, with Retention, not held. Deleting a version drops *)
(* its roots, its outgoing edges and its own pointer                       *)
(* (_delete_version_children); edges into it stay, so a requirer that      *)
(* outlived it reads as broken (requirements_met).                         *)
GC(i, v) ==
    /\ st[i][v] \in {"ready", "superseded", "failed"}
    /\ v < MaxRowVer(i)
    /\ v # LatestReady(i)
    /\ PointersAt(<<i, v>>) = {}
    /\ Retention => <<i, v>> \notin Held
    /\ st' = [st EXCEPT ![i][v] = "absent"]
    /\ blob' = [blob EXCEPT ![i][v] = FALSE]
    /\ ptr' = {e \in ptr : e[1] # <<i, v>>}
    /\ roots' = roots \ {<<i, v>>}
    /\ req' = {e \in req : e[1] # <<i, v>>}
    /\ UNCHANGED <<prov, everReady>>

Next ==
    \/ \E i \in Ids, p \in Provs, R \in (IF Retention THEN SUBSET Refs ELSE {{}}) :
        Create(i, p, R)
    \/ \E i \in Ids, v \in Versions :
        \/ WriteBlob(i, v) \/ Fail(i, v) \/ Finalize(i, v) \/ FinalizeCanonical(i, v)
        \/ (EnableGC /\ GC(i, v))
        \/ Retain(i, v) \/ Release(i, v)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)

\* The partial unique index idx_tenant_provenance_unique.
UniqueReadyPerProv ==
    \A p \in Provs : Cardinality(ReadyWithProv(p)) <= 1

\* A servable row always has bytes: its own, or those of the canonical it
\* reads through its pointer (_blob_key follows superseded_by one step).
ServableHasBlob ==
    \A i \in Ids, v \in Versions :
        st[i][v] \in {"ready", "superseded"} =>
            \/ blob[i][v]
            \/ \E c \in Refs : <<<<i, v>>, c>> \in ptr /\ blob[c[1]][c[2]]

\* A pointer names a stored row, and only one: a sweep never takes a
\* canonical while something reads through it.
PointersResolve ==
    /\ \A e \in ptr : e[2] \in Stored
    /\ \A r \in Refs : Cardinality({c \in Refs : <<r, c>> \in ptr}) <= 1

\* Once an id has had a current value, get_latest_version(id) keeps
\* returning one. A consumer resolves a cell's inputs this way, so a
\* violation is an upstream that silently disappears from a downstream
\* cell's namespace.
CurrentValueDurable ==
    \A i \in Ids : everReady[i] => LatestReady(i) # 0

\* A root points at a stored version: deleting a version drops its roots.
RootsStored ==
    roots \subseteq Stored

\* No step deletes a held version, so a held output folder never loses a
\* file: what it requires is held with it. Stated over steps rather than
\* states because a caller may retain, or require, a version whose own
\* requirements GC already took while nothing held it (TLC finds that in
\* seven steps); the ledger keeps what is held, and whether a version is
\* whole when a caller holds it is the caller's to check (requirements_met).
HeldIsKept ==
    [][\A r \in Held \intersect Stored : r \in Stored']_vars
=============================================================================
