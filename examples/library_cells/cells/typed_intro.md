# Typed library cells (PEP 563)

Type annotations participate in the free-variable check by default , 
`def f(x: SomeType): ...` would block export when `SomeType` isn't
bound in the slice.

Adding `from __future__ import annotations` to the cell relaxes this:
PEP 563 stringifies annotations and `symtable` correctly drops them from
the reference set. That makes it easy to write helpers that hint at types
defined elsewhere (e.g. `pyarrow.Table`) without forcing every consumer
to import the same names.
