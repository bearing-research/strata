"""Core transform abstraction and the in-process transform registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    import pyarrow as pa


class Transform[P: BaseModel](ABC):
    """Abstract base class for all transforms.

    A subclass declares a ``Params`` pydantic model, implements :meth:`execute`,
    and is registered under a ``{name}@{version}`` ``ref`` with
    :func:`register_transform`. Personal mode runs it locally through :meth:`run`;
    service mode runs it through a registered HTTP executor.

    Examples
    --------
    >>> @register_transform("my_transform@v1")
    ... class MyTransform(Transform[MyParams]):
    ...     Params = MyParams
    ...
    ...     def execute(self, inputs: list[pa.Table], params: MyParams) -> pa.Table:
    ...         return inputs[0].filter(...)
    """

    ref: ClassVar[str]
    # Not ClassVar: typing forbids a type variable there, and this is what
    # ties parse_params's result to P.
    Params: type[P]

    def validate(self, inputs: list[pa.Table], params: P) -> None:
        """Validate inputs and parameters before :meth:`execute`; a no-op by default.

        Raises
        ------
        ValueError
            If validation fails.
        """

    @abstractmethod
    def execute(self, inputs: list[pa.Table], params: P) -> pa.Table:
        """Run the transformation logic on the input tables."""
        ...

    def get_input_names(self, num_inputs: int) -> list[str]:
        """Return display names for the inputs.

        Defaults to ``["input0", "input1", ...]``; override for names such as
        ``"left"`` and ``"right"``.
        """
        return [f"input{i}" for i in range(num_inputs)]

    @classmethod
    def parse_params(cls, params: dict[str, Any]) -> P:
        """Parse and validate raw parameters against ``Params``.

        Raises
        ------
        pydantic.ValidationError
            If the parameters do not satisfy the ``Params`` model.
        """
        return cls.Params.model_validate(params)

    def run(
        self,
        inputs: list[pa.Table],
        params: dict[str, Any],
    ) -> pa.Table:
        """Parse ``params``, call :meth:`validate`, then :meth:`execute`."""
        parsed_params = self.parse_params(params)
        self.validate(inputs, parsed_params)
        return self.execute(inputs, parsed_params)


_transforms: dict[str, type[Transform]] = {}


def register_transform(ref: str):
    """Return a class decorator that registers a transform under ``ref`` (``{name}@{version}``).

    Examples
    --------
    >>> @register_transform("my_transform@v1")
    ... class MyTransform(Transform[MyParams]):
    ...     ...
    """

    def decorator(cls: type[Transform]) -> type[Transform]:
        cls.ref = ref
        _transforms[ref] = cls
        return cls

    return decorator


def get_transform(ref: str) -> Transform | None:
    """Return a new instance of the transform registered under ``ref``, or ``None``.

    A ``local://`` prefix on ``ref`` is stripped before lookup.
    """
    if ref.startswith("local://"):
        ref = ref[8:]

    cls = _transforms.get(ref)
    if cls is None:
        return None
    return cls()


def list_transforms() -> list[str]:
    """List the references of all registered transforms."""
    return list(_transforms.keys())


def _run_transform(
    ref: str,
    inputs: list[pa.Table],
    params: dict[str, Any],
) -> pa.Table:
    """Run a registered transform by reference (server build runner and embedded executor).

    Library users should call ``client.materialize`` instead.

    Raises
    ------
    ValueError
        If no transform is registered under ``ref``.
    """
    transform = get_transform(ref)
    if transform is None:
        raise ValueError(f"Unknown transform: {ref}")
    return transform.run(inputs, params)


run_transform = _run_transform
