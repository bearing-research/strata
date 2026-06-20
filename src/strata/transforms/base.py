"""Core transform abstraction and the in-process transform registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    import pyarrow as pa


class Transform[P: BaseModel](ABC):
    """Abstract base class for all transforms.

    A transform encapsulates a parameter schema, optional input validation, and
    the execution logic that maps input Arrow tables to a single result table.
    A subclass declares a ``Params`` model, implements :meth:`execute`, and is
    registered under a ``{name}@{version}`` reference with
    :func:`register_transform`.

    In personal mode a transform runs locally through :meth:`run`. In service
    mode it runs remotely via a registered HTTP executor.

    Attributes
    ----------
    ref : str
        Reference identifying the transform, formatted ``{name}@{version}``
        (for example ``"scan@v1"`` or ``"duckdb_sql@v1"``). Assigned by
        :func:`register_transform`.
    Params : type of pydantic.BaseModel
        Model used to parse and validate the transform's parameters.

    Examples
    --------
    >>> class MyParams(BaseModel):
    ...     foo: str
    ...     bar: int = 10
    >>> @register_transform("my_transform@v1")
    ... class MyTransform(Transform[MyParams]):
    ...     Params = MyParams
    ...
    ...     def execute(self, inputs: list[pa.Table], params: MyParams) -> pa.Table:
    ...         return inputs[0].filter(...)
    """

    ref: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def validate(self, inputs: list[pa.Table], params: P) -> None:
        """Validate inputs and parameters before execution.

        Called by :meth:`run` ahead of :meth:`execute`. The default
        implementation performs no validation; override to add custom checks.

        Parameters
        ----------
        inputs : list of pyarrow.Table
            Input tables for the transform.
        params : P
            Validated parameters.

        Raises
        ------
        ValueError
            If validation fails.
        """

    @abstractmethod
    def execute(self, inputs: list[pa.Table], params: P) -> pa.Table:
        """Run the transformation logic.

        Parameters
        ----------
        inputs : list of pyarrow.Table
            Input tables for the transform.
        params : P
            Validated parameters.

        Returns
        -------
        pyarrow.Table
            The result table.
        """
        ...

    def get_input_names(self, num_inputs: int) -> list[str]:
        """Return display names for the input tables.

        The default is ``["input0", "input1", ...]``. Override to provide
        domain-specific names such as ``"left"`` and ``"right"`` for a join.

        Parameters
        ----------
        num_inputs : int
            Number of inputs.

        Returns
        -------
        list of str
            Name for each input, in order.
        """
        return [f"input{i}" for i in range(num_inputs)]

    @classmethod
    def parse_params(cls, params: dict[str, Any]) -> BaseModel:
        """Parse and validate raw parameters against ``Params``.

        Parameters
        ----------
        params : dict
            Raw parameter mapping.

        Returns
        -------
        pydantic.BaseModel
            A validated ``Params`` instance.

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
        """Parse parameters, validate, and execute the transform.

        The main entry point for running a transform end to end: it parses and
        validates ``params`` against ``Params``, calls :meth:`validate` for any
        custom checks, then calls :meth:`execute`.

        Parameters
        ----------
        inputs : list of pyarrow.Table
            Input tables for the transform.
        params : dict
            Raw parameter mapping.

        Returns
        -------
        pyarrow.Table
            The result table.
        """
        parsed_params = self.parse_params(params)
        self.validate(inputs, parsed_params)
        return self.execute(inputs, parsed_params)


_transforms: dict[str, type[Transform]] = {}


def register_transform(ref: str):
    """Register a transform class under a reference.

    Parameters
    ----------
    ref : str
        Transform reference, formatted ``{name}@{version}`` (for example
        ``"duckdb_sql@v1"``).

    Returns
    -------
    callable
        A class decorator that sets ``ref`` on the class and adds it to the
        registry.

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
    """Look up a transform instance by reference.

    Parameters
    ----------
    ref : str
        Transform reference, with an optional ``local://`` prefix that is
        stripped before lookup.

    Returns
    -------
    Transform or None
        A new instance of the registered transform, or ``None`` if no transform
        is registered under ``ref``.
    """
    if ref.startswith("local://"):
        ref = ref[8:]

    cls = _transforms.get(ref)
    if cls is None:
        return None
    return cls()


def list_transforms() -> list[str]:
    """List the references of all registered transforms.

    Returns
    -------
    list of str
        Every registered transform reference.
    """
    return list(_transforms.keys())


def _run_transform(
    ref: str,
    inputs: list[pa.Table],
    params: dict[str, Any],
) -> pa.Table:
    """Run a registered transform by reference.

    Internal entry point used by the server's build runner and embedded
    executor. Library users should call ``client.materialize`` instead.

    Parameters
    ----------
    ref : str
        Transform reference (for example ``"duckdb_sql@v1"``).
    inputs : list of pyarrow.Table
        Input tables for the transform.
    params : dict
        Raw parameter mapping.

    Returns
    -------
    pyarrow.Table
        The result table.

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
