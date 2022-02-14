"""Type mapping between Python and PySpark types."""

import inspect
from types import UnionType
from typing import (
    Any,
    Callable,
    List,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    no_type_check,
)

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    FloatType,
    IntegerType,
    MapType,
    NullType,
    StringType,
    StructField,
    StructType,
)

import numpy as np
import warnings

IntArray = np.ndarray[1, int]
BooleanArray = np.ndarray[1, bool]
FloatArray = np.ndarray[1, float]
NullableFloatArray = np.ndarray[1, float | None]


def convert_annotation_to_spark(name: str, t: type) -> StructField:
    """Converts a Python type annotation to PySpark DataType."""
    # TODO: Check for numpy.typing.NDArray and others... they're a bit raw for now

    # Scalar types
    if get_origin(t) is None:
        if t is None:
            return StructField(name, NullType(), nullable=True)  # FIXME: allowed?
        if t is Any:
            return StructField(name, NullType())  # FIXME: is this allowed?
        if issubclass(t, str):
            return StructField(name, StringType(), nullable=False)
        if issubclass(t, bool):
            return StructField(name, BooleanType(), nullable=False)
        if issubclass(t, float):
            return StructField(name, FloatType(), nullable=False)
        if issubclass(t, int):
            return StructField(name, IntegerType(), nullable=False)
        # Other things are disallowed
        if isinstance(t, np.ndarray):
            raise TypeError(
                "Cannot specify ndarray without sizes. Use `np.ndarray[1, float]`"
                " or `IntArray`/`BooleanArray`/`[Nullable]FloatArray` instead."
            )
        raise TypeError(f"Unknown type annotation: {t!r}")
    # More complex case
    origin = get_origin(t)
    args = get_args(t)
    # Optional
    if origin in [Union, UnionType]:
        # Check for nullables
        has_none = bool((None in args) or (type(None) in args))
        clean_args = tuple(x for x in args if x not in [None, type(None)])
        # Make sure it's a union with only 1 other value
        if len(clean_args) == 1:
            base = convert_annotation_to_spark(name, clean_args[0])
            return StructField(name, base.dataType, nullable=has_none)
        raise TypeError(f"Unions of multiple types (besides None) not allowed: {t!r}")

    # lists as arrays
    if origin in [List, list]:
        # This can do arrays of arrays
        base = convert_annotation_to_spark(name, args[0])
        at = ArrayType(base.dataType, containsNull=base.nullable)
        return StructField(name, at, nullable=False)

    # numpy array
    if origin is np.ndarray:
        n = args[0]
        if n is Any:
            n = 1  # FIXME: This always assumes ndarray[Any, dtype] is 1D for now
        if not isinstance(n, int):
            raise TypeError(f"Unknown ndarray annotation (n = {n!r}): {t!r}")
        # Make n-dimensional list as array-of-arrays
        new_t = args[1:]
        for _ in range(n):
            new_t = list[new_t]
        return convert_annotation_to_spark(name, new_t)

    # tuples as lists or structs
    if origin in [Tuple, tuple]:
        if ... in args:
            # treat it as a list
            clean_args = tuple(x for x in args if x is not ...)
            return convert_annotation_to_spark(name, list[clean_args])
        else:
            # treat it as a struct
            warnings.warn(f"Unknown names for fields, setting as '{name}_i'.")
            fields = []
            for i, arg in enumerate(args):
                fld = convert_annotation_to_spark(f"{name}_{i}", arg)
                fields.append(fld)
            return StructField(name, StructType(fields), nullable=False)

    raise TypeError(f"Unknown annotation origin {origin!r} for: {t!r}")


@no_type_check
def eval_type(t, globalns=None, localns=None, recursive_guard=frozenset()):
    """Wrapper for typing._eval_type() function."""
    from typing import _eval_type  # noqa

    if localns is None:
        localns = {}
    if globalns is None:
        globalns = globals()

    return _eval_type(t, globalns, localns, recursive_guard=recursive_guard)


def get_param_type(p: inspect.Parameter) -> type:
    """Tries to get the type from a parameter."""
    if p.annotation == p.empty:
        raise TypeError(f"Missing annotation for {p!r}")
    t = eval_type(p.annotation)
    return t
