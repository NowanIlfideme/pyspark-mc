"""Basic distributions."""

from abc import ABCMeta
from types import UnionType
from typing import (
    Any,
    List,
    Type,
    get_args,
    get_origin,
    no_type_check,
    Union,
)
import inspect

import numpy as np
import pyspark.sql.types as T


@no_type_check
def eval_type(t, globalns=None, localns=None, recursive_guard=frozenset()):
    """Wrapper for typing._eval_type() function."""
    from typing import _eval_type  # noqa

    if localns is None:
        localns = {}
    if globalns is None:
        globalns = globals()

    return _eval_type(t, globalns, localns, recursive_guard=recursive_guard)


def get_type(p: inspect.Parameter) -> type:
    """Tries to get the type from a parameter."""
    if p.annotation == p.empty:
        raise TypeError(f"Missing annotation for {p!r}")
    t = eval_type(p.annotation)
    return t


# FIXME: Simplify, use recursive type approach :)
map_types = {
    float: T.FloatType,
    int: T.IntegerType,
    str: T.StringType,
    list[str]: lambda: T.ArrayType(T.StringType(), containsNull=False),
    List[str]: lambda: T.ArrayType(T.StringType(), containsNull=False),
    list[int]: lambda: T.ArrayType(T.IntegerType(), containsNull=False),
    List[int]: lambda: T.ArrayType(T.IntegerType(), containsNull=False),
    list[float]: lambda: T.ArrayType(T.FloatType(), containsNull=False),
    List[float]: lambda: T.ArrayType(T.FloatType(), containsNull=False),
    np.ndarray: lambda: T.ArrayType(T.FloatType(), containsNull=False),
}


def field_for(name: str, t: Type, default=inspect.Parameter.empty) -> T.StructField:
    """Gets a struct field for a given type."""
    nullable = False  # can become true

    if get_origin(t) in [List, list]:
        # Allow: `List[x]`, `list[x]`
        args = get_args(t)
        if len(args) != 1:
            raise TypeError(f"Multiple args given to list annotation: {t!r}")
        # and map_types will fix the rest? TODO: a proper recursive approach.
    elif get_origin(t) in [Union, UnionType]:
        # Allow: `Optional[x]`, `Union[x, None]`, `x|None`
        args = get_args(t)
        if (None in args) or (type(None) in args):
            nullable = True
        clean_args = tuple(x for x in args if x not in [None, type(None)])
        if len(clean_args) == 0:
            raise TypeError(f"Requires at least one type, got {t!r}")
        elif len(clean_args) > 1:
            raise TypeError(f"Multiple types (non-Optional) are unsupported: {args}")
        t = clean_args[0]
    elif get_origin(t) is not None:
        raise TypeError(f"Unsupported type annotation: {t!r}")

    # Default types?
    if default == inspect.Parameter.empty:
        pass
    elif default is None:
        # Disallow implicit nullable default?
        nullable = True
    else:
        # TODO: check default type
        pass

    # Map data types
    dt = map_types.get(t, None)()
    if dt is None:
        raise TypeError(f"Unsupported type annotation: {t!r}")

    return T.StructField(name, dt, nullable=nullable)


class DistroMeta(ABCMeta):
    """Metaclass for distribution objects."""

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        """Creates a new Distribution class."""
        x = super().__new__(cls, name, bases, namespace)

        if inspect.isabstract(x):
            # Abstact - don't do any more magic.
            return x

        # Inspect to get init arguments, and set a struct type
        # TODO: This can be done via a class method, right? No metaclass needed?
        sig = inspect.signature(x.__init__)
        fields = []
        for param in list(sig.parameters.values())[1:]:
            t = get_type(param)
            fld = field_for(param.name, t, default=param.default)
            fields.append(fld)
        SType = T.StructType(fields)
        x.SType = SType

        return x
