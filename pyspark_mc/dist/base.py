"""Disrtribution base objects."""

import inspect
from abc import ABC, abstractmethod
from typing import Final
import numpy as np

from numpy.random import Generator, default_rng
from pyspark.sql.types import StructType, StructField

from pyspark_mc.version import __version__
from pyspark_mc.internals.type_mapping import (
    convert_annotation_to_spark,
    get_param_type,
)

DISTRO_KEY = "distribution"

__distributions__: dict[str, type["Distribution"]] = {}


class Distribution(ABC):
    """Base class for distributions."""

    def __init__(self):
        """Creates the distribution from the arguments."""

    def sample(
        self,
        shape: int | tuple[int, ...] = 1,
        *,
        random_state: int | Generator | None = None,
    ) -> np.ndarray:
        """Samples data from this distribution.

        FIXME: Figure out what the semantics are supposed to be here.
        """
        # Get shape
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)

        # Get RNG
        if isinstance(random_state, Generator):
            rng = random_state
        else:
            rng = default_rng(random_state)
        # Sample
        res = self._sample(shape, rng)
        return res

    @abstractmethod
    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        """Internal sample method, which should be overridden."""

    @classmethod
    def as_struct_type(cls) -> StructType:
        """Gets the corresponding PySpark StructType for this distribution."""
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]
        fields = []
        for p in params:
            t = get_param_type(p)
            fld = convert_annotation_to_spark(p.name, t)
            fields.append(fld)
        return StructType(fields)

    @classmethod
    def as_struct_field(cls, name: str) -> StructField:
        """Creates a StructField with the given name."""
        cn = cls.__qualname__
        dt = cls.as_struct_type()
        metadata = {DISTRO_KEY: cn, "__version__": __version__}
        return StructField(name, dt, nullable=False, metadata=metadata)

    def __init_subclass__(cls) -> None:
        """Checks concrete subclasses for correctness of implementation."""
        if inspect.isabstract(cls):
            return
        global __distributions__
        cn = cls.__qualname__

        # Check for re-definition
        if cn in __distributions__:
            raise TypeError(f"A Distribution called {cn!r} already exists.")

        # Check struct type implementation
        try:
            st = cls.as_struct_type()
            assert isinstance(st, StructType)
        except Exception as e:
            raise TypeError(f"Improperly defined type {cls!r}") from e

        # Register class
        __distributions__[cn] = cls

    def __repr__(self) -> str:
        """Readable representation."""
        cn = type(self).__qualname__
        sig = inspect.signature(self.__init__)
        params = list(sig.parameters.values())[0:]
        mapping = []
        for p in params:
            mapping.append([p.name, getattr(self, p.name, "???")])
        ppack = ", ".join([f"{k!s}={v!r}" for k, v in mapping])
        return f"{cn}({ppack})"


def to_distribution(name: str, *values) -> Distribution:
    """Gets and converts to a distribution by name."""
    cls = __distributions__.get(name)
    if cls is None:
        raise TypeError(f"Unknown distribution: {name!r}")
    # FIXME: This isn't that easy. :D
    res = cls(*values)
    return res
