"""Continuous univariate distributions."""

import numpy as np
from numpy.random import Generator

from .base import Distribution


class Uniform(Distribution):
    """Uniform distribution (continuous)."""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=shape)


class Normal(Distribution):
    """A normal distribution object."""

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size=shape)


# TODO: Other classical distros, e.g.
# https://docs.pymc.io/en/v3/api/distributions/continuous.html
