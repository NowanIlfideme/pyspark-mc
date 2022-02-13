"""Discrete univariate distributions."""

import numpy as np
from numpy.random import Generator


from .base import Distribution


class Bernoulli(Distribution):
    """Bernoulli distribution (coin-flip)."""

    def __init__(self, p: float):
        self.p = p


class Binomial(Distribution):
    """Binomial distribution (n coin-flips)."""

    def __init__(self, p: float, n: int):
        self.p = p
        self.n = n

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.binomial(self.n, self.p, size=shape)


class Poisson(Distribution):
    """Poisson distribution."""

    def __init__(self, mu: float):
        self.mu = mu

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.poisson(self.mu, size=shape)


class Categorical(Distribution):
    """Categorical distribution."""

    def __init__(self, probs: np.ndarray[1, float]):
        self.probs = probs

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.choice(len(self.probs), p=self.probs, size=shape)


class DiscreteUniform(Distribution):
    """Discrete uniform distribution."""

    def __init__(self, n: int):
        self.n = n

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.choice(self.n, size=shape)


# TODO: named categories? e.g. list of str? or do it via field metadata?
# TODO: other classical distributions, e.g.:
# https://docs.pymc.io/en/v3/api/distributions/discrete.html


class Empirical(Distribution):
    """Empirical distribution of values.

    FIXME: Split this into Empirical distributions for `float`, `bool`, `int`, `str`
    """

    def __init__(self, draws: np.ndarray[1, float]):
        self.draws = draws

    def _sample(self, shape: tuple[int, ...], rng: Generator) -> np.ndarray:
        return rng.choice(self.draws, size=shape)
