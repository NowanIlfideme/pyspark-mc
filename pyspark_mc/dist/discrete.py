"""Discrete univariate distributions."""

import numpy as np

from .base import DistroMeta


class Bernoulli(metaclass=DistroMeta):
    """Bernoulli distribution (coin-flip)."""

    def __init__(self, p: float):
        self.p = p


class Binomial(metaclass=DistroMeta):
    """Binomial distribution (n coin-flips)."""

    def __init__(self, p: float, n: int):
        self.p = p
        self.n = n


class Poisson(metaclass=DistroMeta):
    """Poisson distribution."""

    def __init__(self, mu: float):
        self.mu = mu


class Categorical(metaclass=DistroMeta):
    """Categorical distribution."""

    def __init__(self, probs: np.ndarray):
        self.probs = probs


class DiscreteUniform(metaclass=DistroMeta):
    """Discrete uniform distribution."""

    def __init__(self, n: int):
        self.n = n


# TODO: named categories? e.g. list of str? or do it via field metadata?
# TODO: other classical distributions, e.g.:
# https://docs.pymc.io/en/v3/api/distributions/discrete.html


class Empirical(metaclass=DistroMeta):
    """Empirical distribution of values."""

    def __init__(self, draws: np.ndarray):
        self.draws = draws
