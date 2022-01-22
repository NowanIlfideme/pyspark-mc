"""Continuous univariate distributions."""

from .base import DistroMeta


class Uniform(metaclass=DistroMeta):
    """Uniform distribution (continuous)."""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper


class Normal(metaclass=DistroMeta):
    """A normal distribution object."""

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma


# TODO: Other classical distros, e.g.
# https://docs.pymc.io/en/v3/api/distributions/continuous.html
