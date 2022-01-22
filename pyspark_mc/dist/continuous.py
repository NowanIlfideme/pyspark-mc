"""Continuous univariate distributions."""

from .base import DistroMeta


class Normal(metaclass=DistroMeta):
    """A normal distribution object."""

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
