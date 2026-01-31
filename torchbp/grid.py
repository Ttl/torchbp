"""Grid classes for SAR image formation.

This module provides typed grid classes to replace dictionary-based grid definitions
with validation, type safety, and cached computations.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Grid(ABC):
    """Base class for all grid types."""

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Return grid dimensions."""
        pass

    @abstractmethod
    def spacing(self) -> Tuple[float, ...]:
        """Return grid spacing."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to legacy dict format for backward compatibility."""
        pass


class PolarGrid(Grid):
    """Pseudo-polar grid (theta stored as sin(angle)).

    Parameters
    ----------
    r_range : Tuple[float, float]
        Range extent (r0, r1) in meters. r1 must be > r0.
    theta_range : Tuple[float, float]
        Azimuth extent (theta0, theta1) as sin(angle). Must be in [-1, 1].
    nr : int
        Number of range bins. Must be positive.
    ntheta : int
        Number of azimuth bins. Must be positive.

    Examples
    --------
    >>> grid = PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=100, ntheta=200)
    >>> grid.dr  # Range spacing (cached)
    0.5
    >>> grid.dtheta  # Azimuth spacing (cached)
    0.01
    >>> grid.shape()
    (100, 200)
    """

    def __init__(
        self,
        r_range: Tuple[float, float],
        theta_range: Tuple[float, float],
        nr: int,
        ntheta: int
    ):
        # Unpack ranges
        r0, r1 = r_range
        theta0, theta1 = theta_range

        # Validation
        if r1 <= r0:
            raise ValueError(f"r1 ({r1}) must be > r0 ({r0})")
        if nr <= 0 or ntheta <= 0:
            raise ValueError(f"nr ({nr}) and ntheta ({ntheta}) must be positive")
        if not (-1 <= theta0 <= 1 and -1 <= theta1 <= 1):
            raise ValueError(
                f"theta must be in [-1, 1], got theta0={theta0}, theta1={theta1}"
            )
        if theta1 <= theta0:
            raise ValueError(f"theta1 ({theta1}) must be > theta0 ({theta0})")

        # Store ranges
        self.r_range = r_range
        self.theta_range = theta_range
        self.nr = nr
        self.ntheta = ntheta

        # Cache computed properties
        self._dr = (r1 - r0) / nr
        self._dtheta = (theta1 - theta0) / ntheta

    @property
    def r0(self) -> float:
        """Minimum range in meters."""
        return self.r_range[0]

    @property
    def r1(self) -> float:
        """Maximum range in meters."""
        return self.r_range[1]

    @property
    def theta0(self) -> float:
        """Minimum azimuth (sin of angle)."""
        return self.theta_range[0]

    @property
    def theta1(self) -> float:
        """Maximum azimuth (sin of angle)."""
        return self.theta_range[1]

    @property
    def dr(self) -> float:
        """Range spacing in meters (cached)."""
        return self._dr

    @property
    def dtheta(self) -> float:
        """Azimuth spacing (cached)."""
        return self._dtheta

    def shape(self) -> Tuple[int, int]:
        """Return grid dimensions (nr, ntheta)."""
        return (self.nr, self.ntheta)

    def spacing(self) -> Tuple[float, float]:
        """Return grid spacing (dr, dtheta)."""
        return (self.dr, self.dtheta)

    def to_dict(self) -> dict:
        """Convert to legacy dict format.

        Returns
        -------
        dict
            Grid dict with keys: "r", "theta", "nr", "ntheta"
        """
        return {
            "r": self.r_range,
            "theta": self.theta_range,
            "nr": self.nr,
            "ntheta": self.ntheta
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PolarGrid':
        """Create PolarGrid from legacy dict format.

        Parameters
        ----------
        d : dict
            Dict with keys: "r", "theta", "nr", "ntheta"

        Returns
        -------
        PolarGrid
            Grid object
        """
        return cls(d["r"], d["theta"], d["nr"], d["ntheta"])

    def with_doubled_azimuth(self) -> 'PolarGrid':
        """Create grid with 2x azimuth resolution.

        Used in FFBP algorithm for azimuth upsampling.

        Returns
        -------
        PolarGrid
            New grid with ntheta *= 2
        """
        return PolarGrid(self.r_range, self.theta_range, self.nr, 2 * self.ntheta)

    def resize(self, nr: int = None, ntheta: int = None) -> 'PolarGrid':
        """Return new grid with different resolution.

        Parameters
        ----------
        nr : int, optional
            New number of range bins. If None, keep current nr.
        ntheta : int, optional
            New number of azimuth bins. If None, keep current ntheta.

        Returns
        -------
        PolarGrid
            New grid with updated resolution
        """
        return PolarGrid(
            self.r_range,
            self.theta_range,
            nr if nr is not None else self.nr,
            ntheta if ntheta is not None else self.ntheta
        )

    def __repr__(self) -> str:
        return (
            f"PolarGrid(r=({self.r0:.1f}, {self.r1:.1f}), "
            f"theta=({self.theta0:.2f}, {self.theta1:.2f}), "
            f"nr={self.nr}, ntheta={self.ntheta})"
        )


class CartesianGrid(Grid):
    """Cartesian grid.

    Parameters
    ----------
    x_range : Tuple[float, float]
        X extent (x0, x1) in meters. x1 must be > x0.
    y_range : Tuple[float, float]
        Y extent (y0, y1) in meters. y1 must be > y0.
    nx : int
        Number of X samples. Must be positive.
    ny : int
        Number of Y samples. Must be positive.

    Examples
    --------
    >>> grid = CartesianGrid(x_range=(-50, 50), y_range=(-50, 50), nx=100, ny=100)
    >>> grid.dx
    1.0
    >>> grid.dy
    1.0
    >>> grid.shape()
    (100, 100)
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        nx: int,
        ny: int
    ):
        # Unpack ranges
        x0, x1 = x_range
        y0, y1 = y_range

        # Validation
        if x1 <= x0:
            raise ValueError(f"x1 ({x1}) must be > x0 ({x0})")
        if y1 <= y0:
            raise ValueError(f"y1 ({y1}) must be > y0 ({y0})")
        if nx <= 0 or ny <= 0:
            raise ValueError(f"nx ({nx}) and ny ({ny}) must be positive")

        # Store ranges
        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny

        # Cache computed properties
        self._dx = (x1 - x0) / nx
        self._dy = (y1 - y0) / ny

    @property
    def x0(self) -> float:
        """Minimum X coordinate in meters."""
        return self.x_range[0]

    @property
    def x1(self) -> float:
        """Maximum X coordinate in meters."""
        return self.x_range[1]

    @property
    def y0(self) -> float:
        """Minimum Y coordinate in meters."""
        return self.y_range[0]

    @property
    def y1(self) -> float:
        """Maximum Y coordinate in meters."""
        return self.y_range[1]

    @property
    def dx(self) -> float:
        """X spacing in meters (cached)."""
        return self._dx

    @property
    def dy(self) -> float:
        """Y spacing in meters (cached)."""
        return self._dy

    def shape(self) -> Tuple[int, int]:
        """Return grid dimensions (nx, ny)."""
        return (self.nx, self.ny)

    def spacing(self) -> Tuple[float, float]:
        """Return grid spacing (dx, dy)."""
        return (self.dx, self.dy)

    def to_dict(self) -> dict:
        """Convert to legacy dict format.

        Returns
        -------
        dict
            Grid dict with keys: "x", "y", "nx", "ny"
        """
        return {
            "x": self.x_range,
            "y": self.y_range,
            "nx": self.nx,
            "ny": self.ny
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CartesianGrid':
        """Create CartesianGrid from legacy dict format.

        Parameters
        ----------
        d : dict
            Dict with keys: "x", "y", "nx", "ny"

        Returns
        -------
        CartesianGrid
            Grid object
        """
        return cls(d["x"], d["y"], d["nx"], d["ny"])

    def resize(self, nx: int = None, ny: int = None) -> 'CartesianGrid':
        """Return new grid with different resolution.

        Parameters
        ----------
        nx : int, optional
            New number of X samples. If None, keep current nx.
        ny : int, optional
            New number of Y samples. If None, keep current ny.

        Returns
        -------
        CartesianGrid
            New grid with updated resolution
        """
        return CartesianGrid(
            self.x_range,
            self.y_range,
            nx if nx is not None else self.nx,
            ny if ny is not None else self.ny
        )

    def __repr__(self) -> str:
        return (
            f"CartesianGrid(x=({self.x0:.1f}, {self.x1:.1f}), "
            f"y=({self.y0:.1f}, {self.y1:.1f}), "
            f"nx={self.nx}, ny={self.ny})"
        )
