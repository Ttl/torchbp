from typing import Tuple, Optional
import torch
from torch import Tensor
from ..grid import unpack_polar_grid, unpack_cartesian_grid


def get_batch_dims(data: Tensor, pos: Tensor) -> Tuple[int, int, int]:
    """Extract (nbatch, nsweeps, sweep_samples) with validation.

    Handles both batched (3D) and unbatched (2D) inputs.

    Parameters
    ----------
    data : Tensor
        Data tensor, shape (nsweeps, sweep_samples) or (nbatch, nsweeps, sweep_samples)
    pos : Tensor
        Position tensor, shape (nsweeps, 3) or (nbatch, nsweeps, 3)

    Returns
    -------
    tuple
        (nbatch, nsweeps, sweep_samples)

    Raises
    ------
    ValueError
        If data dimensions are invalid or pos shape doesn't match data
    """
    if data.dim() == 2:
        nbatch = 1
        nsweeps, sweep_samples = data.shape
        if pos.shape != (nsweeps, 3):
            raise ValueError(f"Expected pos shape ({nsweeps}, 3), got {pos.shape}")
    elif data.dim() == 3:
        nbatch, nsweeps, sweep_samples = data.shape
        if pos.shape != (nbatch, nsweeps, 3):
            raise ValueError(f"Expected pos shape ({nbatch}, {nsweeps}, 3), got {pos.shape}")
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.dim()}D")

    return nbatch, nsweeps, sweep_samples


def get_batch_dims_img(img: Tensor, dorigin: Tensor) -> int:
    """Extract nbatch from image and dorigin tensors.

    Handles both batched (3D) and unbatched (2D) image inputs.

    Parameters
    ----------
    img : Tensor
        Image tensor, shape (nr, ntheta) or (nbatch, nr, ntheta)
    dorigin : Tensor
        Origin offset tensor, shape (3,) or (nbatch, 3)

    Returns
    -------
    int
        Number of batches

    Raises
    ------
    ValueError
        If dorigin shape doesn't match img
    """
    if img.dim() == 3:
        nbatch = img.shape[0]
        if dorigin.shape != (nbatch, 3):
            raise ValueError(f"Expected dorigin shape ({nbatch}, 3), got {dorigin.shape}")
    else:
        nbatch = 1
        if dorigin.shape != (3,):
            raise ValueError(f"Expected dorigin shape (3,), got {dorigin.shape}")

    return nbatch


class AntennaPattern:
    """Encapsulate antenna gain pattern and extent.

    Simplifies handling of optional antenna pattern parameters across
    multiple prepare functions.

    Parameters
    ----------
    g : Tensor, optional
        Antenna gain pattern, shape (g_nel, g_naz)
    g_extent : list, optional
        Antenna pattern extent [g_el0, g_az0, g_el1, g_az1] in radians.

    Raises
    ------
    ValueError
        If only one of g or g_extent is provided
    """

    def __init__(self, g: Optional[Tensor] = None, g_extent: Optional[list] = None):
        if (g is None) != (g_extent is None):
            raise ValueError("g and g_extent must both be None or both provided")

        self.g = g
        if g is not None:
            self.g_nel, self.g_naz = g.shape
            g_el0, g_az0, g_el1, g_az1 = g_extent
            self.g_el0, self.g_az0 = g_el0, g_az0
            self.g_el1, self.g_az1 = g_el1, g_az1
            self.g_daz = (g_az1 - g_az0) / self.g_naz
            self.g_del = (g_el1 - g_el0) / self.g_nel
        else:
            self.g_nel = self.g_naz = 0
            self.g_daz = self.g_del = 0.0
            self.g_el0 = self.g_az0 = self.g_el1 = self.g_az1 = 0.0

    def to_cpp_args(self) -> Tuple:
        """Convert to tuple for C++ call.

        Returns
        -------
        tuple
            (g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel)
        """
        return (self.g, self.g_az0, self.g_el0, self.g_daz, self.g_del,
                self.g_naz, self.g_nel)

    def to_cpp_args_without_tensor(self) -> Tuple:
        """Convert to tuple for C++ call (excluding tensor).

        Returns
        -------
        tuple
            (g_az0, g_el0, g_daz, g_del, g_naz, g_nel)
        """
        return (self.g_az0, self.g_el0, self.g_daz, self.g_del,
                self.g_naz, self.g_nel)
