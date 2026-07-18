import torch
from torch import Tensor
from typing import TYPE_CHECKING

__all__ = [
    "multilook_polar",
]

if TYPE_CHECKING:
    from ..grid import PolarGrid


def multilook_polar(sar_img: Tensor, kernel: tuple, grid_polar: "PolarGrid | dict") -> tuple[Tensor, dict]:
    """
    Apply multilook (spatial averaging) to polar SAR image.

    Parameters
    ----------
    sar_img : Tensor
        SAR image in polar coordinates.
    kernel : tuple
        Kernel size (nr, ntheta) for averaging.
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object (recommended): ``PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)``
        - dict (legacy): ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view)

    Returns
    -------
    sar_img_ml : Tensor
        Multilooked SAR image.
    grid_out : dict
        Updated grid definition reflecting reduced resolution.
    """
    # Convert Grid object to dict for manipulation
    if hasattr(grid_polar, 'to_dict'):
        grid_polar = grid_polar.to_dict()

    squeeze = False
    if sar_img.dim() == 2:
        squeeze = True
        sar_img = sar_img.unsqueeze(0)

    sar_img = torch.nn.functional.avg_pool2d(
        sar_img.real, kernel, stride=None
    ) + 1j * torch.nn.functional.avg_pool2d(sar_img.imag, kernel, stride=None)
    grid_out = {
        "r": grid_polar["r"],
        "theta": grid_polar["theta"],
        "nr": sar_img.shape[-2],
        "ntheta": sar_img.shape[-1],
    }
    if squeeze:
        sar_img = sar_img.squeeze(0)
    return sar_img, grid_out
