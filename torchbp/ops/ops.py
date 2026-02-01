import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..grid import PolarGrid


def cfar_2d(
    img: Tensor, Navg: tuple, Nguard: tuple, threshold: float, peaks_only: bool = False
) -> Tensor:
    """
    Constant False Alaram Rate detection for 2D image.

    Parameters
    ----------
    img : Tensor
        Absolute valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).
    Nguard : tuple
        Number of guard cells in 2D (N1, N0).
        Both dimensions need to be less or same than the same entry in Navg.
    threshold : float
        Threshold for detection.
    peaks_only : bool
        Reject pixels that are not higher than their immediate neighbors.
        Defaults is False.

    Returns
    -------
    detection : Tensor
        Detections tensor same shape as the input image. For each pixel 0 for no
        detection, positive value is the SNR of detection at that position.
    """
    if img.dim() == 3:
        nbatch = img.shape[0]
        N0 = img.shape[1]
        N1 = img.shape[2]
    elif img.dim() == 2:
        nbatch = 1
        N0 = img.shape[0]
        N1 = img.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if threshold <= 0:
        raise ValueError("Threshold should be positive")
    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if len(Nguard) != 2:
        raise ValueError("Nguard dimension should be 2")
    if Nguard[0] > Navg[0]:
        raise ValueError("Nguard[0] > Navg[0]")
    if Nguard[1] > Navg[1]:
        raise ValueError("Nguard[1] > Navg[1]")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")
    if Nguard[0] < 0:
        raise ValueError("Nguard[0] < 0")
    if Nguard[1] < 0:
        raise ValueError("Nguard[1] < 0")

    return torch.ops.torchbp.cfar_2d.default(
        img,
        nbatch,
        N0,
        N1,
        Navg[0],
        Navg[1],
        Nguard[0],
        Nguard[1],
        threshold,
        peaks_only,
    )


def div_2d_interp_linear(
    img1: Tensor,
    img2: Tensor
) -> Tensor:
    """
    Calculate img1 / img2 and interpolate img2 dimensions to match img1.

    Gradient is not supported.

    Parameters
    ----------
    img1 : Tensor
        2D or 3D image with complex64 or float type.
    img2 : Tensor
        2D or 3D image with complex64 or float type.
        If 3D, the first dimension must match with img1.

    Returns
    -------
    out : Tensor
        img1 / img2.
    """

    if img1.dim() == 3:
        nbatch = img1.shape[0]
        if img2.dim() == 3 and img2.shape[0] != nbatch or img2.dim() == 2 and nbatch != 1:
            raise ValueError(f"Invalid dimensions {img1.shape} and {img2.shape}")
    else:
        nbatch = 1
        if img2.dim() == 3 and img2.shape[0] != 1:
            raise ValueError(f"Invalid dimensions {img1.shape} and {img2.shape}")

    na0 = img1.shape[-2]
    na1 = img1.shape[-1]
    nb0 = img2.shape[-2]
    nb1 = img2.shape[-1]

    return torch.ops.torchbp.div_2d_interp_linear.default(
        img1,
        img2,
        nbatch,
        na0, na1,
        nb0, nb1
    )


def mul_2d_interp_linear(
    img1: Tensor,
    img2: Tensor
) -> Tensor:
    """
    Calculate img1 * img2 and interpolate img2 dimensions to match img1.

    Gradient is not supported.

    Parameters
    ----------
    img1 : Tensor
        2D or 3D image with complex64 or float type.
    img2 : Tensor
        2D or 3D image with complex64 or float type.
        If 3D, the first dimension must match with img1.

    Returns
    -------
    out : Tensor
        img1 * img2.
    """

    if img1.dim() == 3:
        nbatch = img1.shape[0]
        if img2.dim() == 3 and img2.shape[0] != nbatch or img2.dim() == 2 and nbatch != 1:
            raise ValueError(f"Invalid dimensions {img1.shape} and {img2.shape}")
    else:
        nbatch = 1
        if img2.dim() == 3 and img2.shape[0] != 1:
            raise ValueError(f"Invalid dimensions {img1.shape} and {img2.shape}")

    na0 = img1.shape[-2]
    na1 = img1.shape[-1]
    nb0 = img2.shape[-2]
    nb1 = img2.shape[-1]

    return torch.ops.torchbp.mul_2d_interp_linear.default(
        img1,
        img2,
        nbatch,
        na0, na1,
        nb0, nb1
    )


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

    sar_img = torch.nn.functional.avg_pool2d(
        sar_img.real, kernel, stride=None
    ) + 1j * torch.nn.functional.avg_pool2d(sar_img.imag, kernel, stride=None)
    grid_out = {
        "r": grid_polar["r"],
        "theta": grid_polar["theta"],
        "nr": sar_img.shape[-2],
        "ntheta": sar_img.shape[-1],
    }
    return sar_img, grid_out


def subpixel_correlation_op(im_m: Tensor, im_s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if im_m.dim() == 3:
        nbatch = im_m.shape[0]
        Nx = im_m.shape[1]
        Ny = im_m.shape[2]
    elif im_m.dim() == 2:
        nbatch = 1
        Nx = im_m.shape[0]
        Ny = im_m.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {im_m.shape}")

    if im_m.shape != im_s.shape:
        raise ValueError(f"Image shapes are different {im_m.shape} != {im_s.shape}")

    mean_m = torch.mean(im_m, dim=(-2,-1))
    mean_s = torch.mean(im_s, dim=(-2,-1))
    return torch.ops.torchbp.subpixel_correlation.default(
            im_m,
            im_s,
            mean_m,
            mean_s,
            nbatch,
            Nx,
            Ny)


