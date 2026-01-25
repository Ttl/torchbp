import torch
from torch import Tensor


def lee_filter(img: Tensor, wx: int, wy: int, cu: float) -> Tensor:
    """
    Lee filter for speckle noise reduction.

    Parameters
    ----------
    img : Tensor
        Complex or real SAR image.
    wx : int
        Window size in the first dimension.
    wy : int
        Window size in the second dimension.
    cu : float
        Coefficient of variance of the noise-free image.

    Returns
    -------
    img : Tensor
        Filtered input image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        Nx = img.shape[1]
        Ny = img.shape[2]
    elif img.dim() == 2:
        nbatch = 1
        Nx = img.shape[0]
        Ny = img.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    # Half-window size in C++
    wx = wx // 2
    wy = wy // 2

    return torch.ops.torchbp.lee_filter.default(img, nbatch, Nx, Ny, wx, wy, cu)


