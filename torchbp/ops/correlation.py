import torch
from torch import Tensor


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
