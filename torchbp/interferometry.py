from .util import process_image_with_patches
import torch
import torch.nn.functional as F
from torch import Tensor

def _goldstein_patch(patches: Tensor, alpha: float, w: int=3):
    fpatch = torch.fft.fft2(patches)
    fpatch = torch.fft.fftshift(fpatch, dim=(-2, -1))

    C, P, K, _ = fpatch.shape
    # Reshape to combine first two dims for pooled processing
    fpatches_reshaped = fpatch.view(C * P, 1, K, K)

    # Apply box filter (average pooling)
    filtered = F.avg_pool2d(torch.abs(fpatches_reshaped), kernel_size=w, stride=1, padding=w//2)

    filtered = filtered.view(C, P, K, K)

    filtered = fpatch * filtered**alpha
    del fpatch
    del fpatches_reshaped

    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))

    filtered = torch.fft.ifft2(filtered)

    return filtered


def goldstein_filter(igram: Tensor, patch_size: int=64, w: int=3, alpha: float=1, overlap: float=0.75):
    """
    Goldstein phase filter. [1]_

    Parameters
    ----------
    igram: Tensor
        Complex interferogram.
    patch_size : int
        Patch side-length.
    w : int
        Smoothing window size.
    alpha : float
        Smoothing exponent.
    overlap : float
        Overlap between patches.

    References
    ----------
    .. [1] R. M. Goldstein and C. L. Werner, "Radar interferogram filtering for
    geophysical applications," in Geophysical Research Letters, vol. 25, no. 21,
    pp 4035-4038, 1998

    Returns
    ----------
    filtered : Tensor
        Filtered interferogram.
    """

    orig_dim = igram.dim()
    if orig_dim == 2:
        igram = igram.unsqueeze(0)

    f_patch = lambda x : _goldstein_patch(x, alpha, w)
    overlap = int(overlap * patch_size)
    filtered = process_image_with_patches(igram, patch_size, overlap, f_patch)
    if orig_dim == 2:
        filtered = filtered.squeeze(0)
    return filtered

def phase_to_elevation_polar(unw: Tensor, h1: Tensor, h2: Tensor, fc: float, grid: dict) -> Tensor:
    """

    """
    device = unw.device

    c0 = 299792458
    wl = c0 / fc

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    ntheta = grid["ntheta"]
    nr = grid["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)

    look_angle = torch.arctan2(torch.tensor(h1, device=r.device), r)[:,None]

    bt = (h2 - h1) * torch.cos(look_angle)

    r1 = torch.sqrt(r**2 + h1**2)[:,None]
    z = -wl * r1 * torch.sin(look_angle) * unw / (4 * torch.pi * bt)
    return z
