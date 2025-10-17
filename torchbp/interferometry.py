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

def phase_to_elevation(unw: Tensor, coords: Tensor, origin1: Tensor, origin2: Tensor, fc: float) -> Tensor:
    """
    Convert phase unwrapped interferogram to elevation.

    Parameters
    ----------
    unw : Tensor
        Unwrapped phase tensor. Shape: [Nx, Ny].
    coords : Tensor
        Coordinates for each position in image. Shape: [3, Nx, Ny].
    origin1 : Tensor
        3D antenna phase center location of the master image.
    origin2 : Tensor
        3D antenna phase center location of the slave image.
    fc : float
        RF center frequency in Hz.

    Returns
    ----------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    c0 = 299792458
    wl = c0 / fc

    w = (origin2 - origin1)[:,None,None]
    v = coords - origin1[:,None,None]
    w_dot_v = torch.sum(w*v, dim=0)
    r1 = torch.linalg.norm(v, dim=0)

    look_angle = torch.arccos(origin1[2] / r1)

    # All these equations are equal
    #bt = torch.linalg.norm(torch.cross(w, v, dim=0), dim=0) / r1
    #bt = torch.sqrt(torch.linalg.norm(w, dim=0)**2 - w_dot_v**2 / r1**2)
    bt = torch.linalg.norm(w - w_dot_v * v / r1**2, dim=0)

    z = -wl * r1 * torch.sin(look_angle) * unw / (4 * torch.pi * bt)
    return z

def phase_to_elevation_polar(unw: Tensor, origin1: Tensor, origin2: Tensor, fc: float, grid: dict) -> Tensor:
    """
    Convert phase unwrapped interferogram to elevation.

    Parameters
    ----------
    unw : Tensor
        Unwrapped phase tensor. Shape: [Nx, Ny].
    origin1 : Tensor
        3D antenna phase center location of the master image.
    origin2 : Tensor
        3D antenna phase center location of the slave image.
    fc : float
        RF center frequency in Hz.
    grid : dict
        Image grid definition dictionary.

    Returns
    ----------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    ntheta = grid["ntheta"]
    nr = grid["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)
    coords = torch.stack([r[:,None] * torch.sqrt(1 - theta**2)[None,:], r[:,None] * theta[None,:], torch.zeros_like(unw)])

    return phase_to_elevation(unw, coords, origin1, origin2, fc)

def phase_to_elevation_cart(unw: Tensor, origin1: Tensor, origin2: Tensor, fc: float, grid: dict) -> Tensor:
    """
    Convert phase unwrapped interferogram to elevation.

    Parameters
    ----------
    unw : Tensor
        Unwrapped phase tensor. Shape: [Nx, Ny].
    origin1 : Tensor
        3D antenna phase center location of the master image.
    origin2 : Tensor
        3D antenna phase center location of the slave image.
    fc : float
        RF center frequency in Hz.
    grid : dict
        Image grid definition dictionary.

    Returns
    ----------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    x0, x1 = grid["x"]
    y0, y1 = grid["y"]
    nx = grid["nx"]
    ny = grid["ny"]
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    x = x0 + dx * torch.arange(nx, device=device)
    y = y0 + dy * torch.arange(ny, device=device)
    coords = torch.stack(torch.meshgrid([x, y, torch.tensor([0], device=device,
        dtype=x.dtype)], indexing="ij"))
    coords = coords[..., 0]

    return phase_to_elevation(unw, coords, origin1, origin2, fc)
