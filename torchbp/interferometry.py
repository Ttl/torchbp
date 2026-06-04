from .util import process_image_with_patches
from .grid import unpack_polar_grid, unpack_cartesian_grid
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TYPE_CHECKING
from .ops import subpixel_correlation_op
from scipy.optimize import minimize
import numpy as np

if TYPE_CHECKING:
    from .grid import PolarGrid, CartesianGrid

def _goldstein_patch(patches: Tensor, alpha: float, w: int=3) -> Tensor:
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


def goldstein_filter(igram: Tensor, patch_size: int=64, w: int=3, alpha: float=1, overlap: float=0.75) -> Tensor:
    """
    Goldstein phase filter. [1]_

    Parameters
    ----------
    igram: Tensor
        Complex interferogram.
    patch_size : int
        Patch side-length.
    w : int
        Smoothing window size. Must be odd.
    alpha : float
        Smoothing exponent.
    overlap : float
        Overlap between patches as fraction of patch_size.

    References
    ----------
    .. [1] R. M. Goldstein and C. L. Werner, "Radar interferogram filtering for
        geophysical applications," in Geophysical Research Letters, vol. 25, no. 21,
        pp 4035-4038, 1998

    Returns
    -------
    filtered : Tensor
        Filtered interferogram.
    """

    if w % 2 == 0:
        raise ValueError(f"w must be odd, got {w}")

    orig_dim = igram.dim()
    if orig_dim == 2:
        igram = igram.unsqueeze(0)

    # Reflect-pad by one patch on all sides so border pixels get the same
    # multi-patch coverage as the interior. Cropped off again at the end.
    pad = min(patch_size, igram.shape[-2] - 1, igram.shape[-1] - 1)
    if pad > 0:
        igram = F.pad(igram, (pad, pad, pad, pad), mode="reflect")

    f_patch = lambda x : _goldstein_patch(x, alpha, w)
    overlap = int(overlap * patch_size)
    filtered = process_image_with_patches(igram, patch_size, overlap, f_patch)
    if pad > 0:
        filtered = filtered[..., pad:filtered.shape[-2] - pad, pad:filtered.shape[-1] - pad]
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
    -------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    c0 = 299792458
    wl = c0 / fc

    v1 = coords - origin1[:,None,None]
    v2 = coords - origin2[:,None,None]
    r1 = torch.linalg.norm(v1, dim=0)
    r2 = torch.linalg.norm(v2, dim=0)

    # First-order height sensitivity for flat imaging plane:
    #   dphi/dh = (4pi/lambda) · (origin2_z/r2 − origin1_z/r1)
    sensitivity = origin2[2] / r2 - origin1[2] / r1

    z = -wl * unw / (4 * torch.pi * sensitivity)
    return z


def phase_to_elevation_polar(unw: Tensor, origin1: Tensor, origin2: Tensor, fc: float, grid: "PolarGrid | dict") -> Tensor:
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
    grid : PolarGrid or dict
        Image grid definition. PolarGrid object or dictionary.

    Returns
    -------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)

    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)
    coords = torch.stack([r[:,None] * torch.sqrt(1 - theta**2)[None,:], r[:,None] * theta[None,:], torch.zeros_like(unw)])

    return phase_to_elevation(unw, coords, origin1, origin2, fc)


def phase_to_elevation_cart(unw: Tensor, origin1: Tensor, origin2: Tensor, fc: float, grid: "CartesianGrid | dict") -> Tensor:
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
    grid : CartesianGrid or dict
        Image grid definition. CartesianGrid object or dictionary.

    Returns
    -------
    z : Tensor
        Elevation tensor with the same shape as unw.
    """
    device = unw.device

    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)

    x = x0 + dx * torch.arange(nx, device=device)
    y = y0 + dy * torch.arange(ny, device=device)
    coords = torch.stack(torch.meshgrid([x, y, torch.tensor([0], device=device,
        dtype=x.dtype)], indexing="ij"))
    coords = coords[..., 0]

    return phase_to_elevation(unw, coords, origin1, origin2, fc)


def flat_earth_phase_polar(origin1: Tensor, origin2: Tensor, fc: float, grid: "PolarGrid | dict") -> Tensor:
    """
    Compute flat earth interferometric phase for a polar grid.

    For images formed by backprojection on a flat (z=0) grid, the
    interferometric phase contains baseline geometry fringes even over
    flat terrain. This function computes that phase so it can be removed,
    isolating the topographic signal.

    Parameters
    ----------
    origin1 : Tensor
        3D antenna phase center of the master image [x, y, z].
    origin2 : Tensor
        3D antenna phase center of the slave image [x, y, z].
    fc : float
        RF center frequency in Hz.
    grid : PolarGrid or dict
        Polar grid definition.

    Returns
    -------
    phase : Tensor
        Flat earth phase tensor. Shape: [nr, ntheta].
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    device = origin1.device

    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)
    x = r[:, None] * torch.sqrt(1 - theta[None, :] ** 2)
    y = r[:, None] * theta[None, :]

    c0 = 299792458
    d1 = torch.sqrt((x - origin1[0]) ** 2 + (y - origin1[1]) ** 2 + origin1[2] ** 2)
    d2 = torch.sqrt((x - origin2[0]) ** 2 + (y - origin2[1]) ** 2 + origin2[2] ** 2)
    return 4 * torch.pi * fc / c0 * (d1 - d2)


def flat_earth_phase_cart(origin1: Tensor, origin2: Tensor, fc: float, grid: "CartesianGrid | dict") -> Tensor:
    """
    Compute flat earth interferometric phase for a Cartesian grid.

    Parameters
    ----------
    origin1 : Tensor
        3D antenna phase center of the master image [x, y, z].
    origin2 : Tensor
        3D antenna phase center of the slave image [x, y, z].
    fc : float
        RF center frequency in Hz.
    grid : CartesianGrid or dict
        Cartesian grid definition.

    Returns
    -------
    phase : Tensor
        Flat earth phase tensor. Shape: [nx, ny].
    """
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)
    device = origin1.device

    x = x0 + dx * torch.arange(nx, device=device)
    y = y0 + dy * torch.arange(ny, device=device)

    c0 = 299792458
    d1 = torch.sqrt((x[:, None] - origin1[0]) ** 2 + (y[None, :] - origin1[1]) ** 2 + origin1[2] ** 2)
    d2 = torch.sqrt((x[:, None] - origin2[0]) ** 2 + (y[None, :] - origin2[1]) ** 2 + origin2[2] ** 2)
    return 4 * torch.pi * fc / c0 * (d1 - d2)


def elevation_to_phase_slant_polar(
    z: "Tensor | float", origin1: Tensor, origin2: Tensor, fc: float,
    grid: "PolarGrid | dict"
) -> Tensor:
    """
    Compute traditional (slant-range) interferometric phase at given elevation.

    For scatterers at height z above the imaging plane::

        phi = (4pi fc/c0)*(r1 − r2)

    where r_n = slant range in nth image

    Reduces to :func:`flat_earth_phase_polar` when z = 0.

    Parameters
    ----------
    z : Tensor or float
        Elevation map [nr, ntheta] or scalar height.
    origin1 : Tensor
        Master APC [x, y, z].
    origin2 : Tensor
        Slave APC [x, y, z].
    fc : float
        RF center frequency (Hz).
    grid : PolarGrid or dict
        Polar grid definition.

    Returns
    -------
    phase : Tensor
        Interferometric phase [nr, ntheta].
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    device = origin1.device

    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)
    x = r[:, None] * torch.sqrt(1 - theta[None, :] ** 2)
    y = r[:, None] * theta[None, :]

    c0 = 299792458
    d1 = torch.sqrt((x - origin1[0]) ** 2 + (y - origin1[1]) ** 2 + (origin1[2] - z) ** 2)
    d2 = torch.sqrt((x - origin2[0]) ** 2 + (y - origin2[1]) ** 2 + (origin2[2] - z) ** 2)
    return 4 * torch.pi * fc / c0 * (d1 - d2)


def phase_to_elevation_slant_polar(
    unw: Tensor, origin1: Tensor, origin2: Tensor, fc: float,
    grid: "PolarGrid | dict", n_iter: int = 5
) -> Tensor:
    """
    Convert unwrapped topographic phase to elevation via Newton iteration.

    Inverts the traditional interferometric relationship::

        phi_topo(z) = elevation_to_phase_slant_polar(z, ...)
                   − elevation_to_phase_slant_polar(0, ...)

    Starts from the linearised BP-interferometry estimate and refines
    with Newton's method.

    Parameters
    ----------
    unw : Tensor
        Unwrapped topographic phase (flat-earth removed) [nr, ntheta].
    origin1 : Tensor
        Master APC [x, y, z].
    origin2 : Tensor
        Slave APC [x, y, z].
    fc : float
        RF center frequency (Hz).
    grid : PolarGrid or dict
        Polar grid definition.
    n_iter : int
        Number of Newton iterations (default 5).

    Returns
    -------
    z : Tensor
        Estimated elevation [nr, ntheta].
    """
    # Linearised initial estimate.
    # phase_to_elevation_polar uses z = -wl*unw/(4*pi*sens) which has
    # a negation for the BP conjugate-phase convention. Here the
    # phase is in the direct convention, so negate the initial guess.
    z = -phase_to_elevation_polar(unw, origin1, origin2, fc, grid)

    phi_flat = elevation_to_phase_slant_polar(0.0, origin1, origin2, fc, grid)

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    device = unw.device
    r = r0 + dr * torch.arange(nr, device=device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=device)
    x = r[:, None] * torch.sqrt(1 - theta[None, :] ** 2)
    y = r[:, None] * theta[None, :]

    c0 = 299792458
    k = 4 * torch.pi * fc / c0
    a1 = (x - origin1[0]) ** 2 + (y - origin1[1]) ** 2
    a2 = (x - origin2[0]) ** 2 + (y - origin2[1]) ** 2

    for _ in range(n_iter):
        d1 = torch.sqrt(a1 + (origin1[2] - z) ** 2)
        d2 = torch.sqrt(a2 + (origin2[2] - z) ** 2)
        phi_z = k * (d1 - d2)
        residual = unw - (phi_z - phi_flat)
        # dphi/dz = k * ((h2 - z)/r2 - (h1 - z)/r1)
        sensitivity = k * ((origin2[2] - z) / d2 - (origin1[2] - z) / d1)
        z = z + residual / sensitivity

    return z


def subpixel_correlation(im_m: Tensor, im_s: Tensor) -> tuple[Tensor, Tensor]:
    """
    Solve for subpixel offset that maximize coherent correlation between the two
    input images. [1]_

    Parameters
    ----------
    im_m : Tensor
        Master image.
    im_s : Tensor
        Slave image.

    References
    ----------
    .. [1] D. Li and Y. Zhang, "A Fast Offset Estimation Approach for InSAR
        Image Subpixel Registration," in IEEE Geoscience and Remote Sensing Letters,
        vol. 9, no. 2, pp. 267-271, March 2012.

    Returns
    -------
    offsets : Tensor
        Solved X and Y subpixel offsets.
    corrs : Tensor
        Correlation at the best subpixel offset.
    """
    a, b, c = subpixel_correlation_op(im_m, im_s)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    c = c.cpu().numpy()

    corrs = torch.zeros((c.shape[0]), dtype=torch.float32, device=im_m.device)
    offsets = torch.zeros((c.shape[0], 2), dtype=torch.float32, device=im_m.device)
    for i in range(a.shape[0]):
        ai = a[i]
        bi = b[i]
        ci = c[i]
        def opt(x):
            dx, dy = x
            return -np.abs(bi[0] + bi[1]*dx + bi[2]*dy + bi[3]*dx*dy) / np.sqrt(ci
                    * (ai[0] + ai[1]*dx + ai[2]*dy + ai[3]*dx**2 + ai[4]*dy**2
                        + ai[5]*dx*dy + ai[6]*dx**2*dy + ai[7]*dx*dy**2
                        + ai[8]*dx**2*dy**2))
        sol = minimize(opt, [0, 0], method="SLSQP", bounds=[(0, 1), (0, 1)])
        offsets[i, 0] = sol.x[0]
        offsets[i, 1] = sol.x[1]
        corrs[i] = -sol.fun
    return offsets, corrs
