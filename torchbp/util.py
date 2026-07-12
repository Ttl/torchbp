import torch
from torch import Tensor
import torch.nn.functional as F
from math import pi
import numpy as np
from scipy.signal import get_window
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import PolarGrid, CartesianGrid


def bp_polar_range_dealias(
    img: Tensor, origin: Tensor, fc: float, grid_polar: "PolarGrid | dict",
    alias_fmod: float = 0, dem: Tensor | None = None
) -> Tensor:
    """
    De-alias range-axis spectrum of polar SAR image processed with backprojection. [1]_

    Equivalent to the ``dealias`` option of
    :func:`torchbp.ops.backprojection_polar_2d` when ``origin`` is the center
    of the platform positions used for backprojection (``[0, 0, z0]`` when
    the positions were centered) and the same ``dem`` is given.

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth] or
        [nbatch, Range, azimuth].
    origin : Tensor
        Center of the platform position.
    fc : float
        RF center frequency.
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
        - dict: {"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}
    alias_fmod : float
        Range modulation frequency applied to input.
    dem : Tensor or None
        Digital elevation model sampled on the image polar grid, same
        convention as the `dem` input of
        :func:`torchbp.ops.backprojection_polar_2d`: float32
        [dem_nr, dem_ntheta] covering the grid extent, can be coarser than
        the image grid (bilinearly interpolated). When given, the carrier is
        referenced to the pixel at the DEM height, matching the dealias
        carrier of a backprojection with the same `dem`. If None the carrier
        is referenced to the z=0 plane.

    References
    ----------
    .. [1] T. Shi, X. Mao, A. Jakobsson and Y. Liu, "Extended PGA for Spotlight
        SAR-Filtered Backprojection Imagery," in IEEE Geoscience and Remote Sensing
        Letters, vol. 19, pp. 1-5, 2022, Art no. 4516005.

    Returns
    -------
    img : Tensor
        SAR image without range spectrum aliasing.
    """
    from .grid import unpack_polar_grid

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)

    if origin.dim() == 2:
        origin = origin[0]

    if img.dtype == torch.complex64:
        # Custom op: one pass over the image with a per-pixel bilinear DEM
        # lookup, no full-size temporaries beyond the output.
        nbatch = img.shape[0] if img.dim() == 3 else 1
        if dem is not None:
            if dem.dtype != torch.float32:
                raise ValueError(f"dem must be float32, got {dem.dtype}")
            if dem.device != img.device:
                raise ValueError(
                    f"dem must be on the same device as img ({img.device}), "
                    f"got {dem.device}"
                )
        out = torch.ops.torchbp.polar_range_dealias(
            img, dem, nbatch, nr, ntheta, fc, r0, dr, theta0, dtheta,
            float(origin[0]), float(origin[1]), float(origin[2]), alias_fmod
        )
        return out.reshape(img.shape)

    if dem is not None:
        raise NotImplementedError(
            "dem is only supported for complex64 images"
        )

    er = torch.arange(nr, device=img.device)
    r = r0 + dr * er
    theta = theta0 + dtheta * torch.arange(ntheta, device=img.device)

    x = r[:, None] * torch.sqrt(1 - torch.square(theta))[None, :]
    y = r[:, None] * theta[None, :]

    d = torch.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2 + origin[2] ** 2)
    c0 = 299792458
    phase = torch.exp(-1j * 4 * pi * fc * d / c0 + 1j*alias_fmod*er[:,None])
    if img.dim() == 3:
        phase = phase.unsqueeze(0)
    return phase * img


def bp_polar_range_alias(
    img: Tensor, origin: Tensor, fc: float, grid_polar: "PolarGrid | dict",
    alias_fmod: float = 0, dem: Tensor | None = None
) -> Tensor:
    """
    Inverse of bp_polar_range_dealias.

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth] or
        [nbatch, Range, azimuth].
    origin : Tensor
        Center of the platform position.
    fc : float
        RF center frequency.
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
        - dict: {"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}
    alias_fmod : float
        Range modulation frequency applied to output.
    dem : Tensor or None
        Digital elevation model sampled on the image polar grid. See
        :func:`bp_polar_range_dealias`. Must be the same `dem` the image was
        dealiased with.

    Returns
    -------
    img : Tensor
        SAR image with range spectrum aliasing.
    """
    return bp_polar_range_dealias(img, origin, -fc, grid_polar, -alias_fmod, dem=dem)


def diff(x: Tensor, dim: int = -1, same_size: bool = False) -> Tensor:
    """
    ``np.diff`` implemented in torch.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension.
    same_size : bool
        Pad output to same size as input.

    Returns
    -------
    d : Tensor
        Difference tensor.
    """
    if dim == 0:
        if same_size:
            padding = [0 for i in range(2*len(x.shape))]
            padding[-2] = 1
            padding = tuple(padding)
            return torch.nn.functional.pad(x[1:] - x[:-1], padding)
        else:
            return x[1:] - x[:-1]
    if dim != -1:
        raise NotImplementedError("Only dim=0 and dim=-1 is implemented")
    if same_size:
        return torch.nn.functional.pad(x[..., 1:] - x[..., :-1], (1, 0))
    else:
        return x[..., 1:] - x[..., :-1]


def unwrap(phi: Tensor, dim: int = -1) -> Tensor:
    """
    ``np.unwrap`` implemented in torch.

    Parameters
    ----------
    phi : Tensor
        Input tensor.
    dim : int
        Dimension.

    Returns
    -------
    phi : Tensor
        Unwrapped tensor.
    """
    if dim != -1:
        raise NotImplementedError("Only dim=-1 is implemented")
    phi_wrap = ((phi + torch.pi) % (2 * torch.pi)) - torch.pi
    dphi = diff(phi_wrap, same_size=True)
    dphi_m = ((dphi + torch.pi) % (2 * torch.pi)) - torch.pi
    dphi_m[(dphi_m == -torch.pi) & (dphi > 0)] = torch.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < torch.pi] = 0
    return phi_wrap + phi_adj.cumsum(dim)


def unwrap_ref(x: Tensor, y: Tensor) -> Tensor:
    """
    Solve for integer array k such that x + k*2pi is closest to y.
    `k = round((y - x) / (2pi))`.

    Parameters
    ----------
    x : Tensor
        Phase wrapped signal.
    y : Tensor
        Reference signal.

    Returns
    -------
    unwrapped_x : Tensor
        Phase unwrapped x
    """
    k = torch.round((y - x) / (2 * torch.pi))
    unwrapped_x = x + k * 2 * torch.pi

    return unwrapped_x


def quad_interp(a: Tensor, v: int) -> Tensor:
    """
    Quadractic peak interpolation.
    Useful for FFT peak interpolation.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    v : int
        Peak index.

    Returns
    -------
    f : Tensor
        Estimated fractional peak index.
    """
    a1 = a[(v - 1) % len(a)]
    a2 = a[v % len(a)]
    a3 = a[(v + 1) % len(a)]
    return 0.5 * (a1 - a3) / (a1 - 2 * a2 + a3)


def argmax_nd(x: Tensor) -> tuple[int, ...]:
    """
    `torch.argmax` but returns N-dimensional index of the peak
    """
    d = torch.argmax(x).item()
    res = []
    for s in x.shape[::-1]:
        d, m = divmod(d, s)
        res.append(m)
    return tuple(res)[::-1]


def find_image_shift_1d(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """
    Find shift between images that maximizes correlation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    y : Tensor
        Input tensor. Should have same shape as x.
    dim : int
        Dimensions to shift.

    Returns
    -------
    c : Tensor
        Estimated shift.
    """
    if x.shape != y.shape:
        raise ValueError("Input shapes should be identical")
    if dim < 0:
        dim = x.dim() + dim
    fx = torch.fft.fft(x, dim=dim)
    fy = torch.fft.fft(y, dim=dim)
    c = (fx * fy.conj()) / (torch.abs(fx) * torch.abs(fy))
    other_dims = [i for i in range(x.dim()) if i != dim]
    c = torch.abs(torch.fft.ifft(c, dim=dim))
    if len(other_dims) > 0:
        c = torch.mean(c, dim=other_dims)
    return torch.argmax(c)


def subset_cart(
    img: Tensor, grid_cart: "CartesianGrid | dict", x0: float, x1: float, y0: float, y1: float
) -> tuple[Tensor, dict]:
    """Cartesian image subset.

    Parameters
    ----------
    img : Tensor
        Input image.
    grid_cart : CartesianGrid or dict
        Cartesian grid definition. Can be:

        - CartesianGrid object: CartesianGrid(x_range=(x0, x1), y_range=(y0, y1), nx=nx, ny=ny)
        - dict: {"x": (x0, x1), "y": (y0, y1), "nx": nx, "ny": ny}
    x0 : float
        Subset x0.
    x1 : float
        Subset x1.
    y0 : float
        Subset y0.
    y1 : float
        Subset y1.

    Returns
    -------
    img : Tensor
        Subset of input image.
    grid : dict
        Grid
    """
    gx0, gx1 = grid_cart["x"]
    gy0, gy1 = grid_cart["y"]
    nx = grid_cart["nx"]
    ny = grid_cart["ny"]
    dx = (gx1 - gx0) / nx
    dy = (gy1 - gy0) / ny

    nx0 = max(0, min(nx, int((x0 - gx0) / dx)))
    nx1 = max(0, min(nx, int((x1 - gx0) / dx)))
    ny0 = max(0, min(ny, int((y0 - gy0) / dy)))
    ny1 = max(0, min(ny, int((y1 - gy0) / dy)))

    out = img[..., nx0:nx1, ny0:ny1]

    grid_new = {
        "x": (gx0 + dx * nx0, gx0 + dx * nx1),
        "y": (gy0 + ny0 * dy, gy0 + ny1 * dy),
        "nr": out.shape[-2],
        "ntheta": out.shape[-1],
    }

    return out, grid_new


def subset_polar(
    img: Tensor, grid_polar: "PolarGrid | dict", r0: float, r1: float, theta0: float, theta1: float
) -> tuple[Tensor, dict]:
    """Polar image subset.

    Parameters
    ----------
    img : Tensor
        Input image.
    grid_polar : PolarGrid or dict
        Polar grid definition. PolarGrid object or dictionary.
    r0 : float
        Subset r0.
    r1 : float
        Subset r1.
    theta0 : float
        Subset theta0.
    theta1 : float
        Subset theta1.

    Returns
    -------
    img : Tensor
        Subset of input image.
    grid_new : dict
        Grid.
    """
    gr0, gr1 = grid_polar["r"]
    gtheta0, gtheta1 = grid_polar["theta"]
    nr = grid_polar["nr"]
    ntheta = grid_polar["ntheta"]
    dr = (gr1 - gr0) / nr
    dtheta = (gtheta1 - gtheta0) / ntheta

    nr0 = max(0, min(nr, int((r0 - gr0) / dr)))
    nr1 = max(0, min(nr, int((r1 - gr0) / dr)))
    ntheta0 = max(0, min(ntheta, int((theta0 - gtheta0) / dtheta)))
    ntheta1 = max(0, min(ntheta, int((theta1 - gtheta0) / dtheta)))

    out = img[..., nr0:nr1, ntheta0:ntheta1]
    grid_new = {
        "r": (gr0 + dr * nr0, gr0 + dr * nr1),
        "theta": (gtheta0 + ntheta0 * dtheta, gtheta0 + ntheta1 * dtheta),
        "nr": out.shape[-2],
        "ntheta": out.shape[-1],
    }

    return out, grid_new


def find_image_shift_2d(
    x: Tensor, y: Tensor, dim: tuple = (-2, -1), interpolate=False
) -> tuple:
    """
    Find shift between images that maximizes correlation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    y : int
        Input tensor. Should have same shape as x.
    dim : tuple
        Dimension.

    Returns
    -------
    c : tuple
        Estimated shift.
    a : float
        Peak of correlation.
    """
    if x.shape != y.shape:
        raise ValueError("Input shapes should be identical")
    if dim != (-2, -1):
        raise NotImplentedError("dim must be (-2, -1)")
    d2 = []
    for i in dim:
        if i < 0:
            d2.append(x.dim() + i)
        else:
            d2.append(i)
    dims = d2
    fx = torch.fft.fft2(x, dim=dim)
    fy = torch.fft.fft2(y, dim=dim)
    c = (fx * fy.conj()) / (torch.abs(fx) * torch.abs(fy))
    other_dims = [i for i in range(x.dim()) if i not in dim]
    c = torch.abs(torch.fft.ifft2(c, dim=dim))
    idx = argmax_nd(torch.abs(c))
    a = c[idx].item()
    if interpolate:
        # Apply quad_interp to each spatial dimension
        interp_idx = list(idx)

        dim_idx = dims[0]
        # Extract 1D slice along this dimension at the peak location
        slice_indices = list(idx)
        slice_indices[dim_idx] = slice(None)  # Replace with slice for this dimension
        c_slice = c[tuple(slice_indices)]

        delta_0 = quad_interp(c_slice, idx[dim_idx])
        interp_idx[dim_idx] = idx[dim_idx] + delta_0

        # Interpolate along the second spatial dimension (dim[-1])
        dim_idx = dims[1]
        slice_indices = list(idx)
        slice_indices[dim_idx] = slice(None)
        c_slice = c[tuple(slice_indices)]

        delta_1 = quad_interp(c_slice, idx[dim_idx])
        interp_idx[dim_idx] = idx[dim_idx] + delta_1

        idx = tuple([i.item() for i in interp_idx])
    idx = [
        idx[i] - c.shape[i] if idx[i] > c.shape[i] // 2 else idx[i]
        for i in range(len(idx))
    ]
    return idx, a


def fft_peak_1d(x: Tensor, dim: int = -1, fractional: bool = True) -> Tensor:
    """
    Find fractional peak of ``abs(fft(x))``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension to calculate peak.
    fractional : bool
        Estimate peak location with fractional index accuracy.

    Returns
    -------
    a : int or float
        Estimated peak index.
    """
    fx = torch.abs(torch.fft.fft(x, dim=dim))
    a = torch.argmax(fx)
    if fractional:
        a = a + quad_interp(fx, a)
    l = x.shape[dim]
    if a > l // 2:
        a = l - a
    return a


def detrend(x: Tensor) -> Tensor:
    """
    Removes linear trend

    Parameters
    ----------
    x : Tensor
        Input tensor. Should be 1 dimensional.

    Returns
    -------
    x : Tensor
        x with linear trend removed.
    """
    n = x.shape[0]

    k = torch.arange(n, device=x.device, dtype=x.dtype) / n

    # Solve least squares problem: k * a + b = x
    ones = torch.ones(n, device=x.device, dtype=x.dtype)

    A = torch.stack([k, ones], dim=1)
    params = torch.linalg.lstsq(A, x).solution
    a, b = params[0], params[1]

    # Remove linear trend
    return x - (a * k + b)


def weighted_detrend(x: Tensor, w: Tensor | None) -> Tensor:
    """
    Removes weighted linear trend.

    The removed line is the weighted least squares fit, so samples with
    zero weight do not affect the fit but the line is still subtracted
    over the whole tensor. Degenerate weights (all zero or concentrated
    on one sample) remove only the weighted mean.

    Parameters
    ----------
    x : Tensor
        Input tensor. Should be 1 dimensional.
    w : Tensor or None
        Non-negative per-sample weights with the same shape as ``x``.
        None falls back to the unweighted :func:`detrend`.

    Returns
    -------
    x : Tensor
        x with weighted linear trend removed.
    """
    if w is None:
        return detrend(x)
    n = x.shape[0]
    k = torch.arange(n, device=x.device, dtype=x.dtype) / n
    wsum = torch.sum(w)
    if wsum <= 0:
        return x
    km = torch.sum(w * k) / wsum
    xm = torch.sum(w * x) / wsum
    kk = torch.sum(w * (k - km) ** 2)
    if kk <= 0:
        return x - xm
    a = torch.sum(w * (k - km) * (x - xm)) / kk
    return x - (xm + a * (k - km))


def entropy(x: Tensor) -> Tensor:
    """
    Calculates entropy:

    ``-sum(y*log(y))``

    where ``y = abs(x) / sum(abs(x))``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    entropy : Tensor
        Calculated entropy of the input.
    """
    ax = torch.abs(x)
    ax /= torch.sum(ax)
    return -torch.sum(torch.xlogy(ax, ax))


def contrast(x: Tensor, dim: int = -1) -> Tensor:
    """
    Calculates negative contrast:

    ``-mean(std/mu)``

    where ``mu`` is mean and ``std`` is standard deviation of ``abs(x)`` along
    dimension ``dim``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    contrast: Tensor
        Calculated negative contrast of the input.
    """
    std, mu = torch.std_mean(torch.abs(x), dim=dim)
    contrast = torch.mean(std / mu)
    return -contrast


def shift_spectrum(x: Tensor, dim: int = -1) -> Tensor:
    """
    Equivalent to: ``fft(ifftshift(ifft(x, dim), dim), dim)``,
    but avoids calculating FFTs.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    y : Tensor
        Shifted tensor.
    """
    if dim != -1:
        raise NotImplementedError("dim should be -1")
    shape = [1] * len(x.shape)
    shape[dim] = x.shape[dim]
    c = torch.ones(shape, dtype=torch.float32, device=x.device)
    c[..., 1::2] = -1
    return x * c


def generate_fmcw_data(
    target_pos: Tensor,
    target_rcs: Tensor,
    pos: Tensor,
    fc: float,
    bw: float,
    tsweep: float,
    fs: float,
    d0: float = 0,
    g: Tensor | None = None,
    g_extent: list | None = None,
    att: Tensor = None,
    rvp: bool = True,
    vel: Tensor | None = None,
) -> Tensor:
    """
    Generate FMCW radar time-domain IF signal.

    Parameters
    ----------
    target_pos : Tensor
        [ntargets, 3] tensor of target XYZ positions.
    target_rcs : Tensor
        [ntargets, 1] tensor of target reflectivity.
    pos : Tensor
        [nsweeps, 3] tensor of platform positions. When `vel` is provided,
        `pos[s]` is the platform position at the midpoint of sweep `s`.
    fc : float
        RF center frequency in Hz.
    bw : float
        RF bandwidth in Hz.
    tsweep : float
        Length of one sweep in seconds.
    fs : float
        Sampling frequency in Hz.
    d0 : float
        Zero range.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center. Isotropic antenna is assumed if g is None.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3].
        [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    rvp : bool
        True to include residual video phase term.
    vel : Tensor or None
        [nsweeps, 3] tensor of platform velocities in m/s. When given, the
        two-way delay is evaluated per intra-sweep sample at the instantaneous
        platform position `pos[s] + vel[s] * (t_sample - tsweep/2)`, i.e. the
        stop-and-go approximation is removed. None (default) reproduces the
        stop-and-go model where `pos[s]` is held fixed across the chirp.

    Returns
    -------
    data : Tensor
        [nsweeps, nsamples] measurement data.
    """
    if pos.dim() != 2:
        raise ValueError("pos tensor should have 2 dimensions")
    if pos.shape[1] != 3:
        raise ValueError("positions should be 3 dimensional")
    if vel is not None:
        if vel.shape != pos.shape:
            raise ValueError("vel must have the same shape as pos")
    npos = pos.shape[0]
    nsamples = int(fs * tsweep)

    device = pos.device
    data = torch.zeros((npos, nsamples), dtype=torch.complex64, device=device)
    t = torch.arange(nsamples, dtype=torch.float32, device=device) / fs
    k = bw / tsweep

    c0 = 299792458

    use_rvp = 1 if rvp else 0

    antenna_gain = g is not None and att is not None
    if antenna_gain:
        if g_extent is None:
            raise ValueError("g_extent is None, but g is not None")
        if len(g_extent) != 4:
            raise ValueError("g_extent should be a 4 element list")
        g_el0, g_az0, g_el1, g_az1 = g_extent
        nelevation, nazimuth = g.shape

        # Add batch and channel dimensions to g
        g_batch = g.unsqueeze(0).unsqueeze(0)

    t = t[None, :]
    # Sample-time offset from sweep midpoint, used only for non-stop-and-go.
    t_rel = (t - 0.5 * tsweep) if vel is not None else None

    for e, target in enumerate(target_pos):
        rcs_phase = torch.angle(target_rcs[e])
        rcs_abs = torch.sqrt(torch.abs(target_rcs[e]))

        dpos = pos - target[None, :]                               # [nsweeps, 3]
        if vel is None:
            d = torch.linalg.vector_norm(dpos, dim=-1)[:, None] + d0  # [nsweeps, 1]
        else:
            # |dpos + vel * t_rel|^2 = |dpos|^2 + 2<dpos,vel> t_rel + |vel|^2 t_rel^2
            dp_sq = (dpos * dpos).sum(dim=-1, keepdim=True)        # [nsweeps, 1]
            dp_v = (dpos * vel).sum(dim=-1, keepdim=True)          # [nsweeps, 1]
            v_sq = (vel * vel).sum(dim=-1, keepdim=True)           # [nsweeps, 1]
            d = torch.sqrt(dp_sq + 2.0 * dp_v * t_rel + v_sq * t_rel**2) + d0  # [nsweeps, nsamples]
        tau = 2 * d / c0
        if antenna_gain:
            # Antenna gain evaluated at sweep midpoint (slow-varying across chirp).
            d_mid = torch.linalg.vector_norm(dpos, dim=-1)[:, None] + d0
            look_angle = torch.asin(pos[:, 2] / d_mid[:, 0])
            el_deg = -look_angle - att[:, 0]
            az_deg = (
                torch.atan2(target[1] - pos[:, 1], target[0] - pos[:, 0]) - att[:, 2]
            )

            az_norm = 2.0 * (az_deg - g_az0) / (g_az1 - g_az0) - 1.0
            el_norm = 2.0 * (el_deg - g_el0) / (g_el1 - g_el0) - 1.0

            # grid_sample maps grid[..., 0] -> width (azimuth) and
            # grid[..., 1] -> height (elevation) of g, which is [nel, naz].
            grid = torch.stack([az_norm, el_norm], dim=-1)
            grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

            g_a = F.grid_sample(
                g_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            g_a = g_a.reshape(d_mid.shape)
        else:
            g_a = 1

        data += (g_a * rcs_abs / d**2) * torch.exp(
            1j * 2 * pi * (-fc * tau - k * tau * t + use_rvp * 0.5 * k * tau**2)
            + 1j * rcs_phase
        )
    return data


def make_polar_grid(
    r0: float, r1: float, nr: int, ntheta: int, theta_limit: int = 1, squint: float = 0
) -> "PolarGrid":
    """
    Generate PolarGrid object.

    Parameters
    ----------
    r0 : float
        Minimum range in m.
    r1 : float
        Maximum range in m.
    nr : float
        Number of range points.
    ntheta : float
        Number of azimuth points.
    theta_limit : float
        Theta axis limits, symmetrical around zero.
        Units are sin of angle (0 to 1 valid range).
        Default is 1.
    squint : float
        Grid azimuth mean angle, radians.

    Returns
    -------
    grid_polar : PolarGrid
        Polar grid object.
    """
    from .grid import PolarGrid

    t0 = float(np.clip(np.sin(squint) - theta_limit, -1, 1))
    t1 = float(np.clip(np.sin(squint) + theta_limit, -1, 1))
    return PolarGrid(r_range=(r0, r1), theta_range=(t0, t1), nr=nr, ntheta=ntheta)


# Alias for backward compatibility
make_polar_grid_obj = make_polar_grid


def dem_to_polar(
    dem: Tensor,
    dem_grid: "CartesianGrid | dict",
    polar_grid: "PolarGrid | dict",
    origin: Tensor | None = None,
    rotation: float = 0.0,
) -> Tensor:
    """
    Resample a Cartesian DEM onto a polar grid for use as the `dem` input of
    `torchbp.ops.backprojection_polar_2d`.

    Parameters
    ----------
    dem : Tensor
        Cartesian DEM heights, shape [nx, ny] on `dem_grid`. Same x/y
        coordinate frame and z datum as the platform positions before
        subtracting `origin`.
    dem_grid : CartesianGrid or dict
        Cartesian grid of `dem`.
    polar_grid : PolarGrid or dict
        Polar grid to sample to. Should have the same r and theta extent as
        the imaging grid; nr and ntheta can be smaller for a downsampled DEM.
    origin : Tensor or None
        Polar grid origin [x, y, z] in the DEM frame (the origin subtracted
        from the platform positions). Default is zeros.
    rotation : float
        Polar grid rotation in radians, same convention as
        `torchbp.ops.polar_to_cart`.

    Returns
    -------
    dem_polar : Tensor
        DEM heights at the polar grid points relative to the origin z,
        shape [nr, ntheta] float32. Points outside the DEM extent take the
        nearest border value. Guard band points |theta| > 1 sample at the
        clamped angle.
    """
    from .grid import unpack_cartesian_grid, unpack_polar_grid

    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(dem_grid)
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(polar_grid)

    device = dem.device
    if origin is None:
        origin = torch.zeros(3, device=device)

    r = r0 + dr * torch.arange(nr, device=device, dtype=torch.float32)
    sin_t = theta0 + dtheta * torch.arange(ntheta, device=device, dtype=torch.float32)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t * sin_t, min=0.0))
    # Angle in the DEM frame is the grid angle plus the grid rotation.
    cos_rot = float(np.cos(rotation))
    sin_rot = float(np.sin(rotation))
    cos_a = cos_t * cos_rot - sin_t * sin_rot
    sin_a = sin_t * cos_rot + cos_t * sin_rot
    x = origin[0] + r[:, None] * cos_a[None, :]
    y = origin[1] + r[:, None] * sin_a[None, :]

    # grid_sample with align_corners=True: -1 maps to index 0 and +1 to the
    # last sample. Grid samples are at x0 + i*dx (cell start convention).
    gx = 2.0 * (x - x0) / (dx * (nx - 1)) - 1.0
    gy = 2.0 * (y - y0) / (dy * (ny - 1)) - 1.0
    # Input is [1, 1, nx, ny]: y is the innermost (W) axis.
    coords = torch.stack([gy, gx], dim=-1)[None]
    dem_polar = F.grid_sample(
        dem[None, None].to(torch.float32), coords, mode="bilinear",
        padding_mode="border", align_corners=True)[0, 0]
    return dem_polar - origin[2]


def phase_to_distance(p: Tensor, fc: float) -> Tensor:
    """
    Convert radar reflection phase shift to distance.

    Parameters
    ----------
    p : Tensor
        Phase shift tensor.
    fc : float
        RF center frequency.
    """
    c0 = 299792458
    return c0 * p / (4 * torch.pi * fc)

def wiener_normalize(
    sar: Tensor,
    tx_power: Tensor,
    eps: float | None = None,
    calib_quantile: float = 0.1,
) -> Tensor:
    """
    SNR-aware radiometric normalization of a SAR image by an illumination map.

    Plain division ``sar / tx_power`` inverts the illumination, but where the
    illumination is weak (swath edges, antenna nulls) it divides receiver noise
    by a near-zero number and the result blows up. This applies the Wiener (MMSE)
    estimate instead:

    .. math::

        \\hat{s} = \\frac{\\mathrm{sar}\\cdot\\mathrm{tx\\_power}}
                        {\\mathrm{tx\\_power}^2 + \\varepsilon^2}

    which equals ``(sar / tx_power) * SNR / (1 + SNR)`` with the per-pixel power
    SNR ``= (tx_power / eps)**2``. Where the illumination is strong it reduces to
    the full normalization ``sar / tx_power``; where it is weak the gain rolls off
    as ``tx_power**2`` so the output goes to zero instead of amplifying noise. The
    SNR map itself is ``(tx_power / eps)**2``.

    The regularization ``eps`` is the noise-to-signal amplitude ratio
    :math:`\\varepsilon = \\sigma_n / \\sigma_s` **in tx_power units**, where
    :math:`\\sigma_n` is the additive noise amplitude in the image and
    :math:`\\sigma_s` is the reflectivity scale relating image to illumination
    (``sar = s * tx_power + n``). It is the illumination level at which the SNR
    equals one. Note this is *not* simply the noise level :math:`\\sigma_n`: it
    must be divided by the reflectivity scale (which also absorbs the leftover
    radiometric calibration constant of ``tx_power``).

    Parameters
    ----------
    sar : Tensor
        Complex or magnitude SAR image. 2D ``(nr, ntheta)`` or 3D
        ``(nbatch, nr, ntheta)``.
    tx_power : Tensor
        Illumination map from
        :func:`torchbp.ops.backprojection_polar_2d_tx_power` (square root of power
        returned for unit reflectivity), real-valued with the same number of
        dimensions as ``sar``. Its trailing two dimensions may be coarser than
        ``sar``; if so it is bilinearly interpolated up to ``sar``'s grid on the
        fly (via :func:`torchbp.ops.mul_2d_interp_linear` /
        :func:`torchbp.ops.div_2d_interp_linear`) so no full-resolution map is
        allocated. Non-finite entries (un-illuminated pixels) are treated as
        no-data and mapped to zero.
    eps : float or None
        Regularization level :math:`\\sigma_n / \\sigma_s` in ``tx_power`` units.
        If None it is estimated from the data using the identity
        :math:`E|\\mathrm{sar}|^2 = \\sigma_s^2\\,\\mathrm{tx\\_power}^2 +
        \\sigma_n^2`: :math:`\\sigma_s^2` from the brightest ``calib_quantile``
        fraction of pixels and :math:`\\sigma_n^2` from the dimmest fraction
        (with the residual signal subtracted). For real data prefer passing an
        explicit value from a known shadow region and a calibration target.
    calib_quantile : float
        Fraction (0-0.5) of pixels at each illumination extreme used to estimate
        ``eps`` when it is not given. Default 0.1 (dimmest/brightest 10%).

    Returns
    -------
    s_hat : Tensor
        Normalized image, same dtype as ``sar``.
    """
    # When tx_power is coarser than sar image it is interpolated on the fly with
    # the fused interp ops so no full-resolution illumination map is ever
    # materialized.
    interp = tuple(sar.shape[-2:]) != tuple(tx_power.shape[-2:])

    finite = torch.isfinite(tx_power)
    txp = torch.where(finite, tx_power, torch.zeros_like(tx_power))

    if eps is None:
        if interp:
            # Area-average the image power onto the (coarse) tx_power grid so each
            # illumination sample is paired with the mean |sar|^2 it illuminates
            p = sar.abs().to(torch.float32)
            p = p * p
            squeeze = p.dim() == 2
            if squeeze:
                p = p[None]
            p = F.adaptive_avg_pool2d(p, tuple(txp.shape[-2:]))
            if squeeze:
                p = p[0]
            x = txp[finite].flatten().float()
            y = p[finite].flatten()
        else:
            x = txp[finite].flatten().float()
            y = (sar.abs()[finite].flatten().float()) ** 2  # |sar|^2
        # torch.quantile caps at ~16M elements; subsample large images.
        if x.numel() > 1_000_000:
            sel = torch.randperm(x.numel(), device=x.device)[:1_000_000]
            xq, yq = x[sel], y[sel]
        else:
            xq, yq = x, y
        lo = torch.quantile(xq, calib_quantile)
        hi = torch.quantile(xq, 1.0 - calib_quantile)
        him = xq >= hi
        lom = xq <= lo
        # sigma_s^2 from bright (signal-dominated), sigma_n^2 from dim with the
        # residual signal sigma_s^2 * tx_power^2 removed.
        sigma_s2 = yq[him].mean() / (xq[him] ** 2).mean().clamp_min(1e-30)
        sigma_n2 = (yq[lom].mean() - sigma_s2 * (xq[lom] ** 2).mean()).clamp_min(0.0)
        eps = float((sigma_n2 / sigma_s2.clamp_min(1e-30)).sqrt())

    eps2 = float(eps) ** 2

    if interp:
        # Wiener combine with tx_power bilinearly interpolated up to sar's grid,
        # built from the fused interp ops so no full-resolution illumination map
        # is allocated:
        #     s_hat = sar * interp(txp) / interp(txp**2 + eps**2)
        # The denominator uses interp(txp**2) instead of interp(txp)**2, which
        # can have slight error, but interpolation itself is already approximate.
        from .ops import mul_2d_interp_linear, div_2d_interp_linear
        num = mul_2d_interp_linear(sar, txp)
        s_hat = div_2d_interp_linear(num, txp * txp + eps2)
        return s_hat.reshape(sar.shape)

    s_hat = sar * txp / (txp * txp + eps2)
    # Un-illuminated (non-finite tx_power) pixels carry no signal -> zero.
    return torch.where(finite, s_hat, torch.zeros_like(s_hat))


def next_fast_len(n: int) -> int:
    """CuFFT-friendly length (powers of 2,3,5,7)"""
    def is_fast(k: int) -> bool:
        for p in (2,3,5,7):
            while k % p == 0:
                k //= p
        return k == 1

    while not is_fast(n):
        n += 1
    return n


def conv_lowpass_filter(data: Tensor, window_width: int) -> Tensor:
    """
    Time-domain lowpass filter: moving average with Hamming window taps
    along the last axis, with replicate edge padding.

    Time-domain alternative to :func:`fft_lowpass_filter_window`. Note
    the different parameter meaning: ``window_width`` is the filter
    length in samples (cutoff at roughly ``1 / window_width`` cycles per
    sample), while the FFT variant's width is in frequency bins.
    Replicate padding avoids the edge roll-off that zero padding causes
    on short signals. Works for real and complex input.

    Parameters
    ----------
    data : Tensor
        Input tensor, filtered along the last axis.
    window_width : int
        Filter length in samples. Lengths <= 1 (or longer than the
        signal) return the input unchanged.

    Returns
    -------
    filtered_data : Tensor
        Filtered tensor with the same shape as the input.
    """
    if window_width is None or window_width <= 1:
        return data
    L = int(window_width)
    if L > data.shape[-1]:
        return data
    w = torch.hamming_window(L, device=data.device, dtype=torch.float32)
    w = (w / w.sum()).to(data.dtype)
    shape = data.shape
    v = data.reshape(-1, 1, shape[-1])
    vp = torch.nn.functional.pad(
        v, (L // 2, L - L // 2 - 1), mode="replicate"
    )
    out = torch.nn.functional.conv1d(vp, w[None, None])
    return out.reshape(shape)


def fft_lowpass_filter_precalculate_window(
        data_length: int,
        window_width: int,
        device: str,
        window: str | tuple,
        circular_conv: bool = False,
        fast_len: bool = True) -> Tensor:
    """
    Precompute window to be used with `fft_lowpass_filter_window`.

    Returns
    -------
    w : Tensor
        Windowing Tensor.
    """
    half_width = (window_width + 1) // 2

    # Original padding for linear convolution
    pad_size = 2 * half_width if not circular_conv else 0

    # FFT length
    fft_len = data_length + pad_size
    if fast_len:
        fft_len = next_fast_len(fft_len)

    # Window centered at DC (symmetric for zero-phase)
    half_window = get_window(window, 2 * half_width - 1, fftbins=True)[half_width - 1:]
    w = np.zeros(fft_len, dtype=np.float32)
    w[:half_width] = half_window
    w[-half_width + 1:] = np.flip(half_window[1:])

    return torch.tensor(w, device=device)


def fft_lowpass_filter_window(
    target_data: Tensor,
    window: str | tuple | Tensor = "hamming",
    window_width: int = None,
    circular_conv: bool = False,
    fast_len: bool = True,
) -> Tensor:
    """
    FFT low-pass filtering with a configurable window function.
    """
    if window_width is None or window_width > target_data.shape[-1]:
        return target_data

    half_width = (window_width + 1) // 2
    N = target_data.shape[-1]

    # Determine padding and FFT length
    if isinstance(window, Tensor):
        fft_len = window.numel()
        pad_size = 2 * half_width if not circular_conv else 0
    else:
        pad_size = 2 * half_width if not circular_conv else 0
        fft_len = N + pad_size
        if fast_len:
            fft_len = next_fast_len(fft_len)

    fdata = torch.fft.fft(target_data, dim=-1, n=fft_len)

    if isinstance(window, Tensor):
        w = window
    else:
        w = fft_lowpass_filter_precalculate_window(
            N, window_width, target_data.device, window,
            circular_conv=circular_conv, fast_len=fast_len
        )

    fdata *= w
    filtered_data = torch.fft.ifft(fdata, dim=-1)

    # Trim to original length
    filtered_data = filtered_data[..., :N]

    return filtered_data


def center_pos(pos: Tensor) -> tuple[Tensor, Tensor]:
    """
    Center position to origin. Centers X and Y coordinates, but doesn't modify Z.
    Useful for preparing positions for polar backprojection

    Parameters
    ----------
    pos : Tensor
        3D positions. Shape should be [N, 3].

    Returns
    -------
    pos_local : Tensor
        Centered positions.
    origin : Tensor
        Position subtracted from the pos.
    """
    origin = torch.tensor(
        [torch.mean(pos[:, 0]), torch.mean(pos[:, 1]), 0],
        device=pos.device,
        dtype=torch.float32,
    )[None, :]
    pos_local = pos - origin
    return pos_local, origin


def bounding_cart_grid(
    grid_polar: "PolarGrid | dict",
    origin: tuple,
    origin_angle: float,
) -> dict:
    """
    Return the bounding Cartesian grid for polar input grid.

    Parameters
    ----------
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
        - dict: {"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}

        where theta is sin of angle (-1, 1 for 180 degree view).
    origin : tuple
        Origin coordinates of grid_polar in the Cartesian grid.
    origin_angle : float
        Reference direction (radians) that corresponds to s = 0.

    Returns
    -------
    (xmin, ymin, xmax, ymax) : tuple[float, float, float, float]
        Coordinates of the smallest axis‑aligned rectangle containing the grid.
    """
    (r0, r1) = grid_polar["r"]
    (s0, s1) = grid_polar["theta"]

    # Convert the stored sine values back to angles and shift by the origin.
    a0 = origin_angle + np.arcsin(s0)
    a1 = origin_angle + np.arcsin(s1)
    a_min, a_max = (a0, a1) if a0 <= a1 else (a1, a0)

    # Quadrantal angles where x or y may reach an extremum.
    candidate_angles = np.linspace(a_min, a_max, 20, endpoint=True)

    xmin = ymin = float("inf")
    xmax = ymax = -float("inf")

    for r in (r0, r1):
        for a in candidate_angles:
            x = r * np.cos(a) + origin[0]
            y = r * np.sin(a) + origin[1]
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    dr = (grid_polar["r"][1] - grid_polar["r"][0]) / grid_polar["nr"]
    nx = int((xmax - xmin) / dr)
    ny = int((ymax - ymin) / dr)
    grid_cart = {"x": (xmin, xmax), "y": (ymin, ymax), "nx": nx, "ny": ny}
    return grid_cart


def create_triangular_weights(patch_size: int, overlap: int, device: str = "cpu") -> Tensor:
    """
    Create triangular weights for smooth blending of overlapping patches.

    Parameters
    ----------
    patch_size : int
        Side length of patches.
    overlap : int
        Overlap between patches.
    device : str
        Pytorch device.

    Returns
    -------
    weights_2d : Tensor
        Weight tensor of [patch_size, patch_size] with triangular weighting
    """
    K = patch_size
    O = overlap

    if O == 0:
        # No overlap - use uniform weights
        return torch.ones(K, K, device=device)

    # Create 1D triangular weights
    weights_1d = torch.ones(K, device=device)

    # Apply triangular weighting at the edges
    fade_length = O

    # Left edge: linear fade-in
    for i in range(min(fade_length, K)):
        weights_1d[i] = (i + 1) / (fade_length + 1)

    # Right edge: linear fade-out
    for i in range(max(0, K - fade_length), K):
        weights_1d[i] = (K - i) / (fade_length + 1)

    # Create 2D triangular weights using outer product
    weights_2d = torch.outer(weights_1d, weights_1d)

    return weights_2d


def extract_overlapping_patches(img: Tensor, patch_size: int, overlap: int) -> tuple[Tensor, tuple[int, ...]]:
    """
    Extract overlapping patches from a tensor.

    Parameters
    ----------
    img : Tensor
        Input tensor of shape [C, N, M].
    patch_size : int
        Side length of square patches (K).
    overlap : int
        Overlap between patches.

    Returns
    -------
    patches : Tensor
        Tensor of shape [C, P, K, K] where P is the number of patches.
    dim : tuple
        Original image dimensions.
    """
    C, N, M = img.shape
    K = patch_size
    O = overlap

    # Calculate stride (distance between patch centers)
    stride = K - O

    pad_height = (K - N % stride) % stride
    pad_width = (K - M % stride) % stride

    if pad_height > 0 or pad_width > 0:
        # Pad with reflection to avoid edge artifacts
        img_padded = F.pad(img, (0, pad_width, 0, pad_height), mode="reflect")
    else:
        img_padded = img

    # Add batch dimension
    img_batch = img_padded.unsqueeze(0)

    # Extract patches
    patches = F.unfold(img_batch, kernel_size=K, stride=stride)
    patches = patches.squeeze(0)
    patches = patches.view(C, K * K, -1)
    P = patches.shape[2]
    patches = patches.transpose(1, 2).contiguous().view(C, P, K, K)

    return patches, img_padded.shape[1:]  # Return padded dimensions


def merge_patches_with_triangular_weights(
    patches: Tensor, original_shape: tuple[int, int], patch_size: int, overlap: int, padded_shape: tuple[int, int] | None = None
) -> Tensor:
    """
    Merge overlapping patches back into an image using triangular weighting.

    Parameters
    ----------
    patches : Tensor
        Tensor of shape [C, P, K, K] containing patches
    original_shape : tuple
        Original shape of the image (N, M).
    patch_size : int
        Side length of square patches (K).
    overlap : int
        Overlap between patches.
    padded_shape : tuple
        Tuple (N_pad, M_pad) of padded dimensions.

    Returns
    -------
    img : Tensor
        Reconstructed image tensor of shape [C, N, M].
    """
    C, P, K, K_check = patches.shape
    assert K == K_check, "Patches must be square"

    N, M = original_shape
    stride = K - overlap

    # Determine reconstruction dimensions
    if padded_shape is not None:
        N_recon, M_recon = padded_shape
    else:
        N_recon, M_recon = N, M

    if overlap == 0:
        # No overlap case - use uniform weights and simple reconstruction
        weights = torch.ones(K, K, device=patches.device)
        weighted_patches = patches * weights.unsqueeze(0).unsqueeze(0)

        # Reshape patches for fold operation
        weighted_patches_flat = (
            weighted_patches.view(C, P, K * K).transpose(1, 2).contiguous()
        )
        weighted_patches_flat = weighted_patches_flat.view(C * K * K, P)

        # Add batch dimension for fold
        weighted_patches_batch = weighted_patches_flat.unsqueeze(0)

        # Reconstruct using fold
        reconstructed = F.fold(
            weighted_patches_batch,
            output_size=(N_recon, M_recon),
            kernel_size=K,
            stride=stride,
        )

        # Remove batch dimension
        reconstructed = reconstructed.squeeze(0)

        # Crop back to original size if padding was used
        if padded_shape is not None:
            reconstructed = reconstructed[:, :N, :M]

        return reconstructed

    # Overlapping case - use triangular weights
    weights = create_triangular_weights(patch_size, overlap, device=patches.device)

    # Apply weights to patches
    weighted_patches = patches * weights.unsqueeze(0).unsqueeze(0)

    # Reshape patches for fold operation
    weighted_patches_flat = (
        weighted_patches.view(C, P, K * K).transpose(1, 2).contiguous()
    )
    weighted_patches_flat = weighted_patches_flat.view(C * K * K, P)

    # Also create weight patches for normalization
    weight_patches = weights.unsqueeze(0).expand(P, -1, -1).unsqueeze(0)
    weight_patches_flat = weight_patches.view(1, P, K * K).transpose(1, 2).contiguous()
    weight_patches_flat = weight_patches_flat.view(K * K, P)

    # Add batch dimension for fold
    weighted_patches_batch = weighted_patches_flat.unsqueeze(0)
    weight_patches_batch = weight_patches_flat.unsqueeze(0)

    # Reconstruct using fold
    reconstructed = F.fold(
        weighted_patches_batch,
        output_size=(N_recon, M_recon),
        kernel_size=K,
        stride=stride,
    )
    weight_sum = F.fold(
        weight_patches_batch,
        output_size=(N_recon, M_recon),
        kernel_size=K,
        stride=stride,
    )

    # Remove batch dimension
    reconstructed = reconstructed.squeeze(0)
    weight_sum = weight_sum.squeeze(0)

    # Normalize by weight sum to handle overlaps
    epsilon = 1e-8
    reconstructed = reconstructed / (weight_sum + epsilon)

    if padded_shape is not None:
        reconstructed = reconstructed[:, :N, :M]

    return reconstructed


def process_image_with_patches(img: Tensor, patch_size: int, overlap: int, process_fn) -> Tensor:
    """
    Process an image by extracting patches, applying a function, and merging back.

    Parameters
    ----------
    img : Tensor
        Input tensor of shape [C, N, M] or [N, M].
    patch_size : int
        Side length of square patches.
    overlap : int
        Overlap between patches.
    process_fn : function
        Function to apply to patches.

    Returns
    -------
    img : Tensor
        Processed image with same shape as the input.
    """
    N, M = img.shape[1], img.shape[2]

    # Extract patches
    patches, padded_shape = extract_overlapping_patches(img, patch_size, overlap)

    # Apply processing function
    processed_patches = process_fn(patches)

    # Merge back
    result = merge_patches_with_triangular_weights(
        processed_patches, (N, M), patch_size, overlap, padded_shape
    )

    return result
