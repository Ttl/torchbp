import torch
from torch import Tensor
import torch.nn.functional as F
from math import pi
import numpy as np
from scipy.signal import get_window


def bp_polar_range_dealias(
    img: Tensor, origin: Tensor, fc: float, grid_polar: dict
) -> Tensor:
    """
    De-alias range-axis spectrum of polar SAR image processed with backprojection.

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth].
    origin : Tensor
        Center of the platform position.
    fc : float
        RF center frequency.
    grid_polar : dict
        Polar grid definition

    References
    ----------
    .. [#] T. Shi, X. Mao, A. Jakobsson and Y. Liu, "Extended PGA for Spotlight
    SAR-Filtered Backprojection Imagery," in IEEE Geoscience and Remote Sensing
    Letters, vol. 19, pp. 1-5, 2022, Art no. 4516005.

    Returns
    ----------
    img : Tensor
        SAR image without range spectrum aliasing.
    """
    r0, r1 = grid_polar["r"]
    theta0, theta1 = grid_polar["theta"]
    ntheta = grid_polar["ntheta"]
    nr = grid_polar["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    r = r0 + dr * torch.arange(nr, device=img.device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=img.device)

    x = r[:, None] * torch.sqrt(1 - torch.square(theta))[None, :]
    y = r[:, None] * theta[None, :]

    if origin.dim() == 2:
        origin = origin[0]
    d = torch.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2 + origin[2] ** 2)
    c0 = 299792458
    phase = torch.exp(-1j * 4 * pi * fc * d / c0)
    if img.dim() == 3:
        phase = phase.unsqueeze(0)
    return phase * img


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
    ----------
    d : Tensor
        Difference tensor.
    """
    if dim != -1:
        raise NotImplementedError("Only dim=-1 is implemented")
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
    ----------
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
    ----------
    f : float
        Estimated fractional peak index.
    """
    a1 = a[(v - 1) % len(a)]
    a2 = a[v % len(a)]
    a3 = a[(v + 1) % len(a)]
    return 0.5 * (a1 - a3) / (a1 - 2 * a2 + a3)


def argmax_nd(x: Tensor):
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
    y : int
        Input tensor. Should have same shape as x.
    dim : int
        Dimensions to shift.

    Returns
    ----------
    c : int
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
    img: Tensor, grid_cart: dict, x0: float, x1: float, y0: float, y1: float
) -> (Tensor, dict):
    """Cartesian image subset.

    Parameters
    ----------
    img : Tensor
        Input image.
    grid_cart : dict
        Cartesian grid dictionary.
    x0 : float
        Subset x0.
    x1 : float
        Subset x1.
    y0 : float
        Subset y0.
    y1 : float
        Subset y1.

    Returns
    ----------
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
    img: Tensor, grid_polar: dict, r0: float, r1: float, theta0: float, theta1: float
) -> (Tensor, dict):
    """Polar image subset.

    Parameters
    ----------
    img : Tensor
        Input image.
    grid_cart : dict
        Cartesian grid dictionary.
    r0 : float
        Subset r0.
    r1 : float
        Subset r1.
    theta0 : float
        Subset theta0.
    theta1 : float
        Subset tehta1.

    Returns
    ----------
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
    ----------
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
    ----------
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
    ----------
    x : Tensor
        x with linear trend removed.
    """
    n = x.shape[0]
    k = np.arange(n) / n
    a, b = np.polyfit(k, x.cpu().numpy(), 1)
    return x - (a * torch.arange(n, device=x.device, dtype=x.dtype) / n + b)


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
    ----------
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
    ----------
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
    ----------
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
        [nsweeps, 3] tensor of platform positions.
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
        (0, 0) angle is at the beam center.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1]
        g_el0 : float
            grx and gtx elevation axis starting value. Units in radians. -pi/2 if
            including data over the whole sphere.
        g_az0 : float
            grx and gtx azimuth axis starting value. Units in radians. -pi if
            including data over the whole sphere.
        g_el1 : float
            grx and gtx elevation axis end value. Units in radians. +pi/2 if
            including data over the whole sphere.
        g_az1 : float
            grx and gtx azimuth axis end value. Units in radians. +pi if
            including data over the whole sphere.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3].
        [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    rvp : bool
        True to include residual video phase term.

    Returns
    ----------
    data : Tensor
        [nsweeps, nsamples] measurement data.
    """
    if pos.dim() != 2:
        raise ValueError("pos tensor should have 2 dimensions")
    if pos.shape[1] != 3:
        raise ValueError("positions should be 3 dimensional")
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
    for e, target in enumerate(target_pos):
        rcs_phase = torch.angle(target_rcs[e])
        rcs_abs = torch.sqrt(torch.abs(target_rcs[e]))

        d = torch.linalg.vector_norm(pos - target[None, :], dim=-1)[:, None] + d0
        tau = 2 * d / c0
        if antenna_gain:
            look_angle = torch.asin(pos[:, 2] / d[:, 0])
            el_deg = -look_angle - att[:, 0]
            az_deg = (
                torch.atan2(target[1] - pos[:, 1], target[0] - pos[:, 0]) - att[:, 2]
            )

            az_norm = 2.0 * (az_deg - g_az0) / (g_az1 - g_az0) - 1.0
            el_norm = 2.0 * (el_deg - g_el0) / (g_el1 - g_el0) - 1.0

            grid = torch.stack([el_norm, az_norm], dim=-1)
            grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

            g_a = F.grid_sample(
                g_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            g_a = g_a.reshape(d.shape)
        else:
            g_a = 1

        data += (g_a * rcs_abs / d**2) * torch.exp(
            -1j * 2 * pi * (fc * tau - k * tau * t + use_rvp * 0.5 * k * tau**2) + 1j * rcs_phase
        )
    return data


def make_polar_grid(
    r0: float, r1: float, nr: int, ntheta: int, theta_limit: int = 1, squint: float = 0
) -> dict:
    """
    Generate polar grid dict in format understood by other polar functions.

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
    ----------
    grid_polar : dict
        Polar grid dict.
    """
    t0 = np.clip(np.sin(squint) - theta_limit, -1, 1)
    t1 = np.clip(np.sin(squint) + theta_limit, -1, 1)
    grid_polar = {"r": (r0, r1), "theta": (t0, t1), "nr": nr, "ntheta": ntheta}
    return grid_polar


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


def fft_lowpass_filter_window(
    target_data: Tensor, window: str | tuple = "hamming", window_width: int = None
) -> Tensor:
    """
    FFT low-pass filtering with a configurable window function.

    Parameters
    ----------
    target_data : Tensor
        Input data.
    window_type : str
        Window to apply. See scipy.get_window for syntax.
        e.g., 'hann', 'hamming', 'blackman'.
    window_width : int
        Width of the window in samples. If None or larger than signal, returns
        the input unchanged.

    Returns
    ----------
        Filtered tensor (same shape as input)
    """
    fdata = torch.fft.fft(target_data, dim=-1)
    n = target_data.size(-1)

    # If window_width is None, do nothing
    if window_width is None or window_width > n:
        return target_data

    # Window needs to be centered at DC in FFT
    half_width = (window_width + 1) // 2
    half_window = get_window(window, 2 * half_width - 1, fftbins=True)[half_width - 1 :]
    w = np.zeros(n, dtype=np.float32)
    w[:half_width] = half_window
    w[-half_width + 1 :] = np.flip(half_window[1:])

    w = torch.tensor(w).to(target_data.device)
    filtered_data = torch.fft.ifft(fdata * w, dim=-1)
    return filtered_data


def center_pos(pos: Tensor):
    """
    Center position to origin. Centers X and Y coordinates, but doesn't modify Z.
    Useful for preparing positions for polar backprojection

    Parameters
    ----------
    pos : Tensor
        3D positions. Shape should be [N, 3].

    Returns
    ----------
    pos_local : Tensor
        Centered positions.
    origin : Tensor
        Position subtracted from the pos.
    h : Tensor
        Mean height.
    """
    origin = torch.tensor(
        [torch.mean(pos[:, 0]), torch.mean(pos[:, 1]), 0],
        device=pos.device,
        dtype=torch.float32,
    )[None, :]
    pos_local = pos - origin
    return pos_local, origin


def bounding_cart_grid(
    grid_polar: dict,
    origin: tuple,
    origin_angle: float,
) -> dict:
    """
    Return the bounding Cartesian grid for polar input grid.

    Parameters
    ----------
    grid_polar : dict
        {
          "r":      (r0, r1),              # range
          "theta":  (s0, s1),              # sine of azimuth angle
          "nr":     nr,                    # number of samples in r
          "ntheta": ntheta                 # number of samples in theta
        }

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


def create_triangular_weights(patch_size, overlap, device="cpu"):
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
    ----------
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


def extract_overlapping_patches(img, patch_size, overlap):
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
    ----------
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
    patches, original_shape, patch_size, overlap, padded_shape=None
):
    """
    Merge overlapping patches back into an image using triangular weighting.

    Parameters
    ----------
    patches : Tensor
        Tensor of shape [C, P, K, K] containing patches
    original_shape : tuple
        Original shape of the image (N, M).
    overlap : int
        Overlap between patches.
    padded_shape : tuple
        Tuple (N_pad, M_pad) of padded dimensions.

    Returns
    ----------
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


def process_image_with_patches(img, patch_size, overlap, process_fn):
    """
    Process an image by extracting patches, applying a function, and merging back.

    Parameters
    ----------
    img : Tensor
        Input tensor of shape [C, N, M] or [N, M].
    patch_size : int
        Side length of square patches.
    overlap : float
        Overlap between patches as fraction.
    process_fn : function
        Function to apply to patches.

    Returns
    ----------
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
