import torch
from torch import Tensor
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


def find_image_shift_1d(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """
    Find shift between images that maximizes correlation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    y : int
        Input tensor. Should have same shape as x.

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
    return x - (a * torch.arange(n, device=x.device) / n + b)


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

    t = t[None, :]
    for e, target in enumerate(target_pos):
        d = torch.linalg.vector_norm(pos - target[None, :], dim=-1)[:, None] + d0
        tau = 2 * d / c0
        data += (target_rcs[e] / d**4) * torch.exp(
            -1j * 2 * pi * (fc * tau - k * tau * t + use_rvp * 0.5 * k * tau**2)
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
    w = np.zeros(n)
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
        [
            torch.mean(pos[:, 0]),
            torch.mean(pos[:, 1]),
            0
        ],
        device=pos.device,
        dtype=torch.float32,
    )[None, :]
    pos_local = pos - origin
    return pos_local, origin
