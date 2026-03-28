from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
from typing import TYPE_CHECKING, Union, Tuple
from .grid import unpack_polar_grid

if TYPE_CHECKING:
    from .grid import PolarGrid
from .ops import (
    backprojection_polar_2d,
    backprojection_cart_2d,
    gpga_backprojection_2d_core,
    ffbp,
)
from .ops import entropy
from .util import (
    unwrap,
    unwrap_ref,
    detrend,
    fft_lowpass_filter_precalculate_window,
    fft_lowpass_filter_window,
    diff
)
import inspect
from scipy import signal
from copy import deepcopy


def pga_estimator(
    g: Tensor, estimator: str = "wls", eps: float = 1e-6, return_weight: bool = False
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """
    Estimate phase error from set of measurements.

    Parameters
    ----------
    g : Tensor
        Demodulated phase from each target. Shape [Ntargets, Nazimuth].
    estimator : str
        Estimator to use.
            - "pd": Phase difference. [1]_
            - "ml": Maximum likelihood. [2]_
            - "wls": Weighted least squares using estimated signal-to-clutter weighting. [3]_
    eps : float
        Minimum weight for weighted PGA.

    References
    ----------
    .. [1] D. E. Wahl, P. H. Eichel, D. C. Ghiglia and C. V. Jakowatz, "Phase
        gradient autofocus - A robust tool for high resolution SAR phase
        correction," in IEEE Transactions on Aerospace and Electronic Systems, vol.
        30, no. 3, pp. 827-835, July 1994.

    .. [2] Charles V. Jakowatz and Daniel E. Wahl, "Eigenvector method for
        maximum-likelihood estimation of phase errors in synthetic-aperture-radar
        imagery," J. Opt. Soc. Am. A 10, 2539-2546 (1993).

    .. [3] Wei Ye, Tat Soon Yeo and Zheng Bao, "Weighted least-squares
        estimation of phase errors for SAR/ISAR autofocus," in IEEE Transactions on
        Geoscience and Remote Sensing, vol. 37, no. 5, pp. 2487-2494, Sept. 1999.

    Returns
    -------
    phi : Tensor
        Solved phase error.
    """
    if estimator == "ml":
        u, s, v = torch.linalg.svd(g)
        phi = torch.angle(v[0, :])
    elif estimator == "wls":
        c = torch.mean(torch.abs(g), dim=1, keepdim=True)
        d = torch.mean(torch.abs(g) ** 2, dim=1, keepdim=True)
        w = (
            torch.nan_to_num(
                d / (2 * (2 * c**2 - d) - 2 * c * torch.sqrt(4 * c**2 - 3 * d))
            )
            + eps
        )
        gshift = torch.nn.functional.pad(g[..., :-1], (1, 0))
        phidot = torch.angle(
            torch.sum((w / torch.max(w)) * (g * torch.conj(gshift)), dim=0)
        )
        phi = torch.cumsum(phidot, dim=0)
        if return_weight:
            return phi, w
    elif estimator == "pd":
        z = torch.zeros((g.shape[0], 1), device=g.device, dtype=g.dtype)
        gdot = torch.diff(g, prepend=z, dim=-1)
        phidot = torch.sum((torch.conj(g) * gdot).imag, dim=0) / torch.sum(
            torch.abs(g) ** 2, dim=0
        )
        phi = torch.cumsum(phidot, dim=0)
    else:
        raise ValueError(f"Unknown estimator {estimator}")
    return phi


def pga(
    img: Tensor,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.5,
    min_window: int = 5,
    remove_trend: bool = True,
    offload: bool = False,
    estimator: str = "wls",
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Phase gradient autofocus

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth].
    window_width : int
        Initial window width. Default is None which uses full image size.
    max_iter : int
        Maximum number of iterations.
    window_exp : float
        Exponent for decreasing the window size for each iteration.
    min_window : int
        Minimum window size.
    remove_trend : bool
        Remove linear trend that shifts the image.
    offload : bool
        Offload some variable to CPU to save VRAM on GPU at
        the expense of longer running time.
    estimator : str
        Estimator to use.
        See `pga_estimator` function for possible choices.
    eps : float
        Minimum weight for weighted PGA.

    Returns
    -------
    img : Tensor
        Focused image.
    phi : Tensor
        Solved phase error.
    """
    if img.ndim != 2:
        raise ValueError("Input image should be 2D.")
    if window_exp > 1 or window_exp < 0:
        raise ValueError(f"Invalid window_exp {window_exp}")
    nr, ntheta = img.shape
    phi_sum = torch.zeros(ntheta, device=img.device)
    if window_width is None:
        window_width = ntheta
    if window_width > ntheta:
        window_width = ntheta
    x = np.arange(ntheta)
    dev = img.device
    for i in range(max_iters):
        window = int(window_width * window_exp**i)
        if window < min_window:
            break
        # Peak for each range bin
        g = img.clone()
        if offload:
            img = img.to(device="cpu")
        rpeaks = torch.argmax(torch.abs(g), axis=1)
        # Roll theta axis so that peak is at 0 bin
        for j in range(nr):
            g[j, :] = torch.roll(g[j, :], -rpeaks[j].item())
        # Apply window
        g[:, 1 + window // 2 : 1 - window // 2] = 0
        # IFFT across theta
        g = torch.fft.fft(g, axis=-1)
        phi = pga_estimator(g, estimator, eps)
        del g
        if remove_trend:
            phi = detrend(unwrap(phi))
        phi_sum += phi

        if offload:
            img = img.to(device=dev)
        img_ifft = torch.fft.fft(img, axis=-1)
        img_ifft *= torch.exp(-1j * phi[None, :])
        img = torch.fft.ifft(img_ifft, axis=-1)

    return img, phi_sum


def gpga_bp_polar(
    img: Tensor | None,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid_polar: "PolarGrid | dict",
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.7,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    remove_trend: bool = True,
    estimator: str = "pd",
    lowpass_window: str = "boxcar",
    eps: float = 1e-6,
    interp_method: str = "linear",
    data_fmod: float = 0
) -> tuple[Tensor, Tensor]:
    """
    Generalized phase gradient autofocus using 2D polar coordinate
    backprojection image formation. [1]_

    Parameters
    ----------
    img : Tensor or None
        Complex input image. Shape should be: [Range, azimuth].
        If None image is generated from the data.
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
        - dict: {"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}

        where theta is sin of angle (-1, 1 for 180 degree view).
    window_width : int or None
        Initial low-pass filter window width in samples. None for initial
        maximum size.
    max_iters : int
        Maximum number of iterations.
    window_exp : float
        Exponent on window_width decrease for each iteration.
    min_window : int
        Minimum window size.
    d0 : float
        Zero range correction.
    target_threshold_db : float
        Filter out targets that are this many dB below the maximum amplitude
        target.
    remove_trend : bool
        Remove linear trend in phase correction.
    estimator : str
        Estimator to use.
        See `pga_estimator` function for possible choices.
    lowpass_window : str
        FFT window to use for lowpass filtering.
        See `scipy.get_window` for syntax.
    eps : float
        Minimum weight for weighted PGA.
    interp_method : str
        Interpolation method
        "linear": linear interpolation.
        ("lanczos", N): Lanczos interpolation with order 2*N+1.
    data_fmod : float
        Range modulation frequency applied to input data.

    References
    ----------
    .. [1] A. Evers and J. A. Jackson, "A Generalized Phase Gradient Autofocus
        Algorithm," in IEEE Transactions on Computational Imaging, vol. 5, no. 4,
        pp. 606-619, Dec. 2019.

    Returns
    -------
    img : Tensor
        Focused SAR image.
    phi : Tensor
        Solved phase error.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)

    phi_sum = torch.zeros(data.shape[0], dtype=torch.float32, device=data.device)

    theta = theta0 + dtheta * torch.arange(
        ntheta, device=data.device, dtype=torch.float32
    )
    pos_new = pos.clone()

    if window_width is None:
        window_width = data.shape[0]

    if img is None:
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0, data_fmod=data_fmod)[0]

    for i in range(max_iters):
        rpeaks = torch.argmax(torch.abs(img), dim=1)
        a = torch.abs(img[torch.arange(img.size(0)), rpeaks])
        max_a = torch.max(a)

        target_idx = a > max_a * 10 ** (-target_threshold_db / 20)
        target_theta = theta0 + dtheta * rpeaks[target_idx].to(torch.float32)
        target_r = r0 + dr * target_idx.nonzero(as_tuple=True)[0].to(torch.float32)

        x = target_r * torch.sqrt(1 - target_theta**2)
        y = target_r * target_theta
        z = torch.zeros_like(target_r)
        target_pos = torch.stack([x, y, z], dim=1)

        # Get range profile samples for each target
        target_data = gpga_backprojection_2d_core(
            target_pos, data, pos_new, fc, r_res, d0, interp_method=interp_method, data_fmod=data_fmod
        )
        # Filter samples
        if window_width is not None and window_width < target_data.shape[1]:
            target_data = fft_lowpass_filter_window(
                target_data, window=lowpass_window, window_width=window_width
            )
        phi = pga_estimator(target_data, estimator, eps)
        phi_sum = unwrap(phi_sum + phi)
        if remove_trend:
            phi_sum = detrend(phi_sum)
        # Phase to distance
        c0 = 299792458
        d = phi_sum * c0 / (4 * torch.pi * fc)
        pos_new[:, 0] = pos[:, 0] + d

        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0, data_fmod=data_fmod)[0]
        window_width = int(window_width * window_exp)
        if window_width < min_window:
            break
    return img, phi_sum


def gpga_bp_polar_tde(
    img: Tensor | None,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid_polar: "PolarGrid | dict",
    azimuth_divisions: int,
    range_divisions: int,
    window_width: int | None = None,
    rms_error_limit: float = 0.05,
    max_iters: int = 20,
    window_exp: float = 0.7,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    remove_trend: bool = True,
    lowpass_window: str = "boxcar",
    eps: float = 1e-6,
    interp_method: str = "linear",
    estimate_z: bool = True,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    use_ffbp: bool = False,
    ffbp_opts: dict | None = None,
    verbose: bool = False,
    data_fmod: float = 0,
) -> tuple[Tensor, Tensor]:
    """
    Generalized phase gradient autofocus [1]_ using 2D polar coordinate
    backprojection image formation.

    Estimates 3D position error by dividing the image into subimages, estimating
    slant range error to each subimage, and then solving for 3D position error
    from slant range errors. [2]_

    Z-axis estimation requires variable look angles in the image. Set
    `estimate_z` to False if this is not the case, for example ground based
    radar.

    Parameters
    ----------
    img : Tensor or None
        Complex input image. Shape should be: [Range, azimuth].
        If None image is generated from the data.
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    grid_polar : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
        - dict: {"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}

        where theta is sin of angle (-1, 1 for 180 degree view).
    azimuth_divisions : int
        Number of divisions for local images in azimuth direction.
    range_divisions : int
        Number of divisions for local images in range direction.
    window_width : int or None
        Initial low-pass filter window width in samples. None for initial
        maximum size.
    rms_error_limit : float
        Phase RMS error limit in radians for stopping the optimization iteration.
    max_iters : int
        Maximum number of iterations.
    window_exp : float
        Exponent on window_width decrease for each iteration.
    min_window : int
        Minimum window size.
    d0 : float
        Zero range correction.
    target_threshold_db : float
        Filter out targets that are this many dB below the maximum amplitude
        target.
    remove_trend : bool
        Remove linear trend in phase correction.
    lowpass_window : str
        FFT window to use for lowpass filtering.
        See `scipy.get_window` for syntax.
    eps : float
        Minimum weight for weighted PGA.
    interp_method : str
        Interpolation method
        "linear": linear interpolation.
        ("lanczos", N): Lanczos interpolation with order 2*N+1.
    estimate_z : bool
        Estimate Z-axis position error. Default is True.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
-       If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    use_ffbp : bool
        Use fast factorized backprojection for image formation.
    ffbp_opts : dict
        Dictionary of options for ffbp.
    verbose : bool
        Print progress stats.
    data_fmod : float
        Range modulation frequency applied to input data.


    References
    ----------
    .. [1] A. Evers and J. A. Jackson, "A Generalized Phase Gradient Autofocus
        Algorithm," in IEEE Transactions on Computational Imaging, vol. 5, no. 4,
        pp. 606-619, Dec. 2019.
    .. [2] Z. Ding et al., "An Autofocus Approach for UAV-Based Ultrawideband
        Ultrawidebeam SAR Data With Frequency-Dependent and 2-D Space-Variant
        Motion Errors," in IEEE Transactions on Geoscience and Remote Sensing, vol.
        60, pp. 1-18, 2022, Art no. 5203518.

    Returns
    -------
    img : Tensor
        Focused SAR image.
    pos_new : Tensor
        Solved 3D position error.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)

    r = r0 + dr * torch.arange(nr, device=data.device, dtype=torch.float32)
    theta = theta0 + dtheta * torch.arange(
        ntheta, device=data.device, dtype=torch.float32
    )
    pos_new = pos.clone()

    if window_width is None:
        window_width = data.shape[0] // azimuth_divisions

    if img is None:
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0, data_fmod=data_fmod)[0]

    rdiv = img.shape[0] // range_divisions
    azdiv = img.shape[1] // azimuth_divisions

    local_d = torch.zeros(
        (range_divisions * azimuth_divisions, data.shape[0]),
        dtype=torch.float32,
        device=data.device,
    )
    local_centers = torch.zeros(
        (range_divisions * azimuth_divisions, 2),
        dtype=torch.float32,
        device=data.device,
    )
    local_w = torch.zeros(
        (range_divisions * azimuth_divisions, 1),
        dtype=torch.float32,
        device=data.device,
    )
    h = torch.mean(pos[:, 2])

    if h == 0:
        estimate_z = False

    wl = 3e8 / fc

    if verbose:
        print("Iteration, Window width, RMS error")

    for i in range(max_iters):
        lp_w = fft_lowpass_filter_precalculate_window(
            pos_new.shape[0], window_width, img.device, lowpass_window, fast_len=True
        )
        for ir in range(range_divisions):
            for jr in range(azimuth_divisions):
                ir1 = (ir + 1) * rdiv if ir < range_divisions - 1 else -1
                jr1 = (jr + 1) * azdiv if jr < azimuth_divisions - 1 else -1
                local_img = img[ir * rdiv : ir1, jr * azdiv : jr1]

                rpeaks = torch.argmax(torch.abs(local_img), dim=1)
                a = torch.abs(local_img[torch.arange(local_img.size(0)), rpeaks])
                max_a = torch.max(a)

                target_idx = a > max_a * 10 ** (-target_threshold_db / 20)
                target_theta = (
                    theta0
                    + dtheta * jr * azdiv
                    + dtheta * rpeaks[target_idx].to(torch.float32)
                )
                target_r = (
                    r0
                    + dr * ir * rdiv
                    + dr * target_idx.nonzero(as_tuple=True)[0].to(torch.float32)
                )

                x = target_r * torch.sqrt(1 - target_theta**2)
                y = target_r * target_theta
                z = torch.zeros_like(target_r)
                target_pos = torch.stack([x, y, z], dim=1)

                # Get range profile samples for each target
                target_data = gpga_backprojection_2d_core(
                    target_pos,
                    data,
                    pos_new,
                    fc,
                    r_res,
                    d0,
                    interp_method=interp_method,
                    data_fmod=data_fmod,
                )
                # Filter samples
                if window_width is not None and window_width < target_data.shape[1]:
                    target_data = fft_lowpass_filter_window(
                        target_data, window=lp_w, window_width=window_width
                    )
                phi, w = pga_estimator(target_data, "wls", eps, return_weight=True)
                local_w[ir * azimuth_divisions + jr] = 1 / torch.sum(1 / w)
                phi = unwrap(phi)
                phi = detrend(phi)
                # Phase to distance
                c0 = 299792458
                d = phi * c0 / (4 * torch.pi * fc)
                local_d[ir * azimuth_divisions + jr, :] = d

                local_centers[ir * azimuth_divisions + jr, 0] = torch.mean(
                    w * target_r
                ) / torch.mean(w)
                local_centers[ir * azimuth_divisions + jr, 1] = torch.mean(
                    w * target_theta
                ) / torch.mean(w)

        # Local image centers in Cartesian coordinates
        local_y = local_centers[:, 0] * local_centers[:, 1]
        local_x = local_centers[:, 0] * torch.sqrt(1 - local_centers[:, 1] ** 2)
        # Ground range from each position to local image centers
        local_r = torch.sqrt(
            (pos_new[:, 0][:, None] - local_x[None, :]) ** 2
            + (pos_new[:, 1][:, None] - local_y[None, :]) ** 2
        )

        # Local image center azimuth and elevation angles from each data position
        target_el = torch.arctan2(-pos_new[:, 2][:, None], local_r)
        target_az = torch.arctan2(
            local_y[None, :] - pos_new[:, 1][:, None],
            local_x[None, :] - pos_new[:, 0][:, None],
        )
        sin_az = torch.sin(target_az)
        cos_az = torch.cos(target_az)
        cos_el = torch.cos(target_el)
        sin_el = torch.sin(target_el)
        if estimate_z:
            m = torch.stack([cos_az * cos_el, sin_az * cos_el, sin_el], dim=-1)
        else:
            m = torch.stack([cos_az * cos_el, sin_az * cos_el], dim=-1)

        w = torch.sqrt(local_w).unsqueeze(0)
        w = w / torch.max(w)
        s = local_d.unsqueeze(0).transpose(0, 2)

        # Solve for 2D/3D position change for each data position from
        # distances from each position to local image centers
        d_solved = torch.linalg.lstsq(w * m, w * s).solution
        d_solved = d_solved.squeeze().transpose(0, 1)

        if remove_trend:
            d_solved[0] = detrend(d_solved[0])
        rms_error = (
            4 * torch.pi * torch.sqrt(torch.mean(torch.square(d_solved / wl))).item()
        )
        if verbose:
            print(f"{i+1}, {window_width}, {rms_error}")
        if rms_error < rms_error_limit:
            if verbose:
                print("RMS error limit reached")
            break
        pos_new[:, 0] = pos_new[:, 0] + d_solved[0]
        pos_new[:, 1] = pos_new[:, 1] + d_solved[1]
        if estimate_z:
            pos_new[:, 2] = pos_new[:, 2] + d_solved[2]

        if use_ffbp:
            opts = {"stages": 5, "oversample_r": 1.4, "oversample_theta": 1.4}
            if ffbp_opts is not None:
                opts.update(ffbp_opts)
            img = ffbp(data, grid_polar, fc, r_res, pos_new, d0=d0,
                    data_fmod=data_fmod, g=g, g_extent=g_extent, att=att,
                    **opts)
        else:
            img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new,
                    d0=d0, data_fmod=data_fmod, g=g, g_extent=g_extent,
                    att=att)[0]
        window_width = int(window_width * window_exp)
        if window_width < min_window:
            if verbose:
                print("Window width below the minimum size")
            break
    return img, pos_new


def insar_rme(
    img_s: Tensor | None,
    data_m: Tensor,
    data_s: Tensor,
    pos_m: Tensor,
    pos_s: Tensor,
    fc: float,
    r_res: float,
    grid_polar: "PolarGrid | dict",
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.7,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    lowpass_window: str = "boxcar",
    eps: float = 1e-6,
    interp_method: str = "linear",
    verbose: bool = False,
    data_fmod: float = 0,
    img_m: Tensor | None = None,
    spatial_coherence: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    InSAR residual motion error estimation by cross-pass phase comparison.

    Estimates pass 2 range-direction position errors by comparing demodulated
    target phase between passes. For each pass 2 sweep, the nearest pass 1
    sweep (by along-track distance to target) provides a reference. The
    geometric range difference from the baseline is computed and removed,
    leaving only the RME phase.

    Pass 1 should be autofocused first. Both passes must be on the same
    polar grid (same origin).

    Parameters
    ----------
    img_s : Tensor or None
        Pass 2 image [nr, ntheta]. If None, generated from data.
    data_m : Tensor
        Range compressed pass 1 (master) data [nsweeps_m, samples].
    data_s : Tensor
        Range compressed pass 2 (slave) data [nsweeps_s, samples].
    pos_m : Tensor
        Pass 1 corrected positions [nsweeps_m, 3].
    pos_s : Tensor
        Pass 2 positions to correct [nsweeps_s, 3].
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
    grid_polar : PolarGrid or dict
        Polar grid definition.
    window_width : int or None
        Initial low-pass filter window width. None for automatic.
    max_iters : int
        Maximum number of iterations.
    window_exp : float
        Exponent for decreasing window width each iteration.
    min_window : int
        Minimum window size.
    d0 : float
        Zero range correction.
    target_threshold_db : float
        Filter out targets below this threshold (dB below max).
    lowpass_window : str
        FFT window for lowpass filtering.
    eps : float
        Minimum weight.
    interp_method : str
        Interpolation method: "linear" or ("lanczos", N).
    verbose : bool
        Print progress.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    img_s : Tensor
        Focused pass 2 image.
    pos_s_new : Tensor
        Corrected pass 2 positions (range direction).
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)
    c0 = 299792458
    wl = c0 / fc
    k = 4 * torch.pi * fc / c0
    device = data_s.device

    pos_s_new = pos_s.clone()
    nsweeps_m = data_m.shape[0]
    nsweeps_s = data_s.shape[0]

    if window_width is None:
        window_width = nsweeps_s

    if img_s is None:
        img_s = backprojection_polar_2d(
            data_s, grid_polar, fc, r_res, pos_s_new, d0=d0, data_fmod=data_fmod
        )[0]

    # Initial spatial coherence map for target weighting.
    # Use pre-computed if provided; recomputed each iteration from reformed image.
    if spatial_coherence is not None:
        spatial_coh = spatial_coherence
    else:
        spatial_coh = torch.ones(nr, ntheta, device=device, dtype=torch.float32)

    if spatial_coh.dim() > 2:
        spatial_coh = spatial_coh.squeeze()
    spatial_coh = torch.nan_to_num(spatial_coh, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

    # Form master image once for coherence updates
    if img_m is None:
        img_m_internal = backprojection_polar_2d(
            data_m, grid_polar, fc, r_res, pos_m, d0=d0, data_fmod=data_fmod
        )[0]
    else:
        img_m_internal = img_m

    phi_sum = torch.zeros(nsweeps_s, dtype=torch.float32, device=device)

    if verbose:
        coh_mean = spatial_coh.mean().item()
        coh_high = (spatial_coh > 0.3).float().mean().item() * 100
        print(f"InSAR RME: spatial coherence mean={coh_mean:.3f}, "
              f"{coh_high:.0f}% above 0.3")
        print("InSAR RME: Iteration, Window width")

    for iteration in range(max_iters):
        # Find targets: peak in each range bin above threshold
        rpeaks = torch.argmax(torch.abs(img_s), dim=1)
        a = torch.abs(img_s[torch.arange(img_s.size(0)), rpeaks])
        max_a = torch.max(a)
        target_idx = a > max_a * 10 ** (-target_threshold_db / 20)

        target_az_idx = rpeaks[target_idx]
        target_r_idx = target_idx.nonzero(as_tuple=True)[0]
        target_theta = theta0 + dtheta * target_az_idx.to(torch.float32)
        target_r = r0 + dr * target_r_idx.to(torch.float32)

        x = target_r * torch.sqrt(1 - target_theta**2)
        y = target_r * target_theta
        z = torch.zeros_like(target_r)
        target_pos = torch.stack([x, y, z], dim=1)
        ntargets = target_pos.shape[0]

        # Look up spatial coherence for each target
        coh_nr, coh_naz = spatial_coh.shape
        r_idx_clamped = target_r_idx.clamp(0, coh_nr - 1)
        az_idx_clamped = target_az_idx.clamp(0, coh_naz - 1)
        target_coh = spatial_coh[r_idx_clamped, az_idx_clamped]

        # Demodulated phase for both passes
        td_s = gpga_backprojection_2d_core(
            target_pos, data_s, pos_s_new, fc, r_res, d0,
            interp_method=interp_method, data_fmod=data_fmod,
        )  # [ntargets, nsweeps_s]
        td_m = gpga_backprojection_2d_core(
            target_pos, data_m, pos_m, fc, r_res, d0,
            interp_method=interp_method, data_fmod=data_fmod,
        )  # [ntargets, nsweeps_m]

        # For each target, match pass 2 sweeps to pass 1 sweeps by
        # along-track distance and form cross product.
        # cross[t, m] = td_s[t, m] * conj(td_m[t, matched_n])
        # The per-target constant phase (from baseline, sub-pixel offset)
        # is cancelled by the PGA phase gradient estimator.
        cross = torch.zeros(
            (ntargets, nsweeps_s), dtype=torch.complex64, device=device
        )

        # Compute valid sweep range: pass 2 sweeps that have a valid
        # pass 1 match (along-track overlap). Sweeps outside the overlap
        # get zero cross products and don't affect the PGA estimator.
        valid_start = 0
        valid_end = nsweeps_s
        for t in range(ntargets):
            tp = target_pos[t]
            dy_s = pos_s_new[:, 1] - tp[1]
            dy_m = pos_m[:, 1] - tp[1]
            idx_m = torch.searchsorted(dy_m, dy_s).clamp(0, nsweeps_m - 1)
            idx_m_alt = (idx_m - 1).clamp(0, nsweeps_m - 1)
            closer = torch.abs(dy_m[idx_m_alt] - dy_s) < torch.abs(dy_m[idx_m] - dy_s)
            idx_m = torch.where(closer, idx_m_alt, idx_m)

            # Mask out sweeps where match distance exceeds one sweep spacing
            if nsweeps_m > 1:
                m_spacing = torch.median(torch.abs(torch.diff(dy_m)))
            else:
                m_spacing = torch.tensor(1e10, device=device)
            match_dist = torch.abs(dy_m[idx_m] - dy_s)
            valid = match_dist < 2 * m_spacing

            # Track valid range across targets
            if t == 0:
                valid_mask = valid
            else:
                valid_mask = valid_mask & valid

            cross[t, :] = td_s[t, :] * torch.conj(td_m[t, idx_m])
            cross[t, ~valid] = 0

        # Find contiguous valid range
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            valid_start = valid_indices[0].item()
            valid_end = valid_indices[-1].item() + 1
        if verbose and iteration == 0:
            print(f"  Valid sweep range: [{valid_start}, {valid_end}) of {nsweeps_s}")

        # Lowpass filter cross products
        if window_width < nsweeps_s:
            lp_w_cross = fft_lowpass_filter_precalculate_window(
                nsweeps_s, window_width, device, lowpass_window, fast_len=True
            )
            cross = fft_lowpass_filter_window(
                cross, window=lp_w_cross, window_width=window_width
            )

        cross_valid = cross[:, valid_start:valid_end]

        gamma_clamped = target_coh.clamp(eps, 0.999)
        w_coh = gamma_clamped ** 2 / (1 - gamma_clamped ** 2)

        # PGA WLS
        c_w = torch.mean(torch.abs(cross_valid), dim=1)
        d_w = torch.mean(torch.abs(cross_valid) ** 2, dim=1)
        sqrt_arg = (4 * c_w**2 - 3 * d_w).clamp(min=0)
        denom = 2 * (2 * c_w**2 - d_w) - 2 * c_w * torch.sqrt(sqrt_arg)
        w_scr = torch.where(
            torch.abs(denom) > eps,
            d_w / denom,
            torch.ones_like(d_w)
        )
        w_scr = torch.nan_to_num(w_scr, nan=1.0, posinf=1.0, neginf=1.0)

        # Combine: use coherence weight if coherence map is meaningful,
        # otherwise fall back to SCR weight
        if spatial_coh.mean() > 0.01:
            w = w_coh[:, None]
        else:
            w = w_scr[:, None]
        w = w / (torch.max(w) + eps)

        # PGA gradient: weighted sum of g[m]*conj(g[m-1]) across targets
        gshift = torch.nn.functional.pad(cross_valid[..., :-1], (1, 0))
        phidot = torch.angle(
            torch.sum(w * (cross_valid * torch.conj(gshift)), dim=0)
        )
        phi_valid = torch.cumsum(phidot, dim=0)
        phi_valid = detrend(phi_valid)

        # Extend to full sweep range: hold constant at edges to avoid
        # discontinuity between estimated and non-estimated regions
        phi = torch.zeros(nsweeps_s, dtype=torch.float32, device=device)
        phi[valid_start:valid_end] = phi_valid
        if valid_start > 0:
            phi[:valid_start] = phi_valid[0]
        if valid_end < nsweeps_s:
            phi[valid_end:] = phi_valid[-1]
        phi_sum = phi_sum + phi

        # Convert to range correction
        d_corr = phi_sum * c0 / (4 * torch.pi * fc)
        pos_s_new[:, 0] = pos_s[:, 0] + d_corr

        if verbose:
            rms = torch.sqrt(torch.mean(phi**2)).item()
            w_squeeze = w.squeeze()
            n_effective = (w_squeeze > 0.01 * w_squeeze.max()).sum().item()
            print(f"{iteration+1}, {window_width}, rms_phase={rms:.4f} rad, "
                  f"targets={ntargets}, effective={n_effective}, "
                  f"coh=[{target_coh.min().item():.2f}, "
                  f"{target_coh.median().item():.2f}, "
                  f"{target_coh.max().item():.2f}]")

        # Reform image
        img_s = backprojection_polar_2d(
            data_s, grid_polar, fc, r_res, pos_s_new, d0=d0, data_fmod=data_fmod
        )[0]

        # Update spatial coherence from reformed slave image and master
        coh_win = 5
        k_coh = torch.ones(1, 1, coh_win, coh_win, device=device,
                           dtype=torch.float32) / (coh_win * coh_win)
        _igram = img_s * torch.conj(img_m_internal)
        _igram_smooth = torch.nn.functional.conv2d(
            _igram[None, None].real, k_coh, padding=coh_win // 2
        ) + 1j * torch.nn.functional.conv2d(
            _igram[None, None].imag, k_coh, padding=coh_win // 2
        )
        _pow_s = torch.nn.functional.conv2d(
            (torch.abs(img_s) ** 2)[None, None], k_coh, padding=coh_win // 2
        )
        _pow_m = torch.nn.functional.conv2d(
            (torch.abs(img_m_internal) ** 2)[None, None], k_coh, padding=coh_win // 2
        )
        spatial_coh = (torch.abs(_igram_smooth) / (
            torch.sqrt(_pow_s * _pow_m) + eps)).squeeze()
        spatial_coh = torch.nan_to_num(
            spatial_coh, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        del _igram, _igram_smooth, _pow_s, _pow_m

        window_width = int(window_width * window_exp)
        if window_width < min_window:
            if verbose:
                print("Window width below minimum")
            break

    return img_s, pos_s_new


def insar_rme_blocksvd(
    data_s: Tensor,
    pos_s: Tensor,
    img_m: Tensor,
    fc: float,
    r_res: float,
    grid_polar: "PolarGrid | dict",
    n_az_blocks: int = 32,
    n_r_blocks: int = 16,
    d0: float = 0.0,
    data_fmod: float = 0.0,
    row_weight: str = "coherence",
    align_blocks: bool = True,
    aperture_mask: bool = True,
    aperture_pad: float = 1.0,
    phi_lowpass: int = 0,
    interp_method: str = "linear",
    pixel_chunk: int = 256,
    spatial_coherence: Tensor | None = None,
    return_alpha: bool = False,
    return_magnitude: bool = False,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Closed-form block-coherent InSAR residual motion error estimation.

    Estimates per-slave-sweep range-direction position errors by combining
    per-block coherent statistics computed against an existing master image.

    Algorithm
    ---------
    The image is tiled into `n_r_blocks x n_az_blocks` non-overlapping
    blocks. For each block `b`:

    1. The block's pixel positions are passed as targets to
       `gpga_backprojection_2d_core` on the slave data, producing
       ``B[k, m] = data_s[m, r_idx(pix_k, m)] * exp(-j k R(pix_k, m))``,
       i.e. the per-sweep slave backprojection footprint at the block's
       pixels. Master is never demodulated to targets.
    2. The per-block per-sweep statistic is the inner product against the
       existing master image patch:
         ``alpha^{(b)}_m = Sum_k conj(img_m[pix_k]) * B[k, m]``.
       This is the closed-form maximizer (over per-sweep phase) of the
       block's coherent inner product, with the master image acting as the
       optimal pixel weighting.

    The per-block alpha-vectors are then combined into a global per-sweep phase.
    Each block has an unknown complex constant ``c_b`` (per-block baseline
    + topo phase). With ``align_blocks=True``, each block's
    alpha is divided by its mean phase to absorb ``c_b``, and the aligned alphas
    are coherently summed:

        alpha_combined_m = Sum_b alpha^{(b)}_m / e^{j angle(Sum_m alpha^{(b)}_m)}
        phi_m = +angle(alpha_combined_m) (data is corrected by exp(-j phi))

    The corrected slave position is ``pos_s + [dr_m, 0, 0]`` (range-only,
    in the slave's local frame where the +X axis is the radar look
    direction).

    Parameters
    ----------
    data_s : Tensor [nsweeps_s, nsamples]
        Range-compressed slave data.
    pos_s : Tensor [nsweeps_s, 3]
        Slave platform positions in the slave's local frame.
    img_m : Tensor [nr, ntheta] or [1, nr, ntheta]
        Master image already formed on ``grid_polar``. Must be on the
        same grid as the slave will be reformed on; no master/slave image
        interpolation is performed.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
    grid_polar : PolarGrid or dict
        Polar grid definition.
    n_az_blocks, n_r_blocks : int
        Image tiling. The total number of blocks is the product. Use enough
        azimuth blocks to give multi-block coverage of every sweep
        (a few times ``nsweeps_s / aperture_length``); a small number of
        range blocks (5–20) is enough for diversity. Block size should be
        small enough that topographic phase is approximately constant
        within a block (otherwise per-block coherent integration cancels).
    d0 : float
        Zero range correction.
    data_fmod : float
        Range modulation frequency applied to input data.
    row_weight : "coherence" | "power" | "uniform"
        Per-block row weight.
        "coherence" Weight be coherence. Recommended for real data.
        "power" divides by image power.
        "uniform" uses no weighting.
    align_blocks : bool
        If True, divide each block's alpha by its mean phase before combining.
        This absorbs the per-block complex constant ``c_b`` (baseline /
        sub-pixel position) and is essential when topographic phase varies
        across the image.
    aperture_mask : bool
        If True (recommended), zero out per-block alpha entries for slave
        sweeps outside the block's synthetic-aperture window.
    aperture_pad : float
        Multiplier on the polar grid's angular extent used by
        ``aperture_mask``. 1.0 uses the exact grid extent; >1 widens
        the per-block window.
    spatial_coherence : Tensor [nr, ntheta] or None
        Per-pixel coherence map (e.g. from
        `torchbp.ops.power_coherence_2d`) used to downweight
        decorrelated regions. The map is squared (coh**2 weighting) and
        multiplied into the master image patch before forming each
        block's alpha, so contributions from vegetation or shadow areas are
        suppressed pixel-by-pixel.
    phi_lowpass : int
        If > 0, lowpass-filter the recovered phase along the sweep axis
        with a Hamming window of this width (in samples). Use a value
        much smaller than ``nsweeps_s`` and at least a few times smaller
        than the expected RME bandwidth. 0 disables filtering.
    interp_method : str
        Interpolation method passed to ``gpga_backprojection_2d_core``.
    return_alpha : bool
        If True, also return the [nblocks, nsweeps_s] alpha matrix for
        diagnostics.
    verbose : bool
        Print progress.

    Returns
    -------
    pos_s_new : Tensor [nsweeps_s, 3]
        Corrected slave positions (only the range component is modified).
    phi : Tensor [nsweeps_s]
        Estimated per-sweep RME phase, zero-mean.
    alpha : Tensor [nblocks, nsweeps_s], optional
        Returned only when ``return_alpha=True``.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)
    c0 = 299792458.0
    device = data_s.device
    nsweeps_s = data_s.shape[0]

    if img_m.dim() == 3:
        img_m = img_m.squeeze(0)
    if img_m.shape[0] != nr or img_m.shape[1] != ntheta:
        raise ValueError(
            f"img_m shape {tuple(img_m.shape)} does not match "
            f"grid_polar ({nr}, {ntheta})"
        )

    if spatial_coherence is not None:
        if spatial_coherence.dim() > 2:
            spatial_coherence = spatial_coherence.squeeze()
        spatial_coherence = torch.nan_to_num(
            spatial_coherence, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0, 1).to(dtype=torch.float32, device=device)
        coh_weight = spatial_coherence ** 2
        img_m_eff = img_m * coh_weight.to(img_m.dtype)
    else:
        img_m_eff = img_m

    n_az_blocks = max(1, int(n_az_blocks))
    n_r_blocks = max(1, int(n_r_blocks))
    r_edges = torch.linspace(0, nr, n_r_blocks + 1).to(torch.long).tolist()
    az_edges = torch.linspace(0, ntheta, n_az_blocks + 1).to(torch.long).tolist()

    nblocks = n_r_blocks * n_az_blocks
    A = torch.zeros((nblocks, nsweeps_s), dtype=torch.complex64, device=device)
    block_power = torch.zeros(nblocks, device=device)
    pos_s_y = pos_s[:, 1]

    # Optional spatial coherence weighting (downweight decorrelated regions
    # like vegetation). Per-pixel weights are folded into the master image
    # patch when forming alpha, so each block's alpha automatically excludes
    # contributions from low-coherence pixels.

    for ib in range(n_r_blocks):
        ri0, ri1 = r_edges[ib], r_edges[ib + 1]
        if ri1 <= ri0:
            continue
        rc = r0 + dr * (
            torch.arange(ri0, ri1, device=device, dtype=torch.float32) + 0.5
        )
        for jb in range(n_az_blocks):
            ti0, ti1 = az_edges[jb], az_edges[jb + 1]
            if ti1 <= ti0:
                continue
            tc = theta0 + dtheta * (
                torch.arange(ti0, ti1, device=device, dtype=torch.float32) + 0.5
            )
            R, T = torch.meshgrid(rc, tc, indexing="ij")
            x = (R * torch.sqrt(torch.clamp(1.0 - T ** 2, min=0.0))).reshape(-1)
            y = (R * T).reshape(-1)
            z = torch.zeros_like(x)
            img_m_patch = img_m_eff[ri0:ri1, ti0:ti1].reshape(-1)
            kpix = x.shape[0]

            bidx = ib * n_az_blocks + jb
            p_b = torch.sum(torch.abs(img_m_patch) ** 2)
            block_power[bidx] = p_b

            # Pre-slice sweeps by the block's aperture window so the
            # CUDA kernel only visits sweeps that would survive the mask
            if aperture_mask:
                rc_mid = float(r0 + dr * 0.5 * (ri0 + ri1))
                tc_mid = float(theta0 + dtheta * 0.5 * (ti0 + ti1))
                rel_theta = (rc_mid * tc_mid - pos_s_y) / rc_mid
                lo = aperture_pad * theta0
                hi = aperture_pad * theta1
                mask = (rel_theta >= lo) & (rel_theta <= hi)
                sweep_idx = mask.nonzero(as_tuple=True)[0]
                if sweep_idx.numel() == 0:
                    continue
                data_s_b = data_s.index_select(0, sweep_idx)
                pos_s_b = pos_s.index_select(0, sweep_idx)
            else:
                sweep_idx = None
                data_s_b = data_s
                pos_s_b = pos_s

            nsweeps_b = data_s_b.shape[0]
            alpha_b = torch.zeros(
                nsweeps_b, dtype=torch.complex64, device=device
            )
            for k0 in range(0, kpix, pixel_chunk):
                k1 = min(k0 + pixel_chunk, kpix)
                target_pos = torch.stack(
                    [x[k0:k1], y[k0:k1], z[k0:k1]], dim=1
                )
                B = gpga_backprojection_2d_core(
                    target_pos, data_s_b, pos_s_b, fc, r_res, d0,
                    interp_method=interp_method, data_fmod=data_fmod,
                )
                alpha_b = alpha_b + torch.conj(img_m_patch[k0:k1]) @ B
                del B

            # Zero-padded sweeps contribute nothing to num/den, so
            # coherence on alpha_b matches alpha over the full range.
            if row_weight == "power":
                w_b = 1.0 / (torch.sqrt(p_b) + 1e-20)
            elif row_weight == "coherence":
                num = torch.abs(alpha_b.sum()) ** 2
                den = (alpha_b.abs() ** 2).sum() + 1e-30
                w_b = num / den
            else:
                w_b = 1.0

            if sweep_idx is not None:
                A[bidx].index_copy_(0, sweep_idx, alpha_b * w_b)
            else:
                A[bidx] = alpha_b * w_b

    if verbose:
        nz = int((block_power > 0).sum().item())
        print(f"InSAR RME block-SVD: {nblocks} blocks ({nz} non-empty), "
              f"|A| max={A.abs().max().item():.3e}")

    if align_blocks:
        block_phase = torch.angle(A.sum(dim=1, keepdim=True))
        A_aligned = A * torch.exp(-1j * block_phase)
    else:
        A_aligned = A

    # Coherent sum across blocks.
    v_complex = A_aligned.sum(dim=0).conj()
    v_mag = v_complex.abs()
    phi_wrapped = -torch.angle(v_complex)
    phi = phi_wrapped - phi_wrapped.mean()

    if phi_lowpass and phi_lowpass > 1:
        # Real-valued lowpass via Hamming window centered moving average,
        # applied to exp(-j*phi) to avoid wrap discontinuities
        w = torch.hamming_window(int(phi_lowpass), device=device,
                                 dtype=torch.float32)
        w = w / w.sum()
        cphi = torch.exp(1j * phi)
        cphi_padded = torch.nn.functional.pad(
            cphi[None, None],
            (int(phi_lowpass) // 2, int(phi_lowpass) - int(phi_lowpass) // 2 - 1),
            mode="replicate",
        )
        wcomplex = w.to(torch.complex64)[None, None]
        cphi_smooth = torch.nn.functional.conv1d(cphi_padded, wcomplex)[0, 0]
        phi = torch.angle(cphi_smooth)

    # Unwrap along the sweep axis
    phi_np = phi.detach().cpu().numpy()
    phi_np = np.unwrap(phi_np)
    phi = torch.from_numpy(phi_np).to(device=device, dtype=torch.float32)
    phi = phi - phi.mean()

    d_corr = phi * (c0 / (4.0 * torch.pi * fc))
    pos_s_new = pos_s.clone()
    pos_s_new[:, 0] = pos_s[:, 0] + d_corr

    if verbose:
        rms = torch.sqrt(torch.mean(phi ** 2)).item()
        print(f"  phi RMS={rms:.4f} rad, range correction RMS="
              f"{torch.sqrt(torch.mean(d_corr ** 2)).item() * 1000:.2f} mm")

    out = (pos_s_new, phi)
    if return_magnitude:
        out = out + (v_mag,)
    if return_alpha:
        out = out + (A_aligned,)
    return out


def insar_rme_blocksvd_strata(
    data_s: Tensor,
    pos_s: Tensor,
    img_m: Tensor,
    fc: float,
    r_res: float,
    grid_polar: "PolarGrid | dict",
    n_strata: int = 8,
    n_az_blocks_per_strata: int = 32,
    strata_spacing: str = "elevation",
    estimate_z: bool = False,
    d0: float = 0.0,
    data_fmod: float = 0.0,
    phi_lowpass: int = 0,
    delta_lowpass: int = 0,
    spatial_coherence: Tensor | None = None,
    interp_method: str = "linear",
    pixel_chunk: int = 256,
    return_phi_strata: bool = False,
    verbose: bool = False,
    altitude: float | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Stratified closed-form InSAR RME with optional XZ decomposition.

    Splits the polar grid into ``n_strata`` range bands, runs an
    independent `insar_rme_blocksvd` per band to obtain a
    high-SNR per-sweep slant-range error estimate ``Delta r_s(m)``, then per
    sweep solves a small (``K_strata x n_axes``) weighted linear system
    using the look vector from the slave platform to each strata
    centroid. The result is a per-sweep position correction (XZ or X depending
    on estimate_z).

    Same closed-form / no-master-GPGA / no-master-interpolation
    properties as `insar_rme_blocksvd`. Compared to the plain
    blocksvd:

    - X-only typically yields a more accurate X correction
      (per-strata blocksvd is better-conditioned and the LS averages
      across strata).
    - XZ can recover a real Z error when the geometry has
      enough elevation diversity (significant variation of ``sin(el)``
      across the range swath, i.e. low altitude / wide range, or
      tall platform / short range).

    Parameters
    ----------
    data_s, pos_s, img_m, fc, r_res, grid_polar
        Same as :func:`insar_rme_blocksvd`.
    n_strata : int
        Number of range strata. Should be >= 2 for XZ mode (more strata
        → more LS averaging, but each strata has fewer pixels).
    n_az_blocks_per_strata : int
        Number of azimuth blocks per strata, passed to the per-strata
        :func:`insar_rme_blocksvd` call.
    estimate_z : bool
        If True, also solve for Z position error. Requires ``n_strata >= 2`` and
        meaningful elevation-angle diversity across strata.
    d0, data_fmod, phi_lowpass, spatial_coherence, interp_method, pixel_chunk
        Forwarded to per-strata :func:`insar_rme_blocksvd`.
    delta_lowpass : int
        If > 0, Hamming-window lowpass after the LS. Useful to suppress LS noise
        on Z when its conditioning is marginal.
    return_phi_strata : bool
        If True, also return the [n_strata, nsweeps] per-strata phase
        matrix for diagnostics.
    verbose : bool
        Print per-strata progress.
    altitude : float or None
        Sensor altitude for slant-range grids (BP origin at sensor
        altitude, ``pos_s`` z ≈ 0). When set, uses slant-range geometry
        for elevation strata (``sin_el = H/r``) and look vectors.
        When None (default), altitude is inferred from ``pos_s[:, 2]``
        and ground-range geometry is used (``sin_el = H/sqrt(r**2+H**2)``).

    Returns
    -------
    pos_s_new : Tensor [nsweeps, 3]
        Corrected slave positions; X is always corrected, Z corrected
        only when ``estimate_z=True``.
    delta : Tensor [nsweeps, 3]
        Per-sweep XYZ correction added to ``pos_s``. Non-estimated
        axes are zero.
    phi_strata : Tensor [n_strata, nsweeps], optional
        Returned when ``return_phi_strata=True``.
    """
    r0_full, r1_full, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(
        grid_polar
    )
    c0 = 299792458.0
    device = data_s.device
    nsweeps_s = data_s.shape[0]
    n_axes = 1 + int(estimate_z)

    if img_m.dim() == 3:
        img_m = img_m.squeeze(0)
    if img_m.shape[0] != nr or img_m.shape[1] != ntheta:
        raise ValueError(
            f"img_m shape {tuple(img_m.shape)} does not match "
            f"grid_polar ({nr}, {ntheta})"
        )

    if n_strata < n_axes:
        raise ValueError(
            f"n_strata ({n_strata}) must be >= n_axes ({n_axes})"
        )

    if spatial_coherence is not None:
        sc = spatial_coherence.squeeze() if spatial_coherence.dim() > 2 else spatial_coherence
        if sc.shape[-2:] != (nr, ntheta):
            raise ValueError(
                f"spatial_coherence shape {tuple(sc.shape)} does not match "
                f"grid_polar ({nr}, {ntheta})"
            )

    # Strata edges in pixel units. "range" gives uniform Δr (legacy);
    # "elevation" gives uniform sin(el), spending more strata at near
    # range where the geometry varies fast and fewer at far range where
    # sin(el) is nearly constant. The latter minimizes within-strata
    # geometric error.
    if strata_spacing == "elevation":
        h = float(pos_s[:, 2].mean().item())
        h = max(abs(h), 1e-3)
        # Polar grid r is the ground-range coordinate of the target on
        # the z=0 plane. Slant range from a platform at altitude h is
        # sqrt(r^2 + h^2); look-down elevation = atan2(h, r).
        r0_eff = max(float(r0_full), 1e-3)
        r1_eff = max(float(r1_full), r0_eff + 1e-3)
        sin_el_lo = h / np.sqrt(r1_eff ** 2 + h ** 2)   # far range
        sin_el_hi = h / np.sqrt(r0_eff ** 2 + h ** 2)   # near range
        sin_edges = np.linspace(sin_el_lo, sin_el_hi, n_strata + 1)
        # Invert: sin(el) = h / sqrt(r^2 + h^2) => r = h * sqrt(1/s^2 - 1)
        r_edges_m = h * np.sqrt(np.clip(1.0 / sin_edges ** 2 - 1.0, 0, None))
        # Edges are now in decreasing order along r (sin_edges goes
        # low to high i.e. far to near). Reverse to get ascending r.
        r_edges_m = np.sort(r_edges_m)
        # Convert to pixel-row indices
        edges = np.clip(
            np.round((r_edges_m - r0_full) / dr).astype(int), 0, nr
        ).tolist()
        # Ensure first/last cover the full grid
        edges[0] = 0
        edges[-1] = nr
    elif strata_spacing == "range":
        edges = torch.linspace(0, nr, n_strata + 1).to(torch.long).tolist()
    else:
        raise ValueError(
            f"strata_spacing must be 'elevation' or 'range', got {strata_spacing!r}"
        )

    phi_per_strata = torch.zeros(
        (n_strata, nsweeps_s), dtype=torch.float32, device=device
    )
    dr_per_strata = torch.zeros_like(phi_per_strata)
    mag_per_strata = torch.zeros_like(phi_per_strata)
    coh_per_strata = torch.ones(n_strata, dtype=torch.float32, device=device)
    strata_rc = torch.zeros(n_strata, dtype=torch.float32, device=device)
    strata_valid = torch.zeros(n_strata, dtype=torch.bool, device=device)

    if verbose:
        print(f"InSAR RME stratified blocksvd: {n_strata} range strata, "
              f"estimate_z={estimate_z}")

    for s in range(n_strata):
        i0, i1 = int(edges[s]), int(edges[s + 1])
        if i1 <= i0:
            continue
        nr_s = i1 - i0
        r0_s = float(r0_full + dr * i0)
        r1_s = float(r0_full + dr * i1)
        rc = 0.5 * (r0_s + r1_s)

        gp_s = {
            "r": (r0_s, r1_s),
            "theta": (theta0, theta1),
            "nr": nr_s,
            "ntheta": ntheta,
        }
        coh_s = (
            spatial_coherence[i0:i1, :]
            if spatial_coherence is not None else None
        )
        if coh_s is not None:
            # Mean coherence squared in this strata; downweights strata dominated by
            # shadow / decorrelated pixels (e.g. far-range layover).
            coh_per_strata[s] = (coh_s ** 2).mean()

        _pos_s_new, phi_s, v_mag_s = insar_rme_blocksvd(
            data_s, pos_s, img_m[i0:i1, :], fc, r_res, gp_s,
            n_az_blocks=n_az_blocks_per_strata, n_r_blocks=1,
            d0=d0, data_fmod=data_fmod,
            row_weight="coherence", align_blocks=True,
            aperture_mask=True, aperture_pad=1.0,
            phi_lowpass=phi_lowpass,
            interp_method=interp_method,
            pixel_chunk=pixel_chunk,
            spatial_coherence=coh_s,
            return_magnitude=True,
            verbose=False,
        )
        phi_per_strata[s] = phi_s
        dr_per_strata[s] = phi_s * (c0 / (4.0 * torch.pi * fc))
        mag_per_strata[s] = v_mag_s
        strata_rc[s] = rc
        strata_valid[s] = True

        if verbose:
            print(f"  strata {s}: r=[{r0_s:.1f}, {r1_s:.1f}] m, "
                  f"Δr_rms={torch.sqrt(torch.mean(dr_per_strata[s] ** 2)).item() * 1000:.2f} mm, "
                  f"|v|={v_mag_s.mean().item():.2e}, "
                  f"coh²={coh_per_strata[s].item():.3f}")

    valid_idx = torch.where(strata_valid)[0]
    K = int(len(valid_idx))
    if K < n_axes:
        raise ValueError(
            f"Only {K} valid strata, need >= {n_axes}"
        )
    rc_active = strata_rc[valid_idx]

    # Per-sweep look vector from pos_s[m] to broadside strata centroid
    # (rc_s, 0, 0). Slant-range gradient dR/dpos = -look, so the
    # observed slant-range error and the platform shift are related by
    # delta r_s(m) = look_s(m) * dp(m).
    if altitude is not None:
        # Slant-range grid: rc is slant range, pos_s z ≈ 0.
        # Ground target at pixel range rc is at horizontal distance
        # sqrt(rc**2 - H**2) from sub-sensor point, vertical distance H.
        H = altitude
        x_ground = torch.sqrt(torch.clamp(rc_active ** 2 - H ** 2, min=1e-6))
        dxs = x_ground[None, :] - pos_s[:, 0:1]            # [nsweeps, K]
        dys = -pos_s[:, 1:2].expand(-1, K)
        dzs = torch.full_like(dxs, -H)
    else:
        # Ground-range grid: rc is horizontal distance, pos_s z = altitude.
        dxs = rc_active[None, :] - pos_s[:, 0:1]            # [nsweeps, K]
        dys = -pos_s[:, 1:2].expand(-1, K)
        dzs = -pos_s[:, 2:3].expand(-1, K)
    rg = torch.sqrt(dxs ** 2 + dys ** 2)
    rs = torch.sqrt(rg ** 2 + dzs ** 2) + 1e-9
    cos_el = rg / rs
    sin_el = dzs / rs
    cos_az = dxs / (rg + 1e-9)
    # sin_az unused (broadside centroid; LS doesn't estimate Y)

    cols = [cos_az * cos_el]
    if estimate_z:
        cols.append(sin_el)
    M = torch.stack(cols, dim=-1)                            # [nsweeps, K, n_axes]

    y = dr_per_strata[valid_idx].t().unsqueeze(-1)           # [nsweeps, K, 1]

    # Per-(strata, sweep) SNR weight from per-strata coherent magnitude.
    # Strata at far range have poorer SNR and contribute less to the LS.
    W = mag_per_strata[valid_idx].t()                        # [nsweeps, K]
    # Multiply by per-strata mean coherence (γ**2): downweights strata
    # dominated by shadowed / decorrelated regions even when their
    # coherent magnitude is non-trivial due to bright clutter.
    coh_w = coh_per_strata[valid_idx]                        # [K]
    W = W * coh_w[None, :]
    # Per-sweep normalization keeps the LS scale consistent
    W = W / (W.amax(dim=1, keepdim=True) + 1e-30)
    sqrtW = W.sqrt().unsqueeze(-1)                           # [nsweeps, K, 1]
    Mw = M * sqrtW
    yw = y * sqrtW

    if K == n_axes:
        # Closed-form when square; fall back to lstsq for ill-conditioned
        try:
            sol = torch.linalg.solve(Mw, yw)
        except Exception:
            sol = torch.linalg.lstsq(Mw, yw).solution
    else:
        sol = torch.linalg.lstsq(Mw, yw).solution
    sol = sol.squeeze(-1)                                    # [nsweeps, n_axes]

    delta = torch.zeros((nsweeps_s, 3), dtype=torch.float32, device=device)
    delta[:, 0] = sol[:, 0]
    if estimate_z:
        delta[:, 2] = sol[:, 1]
    delta = delta - delta.mean(dim=0, keepdim=True)

    if delta_lowpass and delta_lowpass > 1:
        L = int(delta_lowpass)
        w = torch.hamming_window(L, device=device, dtype=torch.float32)
        w = w / w.sum()
        for ax in range(3):
            if delta[:, ax].abs().max() == 0:
                continue
            v = delta[:, ax]
            pad_l = L // 2
            pad_r = L - pad_l - 1
            vp = torch.nn.functional.pad(
                v[None, None], (pad_l, pad_r), mode="replicate"
            )
            delta[:, ax] = torch.nn.functional.conv1d(
                vp, w[None, None]
            )[0, 0]

    pos_s_new = pos_s + delta

    if verbose:
        rms = [f"X={torch.sqrt(torch.mean(delta[:, 0] ** 2)).item() * 1000:.2f}"]
        if estimate_z:
            rms.append(
                f"Z={torch.sqrt(torch.mean(delta[:, 2] ** 2)).item() * 1000:.2f}"
            )
        print(f"  position correction RMS (mm): " + ", ".join(rms))

    if return_phi_strata:
        return pos_s_new, delta, phi_per_strata
    return pos_s_new, delta


def _get_kwargs() -> dict:
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


def minimum_entropy_grad_autofocus(
    f,
    data: Tensor,
    data_time: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid: "PolarGrid | dict",
    wa: Tensor,
    tx_norm: Tensor = None,
    max_steps: float = 100,
    lr_max: float = 10000,
    d0: float = 0,
    pos_reg: float = 1,
    lr_reduce: float = 0.8,
    verbose: bool = True,
    convergence_limit: float = 0.01,
    max_step_limit: float = 0.25,
    grad_limit_quantile: float = 0.9,
    fixed_pos: int = 0,
    minimize_only: bool = False,
    data_fmod: float = 0
) -> tuple[Tensor, Tensor, Tensor, int]:
    """
    Minimum entropy autofocus using backpropagation optimization through image
    formation.

    Parameters
    ----------
    f : function
        Radar image generation function.
    data : Tensor
        Radar data.
    data_time : Tensor
        Recording time of each data sample.
    pos : Tensor
        Position at each data sample.
    fc : float
        RF frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    grid : PolarGrid or dict
        Grid definition. Correct definition depends on the radar image function.
    wa : Tensor
        Azimuth windowing function.
        Should be applied to data already, used for scaling gradient.
    tx_norm : Tensor
        Radar image is divided by this tensor before calculating entropy.
        If None no division is done.
    max_steps : int
        Maximum number of optimization steps.
    lr_max : float
        Maximum learning rate.
        Too large learning rate is scaled automatically.
    d0 : float
        Zero range correction.
    pos_reg : float
        Position regularization value.
    lr_reduce : float
        Learning rate is multiplied with this value if new entropy is larger than previously.
    verbose : bool
        Print progress during optimization.
    convergence_limit : float
        If maximum position change is below this value stop optimization.
        Units in wavelengths.
    max_step_limit : float
        Maximum step size in wavelengths.
    grad_limit_quantile : float
        Quantile used for maximum step size calculation.
        0 to 1 range.
    fixed_pos : int
        First `fixed_pos` positions are kept fixed and are not optimized.
    minimize_only : bool
        Reject steps that would increase image entropy.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    sar_img : Tensor
        Optimized radar image.
    origin : Tensor
        Mean of position tensor.
    pos : Tensor
        Platform position.
    step : int
        Number of steps.
    """
    dev = data.device
    t = data_time.unsqueeze(1)
    dt = torch.diff(t, dim=0, prepend=t[0].unsqueeze(0))
    dt[0] = dt[1]
    vopt = torch.diff(pos, dim=0, prepend=pos[0].unsqueeze(0)) / dt
    pos_mean = torch.mean(pos, dim=0)

    if fixed_pos > 0:
        v_fixed = vopt[:fixed_pos].detach().clone()

    pos_orig = pos.clone()
    vopt.requires_grad = True

    wl = 3e8 / fc
    lr = lr_max

    opt = torch.optim.SGD([vopt], momentum=0, lr=1)

    def lr_sch(epoch):
        p = int(0.75 * max_steps)
        if epoch > p:
            a = -lr / (max_steps + 1 - p)
            b = lr * max_steps / (max_steps + 1 - p)
            return a * epoch + b
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_sch)

    last_entr = None
    v_prev = vopt.detach().clone()

    try:
        for step in range(max_steps):
            if fixed_pos > 0:
                v = torch.cat([v_fixed, vopt[fixed_pos:]], dim=0)
            else:
                v = vopt
            pos = torch.cumsum(v * dt, 0)
            pos = pos - torch.mean(pos, dim=0) + pos_mean
            # pos_d2 = torch.diff(pos, n=2, dim=0) / dt

            pos_loss = pos_reg * torch.mean(torch.square(pos - pos_orig))
            # acc_loss = acc_reg * torch.mean(torch.square(pos_d2[1:]))

            origin = torch.mean(pos, axis=0, keepdim=True)
            origin[:,2] = 0
            pos_centered = pos - origin

            sar_img = f(data, grid, fc, r_res, pos_centered, d0, data_fmod=data_fmod).squeeze()
            if tx_norm is not None:
                entr = entropy(sar_img / tx_norm)
            else:
                entr = entropy(sar_img)
            loss = entr + pos_loss  # + acc_loss
            if last_entr is not None and entr > last_entr:
                lr *= lr_reduce
                if minimize_only:
                    vopt.data = v_prev.data
                    continue
            last_entr = entr
            v_prev = vopt.detach().clone()
            if step < max_steps - 1:
                loss.backward()
                l = scheduler.get_last_lr()[0]
                with torch.no_grad():
                    vopt.grad /= wa[:, None]
                    g = vopt.grad.detach()
                    gpos = torch.cumsum(l * g * dt, 0)
                    dp = torch.abs(gpos[:, 0])
                    maxd = torch.quantile(dp, grad_limit_quantile)
                    dp = torch.linalg.vector_norm(gpos, dim=-1)
                    maxd2 = torch.quantile(dp, grad_limit_quantile)
                    s = max_step_limit * wl / (1e-5 + maxd)
                    if maxd < convergence_limit * wl:
                        if verbose:
                            print("Optimization converged")
                        break
                    if s < 1:
                        vopt.grad *= s
                        lr *= s.item()
                opt.step()
                opt.zero_grad()
                scheduler.step()
            if verbose:
                print(
                    step,
                    "Entropy",
                    entr.detach().cpu().numpy(),
                    "loss",
                    loss.detach().cpu().numpy(),
                )
    except KeyboardInterrupt:
        print("Interrupted")
        pass

    return sar_img.detach(), origin.detach(), pos.detach(), step


def bp_polar_grad_minimum_entropy(
    data: Tensor,
    data_time: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid: "PolarGrid | dict",
    wa: Tensor,
    tx_norm: Tensor = None,
    max_steps: float = 100,
    lr_max: float = 10000,
    d0: float = 0,
    pos_reg: float = 1,
    lr_reduce: float = 0.8,
    verbose: bool = True,
    convergence_limit: float = 0.01,
    max_step_limit: float = 0.25,
    grad_limit_quantile: float = 0.9,
    fixed_pos: int = 0,
    data_fmod: float = 0
) -> tuple[Tensor, Tensor, Tensor, int]:
    """
    Minimum entropy autofocus optimization autofocus.

    Wrapper around `minimum_entropy_autofocus`.

    Parameters
    ----------
    data : Tensor
        Radar data.
    data_time : Tensor
        Recording time of each data sample.
    pos : Tensor
        Position at each data sample.
    fc : float
        RF frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    grid : PolarGrid or dict
        Grid definition. Correct definition depends on the radar image function.
    wa : Tensor
        Azimuth windowing function.
        Should be applied to data already, used for scaling gradient.
    tx_norm : Tensor
        Radar image is divided by this tensor before calculating entropy.
        If None no division is done.
    max_steps : int
        Maximum number of optimization steps.
    lr_max : float
        Maximum learning rate.
        Too large learning rate is scaled automatically.
    d0 : float
        Zero range correction.
    pos_reg : float
        Position regularization value.
    lr_reduce : float
        Learning rate is multiplied with this value if new entropy is larger than previously.
    verbose : bool
        Print progress during optimization.
    convergence_limit : float
        If maximum position change is below this value stop optimization.
        Units in wavelengths.
    max_step_limit : float
        Maximum step size in wavelengths.
    grad_limit_quantile : float
        Quantile used for maximum step size calculation.
        0 to 1 range.
    fixed_pos : int
        First `fixed_pos` positions are kept fixed and are not optimized.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    sar_img : Tensor
        Optimized radar image.
    origin : Tensor
        Mean of position tensor.
    pos : Tensor
        Platform position.
    step : int
        Number of steps.
    """
    kw = _get_kwargs()
    return minimum_entropy_grad_autofocus(backprojection_polar_2d, **kw)


def bp_cart_grad_minimum_entropy(
    data: Tensor,
    data_time: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid: "PolarGrid | dict",
    wa: Tensor,
    tx_norm: Tensor = None,
    max_steps: float = 100,
    lr_max: float = 10000,
    d0: float = 0,
    pos_reg: float = 1,
    lr_reduce: float = 0.8,
    verbose: bool = True,
    convergence_limit: float = 0.01,
    max_step_limit: float = 0.25,
    grad_limit_quantile: float = 0.9,
    fixed_pos: int = 0,
    data_fmod: float = 0
) -> tuple[Tensor, Tensor, Tensor, int]:
    """
    Minimum entropy autofocus optimization autofocus.

    Wrapper around `minimum_entropy_autofocus`.

    Parameters
    ----------
    data : Tensor
        Radar data.
    data_time : Tensor
        Recording time of each data sample.
    pos : Tensor
        Position at each data sample.
    fc : float
        RF frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    grid : PolarGrid or dict
        Grid definition. Correct definition depends on the radar image function.
    wa : Tensor
        Azimuth windowing function.
        Should be applied to data already, used for scaling gradient.
    tx_norm : Tensor
        Radar image is divided by this tensor before calculating entropy.
        If None no division is done.
    max_steps : int
        Maximum number of optimization steps.
    lr_max : float
        Maximum learning rate.
        Too large learning rate is scaled automatically.
    d0 : float
        Zero range correction.
    pos_reg : float
        Position regularization value.
    lr_reduce : float
        Learning rate is multiplied with this value if new entropy is larger than previously.
    verbose : bool
        Print progress during optimization.
    convergence_limit : float
        If maximum position change is below this value stop optimization.
        Units in wavelengths.
    max_step_limit : float
        Maximum step size in wavelengths.
    grad_limit_quantile : float
        Quantile used for maximum step size calculation.
        0 to 1 range.
    fixed_pos : int
        First `fixed_pos` positions are kept fixed and are not optimized.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    sar_img : Tensor
        Optimized radar image.
    origin : Tensor
        Mean of position tensor.
    pos : Tensor
        Platform position.
    step : int
        Number of steps.
    """
    kw = _get_kwargs()
    return minimum_entropy_grad_autofocus(backprojection_cart_2d, **kw)
