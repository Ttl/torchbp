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
    blocksvd_alpha,
    ffbp,
)
from .ops import entropy
from .util import (
    unwrap,
    unwrap_ref,
    detrend,
    conv_lowpass_filter,
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
        lp_w = fft_lowpass_filter_precalculate_window(
            pos_new.shape[0], window_width, img.device, lowpass_window, fast_len=True
        )
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
                target_data, window=lp_w, window_width=window_width
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
        If TX antenna equals RX antenna, then this should be just antenna gain.
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
    spatial_coherence: Tensor | None = None,
    return_alpha: bool = False,
    return_magnitude: bool = False,
    return_complex: bool = False,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Closed-form block-coherent InSAR residual motion error estimation.

    Estimates per-slave-sweep range-direction position errors by combining
    per-block coherent statistics computed against an existing master image.

    Notes
    -----
    The image is tiled into `n_r_blocks x n_az_blocks` non-overlapping
    blocks. For each block `b`:

    1. The per-block per-sweep statistic is the inner product of the
       slave data's backprojection footprint at the block's pixels
       against the existing master image patch:
       ``alpha^{(b)}_m = Sum_k conj(img_m[pix_k]) * data_s[m, r_idx(pix_k, m)]
       * exp(j k R(pix_k, m))``, computed by the fused
       :func:`torchbp.ops.blocksvd_alpha` kernel (master is never
       demodulated to targets and the per-pixel footprint matrix is
       never materialized).
    2. This is the closed-form maximizer (over per-sweep phase) of the
       block's coherent inner product, with the master image acting as the
       optimal pixel weighting.

    The per-block alpha-vectors are then combined into a global per-sweep phase.
    Each block has an unknown complex constant ``c_b`` (per-block baseline
    + topo phase). With ``align_blocks=True``, each block's
    alpha is divided by its mean phase to absorb ``c_b``, and the aligned alphas
    are coherently summed::

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
    return_alpha : bool
        If True, also return the [nblocks, nsweeps_s] alpha matrix for
        diagnostics.
    return_complex : bool
        If True, also return the raw per-sweep coherent sum ``v`` before
        any lowpass filtering or unwrapping. Used by
        :func:`insar_rme_blocksvd_strata` to do its own phase
        post-processing.
    verbose : bool
        Print progress.

    Returns
    -------
    pos_s_new : Tensor [nsweeps_s, 3]
        Corrected slave positions (only the range component is modified).
    phi : Tensor [nsweeps_s]
        Estimated per-sweep RME phase, zero-mean.
    v_mag : Tensor [nsweeps_s], optional
        Returned only when ``return_magnitude=True``.
    alpha : Tensor [nblocks, nsweeps_s], optional
        Returned only when ``return_alpha=True``.
    v_complex : Tensor [nsweeps_s], optional
        Returned only when ``return_complex=True``. The RME phase is
        ``-angle(v_complex)`` (before mean removal and unwrapping).
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
    block_power = torch.zeros(nblocks, device=device)
    pos_s_y = pos_s[:, 1]

    # Optional spatial coherence weighting (downweight decorrelated regions
    # like vegetation). Per-pixel weights are folded into the master image
    # patch when forming alpha, so each block's alpha automatically excludes
    # contributions from low-coherence pixels.

    # Per-block pixel rectangles and aperture sweep windows for the fused
    # kernel. The masked sweep set is an index interval when the
    # along-track positions are monotonic; the kernel gets its hull
    # [lo, hi) and the exact boolean mask is re-applied to alpha after,
    # so non-monotonic tracks stay correct. Degenerate or fully masked
    # blocks keep lo == hi == 0 and produce zero alpha rows.
    blocks = torch.zeros((nblocks, 6), dtype=torch.int32)
    if aperture_mask:
        mask = torch.zeros(
            (nblocks, nsweeps_s), dtype=torch.bool, device=device
        )
    for ib in range(n_r_blocks):
        ri0, ri1 = r_edges[ib], r_edges[ib + 1]
        for jb in range(n_az_blocks):
            ti0, ti1 = az_edges[jb], az_edges[jb + 1]
            if ri1 <= ri0 or ti1 <= ti0:
                continue
            bidx = ib * n_az_blocks + jb
            block_power[bidx] = torch.sum(
                torch.abs(img_m_eff[ri0:ri1, ti0:ti1]) ** 2
            )
            if aperture_mask:
                rc_mid = float(r0 + dr * 0.5 * (ri0 + ri1))
                tc_mid = float(theta0 + dtheta * 0.5 * (ti0 + ti1))
                rel_theta = (rc_mid * tc_mid - pos_s_y) / rc_mid
                lo = aperture_pad * theta0
                hi = aperture_pad * theta1
                mask_b = (rel_theta >= lo) & (rel_theta <= hi)
                sweep_idx = mask_b.nonzero(as_tuple=True)[0]
                if sweep_idx.numel() == 0:
                    continue
                mask[bidx] = mask_b
                s_lo = int(sweep_idx[0])
                s_hi = int(sweep_idx[-1]) + 1
            else:
                s_lo, s_hi = 0, nsweeps_s
            blocks[bidx] = torch.tensor(
                [ri0, ri1, ti0, ti1, s_lo, s_hi], dtype=torch.int32
            )

    A_raw = blocksvd_alpha(
        img_m_eff.to(torch.complex64), data_s, pos_s, blocks.to(device),
        fc, r_res, r0, dr, theta0, dtheta, d0=d0, data_fmod=data_fmod,
    )
    if aperture_mask:
        A_raw = A_raw * mask

    # Zero-padded sweeps contribute nothing to num/den, so the row
    # weights match the values over each block's aperture window only.
    if row_weight == "power":
        w_b = 1.0 / (torch.sqrt(block_power) + 1e-20)
    elif row_weight == "coherence":
        num = torch.abs(A_raw.sum(dim=1)) ** 2
        den = (A_raw.abs() ** 2).sum(dim=1) + 1e-30
        w_b = num / den
    else:
        w_b = torch.ones(nblocks, device=device)
    A = A_raw * w_b[:, None]

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
        # Lowpass applied to exp(j*phi) to avoid wrap discontinuities
        phi = torch.angle(
            conv_lowpass_filter(torch.exp(1j * phi), phi_lowpass)
        )

    # Unwrap along the sweep axis
    phi = unwrap(phi)
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
    if return_complex:
        out = out + (v_complex,)
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
    z_lowpass: int | None = None,
    ls_reg: float = 0.3,
    robust_iters: int = 2,
    spatial_coherence: Tensor | None = None,
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

    The per-sweep system is solved as a ridge-regularized weighted least
    squares with optional Huber reweighting across strata. The X and Z
    columns of the system are strongly anti-correlated (both ``cos(el)``
    and ``sin(el)`` are monotonic in range), so without regularization
    differential noise between strata is amplified into large
    anti-correlated X/Z errors. The ridge shrinks the solution toward
    zero where all strata have low SNR, and the Huber pass rejects
    strata whose residuals are inconsistent with the rest (layover,
    decorrelated region).

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
    d0, data_fmod, spatial_coherence
        Forwarded to per-strata :func:`insar_rme_blocksvd`.
    phi_lowpass : int
        If > 1, lowpass the per-strata phase with a Hamming window of
        this width (sweeps) and use the result as the unwrapping
        reference only: the full-band phase is kept as
        ``unwrap(lowpass(phi)) + wrap(phi - lowpass(phi))``. This makes
        the unwrap robust to per-sweep phase noise without discarding
        high-frequency RME content. If 0, the raw phase is unwrapped
        directly, which is fragile at low SNR.
    delta_lowpass : int
        If > 0, Hamming-window lowpass after the LS. Useful to suppress LS noise
        on Z when its conditioning is marginal.
    z_lowpass : int or None
        If > 1, additional Hamming-window lowpass applied to the Z
        correction only. Z is observed through the band-differential of
        the strata phases, which is much noisier than their common mode,
        so it tolerates a tighter bandwidth than X. None (default) uses
        ``phi_lowpass``; 0 disables.
    ls_reg : float
        Ridge regularization of the per-sweep least squares, relative to
        the median per-(strata, sweep) weight. Shrinks the correction
        toward zero where all strata have low SNR. Set to 0 to disable
        (weights then revert to per-sweep normalization).
    robust_iters : int
        Number of Huber IRLS reweighting passes over the per-sweep LS.
        Strata whose weighted residuals exceed 1.345 times the global
        MAD scale are downweighted proportionally, rejecting
        decorrelated or layover-dominated strata. 0 disables.
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

    # Strata edges in pixel units. "range" gives uniform in ground range.
    # "elevation" gives uniform sin(el), spending more strata at near
    # range where the geometry varies fast and fewer at far range where
    # sin(el) is nearly constant.
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
            # Power-weighted coherence squared in this strata: bright
            # coherent scatterers dominate alpha, so weight the
            # coherence by image power instead of taking a plain mean,
            # which would underweight strata that are mostly dark
            # decorrelated pixels but contain strong coherent targets.
            p_img = img_m[i0:i1, :].abs() ** 2
            coh_per_strata[s] = (
                (p_img * coh_s ** 2).sum() / (p_img.sum() + 1e-30)
            )

        _pos_s_new, _phi_unused, v_s = insar_rme_blocksvd(
            data_s, pos_s, img_m[i0:i1, :], fc, r_res, gp_s,
            n_az_blocks=n_az_blocks_per_strata, n_r_blocks=1,
            d0=d0, data_fmod=data_fmod,
            row_weight="coherence", align_blocks=True,
            aperture_mask=True, aperture_pad=1.0,
            phi_lowpass=0,
            spatial_coherence=coh_s,
            return_complex=True,
            verbose=False,
        )
        # Full-band phase with lowpass-referenced unwrap: the lowpassed
        # phase picks the 2*pi branch, the wrapped residual keeps the
        # high-frequency content that a plain lowpass would discard.
        phi_raw = -torch.angle(v_s)
        phi_raw = phi_raw - phi_raw.mean()
        if phi_lowpass and phi_lowpass > 1:
            phi_ref = torch.angle(
                conv_lowpass_filter(torch.exp(1j * phi_raw), phi_lowpass)
            )
            phi_ref = unwrap(phi_ref)
            phi_s = phi_ref + torch.angle(
                torch.exp(1j * (phi_raw - phi_ref))
            )
        else:
            phi_s = unwrap(phi_raw)
        phi_s = phi_s - phi_s.mean()
        phi_per_strata[s] = phi_s
        dr_per_strata[s] = phi_s * (c0 / (4.0 * torch.pi * fc))
        mag_per_strata[s] = v_s.abs()
        strata_rc[s] = rc
        strata_valid[s] = True

        if verbose:
            print(f"  strata {s}: r=[{r0_s:.1f}, {r1_s:.1f}] m, "
                  f"Δr_rms={torch.sqrt(torch.mean(dr_per_strata[s] ** 2)).item() * 1000:.2f} mm, "
                  f"|v|={mag_per_strata[s].mean().item():.2e}, "
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
    # Multiply by per-strata power-weighted coherence (γ**2): downweights
    # strata dominated by shadowed / decorrelated regions even when their
    # coherent magnitude is non-trivial due to bright clutter.
    coh_w = coh_per_strata[valid_idx]                        # [K]
    W = W * coh_w[None, :]
    if ls_reg > 0:
        # Global normalization keeps absolute SNR information so the
        # ridge shrinks genuinely low-SNR sweeps toward zero instead of
        # amplifying noise through the ill-conditioned X/Z inverse.
        W = W / (W.median() + 1e-30)
    else:
        # Per-sweep normalization keeps the LS scale consistent
        W = W / (W.amax(dim=1, keepdim=True) + 1e-30)
    sqrtW = W.sqrt().unsqueeze(-1)                           # [nsweeps, K, 1]

    if ls_reg > 0:
        reg_rows = ls_reg * torch.eye(n_axes, device=device).unsqueeze(
            0
        ).expand(nsweeps_s, -1, -1)
        reg_rhs = torch.zeros((nsweeps_s, n_axes, 1), device=device)

    sqrtW_cur = sqrtW
    for it in range(max(0, int(robust_iters)) + 1):
        Mw = M * sqrtW_cur
        yw = y * sqrtW_cur
        if ls_reg > 0:
            Mw = torch.cat([Mw, reg_rows], dim=1)
            yw = torch.cat([yw, reg_rhs], dim=1)
        sol = torch.linalg.lstsq(Mw, yw).solution            # [nsweeps, n_axes, 1]
        if it >= robust_iters:
            break
        # Huber IRLS on the weighted residuals: strata inconsistent
        # with the per-sweep solution (decorrelated regions, layover)
        # get their weight reduced proportionally to the excess over
        # 1.345x the global MAD scale.
        u = ((y - M @ sol) * sqrtW).squeeze(-1).abs()        # [nsweeps, K]
        scale = 1.4826 * u.median() + 1e-30
        w_h = torch.clamp(1.345 * scale / (u + 1e-30), max=1.0)
        sqrtW_cur = sqrtW * w_h.sqrt().unsqueeze(-1)
    sol = sol.squeeze(-1)                                    # [nsweeps, n_axes]

    delta = torch.zeros((nsweeps_s, 3), dtype=torch.float32, device=device)
    delta[:, 0] = sol[:, 0]
    if estimate_z:
        delta[:, 2] = sol[:, 1]
    delta = delta - delta.mean(dim=0, keepdim=True)

    if delta_lowpass and delta_lowpass > 1:
        for ax in range(3):
            if delta[:, ax].abs().max() == 0:
                continue
            delta[:, ax] = conv_lowpass_filter(delta[:, ax], delta_lowpass)

    z_lp = phi_lowpass if z_lowpass is None else z_lowpass
    if estimate_z and z_lp and z_lp > 1:
        delta[:, 2] = conv_lowpass_filter(delta[:, 2], z_lp)

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


def _interp1_linear(xq: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """
    Linear interpolation of `y` sampled at ascending `x`, evaluated at `xq`.
    `y` shape [..., len(x)]. Values outside `x` range are clamped to the
    edge values.
    """
    idx = torch.searchsorted(x.contiguous(), xq.contiguous()).clamp(
        1, x.shape[0] - 1
    )
    x0 = x[idx - 1]
    x1 = x[idx]
    t = ((xq - x0) / (x1 - x0)).clamp(0, 1)
    return y[..., idx - 1] + t * (y[..., idx] - y[..., idx - 1])


def insar_rme_multisquint(
    img_m: Tensor,
    img_s: Tensor,
    pos_s: Tensor,
    fc: float,
    grid_polar: "PolarGrid | dict",
    n_looks: int = 32,
    look_width: float = 2.0,
    n_r_bands: int = 4,
    band_spacing: str = "elevation",
    estimate_z: bool = False,
    n_z_basis: int = 8,
    aperture_mask: bool = True,
    aperture_pad: float = 1.0,
    patch_theta: int | None = None,
    spatial_coherence: Tensor | None = None,
    data_s: Tensor | None = None,
    r_res: float | None = None,
    max_iters: int = 1,
    d0: float = 0.0,
    data_fmod: float = 0.0,
    dealias: bool = False,
    alias_fmod: float = 0.0,
    delta_lowpass: int = 0,
    ls_reg: float = 0.1,
    remove_trend: bool = True,
    altitude: float | None = None,
    return_phi_bands: bool = False,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Multisquint InSAR residual motion error estimation [1]_ [2]_.

    Image-domain method: in a polar-grid backprojection image the azimuth
    (theta) spectrum maps linearly to along-track platform position,
    ``f_theta = -(2/wl) * cos(el) * y``, so squinted sub-looks are bandpass
    windows of the theta-axis FFT. For each range band and each adjacent
    pair of looks, the double difference of the per-look interferograms

        ``D_j = (L_m_j * conj(L_s_j)) * conj(L_m_(j-1) * conj(L_s_(j-1)))``

    cancels the look-independent topographic / baseline phase, leaving the
    difference of the slave RME phase between the two look positions.
    Coherently averaging ``D_j`` over the band and integrating the steps
    gives the per-band slant-range error profile along the track; a
    per-sweep weighted least squares across range bands separates X (and
    optionally Z) position error using elevation angle diversity, as in
    :func:`insar_rme_blocksvd_strata`.

    Because the observable is a phase *difference* between adjacent looks,
    no phase unwrapping is needed as long as the RME phase changes by less
    than pi between look centers. Cost is a few FFTs per look — no
    backprojection of slave data is required (unless ``max_iters > 1``).

    The master image is assumed to be focused (autofocused first) and both
    images must be formed on the same polar grid in the same coordinate
    frame. Constant slant-range error is unobservable (absorbed into the
    baseline estimate); the returned correction is zero-mean per axis.
    Along-track (Y) error is not estimated; its first-order effect on the
    look phases averages out over a symmetric azimuth extent.

    Parameters
    ----------
    img_m : Tensor
        Master image [nr, ntheta] formed on ``grid_polar``. Assumed free of
        motion error.
    img_s : Tensor
        Slave image [nr, ntheta] formed on the same grid with positions
        ``pos_s``.
    pos_s : Tensor
        Slave platform positions [nsweeps, 3] in the backprojection frame.
        Along-track (Y) coordinates must be ascending.
    fc : float
        RF center frequency in Hz.
    grid_polar : PolarGrid or dict
        Polar grid definition.
    n_looks : int
        Number of squinted looks across the along-track extent. The
        recovered error profile has ``n_looks`` samples along the track,
        interpolated to sweeps; RME components faster than
        ``nsweeps / (2 * n_looks)`` cycles per aperture are not resolved.
    look_width : float
        Look bandwidth in units of look spacing. Wider looks have more
        spectral bins (less phase noise) but lower along-track
        resolution. The double differences are taken between looks
        ``ceil(look_width)`` apart so the differenced windows share no
        spectral bins; shared speckle content would bias the phase
        steps toward zero.
    n_r_bands : int
        Number of range bands. Must be >= 2 for ``estimate_z``.
    band_spacing : "elevation" | "range"
        Band edge spacing, see :func:`insar_rme_blocksvd_strata`.
    estimate_z : bool
        Also solve for Z position error. Requires elevation-angle
        diversity across the range swath.
    n_z_basis : int
        Number of linear-interpolation basis knots for the Z error
        profile. Z is observed only through the band-differential of the
        look phases, which is noisier than their common mode; a coarse
        basis trades Z bandwidth for stability. Z components faster than
        ``n_z_basis / 2`` cycles per aperture are not resolved.
    aperture_mask : bool
        Zero out pixels outside a look's angular aperture (using the grid
        theta extent as the beam proxy) when averaging the double
        differences.
    aperture_pad : float
        Multiplier on the grid theta extent used by ``aperture_mask``.
    patch_theta : int or None
        Azimuth multilook patch length (pixels) applied to the per-look
        interferograms before double-differencing. None (default)
        chooses about one look azimuth-resolution cell. Must be small
        enough that topographic phase is approximately constant over the
        patch along theta.
    spatial_coherence : Tensor [nr, ntheta] or None
        Optional per-pixel coherence map; squared and used as a pixel
        weight in the double-difference averages.
    data_s : Tensor or None
        Range-compressed slave data [nsweeps, samples]. Only needed when
        ``max_iters > 1`` to reform the slave image between iterations.
    r_res : float or None
        Range bin resolution of ``data_s``. Required when ``max_iters > 1``.
    max_iters : int
        Number of estimation iterations. Iterating reforms the slave image
        at the corrected positions, which sharpens the looks when the
        initial error is large.
    d0, data_fmod, dealias, alias_fmod
        Backprojection parameters used to reform the slave image between
        iterations. Must match how ``img_s`` was formed.
    delta_lowpass : int
        If > 0, Hamming-window lowpass of this width (sweeps) applied to
        the per-sweep correction.
    ls_reg : float
        Ridge regularization of the per-step least squares, relative to
        the median double-difference weight. Shrinks the correction
        toward zero where all bands have low SNR (track edges).
    remove_trend : bool
        Remove the weighted mean phase step per band before solving.
        This removes linear error trends (unobservable, absorbed into
        the baseline estimate) together with the deterministic
        baseline-induced ramp of the look phases, which is
        range-dependent and would otherwise leak into a slow X/Z drift.
    altitude : float or None
        Sensor altitude for slant-range grids, see
        :func:`insar_rme_blocksvd_strata`. None infers altitude from
        ``pos_s[:, 2]`` and uses ground-range geometry.
    return_phi_bands : bool
        Also return the per-band slant-range error phase interpolated to
        sweeps, shape [n_r_bands, nsweeps].
    verbose : bool
        Print progress.

    References
    ----------
    .. [1] P. Prats and J. J. Mallorqui, "Estimation of azimuth phase
        undulations with multisquint multitemporal coregistration," in IEEE
        Geoscience and Remote Sensing Letters, vol. 1, no. 4, pp. 268-271,
        Oct. 2004.
    .. [2] J. M. de Macedo, R. Scheiber and A. Moreira, "An Autofocus
        Approach for Residual Motion Errors With Application to Airborne
        Repeat-Pass SAR Interferometry," in IEEE Transactions on Geoscience
        and Remote Sensing, vol. 46, no. 10, pp. 3151-3162, Oct. 2008.

    Returns
    -------
    pos_s_new : Tensor [nsweeps, 3]
        Corrected slave positions; X always corrected, Z only when
        ``estimate_z=True``.
    delta : Tensor [nsweeps, 3]
        Total per-sweep XYZ correction added to ``pos_s``. Non-estimated
        axes are zero.
    phi_bands : Tensor [n_r_bands, nsweeps], optional
        Returned when ``return_phi_bands=True``. Last iteration's
        per-band RME phase.
    """
    r0_full, r1_full, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(
        grid_polar
    )
    c0 = 299792458.0
    wl = c0 / fc
    k = 4.0 * torch.pi / wl
    device = img_s.device
    nsweeps = pos_s.shape[0]
    n_axes = 1 + int(estimate_z)

    if img_m.dim() == 3:
        img_m = img_m.squeeze(0)
    if img_s.dim() == 3:
        img_s = img_s.squeeze(0)
    for name, im in (("img_m", img_m), ("img_s", img_s)):
        if im.shape[0] != nr or im.shape[1] != ntheta:
            raise ValueError(
                f"{name} shape {tuple(im.shape)} does not match "
                f"grid_polar ({nr}, {ntheta})"
            )
    if n_looks < 3:
        raise ValueError(f"n_looks ({n_looks}) must be >= 3")
    if n_r_bands < n_axes:
        raise ValueError(f"n_r_bands ({n_r_bands}) must be >= {n_axes}")
    if max_iters > 1 and (data_s is None or r_res is None):
        raise ValueError("data_s and r_res are required when max_iters > 1")

    if spatial_coherence is not None:
        sc = (
            spatial_coherence.squeeze()
            if spatial_coherence.dim() > 2
            else spatial_coherence
        )
        if sc.shape != (nr, ntheta):
            raise ValueError(
                f"spatial_coherence shape {tuple(sc.shape)} does not match "
                f"grid_polar ({nr}, {ntheta})"
            )
        coh_weight = (
            torch.nan_to_num(sc, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1) ** 2
        ).to(dtype=torch.float32, device=device)
    else:
        coh_weight = None

    if altitude is not None:
        h = float(altitude)
    else:
        h = float(pos_s[:, 2].mean().item())
    h = max(abs(h), 1e-3)

    # Range band edges in pixel rows
    if band_spacing == "elevation":
        r0_eff = max(float(r0_full), 1e-3)
        r1_eff = max(float(r1_full), r0_eff + 1e-3)
        if altitude is not None:
            # Slant-range grid: sin(el) = h / r
            sin_el_lo = h / r1_eff
            sin_el_hi = min(h / r0_eff, 1.0)
        else:
            sin_el_lo = h / np.sqrt(r1_eff ** 2 + h ** 2)
            sin_el_hi = h / np.sqrt(r0_eff ** 2 + h ** 2)
        sin_edges = np.linspace(sin_el_lo, sin_el_hi, n_r_bands + 1)
        if altitude is not None:
            r_edges_m = h / np.clip(sin_edges, 1e-6, None)
        else:
            r_edges_m = h * np.sqrt(np.clip(1.0 / sin_edges ** 2 - 1.0, 0, None))
        r_edges_m = np.sort(r_edges_m)
        edges = np.clip(
            np.round((r_edges_m - r0_full) / dr).astype(int), 0, nr
        ).tolist()
        edges[0] = 0
        edges[-1] = nr
    elif band_spacing == "range":
        edges = torch.linspace(0, nr, n_r_bands + 1).to(torch.long).tolist()
    else:
        raise ValueError(
            f"band_spacing must be 'elevation' or 'range', got {band_spacing!r}"
        )

    # Look centers along track. End looks are pulled in by half the look
    # bandwidth so their windows stay supported by the track; the profile
    # is extrapolated as constant beyond them.
    y_lo = float(pos_s[:, 1].min().item())
    y_hi = float(pos_s[:, 1].max().item())
    dy_look = (y_hi - y_lo) / (n_looks - 1)
    half_bw_y = 0.5 * look_width * dy_look
    y_looks = torch.linspace(
        y_lo + half_bw_y,
        y_hi - half_bw_y,
        n_looks,
        device=device,
        dtype=torch.float32,
    )
    dy_look = float(y_looks[1] - y_looks[0])
    half_bw_y = 0.5 * look_width * dy_look
    # Lag between differenced looks: smallest separation at which the
    # two spectral windows share no bins
    lag = max(1, int(np.ceil(look_width - 1e-6)))
    if n_looks < lag + 2:
        raise ValueError(
            f"n_looks ({n_looks}) must be >= ceil(look_width) + 2 ({lag + 2})"
        )

    # Theta-spectrum frequency axis, cycles per theta-unit
    f_grid = torch.fft.fftfreq(ntheta, d=dtheta, device=device).to(torch.float32)
    f_nyq = 0.5 / dtheta
    theta_axis = theta0 + dtheta * torch.arange(
        ntheta, device=device, dtype=torch.float32
    )

    pos_s_new = pos_s.clone()
    delta = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
    img_s_cur = img_s
    S_m = torch.fft.fft(img_m, dim=1)
    phi_bands_swp = None

    for iteration in range(max_iters):
        S_s = torch.fft.fft(img_s_cur, dim=1)

        # Per-row spectral mapping scale: the contribution from a sweep
        # at along-track position y lands at theta-frequency
        # f = -s(r) * y, with s varying across range. Using a per-row
        # scale (instead of a band-center constant) keeps the selected y
        # from drifting across rows, which would smear the look phases
        # at the track ends.
        r_rows = r0_full + dr * (
            torch.arange(nr, device=device, dtype=torch.float32) + 0.5
        )
        if altitude is not None:
            # Slant-range grid: r is slant range
            cos_el_rows = (
                torch.sqrt(torch.clamp(r_rows ** 2 - h ** 2, min=0.0)) / r_rows
            )
        else:
            cos_el_rows = r_rows / torch.sqrt(r_rows ** 2 + h ** 2)
        s_rows = (2.0 / wl) * cos_el_rows                        # [nr]
        if verbose and iteration == 0:
            f_need = float(s_rows.max()) * (
                float(y_looks.abs().max()) + half_bw_y
            )
            if f_need > f_nyq:
                print(
                    "  look band exceeds theta Nyquist at far range, "
                    "edge looks will be attenuated"
                )

        # Azimuth patch length for multilooking the per-look
        # interferograms before double-differencing. The raw per-pixel
        # product G_j * conj(G_(j-1)) is dominated by speckle terms
        # common to the two overlapping looks, which attenuates the
        # phase steps; summing G over patches of about one look
        # azimuth-resolution cell first recovers the full step (the
        # patch sum is the multilooked interferogram of the look pair).
        # Patch must stay small enough that topographic phase is
        # approximately constant within it along theta.
        if patch_theta is None:
            s_mean = float(s_rows.mean())
            L_patch = int(round(0.5 / (s_mean * half_bw_y * dtheta)))
            L_patch = max(4, min(L_patch, ntheta // 8))
        else:
            L_patch = max(1, int(patch_theta))
        npatch = ntheta // L_patch
        ntheta_p = npatch * L_patch

        # Double differences are taken between looks `lag` apart so the
        # two spectral windows share no bins. Shared speckle content
        # between overlapping looks would otherwise add a zero-phase
        # rail to the double difference (the master and slave
        # self-overlap terms are real positive and survive all
        # averaging), diluting every phase step by the shared-energy
        # fraction.
        ndiff = n_looks - lag
        dd_bands = torch.zeros(
            (n_r_bands, ndiff), dtype=torch.complex64, device=device
        )
        P_hist = [None] * n_looks
        for j in range(n_looks):
            xw = (f_grid[None, :] + s_rows[:, None] * float(y_looks[j])) / (
                s_rows[:, None] * half_bw_y
            )
            W = torch.where(
                xw.abs() < 1.0,
                0.5 * (1.0 + torch.cos(torch.pi * xw)),
                torch.zeros_like(xw),
            ).to(torch.complex64)                                # [nr, ntheta]
            L_m = torch.fft.ifft(S_m * W, dim=1)
            L_s = torch.fft.ifft(S_s * W, dim=1)
            G = L_m * torch.conj(L_s)
            if coh_weight is not None:
                G = G * coh_weight
            if aperture_mask:
                theta_rel = (
                    theta_axis[None, :] - float(y_looks[j]) / r_rows[:, None]
                )
                mask = (theta_rel >= aperture_pad * theta0) & (
                    theta_rel <= aperture_pad * theta1
                )
                G = G * mask.to(G.dtype)
            # Multilooked look interferogram: sum over azimuth patches
            P = G[:, :ntheta_p].reshape(nr, npatch, L_patch).sum(dim=2)
            P_hist[j] = P
            if j >= lag:
                # angle of the double difference = phi_err(y_j) -
                # phi_err(y_{j-lag}) projected on the look vector;
                # topographic / baseline phase cancels. No unwrapping
                # needed for sub-pi differences.
                D = P * torch.conj(P_hist[j - lag])
                P_hist[j - lag] = None
                Dr = torch.cumsum(torch.sum(D, dim=1), dim=0)
                for b in range(n_r_bands):
                    i0, i1 = int(edges[b]), int(edges[b + 1])
                    if i1 <= i0:
                        continue
                    lo = Dr[i0 - 1] if i0 > 0 else 0.0
                    dd_bands[b, j - lag] = Dr[i1 - 1] - lo

        steps_bands = torch.angle(dd_bands)
        w_steps = dd_bands.abs()
        band_rc = torch.zeros(n_r_bands, dtype=torch.float32, device=device)
        band_valid = torch.zeros(n_r_bands, dtype=torch.bool, device=device)
        for b in range(n_r_bands):
            i0, i1 = int(edges[b]), int(edges[b + 1])
            if i1 <= i0:
                continue
            band_rc[b] = r0_full + dr * 0.5 * (i0 + i1)
            band_valid[b] = True

        valid_idx = torch.where(band_valid)[0]
        K = int(len(valid_idx))
        if K < n_axes:
            raise ValueError(f"Only {K} valid range bands, need >= {n_axes}")

        if remove_trend:
            # A constant step per band is a linear trend in the error
            # profile: unobservable RME (absorbed into the baseline
            # estimate) plus the deterministic baseline-induced phase
            # ramp, which is range-dependent and would otherwise leak
            # into a slow X/Z drift through the LS. Remove the weighted
            # mean difference per band.
            wm = torch.sum(w_steps * steps_bands, dim=1, keepdim=True) / (
                torch.sum(w_steps, dim=1, keepdim=True) + 1e-30
            )
            steps_bands = steps_bands - wm

        # Each lag difference is the sum of `lag` adjacent look steps:
        # S_map [ndiff, npairs] with ones on the lag-wide band.
        npairs = n_looks - 1
        ii = torch.arange(ndiff, device=device)
        jj_p = torch.arange(npairs, device=device)
        S_map = (
            (jj_p[None, :] >= ii[:, None]) & (jj_p[None, :] < ii[:, None] + lag)
        ).to(torch.float32)                                      # [ndiff, npairs]

        if verbose or return_phi_bands:
            # Per-band profiles for diagnostics: solve adjacent steps
            # from the lag differences per band, then integrate.
            S_aug = torch.cat(
                [S_map, 0.1 * torch.eye(npairs, device=device)], dim=0
            )
            rhs = torch.cat(
                [
                    steps_bands.t(),
                    torch.zeros((npairs, n_r_bands), device=device),
                ],
                dim=0,
            )
            steps_diag = torch.linalg.lstsq(S_aug, rhs).solution  # [npairs, nbands]
            phi_looks = torch.cat(
                [
                    torch.zeros((n_r_bands, 1), device=device),
                    torch.cumsum(steps_diag.t(), dim=1),
                ],
                dim=1,
            )
            phi_looks = phi_looks - phi_looks.mean(dim=1, keepdim=True)
            if verbose:
                for b in valid_idx.tolist():
                    dr_rms = (
                        torch.sqrt(torch.mean(phi_looks[b] ** 2)).item()
                        / k * 1000
                    )
                    print(
                        f"  iter {iteration} band {b}: "
                        f"r=[{r0_full + dr * edges[b]:.1f}, "
                        f"{r0_full + dr * edges[b + 1]:.1f}] m, "
                        f"Δr_rms={dr_rms:.2f} mm"
                    )

        # Position error difference over each lag interval, observed in
        # slant range per band. The double difference phase is
        # +k * l . (eps_j - eps_(j-lag)) where eps is the slave position
        # error; the correction is its negation.
        dr_diffs = -steps_bands[valid_idx] / k                   # [K, ndiff]

        # Look vector from lag-interval midpoint to broadside band
        # centroid, same conventions as insar_rme_blocksvd_strata
        y_mid = 0.5 * (y_looks[lag:] + y_looks[:-lag])           # [ndiff]
        rc_active = band_rc[valid_idx]
        if altitude is not None:
            H = float(altitude)
            x_ground = torch.sqrt(
                torch.clamp(rc_active ** 2 - H ** 2, min=1e-6)
            )
            dxs = x_ground[None, :].expand(ndiff, -1)            # [ndiff, K]
            dys = -y_mid[:, None].expand(-1, K)
            dzs = torch.full_like(dxs, -H)
        else:
            dxs = (rc_active[None, :] - float(pos_s[:, 0].mean())).expand(
                ndiff, -1
            )
            dys = -y_mid[:, None].expand(-1, K)
            dzs = torch.full_like(dxs, -float(pos_s[:, 2].mean()))
        rg = torch.sqrt(dxs ** 2 + dys ** 2)
        rs = torch.sqrt(rg ** 2 + dzs ** 2) + 1e-9
        cos_el_m = rg / rs
        sin_el_m = dzs / rs
        cos_az_m = dxs / (rg + 1e-9)

        m_x = (cos_az_m * cos_el_m).t()                          # [K, ndiff]
        m_z = sin_el_m.t()

        # Global weight normalization keeps the per-difference absolute
        # SNR information so the ridge term can shrink low-SNR
        # observations (track edges) toward zero instead of amplifying
        # noise through the LS inverse.
        W_ls = w_steps[valid_idx]                                # [K, ndiff]
        W_ls = W_ls / (W_ls.median() + 1e-30)
        sqrtW = W_ls.sqrt()

        # Joint LS over all (band, difference) observations: X error
        # step per adjacent look pair, Z error steps on a coarse linear
        # basis. Z observability comes only from the band-differential
        # of the observations, which is much noisier than their common
        # mode, so Z gets fewer degrees of freedom than X.
        nobs = K * ndiff
        if estimate_z:
            nz = int(max(2, min(n_z_basis, npairs)))
            knots = torch.linspace(0, npairs - 1, nz, device=device)
            jj = torch.arange(npairs, device=device, dtype=torch.float32)
            dk = float(knots[1] - knots[0])
            B_z = (1.0 - (jj[:, None] - knots[None, :]).abs() / dk).clamp(
                min=0.0
            )                                                    # [npairs, nz]
            SB_z = S_map @ B_z                                   # [ndiff, nz]
        else:
            nz = 0
        nunk = npairs + nz

        A_blocks = []
        for kb in range(K):
            wb = sqrtW[kb][:, None]                              # [ndiff, 1]
            A_x = S_map * (m_x[kb][:, None] * wb)                # [ndiff, npairs]
            if estimate_z:
                A_zb = SB_z * (m_z[kb][:, None] * wb)            # [ndiff, nz]
                A_blocks.append(torch.cat([A_x, A_zb], dim=1))
            else:
                A_blocks.append(A_x)
        A = torch.cat(
            A_blocks + [ls_reg * torch.eye(nunk, device=device)], dim=0
        )
        yv = torch.cat(
            [
                (dr_diffs * sqrtW).reshape(-1),
                torch.zeros(nunk, device=device),
            ]
        ).unsqueeze(-1)
        sol = torch.linalg.lstsq(A, yv).solution.squeeze(-1)

        steps_x = sol[:npairs]
        steps_z = B_z @ sol[npairs:] if estimate_z else None

        # Integrate the solved steps to the error profile at look centers
        # and interpolate to sweeps
        sol_steps = [steps_x]
        if estimate_z:
            sol_steps.append(steps_z)
        sol_steps = torch.stack(sol_steps, dim=-1)               # [npairs, n_axes]
        prof = torch.cat(
            [
                torch.zeros((1, n_axes), device=device),
                torch.cumsum(sol_steps, dim=0),
            ],
            dim=0,
        )                                                        # [n_looks, n_axes]
        y_s = pos_s[:, 1].to(torch.float32)
        delta_swp = _interp1_linear(y_s, y_looks, prof.t())      # [n_axes, nsweeps]

        delta_it = torch.zeros(
            (nsweeps, 3), dtype=torch.float32, device=device
        )
        delta_it[:, 0] = delta_swp[0]
        if estimate_z:
            delta_it[:, 2] = delta_swp[1]
        delta_it = delta_it - delta_it.mean(dim=0, keepdim=True)

        if delta_lowpass and delta_lowpass > 1:
            for ax in range(3):
                if delta_it[:, ax].abs().max() == 0:
                    continue
                delta_it[:, ax] = conv_lowpass_filter(
                    delta_it[:, ax], delta_lowpass
                )

        rms_it = torch.sqrt(torch.mean(torch.sum(delta_it ** 2, dim=1)))
        if iteration > 0 and rms_it >= 0.9 * rms_prev:
            # No further improvement: the remaining correction is at the
            # per-iteration measurement noise floor and applying it
            # would only add noise.
            if verbose:
                print(
                    f"  iter {iteration}: correction RMS not decreasing, "
                    "stopping"
                )
            break
        rms_prev = rms_it

        delta = delta + delta_it
        pos_s_new = pos_s_new + delta_it
        if return_phi_bands:
            phi_bands_swp = _interp1_linear(y_s, y_looks, phi_looks[valid_idx])

        if verbose:
            rms = [
                f"X={torch.sqrt(torch.mean(delta_it[:, 0] ** 2)).item() * 1000:.2f}"
            ]
            if estimate_z:
                rms.append(
                    f"Z={torch.sqrt(torch.mean(delta_it[:, 2] ** 2)).item() * 1000:.2f}"
                )
            print(
                f"  iter {iteration} correction RMS (mm): " + ", ".join(rms)
            )

        if iteration < max_iters - 1:
            img_s_cur = backprojection_polar_2d(
                data_s, grid_polar, fc, r_res, pos_s_new, d0=d0,
                dealias=dealias, data_fmod=data_fmod, alias_fmod=alias_fmod,
            )[0]

    if return_phi_bands:
        return pos_s_new, delta, phi_bands_swp
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
