from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
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
    diff,
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
        "pd": Phase difference. [1]_
        "ml": Maximum likelihood. [2]_
        "wls": Weighted least squares using estimated signal-to-clutter weighting. [3]_
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
    ----------
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
    eps=1e-6,
) -> (Tensor, Tensor):
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
    ----------
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
    grid_polar: dict,
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
) -> (Tensor, Tensor):
    """
    Generalized phase gradient autofocus using 2D polar coordinate
    backprojection image formation.

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
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
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

    References
    ----------
    .. [#] A. Evers and J. A. Jackson, "A Generalized Phase Gradient Autofocus
    Algorithm," in IEEE Transactions on Computational Imaging, vol. 5, no. 4,
    pp. 606-619, Dec. 2019.

    Returns
    ----------
    img : Tensor
        Focused SAR image.
    phi : Tensor
        Solved phase error.
    """
    r0, r1 = grid_polar["r"]
    theta0, theta1 = grid_polar["theta"]
    ntheta = grid_polar["ntheta"]
    nr = grid_polar["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    phi_sum = torch.zeros(data.shape[0], dtype=torch.float32, device=data.device)

    theta = theta0 + dtheta * torch.arange(
        ntheta, device=data.device, dtype=torch.float32
    )
    pos_new = pos.clone()

    if window_width is None:
        window_width = data.shape[0]

    if img is None:
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new)[0]

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
            target_pos, data, pos_new, fc, r_res, d0, interp_method=interp_method
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

        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0)[0]
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
    grid_polar: dict,
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
    use_ffbp: bool = False,
    ffbp_opts: dict | None = None,
    verbose: bool = False,
) -> (Tensor, Tensor):
    """
    Generalized phase gradient autofocus [0]_ using 2D polar coordinate
    backprojection image formation.

    Estimates 3D position error by dividing the image into subimages, estimating
    slant range error to each subimage, and then solving for 3D position error
    from slant range errors. [1]_

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
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
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
    use_ffbp : bool
        Use fast factorized backprojection for image formation.
    ffbp_opts : dict
        Dictionary of options for ffbp.
    verbose : bool
        Print progress stats.

    References
    ----------
    .. [0] A. Evers and J. A. Jackson, "A Generalized Phase Gradient Autofocus
    Algorithm," in IEEE Transactions on Computational Imaging, vol. 5, no. 4,
    pp. 606-619, Dec. 2019.
    .. [1] Z. Ding et al., "An Autofocus Approach for UAV-Based Ultrawideband
    Ultrawidebeam SAR Data With Frequency-Dependent and 2-D Space-Variant
    Motion Errors," in IEEE Transactions on Geoscience and Remote Sensing, vol.
    60, pp. 1-18, 2022, Art no. 5203518.

    Returns
    ----------
    img : Tensor
        Focused SAR image.
    pos_new : Tensor
        Solved 3D position error.
    """
    r0, r1 = grid_polar["r"]
    theta0, theta1 = grid_polar["theta"]
    ntheta = grid_polar["ntheta"]
    nr = grid_polar["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    r = r0 + dr * torch.arange(nr, device=data.device, dtype=torch.float32)
    theta = theta0 + dtheta * torch.arange(
        ntheta, device=data.device, dtype=torch.float32
    )
    pos_new = pos.clone()

    if window_width is None:
        window_width = data.shape[0] // azimuth_divisions

    if img is None:
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new)[0]

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
            pos_new.shape[0], window_width, img.device, lowpass_window
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
            opts = {"stages": 5, "oversample_r": 1.6, "oversample_theta": 1.6}
            if ffbp_opts is not None:
                opts.update(ffbp_opts)
            img = ffbp(data, grid_polar, fc, r_res, pos_new, d0=d0, **opts)
        else:
            img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0)[
                0
            ]
        window_width = int(window_width * window_exp)
        if window_width < min_window:
            if verbose:
                print("Window width below the minimum size")
            break
    return img, pos_new


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
    grid: dict,
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
) -> (Tensor, Tensor, Tensor, int):
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
    grid : dict
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

    Returns
    ----------
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

            sar_img = f(data, grid, fc, r_res, pos_centered, d0).squeeze()
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
    grid: dict,
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
):
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
    grid : dict
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

    Returns
    ----------
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
    grid: dict,
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
):
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
    grid : dict
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

    Returns
    ----------
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
