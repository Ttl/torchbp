import torch
import numpy as np
from torch import Tensor
from .ops import (
    backprojection_polar_2d,
    backprojection_cart_2d,
    gpga_backprojection_2d_core,
)
from .ops import entropy
from .util import unwrap, detrend, fft_lowpass_filter_window
import inspect
from scipy import signal


def pga_pd(
    img: Tensor,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.5,
    min_window: int = 5,
    remove_trend: bool = True,
    offload: bool = False,
) -> (Tensor, Tensor):
    """
    Phase gradient autofocus

    Phase difference estimator.

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

    References
    ----------
    .. [#] D. E. Wahl, P. H. Eichel, D. C. Ghiglia and C. V. Jakowatz, "Phase
    gradient autofocus - A robust tool for high resolution SAR phase
    correction," in IEEE Transactions on Aerospace and Electronic Systems, vol.
    30, no. 3, pp. 827-835, July 1994.

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
        gdot = torch.diff(g, prepend=g[:, 0][:, None], axis=-1)
        # Weighted sum over range
        phidot = torch.sum((torch.conj(g) * gdot).imag, axis=0) / torch.sum(
            torch.abs(g) ** 2, axis=0
        )
        phi = torch.cumsum(phidot, dim=0)
        if remove_trend:
            phi = detrend(unwrap(phi))
        phi_sum += phi

        del phidot
        del gdot
        del g
        if offload:
            img = img.to(device=dev)
        img_ifft = torch.fft.fft(img, axis=-1)
        img_ifft *= torch.exp(-1j * phi[None, :])
        img = torch.fft.ifft(img_ifft, axis=-1)

    return img, phi_sum


def pga_ml(
    img: Tensor,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.5,
    min_window: int = 5,
    remove_trend: bool = True,
    offload: bool = False,
) -> (Tensor, Tensor):
    """
    Phase gradient autofocus

    Maximum likelihood estimator.

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

    References
    ----------
    .. [#] Charles V. Jakowatz and Daniel E. Wahl, "Eigenvector method for
    maximum-likelihood estimation of phase errors in synthetic-aperture-radar
    imagery," J. Opt. Soc. Am. A 10, 2539-2546 (1993).

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
        u, s, v = torch.linalg.svd(g)
        phi = torch.angle(v[0, :])
        if remove_trend:
            phi = detrend(unwrap(phi))
        phi_sum += phi

        del g
        if offload:
            img = img.to(device=dev)
        img_ifft = torch.fft.fft(img, axis=-1)
        img_ifft *= torch.exp(-1j * phi[None, :])
        img = torch.fft.ifft(img_ifft, axis=-1)

    return img, phi_sum


def gpga_2d_iter(
    target_pos: Tensor,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    window_width: int | None = None,
    d0: float = 0.0,
    estimator: str = "ml",
    lowpass_window="boxcar",
) -> Tensor:
    """
    Single generalized phase gradient iteration.

    Parameters
    ----------
    target_pos : Tensor
        Positions of point-like targets to use to focus the image.
        3D Cartesian coordinates (x, y, z). Dimensions: [ntargets, 3].
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
    window_width : int or None
        Low-pass filter window width in samples. None or more than nsweeps for
        no low-pass filtering.
    d0 : float
        Zero range correction.
    estimator : str
        Estimator to use.
        "ml": Maximum likelihood.
        "pd": Phase difference.
    lowpass_window : str
        FFT window to use for lowpass filtering.
        See `scipy.get_window` for syntax.

    Returns
    ----------
    phi : Tensor
        Solved phase error.
    """
    # Get range profile samples for each target
    target_data = gpga_backprojection_2d_core(
        target_pos, data, pos, fc, r_res, d0
    )
    # Filter samples
    if window_width is not None and window_width < target_data.shape[1]:
        target_data = fft_lowpass_filter_window(
            target_data, window=lowpass_window, window_width=window_width
        )
    if estimator == "ml":
        u, s, v = torch.linalg.svd(target_data)
        phi = torch.angle(v[0, :])
    else:
        g = target_data
        gdot = torch.diff(g, prepend=g[:, 0][:, None], axis=-1)
        phidot = torch.sum((torch.conj(g) * gdot).imag, axis=0) / torch.sum(
            torch.abs(g) ** 2, axis=0
        )
        phi = torch.cumsum(phidot, dim=0)
    return phi


def gpga_ml_bp_polar(
    img: Tensor | None,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid_polar: dict,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.8,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    remove_trend: bool = True,
    estimator: str = "pd",
    lowpass_window: str = "boxcar",
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
        "ml": Maximum likelihood.
        "pd": Phase difference.
    lowpass_window : str
        FFT window to use for lowpass filtering.
        See `scipy.get_window` for syntax.

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

    r = r0 + dr * torch.arange(nr, device=data.device)
    theta = theta0 + dtheta * torch.arange(ntheta, device=data.device)
    pos_new = pos.clone()

    if window_width is None:
        window_width = data.shape[0]

    if img is None:
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new)
        img = img.squeeze()

    for i in range(max_iters):
        rpeaks = torch.argmax(torch.abs(img), axis=1)
        a = torch.abs(img[torch.arange(img.size(0)), rpeaks])
        max_a = torch.max(a)

        target_idx = a > max_a * 10 ** (-target_threshold_db / 20)
        target_theta = theta0 + dtheta * rpeaks[target_idx]
        target_r = r[target_idx]

        x = target_r * torch.sqrt(1 - target_theta**2)
        y = target_r * target_theta
        z = torch.zeros_like(target_r)
        target_pos = torch.stack([x, y, z], dim=1)

        phi = gpga_2d_iter(
            target_pos,
            data,
            pos_new,
            fc,
            r_res,
            window_width,
            d0,
            estimator=estimator,
            lowpass_window=lowpass_window,
        )
        phi_sum = unwrap(phi_sum + phi)
        if remove_trend:
            phi_sum = detrend(phi_sum)
        # Phase to distance
        d = phi_sum * 3e8 / (4 * torch.pi * fc)
        d = d - torch.mean(d)

        pos_new[:, 0] = pos[:, 0] + d
        img = backprojection_polar_2d(data, grid_polar, fc, r_res, pos_new, d0=d0)
        img = img.squeeze()
        window_width = int(window_width**window_exp)
        if window_width < min_window:
            break
    return img, phi_sum


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

            origin = torch.tensor(
                [torch.mean(pos[:, 0]), torch.mean(pos[:, 1]), 0],
                device=dev,
                dtype=torch.float32,
            )[None, :]
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
