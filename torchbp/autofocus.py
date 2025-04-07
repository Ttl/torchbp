import torch
import numpy as np
from torch import Tensor
from .ops import backprojection_polar_2d, backprojection_cart_2d
from .ops import entropy
from .util import quad_interp, fft_peak_1d, unwrap, detrend
import inspect

def pga_pd(img, window_width=None, max_iters=10, window_exp=0.5, min_window=5, remove_trend=True, offload=False):
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
            g[j, :] = torch.roll(g[j,:], -rpeaks[j].item())
        # Apply window
        g[:, 1+window//2:1-window//2] = 0
        # IFFT across theta
        g = torch.fft.fft(g, axis=-1)
        gdot = torch.diff(g, prepend=g[:,0][:,None], axis=-1)
        # Weighted sum over range
        phidot = torch.sum((torch.conj(g) * gdot).imag, axis=0) / torch.sum(torch.abs(g)**2, axis=0)
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
        img_ifft *= torch.exp(-1j*phi[None, :])
        img = torch.fft.ifft(img_ifft, axis=-1)

    return img, phi_sum

def pga_ml(img, window_width=None, max_iters=10, window_exp=0.5, min_window=5, remove_trend=True, offload=False):
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
            g[j, :] = torch.roll(g[j,:], -rpeaks[j].item())
        # Apply window
        g[:, 1+window//2:1-window//2] = 0
        # IFFT across theta
        g = torch.fft.fft(g, axis=-1)
        u,s,v = torch.linalg.svd(g)
        phi = torch.angle(v[0,:])
        if remove_trend:
            phi = detrend(unwrap(phi))
        phi_sum += phi

        del g
        if offload:
            img = img.to(device=dev)
        img_ifft = torch.fft.fft(img, axis=-1)
        img_ifft *= torch.exp(-1j*phi[None, :])
        img = torch.fft.ifft(img_ifft, axis=-1)

    return img, phi_sum

def _get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def minimum_entropy_grad_autofocus(f, data: Tensor, data_time: Tensor, pos: Tensor, vel:
        Tensor, att: Tensor, fc: float, r_res: float, grid: dict, wa:
        Tensor, tx_norm: Tensor=None, max_steps: float=100, lr_max: float=10000,
        d0: float=0, ant_tx_dy: float=0, pos_reg: float=1,
        lr_reduce: float=0.8, verbose: bool=True, convergence_limit: float=0.01,
        max_step_limit: float=0.25, grad_limit_quantile: float=0.9, fixed_pos: int=0,
        minimize_only: bool=False):
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
    vel : Tensor
        Velocity at each data sample.
    att : Tensor
        Antenna attitude at each data sample.
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
    ant_tx_dy : float
        TX antenna distance from RX antenna in cross-range direction.
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

    wl = 3e8/fc
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
            #pos_d2 = torch.diff(pos, n=2, dim=0) / dt

            pos_loss = pos_reg * torch.mean(torch.square(pos - pos_orig))
            #acc_loss = acc_reg * torch.mean(torch.square(pos_d2[1:]))

            origin = torch.tensor([torch.mean(pos[:,0]), torch.mean(pos[:,1]), 0], device=dev, dtype=torch.float32)[None,:]
            pos_centered = pos - origin

            sar_img = f(data, grid, fc, r_res, pos_centered, vel, att, d0, ant_tx_dy).squeeze()
            if tx_norm is not None:
                entr = entropy(sar_img / tx_norm)
            else:
                entr = entropy(sar_img)
            loss = entr + pos_loss# + acc_loss
            if last_entr is not None and entr > last_entr:
                lr *= lr_reduce
                if minimize_only:
                    vopt.data = v_prev.data
                    continue
            last_entr = entr
            v_prev = vopt.detach().clone()
            if step < max_steps-1:
                loss.backward()
                l = scheduler.get_last_lr()[0]
                with torch.no_grad():
                    vopt.grad /= wa[:, None]
                    g = vopt.grad.detach()
                    gpos = torch.cumsum(l*g* dt, 0)
                    dp = torch.abs(gpos[:,0])
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
                print(step, "Entropy", entr.detach().cpu().numpy(), "loss", loss.detach().cpu().numpy())
    except KeyboardInterrupt:
        print("Interrupted")
        pass

    return sar_img.detach(), origin, pos.detach(), step

def bp_polar_grad_minimum_entropy(data: Tensor, data_time: Tensor, pos: Tensor, vel:
        Tensor, att: Tensor, fc: float, r_res: float, grid: dict, wa:
        Tensor, tx_norm: Tensor=None, max_steps: float=100, lr_max: float=10000,
        d0: float=0, ant_tx_dy: float=0, pos_reg: float=1,
        lr_reduce: float=0.8, verbose: bool=True, convergence_limit: float=0.01,
        max_step_limit: float=0.25, grad_limit_quantile: float=0.9, fixed_pos: int=0):
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
    vel : Tensor
        Velocity at each data sample.
    att : Tensor
        Antenna attitude at each data sample.
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
    ant_tx_dy : float
        TX antenna distance from RX antenna in cross-range direction.
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

def bp_cart_grad_minimum_entropy(data: Tensor, data_time: Tensor, pos: Tensor, vel:
        Tensor, att: Tensor, fc: float, r_res: float, grid: dict, wa:
        Tensor, tx_norm: Tensor=None, max_steps: float=100, lr_max: float=10000,
        d0: float=0, ant_tx_dy: float=0, pos_reg: float=1,
        lr_reduce: float=0.8, verbose: bool=True, convergence_limit: float=0.01,
        max_step_limit: float=0.25, grad_limit_quantile: float=0.9, fixed_pos: int=0):
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
    vel : Tensor
        Velocity at each data sample.
    att : Tensor
        Antenna attitude at each data sample.
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
    ant_tx_dy : float
        TX antenna distance from RX antenna in cross-range direction.
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
