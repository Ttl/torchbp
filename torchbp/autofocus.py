from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
from typing import TYPE_CHECKING, Union, Tuple
from .grid import unpack_polar_grid, unpack_cartesian_grid

if TYPE_CHECKING:
    from .grid import PolarGrid, CartesianGrid
from .ops import (
    backprojection_polar_2d,
    backprojection_cart_2d,
    gpga_backprojection_2d_core,
    blocksvd_alpha,
    ffbp,
    afbp,
    cfbp,
    cfbp_adaptive,
)
from .ops import entropy
from .util import (
    unwrap,
    unwrap_ref,
    detrend,
    weighted_detrend,
    conv_lowpass_filter,
    fft_lowpass_filter_precalculate_window,
    fft_lowpass_filter_window,
    diff
)
import inspect
from scipy import signal
from copy import deepcopy


def pga_estimator(
    g: Tensor,
    estimator: str = "wls",
    eps: float = 1e-6,
    return_weight: bool = False,
    weight: Tensor | None = None,
    weight_gate: float = 0.2,
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
    weight : Tensor or None
        Optional non-negative per-sample weight, shape [Ntargets, Nazimuth],
        e.g. the two-way antenna amplitude toward each target at each
        measurement. Weights are normalized per target. Samples with zero
        weight (target outside the antenna beam) do not contribute to the
        phase estimate; measurements where no target has weight get zero
        phase gradient. With "wls" the signal-to-clutter weights are
        estimated from the antenna-envelope-flattened amplitudes of the
        in-beam samples only. With "ml" the samples are gated by the weight
        before the SVD.
    weight_gate : float
        Relative weight (to each target's maximum) below which a sample is
        considered out of beam for the "wls" amplitude statistics and the
        "ml" gating.

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
    tiny = 1e-12
    if weight is not None:
        wn = weight / torch.clamp(
            torch.amax(weight, dim=1, keepdim=True), min=tiny
        )
        in_beam = wn > weight_gate
        # Weight for the phase-difference products between consecutive
        # measurements: both samples of the pair must be in the beam.
        wpair = wn * torch.nn.functional.pad(wn[..., :-1], (1, 0))
    if estimator == "ml":
        gw = g if weight is None else g * in_beam
        u, s, v = torch.linalg.svd(gw)
        phi = torch.angle(v[0, :])
    elif estimator == "wls":
        if weight is None:
            ga = torch.abs(g)
            c = torch.mean(ga, dim=1, keepdim=True)
            d = torch.mean(torch.square(ga), dim=1, keepdim=True)
        else:
            # Flatten the antenna envelope so the amplitude statistics
            # measure signal-to-clutter ratio instead of the beam
            # modulation, and use only in-beam samples.
            ga = in_beam * torch.abs(g) / torch.clamp(wn, min=weight_gate)
            n_in = torch.clamp(
                torch.sum(in_beam, dim=1, keepdim=True), min=1
            )
            c = torch.sum(ga, dim=1, keepdim=True) / n_in
            d = torch.sum(torch.square(ga), dim=1, keepdim=True) / n_in
        w = (
            torch.nan_to_num(
                d / (2 * (2 * c**2 - d) - 2 * c * torch.sqrt(4 * c**2 - 3 * d))
            )
            + eps
        )
        # Pairwise products on views; the first product (against the
        # zero-padded sample) is identically zero, so only pad phidot.
        prod = g[..., 1:] * torch.conj(g[..., :-1])
        if weight is not None:
            prod = wpair[..., 1:] * prod
        phidot = torch.angle(torch.sum((w / torch.max(w)) * prod, dim=0))
        phidot = torch.nn.functional.pad(phidot, (1, 0))
        phi = torch.cumsum(phidot, dim=0)
        if return_weight:
            return phi, w
    elif estimator == "pd":
        z = torch.zeros((g.shape[0], 1), device=g.device, dtype=g.dtype)
        gdot = torch.diff(g, prepend=z, dim=-1)
        if weight is None:
            num = torch.sum((torch.conj(g) * gdot).imag, dim=0)
            den = torch.sum(torch.abs(g) ** 2, dim=0)
        else:
            num = torch.sum(wpair * (torch.conj(g) * gdot).imag, dim=0)
            den = torch.sum(wpair * torch.abs(g) ** 2, dim=0)
        phidot = torch.where(den > tiny, num / torch.clamp(den, min=tiny), 0.0)
        phi = torch.cumsum(phidot, dim=0)
    else:
        raise ValueError(f"Unknown estimator {estimator}")
    return phi


def pga_window_estimate(
    img: Tensor,
    thr_db: float = 10.0,
    margin: float = 2.0,
    min_window: int = 8,
) -> int:
    """
    Estimate the phase gradient autofocus window width from the image.

    Measures the width of the average defocus kernel from the noncoherent
    sum of the peak-centered image rows [1]_: with each row circularly
    shifted so its strongest pixel is at bin 0, the phase of the blur
    cancels in the intensity sum but its width survives, riding on a flat
    clutter pedestal. Rows are weighted by a signal-to-clutter proxy
    (peak-to-mean ratio squared) so target-free rows do not dilute the
    kernel, the pedestal is estimated as the median of the summed profile,
    and the width is measured at ``thr_db`` below the pedestal-subtracted
    peak.

    A sinusoidal phase error produces discrete paired echoes rather than
    a continuous blur, and the dips between echoes would end the width
    measurement early. The profile is therefore smoothed with a window
    scaled to the current width estimate and re-measured until the
    estimate is stable, so the smoothing bridges the echo spacing
    whatever it is.

    On a well focused image the estimate collapses to the target mainlobe
    width, so it can also be used as a convergence indicator.

    References
    ----------
    .. [1] D. E. Wahl, P. H. Eichel, D. C. Ghiglia and C. V. Jakowatz,
        "Phase gradient autofocus - A robust tool for high resolution SAR
        phase correction," in IEEE Transactions on Aerospace and
        Electronic Systems, vol. 30, no. 3, pp. 827-835, July 1994.

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth].
    thr_db : float
        Width is measured where the pedestal-subtracted kernel profile
        drops this many dB below its peak.
    margin : float
        Safety factor applied to the measured width.
    min_window : int
        Lower bound of the returned window width.

    Returns
    -------
    window_width : int
        Estimated window width in azimuth bins, at most the azimuth size.
    """
    if img.ndim != 2:
        raise ValueError("Input image should be 2D.")
    nr, ntheta = img.shape
    a = torch.abs(img)
    rpeaks = torch.argmax(a, dim=1)
    pk = a[torch.arange(nr, device=img.device), rpeaks]
    w = (pk / (torch.mean(a, dim=1) + 1e-30)) ** 2
    w = w / torch.clamp(torch.sum(w), min=1e-30)
    idx = (
        torch.arange(ntheta, device=img.device)[None, :] + rpeaks[:, None]
    ) % ntheta
    s0 = torch.sum(w[:, None] * torch.gather(a, 1, idx) ** 2, dim=0)
    ws = max(3, ntheta // 4096)
    width = ntheta
    for _ in range(6):
        s = sum(
            torch.roll(s0, j) for j in range(-(ws // 2), ws // 2 + 1)
        ) / (2 * (ws // 2) + 1)
        ped = torch.median(s)
        thr = ped + (torch.max(s) - ped) * 10 ** (-thr_db / 10)
        below = s <= thr
        # First crossing walking outward from the centered peak at bin 0
        # in both circular directions.
        right = torch.nonzero(below[: ntheta // 2])
        left = torch.nonzero(below[ntheta // 2 :].flip(0))
        k_r = int(right[0]) if len(right) else ntheta // 2
        k_l = int(left[0]) if len(left) else ntheta // 2
        width = k_r + k_l
        ws_new = max(3, width // 2)
        if ws_new <= ws:
            break
        ws = ws_new
    return max(min_window, min(ntheta, int(margin * width)))


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
    spectrum_support: bool = True,
    support_gate: float = 0.01,
    max_targets: int | None = None,
    truncate: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Phase gradient autofocus

    Parameters
    ----------
    img : Tensor
        Complex input image. Shape should be: [Range, azimuth].
    window_width : int
        Initial window width. Default is None which measures the blur
        width from the image with :func:`pga_window_estimate`. Set
        explicitly to override, e.g. to the image size for the previous
        unwindowed first-iteration behavior.
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
    spectrum_support : bool
        Restrict the estimate to azimuth spectrum bins that have signal.
        If the azimuth axis is oversampled (e.g. a ``cfbp`` image resampled
        with ``cart_to_polar``), part of the spectrum is noise only. Without
        gating, the noise bins corrupt the estimator statistics and bias the
        trend removal, which shifts the image. The support is measured from
        the mean azimuth power spectrum, the phase estimate is integrated
        along the occupied band and the trend fit is weighted by the
        spectrum power.
    support_gate : float
        Azimuth spectrum bins with mean power below this fraction of the
        maximum are considered unoccupied.
    max_targets : int or None
        Estimate the phase error only from this many range rows with the
        highest peak-to-mean amplitude ratio. None (default) uses every
        row. Rows without a strong target get a near-zero weight from the
        "wls" estimator but still cost a windowed FFT and estimator pass,
        so on large images a few hundred rows gives nearly the same
        estimate much faster.
    truncate : bool
        When the window is much smaller than the image, estimate the
        phase error from a zero-padded FFT of only the windowed samples
        instead of a full-length FFT and interpolate the estimate back
        onto the full azimuth spectrum. The windowed signal supports only
        about ``window`` independent estimate values, so a 4x oversampled
        short FFT loses almost nothing. Only affects speed; disable to
        reproduce the exact full-length estimates.

    Returns
    -------
    img : Tensor
        Focused image.
    phi : Tensor
        Solved phase error. Only meaningful over the occupied azimuth
        spectrum bins.
    """
    if img.ndim != 2:
        raise ValueError("Input image should be 2D.")
    if window_exp > 1 or window_exp < 0:
        raise ValueError(f"Invalid window_exp {window_exp}")
    nr, ntheta = img.shape
    phi_sum = torch.zeros(ntheta, device=img.device, dtype=img.real.dtype)
    if window_width is None:
        window_width = pga_window_estimate(img)
    if window_width > ntheta:
        window_width = ntheta
    dev = img.device

    # The image spectrum is the loop state: the correction multiplies it in
    # place and the image domain is materialized once per iteration for the
    # peak search, instead of a separate fft+ifft round trip per correction.
    F = torch.fft.fft(img, axis=-1)

    # Spectrum magnitude is invariant over the iterations since only phase
    # corrections are applied, so the support is solved only once.
    est_weight = None
    det_weight = None
    mask = None
    k_gap = 0
    if spectrum_support:
        spec_pwr = torch.mean(torch.abs(F) ** 2, dim=0)
        support = spec_pwr / torch.clamp(torch.max(spec_pwr), min=1e-30)
        mask = support > support_gate
        # Start the phase integration and unwrapping from the least occupied
        # bin so that they traverse the occupied band contiguously even when
        # the band wraps around the array edge (spectrum not centered).
        w0 = max(3, ntheta // 64)
        s_smooth = sum(
            torch.roll(spec_pwr, j) for j in range(-(w0 // 2), w0 // 2 + 1)
        )
        k_gap = int(torch.argmin(s_smooth))
        est_weight = torch.roll(mask, -k_gap).to(img.real.dtype)[None, :]
        det_weight = torch.roll(support * mask, -k_gap)

    corrected = False
    for i in range(max_iters):
        window = int(window_width * window_exp**i)
        if window < min_window:
            break
        if corrected:
            img = torch.fft.ifft(F, axis=-1)
        # Peak for each range bin
        a_img = torch.abs(img)
        rpeaks = torch.argmax(a_img, axis=1)
        if max_targets is not None and max_targets < nr:
            # Rows with the highest peak-to-mean ratio carry nearly all of
            # the "wls" weight; skip the rest.
            peak = a_img[torch.arange(nr, device=dev), rpeaks]
            score = peak / (torch.mean(a_img, dim=1) + 1e-30)
            rows = torch.topk(score, max_targets).indices
            rpeaks = rpeaks[rows]
            img_est = img[rows]
        else:
            img_est = img
        del a_img
        w2 = window // 2
        m_len = ntheta
        if truncate:
            m_len = 1 << max(4, (4 * (window + 1) - 1).bit_length())
        if truncate and m_len <= ntheta // 2:
            # The windowed rows are zero outside `window` samples around
            # the peak, so an oversampled short FFT of just those samples
            # carries the same information as the full-length FFT. Keep
            # the circular sample layout (tail at negative indices) so a
            # peak shift stays a linear phase.
            offs = torch.cat(
                [
                    torch.arange(0, w2 + 1, device=dev),
                    torch.arange(ntheta - w2, ntheta, device=dev),
                ]
            )
            idx = (rpeaks[:, None] + offs[None, :]) % ntheta
            gs = torch.gather(img_est, 1, idx)
            g = torch.zeros(
                gs.shape[0], m_len, dtype=img.dtype, device=dev
            )
            g[:, : w2 + 1] = gs[:, : w2 + 1]
            if w2 > 0:
                g[:, m_len - w2 :] = gs[:, w2 + 1 :]
            del gs
        else:
            m_len = ntheta
            # Roll theta axis so that peak is at 0 bin. Batched gather
            # instead of a per-row roll: one kernel instead of nr launches
            # and syncs.
            idx = (
                torch.arange(ntheta, device=dev)[None, :] + rpeaks[:, None]
            ) % ntheta
            g = torch.gather(img_est, 1, idx)
            # Apply window
            g[:, 1 + w2 : ntheta - w2] = 0
        del img_est, idx
        if offload:
            F = F.to(device="cpu")
        # IFFT across theta
        g = torch.fft.fft(g, axis=-1)
        k_gap_m = 0
        est_w = est_weight
        if spectrum_support:
            if m_len == ntheta:
                k_gap_m = k_gap
            else:
                # The short FFT samples the same spectrum on a coarser
                # grid; resample the support mask and gap onto it.
                k_gap_m = int(round(k_gap * m_len / ntheta))
                pos = torch.round(
                    torch.arange(m_len, device=dev) * (ntheta / m_len)
                ).long() % ntheta
                est_w = torch.roll(mask[pos], -k_gap_m).to(
                    img.real.dtype
                )[None, :]
            g = torch.roll(g, -k_gap_m, dims=-1)
        phi = pga_estimator(g, estimator, eps, weight=est_w)
        del g
        # Unwrap in the rolled frame where the occupied band is contiguous
        # and the estimate is smooth.
        phi = unwrap(phi)
        if m_len != ntheta:
            # Interpolate the estimate onto the full spectrum, still in
            # the rolled frame (sub-bin offset from the gap rounding
            # included).
            xq = torch.remainder(
                (torch.arange(ntheta, device=dev, dtype=phi.dtype) + k_gap)
                * (m_len / ntheta)
                - k_gap_m,
                m_len,
            )
            phi = _interp1_linear(
                xq,
                torch.arange(m_len, device=dev, dtype=phi.dtype),
                phi,
            )
        if remove_trend:
            phi = weighted_detrend(phi, det_weight)
        if spectrum_support:
            phi = torch.roll(phi, k_gap, dims=-1)
        phi_sum += phi

        if offload:
            F = F.to(device=dev)
        F *= torch.exp(-1j * phi[None, :])
        corrected = True

    if corrected:
        img = torch.fft.ifft(F, axis=-1)
    return img, phi_sum


def pga_xz(
    img: Tensor,
    grid: "PolarGrid | dict",
    fc: float,
    h: float,
    range_divisions: int = 4,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.5,
    min_window: int = 5,
    remove_trend: bool = True,
    estimate_z: bool = True,
    solve_threshold: float = 3e-3,
    eps: float = 1e-6,
    spectrum_support: bool = True,
    support_gate: float = 0.01,
    truncate: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Phase gradient autofocus with range-direction and vertical (x, z)
    position error estimation.

    Image-domain variant of :func:`gpga_tde` restricted to the
    range-direction (x) and vertical (z) platform position error
    components. For these two components the phase error of a pixel
    depends on the pixel direction only through the elevation angle,
    which on a polar grid is a function of the range row alone (scene
    assumed at ``z = 0``), so the error decomposes over the image: at
    range row ``r`` and azimuth spectrum bin ``k``

    ``phi(k, r) = (4 pi fc / c) * (cos(az_c) * cos(el(r)) * dx(k) + sin(el(r)) * dz(k))``

    where ``az_c`` is the grid center azimuth. The profiles ``dx`` and
    ``dz`` are estimated by running the PGA estimator on blocks of range
    rows and solving the two components per spectrum bin from the block
    phases, and the correction is applied with per-row multiplies in the
    azimuth spectrum domain. Unlike :func:`gpga` / :func:`gpga_tde` no
    raw data is needed and no image re-formation is done; the whole
    algorithm runs on the image with FFTs.

    The azimuth spectral variable of a target scales with ``cos(el)``:
    the same along-track platform offset lands on different spectrum
    bins at different range rows. Block estimates are resampled onto a
    common spectral axis referenced to the row with the largest
    ``cos(el)`` before the per-bin solve, and the correction is
    resampled back per row.

    The along-track (y) error component is not modeled: its phase error
    is proportional to the theta coordinate within the image, which does
    not decompose over range rows. At broadside it is also the component
    that defocuses least.

    Parameters
    ----------
    img : Tensor
        Complex input image on a polar grid. Shape: [range, azimuth].
    grid : PolarGrid or dict
        Polar grid definition of the image.
    fc : float
        RF center frequency in Hz.
    h : float
        Mean platform altitude above the scene plane in meters
        (e.g. ``torch.mean(pos[:, 2])``).
    range_divisions : int
        Number of range blocks the phase error is estimated from. Must
        be at least 2 when `estimate_z` is True. More blocks improve the
        conditioning of the x/z separation at the cost of fewer rows per
        block estimate.
    window_width : int
        Initial window width. Default is None which measures the blur
        width from the image with :func:`pga_window_estimate`. Avoid an
        unwindowed (image sized) estimate on a large image: it is
        dominated by phase-difference random-walk noise, which the x/z
        solve amplifies (an unwindowed first pass therefore solves the
        range-direction component only), and it is also the expensive
        iteration the truncated estimation cannot shorten.
    max_iters : int
        Maximum number of iterations.
    window_exp : float
        Exponent for decreasing the window size for each iteration.
    min_window : int
        Minimum window size.
    remove_trend : bool
        Remove linear trend of the solved profiles that only shifts the
        image.
    estimate_z : bool
        Estimate the vertical error component. Requires elevation angle
        variation over the range swath. Automatically disabled when
        ``h == 0``.
    solve_threshold : float
        Relative eigenvalue threshold of the per-bin (x, z) solve. If
        the elevation angle spread over the range swath is too small to
        separate z from x, the weakly observed direction gets zero
        update instead of amplified noise.
    eps : float
        Minimum weight for weighted PGA.
    spectrum_support : bool
        Restrict the estimate to azimuth spectrum bins that have signal.
        See :func:`pga`.
    support_gate : float
        Azimuth spectrum bins with mean power below this fraction of the
        maximum are considered unoccupied.
    truncate : bool
        When the window is much smaller than the image, estimate the
        block phases from a zero-padded FFT of only the windowed samples
        instead of a full-length FFT, as in :func:`pga`. The short-grid
        estimate is interpolated onto the common spectral axis by the
        same resampling step that handles the per-block ``cos(el)``
        scaling. Only affects speed; disable to reproduce the exact
        full-length estimates.

    Returns
    -------
    img : Tensor
        Focused image.
    d : Tensor
        Solved position error profiles in meters, shape [2, ntheta].
        ``d[0]`` is the range-direction (x) and ``d[1]`` the vertical
        (z) error as a function of fftshifted azimuth spectrum bin on
        the common spectral axis (bin ``ntheta // 2`` is zero azimuth
        frequency, the bin is proportional to along-track position with
        unknown scale and sign). Only meaningful over the occupied
        spectrum bins. With `remove_trend` the unobservable linear
        trends are removed.
    """
    if img.ndim != 2:
        raise ValueError("Input image should be 2D.")
    if not _grid_is_polar(grid):
        raise ValueError("pga_xz requires a polar grid.")
    if window_exp > 1 or window_exp < 0:
        raise ValueError(f"Invalid window_exp {window_exp}")
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    if img.shape[0] != nr or img.shape[1] != ntheta:
        raise ValueError(
            f"Image shape {tuple(img.shape)} does not match the grid "
            f"({nr}, {ntheta})."
        )
    if h == 0:
        estimate_z = False
    if estimate_z and range_divisions < 2:
        raise ValueError("estimate_z requires range_divisions >= 2.")
    dev = img.device
    rdtype = img.real.dtype
    c0 = 299792458.0
    k_wave = 4 * torch.pi * fc / c0

    # Per-row look geometry: grid r is ground range from the platform
    # reference (origin), scene assumed at z=0.
    r_rows = r0 + dr * torch.arange(nr, device=dev, dtype=rdtype)
    slant = torch.sqrt(r_rows**2 + h**2)
    cos_el = r_rows / slant
    sin_el = -h / slant
    theta_c = 0.5 * (theta0 + theta1)
    cos_az_c = float(np.sqrt(max(1.0 - theta_c**2, 0.0)))
    mx_rows = cos_az_c * cos_el
    cos_el_ref = torch.max(cos_el)
    # Signed spectral bin axis of the fftshifted azimuth spectrum.
    f_axis = torch.arange(ntheta, device=dev, dtype=rdtype) - ntheta // 2

    if window_width is None:
        window_width = pga_window_estimate(img)
    if window_width > ntheta:
        window_width = ntheta
    nb = range_divisions
    rdiv = nr // nb

    # The fftshifted image spectrum is the loop state: the correction
    # multiplies it in place and the image domain is materialized once per
    # iteration for the peak search, instead of a separate fft+ifft round
    # trip per correction.
    F = torch.fft.fftshift(torch.fft.fft(img, axis=-1), dim=-1)

    # Spectrum magnitude is invariant over the iterations since only
    # phase corrections are applied, so the support is solved only once.
    est_weight = None
    det_weight = None
    if spectrum_support:
        spec_pwr = torch.mean(torch.abs(F) ** 2, dim=0)
        support = spec_pwr / torch.clamp(torch.max(spec_pwr), min=1e-30)
        mask = support > support_gate
        est_weight = mask.to(rdtype)[None, :]
        det_weight = support * mask

    # The correction resamples the solved profiles onto each row's spectral
    # scale on a uniform grid whose geometry does not change over the
    # iterations; precompute the per-pixel interpolation indices.
    gamma_rows = torch.clamp(cos_el / cos_el_ref, min=1e-6)
    qi = (f_axis[None, :] / gamma_rows[:, None] - f_axis[0]).clamp(
        0, ntheta - 1.001
    )
    qi0 = qi.floor().long()
    qt = (qi - qi0).to(rdtype)
    del qi

    d_sum = torch.zeros(2, ntheta, device=dev, dtype=rdtype)
    local_d = torch.zeros(nb, ntheta, device=dev, dtype=rdtype)
    local_w = torch.zeros(nb, device=dev, dtype=rdtype)
    local_m = torch.zeros(nb, 2, device=dev, dtype=rdtype)

    corrected = False
    for i in range(max_iters):
        window = int(window_width * window_exp**i)
        if window < min_window:
            break
        if corrected:
            img = torch.fft.ifft(torch.fft.ifftshift(F, dim=-1), axis=-1)
        w2 = window // 2
        # The windowed rows are zero outside `window` samples around the
        # peak, so an oversampled short FFT of just those samples carries
        # the same information as the full-length FFT (see pga). The
        # short-grid estimate lands on the common axis through the same
        # interpolation that handles the per-block cos(el) scaling.
        m_len = ntheta
        if truncate:
            m_len = 1 << max(4, (4 * (window + 1) - 1).bit_length())
        use_trunc = truncate and m_len <= ntheta // 2
        if use_trunc:
            f_m = torch.arange(m_len, device=dev, dtype=rdtype) - m_len // 2
            est_w = None
            if spectrum_support:
                # Sample the support mask onto the short grid.
                pos = torch.clamp(
                    torch.round(f_m * (ntheta / m_len)).long() + ntheta // 2,
                    0,
                    ntheta - 1,
                )
                est_w = est_weight[:, pos]
            offs = torch.cat(
                [
                    torch.arange(0, w2 + 1, device=dev),
                    torch.arange(ntheta - w2, ntheta, device=dev),
                ]
            )
        else:
            m_len = ntheta
            f_m = f_axis
            est_w = est_weight
        for b in range(nb):
            b1 = (b + 1) * rdiv if b < nb - 1 else nr
            sub = img[b * rdiv : b1]
            # Peak for each range bin, rolled to bin 0 with a batched
            # gather as in pga.
            rpeaks = torch.argmax(torch.abs(sub), axis=1)
            if use_trunc:
                # Keep the circular sample layout (tail at negative
                # indices) so a peak shift stays a linear phase.
                idx = (rpeaks[:, None] + offs[None, :]) % ntheta
                gs = torch.gather(sub, 1, idx)
                g = torch.zeros(
                    gs.shape[0], m_len, dtype=img.dtype, device=dev
                )
                g[:, : w2 + 1] = gs[:, : w2 + 1]
                if w2 > 0:
                    g[:, m_len - w2 :] = gs[:, w2 + 1 :]
                del gs
            else:
                idx = (
                    torch.arange(ntheta, device=dev)[None, :]
                    + rpeaks[:, None]
                ) % ntheta
                g = torch.gather(sub, 1, idx)
                # Apply window
                g[:, 1 + w2 : ntheta - w2] = 0
            g = torch.fft.fftshift(torch.fft.fft(g, axis=-1), dim=-1)
            phi, w = pga_estimator(
                g, "wls", eps, return_weight=True, weight=est_w
            )
            phi = unwrap(phi)
            # SCR-weighted block geometry: where the strong targets are.
            wn = (w[:, 0] / torch.max(w)).to(rdtype)
            r_b = torch.sum(wn * r_rows[b * rdiv : b1]) / torch.sum(wn)
            slant_b = torch.sqrt(r_b**2 + h**2)
            gamma_b = (r_b / slant_b) / cos_el_ref
            # Resample the block estimate onto the common spectral axis
            # (common bin s maps to short-grid bin s*gamma_b*m_len/ntheta).
            phi = _interp1_linear(
                f_axis * (gamma_b * m_len / ntheta), f_m, phi
            )
            local_d[b] = phi / k_wave
            local_w[b] = 1 / torch.sum(1 / w)
            local_m[b, 0] = cos_az_c * r_b / slant_b
            local_m[b, 1] = -h / slant_b

        # Per-bin weighted solve of the (x, z) profiles from the block
        # range errors. The model matrix is bin-independent so the
        # truncated eigenvalue solve of the normal equations (as in
        # gpga_tde) is a single small solve applied to all bins: if the
        # elevation angle spread is too small to separate z from x, the
        # weak eigen-direction gets zero update instead of noise.
        # On the unwindowed first pass the targets are not isolated at all
        # and the block estimates are at their noisiest; the weakly
        # observed z direction amplifies that noise into a correction that
        # can defocus a large image. Solve only the range-direction
        # component there and let the windowed iterations separate z.
        ncomp = 2 if (estimate_z and window < ntheta) else 1
        ws = torch.sqrt(
            local_w / torch.clamp(torch.max(local_w), min=1e-30)
        )[:, None]
        A = ws * local_m[:, :ncomp]
        B = ws * local_d
        AtA = A.T @ A
        Atb = A.T @ B
        evals, evecs = torch.linalg.eigh(AtA)
        evinv = torch.where(
            evals > solve_threshold * torch.max(evals), 1 / evals, 0.0
        )
        p = evecs @ (evinv[:, None] * (evecs.T @ Atb))
        if remove_trend:
            for c in range(ncomp):
                p[c] = weighted_detrend(p[c], det_weight)
        d_sum[:ncomp] += p

        # Apply the correction with per-row spectrum multiplies,
        # resampling the profiles back to each row's spectral scale with
        # the precomputed uniform-grid interpolation.
        px = p[0][qi0] * (1 - qt) + p[0][qi0 + 1] * qt
        phi_corr = k_wave * mx_rows[:, None] * px
        if ncomp == 2:
            pz = p[1][qi0] * (1 - qt) + p[1][qi0 + 1] * qt
            phi_corr = phi_corr + k_wave * sin_el[:, None] * pz
        F *= torch.exp(-1j * phi_corr)
        corrected = True

    if corrected:
        img = torch.fft.ifft(torch.fft.ifftshift(F, dim=-1), axis=-1)
    return img, d_sum


def _grid_is_polar(grid: "PolarGrid | CartesianGrid | dict") -> bool:
    """Return True for a polar grid, False for a Cartesian one.

    Uses the same duck typing as :func:`torchbp.grid.unpack_polar_grid` /
    :func:`torchbp.grid.unpack_cartesian_grid` so both grid objects and legacy
    dicts work.
    """
    if hasattr(grid, "r0") or hasattr(grid, "x0"):
        return hasattr(grid, "r0")
    if isinstance(grid, dict):
        if "r" in grid and "theta" in grid:
            return True
        if "x" in grid and "y" in grid:
            return False
    raise ValueError(
        "grid must be a PolarGrid, CartesianGrid, or an equivalent dict"
    )


def _dem_at_pixels(
    dem: Tensor,
    grid: "PolarGrid | CartesianGrid | dict",
    i_idx: Tensor,
    j_idx: Tensor,
) -> Tensor:
    """Sample DEM heights at (fractional) image pixel indices.

    Matches the backprojection kernels' convention: the DEM shares the grid
    extent, so pixel index maps to DEM index by the constant ratio
    ``dem_n / grid_n`` per axis, sampled bilinearly with edge clamping.
    """
    if _grid_is_polar(grid):
        _, _, _, _, n0, n1, _, _ = unpack_polar_grid(grid)
    else:
        _, _, _, _, n0, n1, _, _ = unpack_cartesian_grid(grid)
    dem_n0, dem_n1 = dem.shape
    f0 = i_idx * (dem_n0 / n0)
    f1 = j_idx * (dem_n1 / n1)
    i0 = torch.clamp(torch.floor(f0).long(), 0, dem_n0 - 1)
    j0 = torch.clamp(torch.floor(f1).long(), 0, dem_n1 - 1)
    i1 = torch.clamp(i0 + 1, max=dem_n0 - 1)
    j1 = torch.clamp(j0 + 1, max=dem_n1 - 1)
    wi = f0 - i0
    wj = f1 - j0
    za = dem[i0, j0] + wj * (dem[i0, j1] - dem[i0, j0])
    zb = dem[i1, j0] + wj * (dem[i1, j1] - dem[i1, j0])
    return za + wi * (zb - za)


def _pixel_to_world(
    grid: "PolarGrid | CartesianGrid | dict",
    i_idx: Tensor,
    j_idx: Tensor,
    dem: Tensor | None = None,
) -> Tensor:
    """Map image pixel indices to Cartesian world coordinates.

    ``i_idx`` indexes the first image axis (range for polar, x for Cartesian),
    ``j_idx`` the second (azimuth/theta for polar, y for Cartesian). Indices may
    be fractional (float tensors) so block centers work too. Returns a
    ``[N, 3]`` tensor of ``(x, y, z)`` with ``z`` sampled from ``dem`` if
    given, else ``z = 0``.
    """
    if _grid_is_polar(grid):
        r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
        r = r0 + dr * i_idx
        theta = theta0 + dtheta * j_idx
        x = r * torch.sqrt(1 - theta**2)
        y = r * theta
    else:
        x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)
        x = x0 + dx * i_idx
        y = y0 + dy * j_idx
    if dem is not None:
        z = _dem_at_pixels(dem, grid, i_idx, j_idx)
    else:
        z = torch.zeros_like(x)
    return torch.stack([x, y, z], dim=1)


def _antenna_weights(
    target_pos: Tensor, pos: Tensor, att: Tensor, g: Tensor, g_extent: list
) -> Tensor:
    """Two-way antenna amplitude toward each target from each position.

    Mirrors the bilinear antenna gain lookup in the backprojection kernels:
    elevation is the look angle relative to the antenna roll, azimuth the
    target bearing relative to the antenna yaw, and ``g`` is sampled
    bilinearly over ``g_extent``. Angles outside the pattern get zero
    weight. Pitch is ignored, matching the kernels.

    Parameters
    ----------
    target_pos : Tensor
        Target positions. Shape [ntargets, 3].
    pos : Tensor
        Platform positions. Shape [nsweeps, 3].
    att : Tensor
        Antenna rotation [roll, pitch, yaw] per sweep. Shape [nsweeps, 3].
    g : Tensor
        Square-root of two-way antenna gain. Shape [elevation, azimuth].
    g_extent : list
        [g_el0, g_az0, g_el1, g_az1] pattern extent in radians.

    Returns
    -------
    w : Tensor
        Antenna amplitude weight. Shape [ntargets, nsweeps].
    """
    g_el0, g_az0, g_el1, g_az1 = g_extent
    g_nel, g_naz = g.shape
    g_del = (g_el1 - g_el0) / g_nel
    g_daz = (g_az1 - g_az0) / g_naz

    dx = target_pos[:, 0][:, None] - pos[None, :, 0]
    dy = target_pos[:, 1][:, None] - pos[None, :, 1]
    dz = target_pos[:, 2][:, None] - pos[None, :, 2]
    d = torch.sqrt(dx**2 + dy**2 + dz**2)
    el = torch.arcsin(torch.clamp(dz / d, -1.0, 1.0)) - att[None, :, 0]
    az = torch.arctan2(dy, dx) - att[None, :, 2]

    el_idx = (el - g_el0) / g_del
    az_idx = (az - g_az0) / g_daz
    el0 = torch.floor(el_idx)
    az0 = torch.floor(az_idx)
    ef = el_idx - el0
    af = az_idx - az0
    valid = (
        (el_idx >= 0)
        & (el0 + 1 < g_nel)
        & (az_idx >= 0)
        & (az0 + 1 < g_naz)
    )
    el0 = torch.clamp(el0.long(), 0, g_nel - 2)
    az0 = torch.clamp(az0.long(), 0, g_naz - 2)
    v00 = g[el0, az0]
    v01 = g[el0, az0 + 1]
    v10 = g[el0 + 1, az0]
    v11 = g[el0 + 1, az0 + 1]
    w = (
        v00 * (1 - ef) * (1 - af)
        + v01 * (1 - ef) * af
        + v10 * ef * (1 - af)
        + v11 * ef * af
    )
    return torch.where(valid, w, torch.zeros_like(w))


class _AntennaWeightCache:
    """Cache of :func:`_antenna_weights` rows across gpga iterations.

    The weight row of a target is insensitive to look-angle changes well
    below the pattern table cell size, and between autofocus iterations
    both endpoints move far less than that: the track by the per-iteration
    position correction, and the target by however far the per-row argmax
    reselection drifts (typically a pixel). Rows are therefore cached by
    the target's image row (selection picks at most one target per row)
    and reused while both stay inside an angular budget of half a pattern
    cell at the nearest cached target range: the whole cache is emptied
    when the track drifts past it, and an individual row is recomputed
    when its reselected target moved past it. Only targets failing these
    checks are recomputed, so the weight error is bounded by about one
    pattern cell's worth of the cell-to-cell variation.
    """

    def __init__(self, att: Tensor, g: Tensor, g_extent: list):
        self._att = att
        self._g = g
        self._g_extent = g_extent
        g_el0, g_az0, g_el1, g_az1 = g_extent
        self._tol = 0.5 * min(
            (g_el1 - g_el0) / g.shape[0], (g_az1 - g_az0) / g.shape[1]
        )
        self._pos_ref = None
        self._r_min = float("inf")
        # key -> (target position [3], weight row [nsweeps])
        self._rows = {}

    def get(self, keys: list, target_pos: Tensor, pos: Tensor) -> Tensor:
        """Weight rows for the targets identified by hashable ``keys``.

        Equal to ``_antenna_weights(target_pos, pos, ...)`` up to the
        drift tolerance. Entries whose key is not in ``keys`` are dropped,
        so the cache holds one iteration's worth of rows.
        """
        if len(keys) == 0:
            return _antenna_weights(
                target_pos, pos, self._att, self._g, self._g_extent
            )
        if self._pos_ref is not None:
            drift = float(
                torch.max(torch.linalg.norm(pos - self._pos_ref, dim=-1))
            )
            if drift > self._r_min * self._tol:
                self._rows.clear()
                self._pos_ref = None
        if self._pos_ref is None:
            self._pos_ref = pos.clone()
            self._r_min = float("inf")
        missing = [i for i, k in enumerate(keys) if k not in self._rows]
        hits = [i for i, k in enumerate(keys) if k in self._rows]
        if hits:
            # One vectorized distance check (and one device sync) for all
            # cached candidates: a reselected target that moved past the
            # angular budget gets its row recomputed.
            cached_pos = torch.stack([self._rows[keys[i]][0] for i in hits])
            far = (
                torch.linalg.norm(target_pos[hits] - cached_pos, dim=-1)
                > self._r_min * self._tol
            ).tolist()
            missing += [i for i, f in zip(hits, far) if f]
            missing.sort()
        if missing:
            miss_pos = target_pos[missing]
            w_new = _antenna_weights(
                miss_pos, pos, self._att, self._g, self._g_extent
            )
            # Conservative lower bound of the target range for the drift
            # tolerance: distance to the track centroid minus the track
            # radius.
            c = torch.mean(pos, dim=0)
            track_r = torch.max(torch.linalg.norm(pos - c, dim=-1))
            d_min = torch.min(torch.linalg.norm(miss_pos - c, dim=-1))
            self._r_min = min(
                self._r_min, max(float(d_min - track_r), 1.0)
            )
            for j, i in enumerate(missing):
                self._rows[keys[i]] = (target_pos[i], w_new[j])
        out = torch.stack([self._rows[k][1] for k in keys])
        self._rows = {k: self._rows[k] for k in keys}
        return out


def _select_targets(
    img: Tensor,
    target_threshold_db: float,
    isolation_db: float = 0.0,
    isolation_guard: int = 5,
    isolation_window: int = 30,
) -> tuple[Tensor, Tensor]:
    """Pick autofocus targets from a (sub)image: at most one per row.

    Takes the strongest pixel of each row (second image axis), keeps the
    ones within ``target_threshold_db`` of the strongest pick, and when
    ``isolation_db > 0`` additionally requires the peak to exceed the mean
    amplitude of the surrounding cells along the row (excluding a
    ``isolation_guard`` half-width around the peak, out to
    ``isolation_window``) by ``isolation_db``. This rejects targets embedded
    in extended clutter, e.g. building walls, that violate the point-target
    assumption. If the isolation screen rejects every candidate (e.g. a
    heavily defocused early iteration), it falls back to amplitude-only
    selection.

    Returns
    -------
    rows, cols : Tensor
        Integer image indices of the selected targets.
    """
    a_img = torch.abs(img)
    rpeaks = torch.argmax(a_img, dim=1)
    rows = torch.arange(img.shape[0], device=img.device)
    a = a_img[rows, rpeaks]
    keep = a > torch.max(a) * 10 ** (-target_threshold_db / 20)
    if isolation_db > 0 and img.shape[1] > 2 * (isolation_guard + 1):
        window = min(isolation_window, img.shape[1] // 2)
        guard = min(isolation_guard, window - 1)
        offsets = torch.cat(
            [
                torch.arange(-window, -guard, device=img.device),
                torch.arange(guard + 1, window + 1, device=img.device),
            ]
        )
        cols = torch.clamp(rpeaks[:, None] + offsets[None, :], 0, img.shape[1] - 1)
        clutter = torch.mean(torch.gather(a_img, 1, cols), dim=1)
        isolated = a > clutter * 10 ** (isolation_db / 20)
        if torch.any(keep & isolated):
            keep = keep & isolated
    return rows[keep], rpeaks[keep]


# Image formation algorithm registry for GPGA. Maps algorithm name to
# (function, accepts antenna pattern, accepts dem, default image_opts). "bp"
# resolves to the polar or Cartesian direct backprojection by grid type.
_GPGA_POLAR_ALGOS = {
    "bp": (backprojection_polar_2d, True, True, {}),
    "ffbp": (ffbp, True, True, {"stages": 5, "oversample_r": 1.4, "oversample_theta": 1.4}),
    "afbp": (afbp, True, False, {}),
}
_GPGA_CART_ALGOS = {
    "bp": (backprojection_cart_2d, False, False, {}),
    "cfbp": (cfbp, False, False, {"stages": 4}),
    "cfbp_adaptive": (cfbp_adaptive, False, False, {"stages": 4}),
}


def _make_image_former(
    algorithm: str,
    grid: "PolarGrid | CartesianGrid | dict",
    data: Tensor,
    fc: float,
    r_res: float,
    d0: float,
    data_fmod: float,
    att: Tensor | None,
    g: Tensor | None,
    g_extent: list | None,
    image_opts: dict | None,
    dem: Tensor | None = None,
):
    """Build a ``form_image(pos) -> 2D image`` closure for GPGA.

    Selects the image formation algorithm by name and grid type, merges
    ``image_opts`` over the algorithm's defaults, forwards antenna pattern
    and DEM arguments only to algorithms that accept them, and normalizes the
    output to a bare 2D tensor.
    """
    is_polar = _grid_is_polar(grid)
    table = _GPGA_POLAR_ALGOS if is_polar else _GPGA_CART_ALGOS
    if algorithm not in table:
        kind = "polar" if is_polar else "Cartesian"
        raise ValueError(
            f"algorithm {algorithm!r} is not available for a {kind} grid. "
            f"Choose one of {sorted(table)}."
        )
    func, accepts_antenna, accepts_dem, defaults = table[algorithm]

    has_antenna = att is not None or g is not None or g_extent is not None
    if has_antenna and not accepts_antenna:
        raise ValueError(
            f"algorithm {algorithm!r} does not support antenna pattern "
            "arguments (att/g/g_extent)."
        )
    if dem is not None and not accepts_dem:
        kind = "polar" if is_polar else "Cartesian"
        raise ValueError(
            f"algorithm {algorithm!r} ({kind} grid) does not support dem. "
            "Use algorithm 'bp' or 'ffbp' with a polar grid."
        )

    opts = dict(defaults)
    if image_opts is not None:
        opts.update(image_opts)

    if algorithm == "afbp" and "nsub" not in opts:
        raise ValueError(
            "algorithm 'afbp' requires the number of subapertures; pass "
            "image_opts={'nsub': N}."
        )

    def form_image(p):
        kwargs = dict(opts)
        kwargs.update(d0=d0, data_fmod=data_fmod)
        if accepts_antenna:
            kwargs.update(att=att, g=g, g_extent=g_extent)
        if dem is not None:
            kwargs.update(dem=dem)
        img = func(data, grid, fc, r_res, p, **kwargs)
        if img.dim() == 3:
            img = img[0]
        return img

    return form_image


def gpga(
    img: Tensor | None,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid: "PolarGrid | CartesianGrid | dict",
    algorithm: str = "bp",
    image_opts: dict | None = None,
    window_width: int | None = None,
    max_iters: int = 10,
    window_exp: float = 0.7,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    isolation_db: float = 6.0,
    isolation_guard: int = 5,
    isolation_window: int = 30,
    beam_gate: float = 0.2,
    remove_trend: bool = True,
    estimator: str = "pd",
    lowpass_window: str = "boxcar",
    eps: float = 1e-6,
    interp_method: str = "linear",
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    dem: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Generalized phase gradient autofocus. [1]_

    Works with any image formation algorithm in the library and with both
    polar and Cartesian grids. The grid type is detected automatically and the
    image formation algorithm is selected with ``algorithm``.

    Parameters
    ----------
    img : Tensor or None
        Complex input image. Shape should be: [Range, azimuth] for a polar
        grid or [x, y] for a Cartesian grid. If None image is generated from
        the data.
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
    grid : PolarGrid, CartesianGrid or dict
        Image grid definition. Can be:

        - PolarGrid object: PolarGrid(r_range=(r0, r1), theta_range=(theta0, theta1), nr=nr, ntheta=ntheta)
          (theta is sin of angle, -1 to 1 for 180 degree view), or the
          equivalent dict {"r": ..., "theta": ..., "nr": ..., "ntheta": ...}.
        - CartesianGrid object: CartesianGrid(x_range=(x0, x1), y_range=(y0, y1), nx=nx, ny=ny),
          or the equivalent dict {"x": ..., "y": ..., "nx": ..., "ny": ...}.
    algorithm : str
        Image formation algorithm. For a polar grid: "bp"
        (:func:`torchbp.ops.backprojection_polar_2d`, default), "ffbp"
        (:func:`torchbp.ops.ffbp`) or "afbp" (:func:`torchbp.ops.afbp`). For a
        Cartesian grid: "bp" (:func:`torchbp.ops.backprojection_cart_2d`,
        default), "cfbp" (:func:`torchbp.ops.cfbp`) or "cfbp_adaptive"
        (:func:`torchbp.ops.cfbp_adaptive`).
    image_opts : dict or None
        Extra keyword arguments for the image formation algorithm, merged over
        per-algorithm defaults. "ffbp" defaults to
        {"stages": 5, "oversample_r": 1.4, "oversample_theta": 1.4};
        "cfbp"/"cfbp_adaptive" default to {"stages": 4}; "afbp" requires
        {"nsub": N}. The range interpolation method of the data is also set
        here, e.g. {"interp_method": ("knab", 6, 2.0)} for polar "bp" or
        {"data_interp_method": ("knab", 6, 2.0)} for "ffbp"/"afbp".
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
    isolation_db : float
        Reject targets whose peak is less than this many dB above the mean
        amplitude of the surrounding cells along the second image axis
        (excluding a guard band around the peak). Screens out targets
        embedded in extended clutter, e.g. building walls, that violate
        the point-target assumption. 0 disables the screen. If no target
        passes the screen, selection falls back to amplitude only.
    isolation_guard : int
        Half-width in pixels of the guard band around the peak excluded
        from the clutter estimate.
    isolation_window : int
        Half-width in pixels of the clutter estimation window.
    beam_gate : float
        When the antenna pattern (``att``, ``g``, ``g_extent``) is given,
        target samples whose two-way antenna amplitude is below ``beam_gate``
        times that target's maximum are zeroed and excluded from the phase
        estimate. This makes the estimate robust to discontinuous target
        illumination (stripmap-like collections), where out-of-beam samples
        contain unrelated clutter from the same range ring.
    remove_trend : bool
        Remove linear trend in phase correction.
    estimator : str
        Estimator to use.
        See `pga_estimator` function for possible choices.
        With discontinuous illumination (antenna weighting on a
        stripmap-like collection) "wls" is recommended: "pd" accumulates a
        drift as targets enter and leave the beam.
    lowpass_window : str
        FFT window to use for lowpass filtering.
        See `scipy.get_window` for syntax.
    eps : float
        Minimum weight for weighted PGA.
    interp_method : str
        Interpolation method
        "linear": linear interpolation.
        ("lanczos", N): Lanczos interpolation with order 2*N+1.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
        When given together with ``att`` and ``g_extent``, the pattern is
        also used to weight the per-target phase history samples (see
        ``beam_gate``).
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
        Antenna pattern arguments are only supported by the polar algorithms.
    data_fmod : float
        Range modulation frequency applied to input data.
    dem : Tensor or None
        Digital elevation model sampled on the image grid. Shape
        [dem_nr, dem_ntheta], covering the same extent as `grid`; can be
        coarser than the image grid (bilinearly interpolated). Values are
        pixel z coordinates in the same frame as `pos`. Used both for the
        image formation and for the autofocus target positions. Only
        supported with a polar grid and algorithm "bp" or "ffbp". See
        :func:`torchbp.util.dem_to_polar` for resampling a Cartesian DEM
        onto the polar grid. If None (default) targets are assumed to lie
        on the z=0 plane.

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
    # A lazy conjugate view (e.g. data straight from fft(...).conj()) is
    # materialized by the dispatcher on every custom-op call, which the
    # iteration loop makes hundreds of times; resolve it once. No-op for
    # regular tensors.
    data = data.resolve_conj()

    form_image = _make_image_former(
        algorithm, grid, data, fc, r_res, d0, data_fmod,
        att, g, g_extent, image_opts, dem,
    )

    phi_sum = torch.zeros(data.shape[0], dtype=torch.float32, device=data.device)

    pos_new = pos.clone()

    use_antenna_weight = (
        att is not None and g is not None and g_extent is not None
    )

    ant_cache = (
        _AntennaWeightCache(att, g, g_extent) if use_antenna_weight else None
    )

    if window_width is None:
        window_width = data.shape[0]

    if img is None:
        img = form_image(pos_new)

    for i in range(max_iters):
        lp_w = fft_lowpass_filter_precalculate_window(
            pos_new.shape[0], window_width, img.device, lowpass_window, fast_len=True
        )
        rows, cols = _select_targets(
            img, target_threshold_db, isolation_db, isolation_guard,
            isolation_window,
        )
        target_pos = _pixel_to_world(
            grid, rows.to(torch.float32), cols.to(torch.float32), dem
        )

        # Get range profile samples for each target
        target_data = gpga_backprojection_2d_core(
            target_pos, data, pos_new, fc, r_res, d0, interp_method=interp_method, data_fmod=data_fmod
        )
        target_w = None
        beam_w = None
        if use_antenna_weight:
            keys = rows.tolist()
            target_w = ant_cache.get(keys, target_pos, pos_new)
            # Zero the out-of-beam samples before lowpass filtering so
            # unrelated clutter at the target's range ring is not smeared
            # into the illuminated interval.
            wn = target_w / torch.clamp(
                torch.amax(target_w, dim=1, keepdim=True), min=1e-12
            )
            target_data = target_data * (wn > beam_gate)
            # Per-sweep illumination for the trend fit. Sweeps that no
            # target illuminates only hold a constant extrapolated phase
            # and would bias an unweighted line fit.
            beam_w = torch.sum(wn**2, dim=0)
        # Filter samples
        if window_width is not None and window_width < target_data.shape[1]:
            target_data = fft_lowpass_filter_window(
                target_data, window=lp_w, window_width=window_width
            )
        phi = pga_estimator(
            target_data, estimator, eps, weight=target_w, weight_gate=beam_gate
        )
        phi_sum = unwrap(phi_sum + phi)
        if remove_trend:
            phi_sum = weighted_detrend(phi_sum, beam_w)
        # Phase to distance
        c0 = 299792458
        d = phi_sum * c0 / (4 * torch.pi * fc)
        pos_new[:, 0] = pos[:, 0] + d

        img = form_image(pos_new)
        window_width = int(window_width * window_exp)
        if window_width < min_window:
            break
    return img, phi_sum


def gpga_tde(
    img: Tensor | None,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    grid: "PolarGrid | CartesianGrid | dict",
    azimuth_divisions: int,
    range_divisions: int,
    algorithm: str = "bp",
    image_opts: dict | None = None,
    window_width: int | None = None,
    rms_error_limit: float = 0.05,
    max_iters: int = 20,
    window_exp: float = 0.7,
    min_window: int = 5,
    d0: float = 0.0,
    target_threshold_db: float = 20,
    isolation_db: float = 6.0,
    isolation_guard: int = 5,
    isolation_window: int = 30,
    beam_gate: float = 0.2,
    remove_trend: bool = True,
    lowpass_window: str = "boxcar",
    eps: float = 1e-6,
    interp_method: str = "linear",
    estimate_z: bool = True,
    solve_threshold: float = 3e-3,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    verbose: bool = False,
    data_fmod: float = 0,
    dem: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Generalized phase gradient autofocus [1]_ with time-domain error (TDE) 3D
    position estimation.

    Works with any image formation algorithm in the library and with both
    polar and Cartesian grids (see :func:`gpga` for ``algorithm`` /
    ``image_opts`` / ``grid``).

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
    grid : PolarGrid, CartesianGrid or dict
        Image grid definition (polar or Cartesian). See :func:`gpga`.
    azimuth_divisions : int
        Number of divisions for local images in azimuth direction.
    range_divisions : int
        Number of divisions for local images in range direction.
    algorithm : str
        Image formation algorithm. See :func:`gpga`.
    image_opts : dict or None
        Extra keyword arguments for the image formation algorithm. See
        :func:`gpga`.
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
    isolation_db : float
        Reject targets whose peak is less than this many dB above the mean
        amplitude of the surrounding cells along the second image axis
        (excluding a guard band around the peak). Screens out targets
        embedded in extended clutter, e.g. building walls, that violate
        the point-target assumption. 0 disables the screen. If no target
        in a block passes the screen, selection falls back to amplitude
        only.
    isolation_guard : int
        Half-width in pixels of the guard band around the peak excluded
        from the clutter estimate.
    isolation_window : int
        Half-width in pixels of the clutter estimation window.
    beam_gate : float
        When the antenna pattern (``att``, ``g``, ``g_extent``) is given,
        target samples whose two-way antenna amplitude is below ``beam_gate``
        times that target's maximum are zeroed and excluded from the phase
        estimate, and each block's contribution to the position solve is
        weighted per sweep by its illumination. This makes the estimate
        robust to discontinuous target illumination (stripmap-like
        collections), where out-of-beam samples contain unrelated clutter
        from the same range ring.
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
    solve_threshold : float
        Relative eigenvalue threshold of the per-sweep position solve.
        Directions of the per-sweep normal matrix with eigenvalues below
        this fraction of the largest eigenvalue over all sweeps are
        considered unobservable at that sweep and get zero position
        update. Raise it if unobservable directions (e.g. along-track at
        broadside with a narrow beam) accumulate noise; lower it if a
        weakly observed direction that should be estimated is being
        suppressed.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
        When given together with ``att`` and ``g_extent``, the pattern is
        also used to weight the per-target phase history samples and the
        per-sweep position solve (see ``beam_gate``).
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
        Antenna pattern arguments are only supported by the polar algorithms.
    verbose : bool
        Print progress stats.
    data_fmod : float
        Range modulation frequency applied to input data.
    dem : Tensor or None
        Digital elevation model sampled on the image grid. Shape
        [dem_nr, dem_ntheta], covering the same extent as `grid`; can be
        coarser than the image grid (bilinearly interpolated). Values are
        pixel z coordinates in the same frame as `pos`. Used for the image
        formation, the autofocus target positions and the block-center
        geometry of the position solve. Only supported with a polar grid
        and algorithm "bp" or "ffbp". See :func:`torchbp.util.dem_to_polar`
        for resampling a Cartesian DEM onto the polar grid. If None
        (default) targets are assumed to lie on the z=0 plane.


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
    # A lazy conjugate view (e.g. data straight from fft(...).conj()) is
    # materialized by the dispatcher on every custom-op call, which the
    # iteration loop makes hundreds of times; resolve it once. No-op for
    # regular tensors.
    data = data.resolve_conj()

    form_image = _make_image_former(
        algorithm, grid, data, fc, r_res, d0, data_fmod,
        att, g, g_extent, image_opts, dem,
    )

    pos_new = pos.clone()

    use_antenna_weight = (
        att is not None and g is not None and g_extent is not None
    )
    # Per-block caches: target keys are block-local row indices.
    ant_caches = None
    if use_antenna_weight:
        ant_caches = [
            _AntennaWeightCache(att, g, g_extent)
            for _ in range(range_divisions * azimuth_divisions)
        ]

    if window_width is None:
        window_width = data.shape[0] // azimuth_divisions

    if img is None:
        img = form_image(pos_new)

    rdiv = img.shape[0] // range_divisions
    azdiv = img.shape[1] // azimuth_divisions

    local_d = torch.zeros(
        (range_divisions * azimuth_divisions, data.shape[0]),
        dtype=torch.float32,
        device=data.device,
    )
    local_centers = torch.zeros(
        (range_divisions * azimuth_divisions, 3),
        dtype=torch.float32,
        device=data.device,
    )
    # Per-block, per-sweep weight for the position solve. Without antenna
    # information a block has the same weight at every sweep; with it the
    # weight follows the block's illumination so sweeps that never saw the
    # block do not constrain the solution.
    local_w = torch.zeros(
        (range_divisions * azimuth_divisions, data.shape[0]),
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
                ir1 = (ir + 1) * rdiv if ir < range_divisions - 1 else None
                jr1 = (jr + 1) * azdiv if jr < azimuth_divisions - 1 else None
                local_img = img[ir * rdiv : ir1, jr * azdiv : jr1]

                rows, cols = _select_targets(
                    local_img, target_threshold_db, isolation_db,
                    isolation_guard, isolation_window,
                )
                if rows.numel() == 0:
                    # No usable targets in this block (e.g. never
                    # illuminated). Zero weight excludes it from the
                    # position solve; centers just need to be finite.
                    local_w[ir * azimuth_divisions + jr, :] = 0
                    local_d[ir * azimuth_divisions + jr, :] = 0
                    mid = _pixel_to_world(
                        grid,
                        torch.tensor(
                            [ir * rdiv + local_img.shape[0] / 2],
                            device=data.device,
                        ),
                        torch.tensor(
                            [jr * azdiv + local_img.shape[1] / 2],
                            device=data.device,
                        ),
                        dem,
                    )[0]
                    local_centers[ir * azimuth_divisions + jr] = mid
                    continue
                # Absolute image indices of the targets in this block.
                i_idx = ir * rdiv + rows.to(torch.float32)
                j_idx = jr * azdiv + cols.to(torch.float32)
                target_pos = _pixel_to_world(grid, i_idx, j_idx, dem)

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
                target_w = None
                if use_antenna_weight:
                    keys = rows.tolist()
                    target_w = ant_caches[
                        ir * azimuth_divisions + jr
                    ].get(keys, target_pos, pos_new)
                    # Zero the out-of-beam samples before lowpass filtering
                    # so unrelated clutter at the target's range ring is not
                    # smeared into the illuminated interval.
                    awn = target_w / torch.clamp(
                        torch.amax(target_w, dim=1, keepdim=True), min=1e-12
                    )
                    target_data = target_data * (awn > beam_gate)
                # Filter samples
                if window_width is not None and window_width < target_data.shape[1]:
                    target_data = fft_lowpass_filter_window(
                        target_data, window=lp_w, window_width=window_width
                    )
                phi, w = pga_estimator(
                    target_data, "wls", eps, return_weight=True,
                    weight=target_w, weight_gate=beam_gate,
                )
                block_w = 1 / torch.sum(1 / w)
                beam_w = None
                if use_antenna_weight:
                    # Per-sweep illumination of this block: SCR-weighted
                    # two-way antenna power over its targets.
                    wn_t = w[:, 0] / torch.max(w)
                    beam_w = torch.sum(wn_t[:, None] * awn**2, dim=0)
                    beam_w = beam_w / torch.clamp(torch.max(beam_w), min=1e-12)
                    local_w[ir * azimuth_divisions + jr, :] = block_w * beam_w
                else:
                    local_w[ir * azimuth_divisions + jr, :] = block_w
                phi = unwrap(phi)
                phi = weighted_detrend(phi, beam_w)
                # Phase to distance
                c0 = 299792458
                d = phi * c0 / (4 * torch.pi * fc)
                local_d[ir * azimuth_divisions + jr, :] = d

                # Normalize to avoid overflow with near-noiseless targets.
                # Average the target world coordinates directly so the block
                # center is grid-agnostic (polar or Cartesian).
                wn = w[:, 0] / torch.max(w)
                local_centers[ir * azimuth_divisions + jr] = torch.sum(
                    wn[:, None] * target_pos, dim=0
                ) / torch.sum(wn)

        # Local image centers in Cartesian world coordinates
        local_x = local_centers[:, 0]
        local_y = local_centers[:, 1]
        local_z = local_centers[:, 2]
        # Ground range from each position to local image centers
        local_r = torch.sqrt(
            (pos_new[:, 0][:, None] - local_x[None, :]) ** 2
            + (pos_new[:, 1][:, None] - local_y[None, :]) ** 2
        )

        # Local image center azimuth and elevation angles from each data position
        target_el = torch.arctan2(
            local_z[None, :] - pos_new[:, 2][:, None], local_r
        )
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

        w = torch.sqrt(local_w).transpose(0, 1).unsqueeze(-1)  # [nsweeps, nblocks, 1]
        w = w / torch.clamp(torch.max(w), min=1e-12)
        s = local_d.unsqueeze(0).transpose(0, 2)

        # Solve for 2D/3D position change for each data position from
        # distances from each position to local image centers. Truncated
        # eigenvalue solve of the normal equations instead of a plain
        # lstsq: with a narrow beam (stripmap) all blocks a sweep can see
        # lie in nearly the same direction, so some position directions
        # (e.g. along-track at broadside) are unobservable at that sweep
        # and a full-rank solve amplifies noise into them. Truncation
        # leaves unobserved directions, and sweeps no block observed at
        # all, with zero update. The threshold is relative to the largest
        # eigenvalue over all sweeps: noise amplification scales with the
        # absolute eigenvalue, and a per-sweep relative threshold would
        # also truncate weakly but usefully observed directions (e.g. z
        # from a limited elevation angle spread) on sweeps where every
        # direction is observed about equally well.
        A = w * m
        b = w * s
        AtA = A.transpose(-1, -2) @ A
        Atb = A.transpose(-1, -2) @ b
        evals, evecs = torch.linalg.eigh(AtA)
        evinv = torch.where(
            evals > solve_threshold * torch.max(evals), 1 / evals, 0.0
        )
        d_solved = evecs @ (
            evinv.unsqueeze(-1) * (evecs.transpose(-1, -2) @ Atb)
        )
        d_solved = d_solved.squeeze(-1).transpose(0, 1)

        # Per-sweep coverage for trend removal and the convergence metric so
        # sweeps that no block observed do not dilute them.
        coverage = torch.sum(local_w, dim=0)
        coverage_sum = torch.clamp(torch.sum(coverage), min=1e-12)
        if remove_trend:
            d_solved[0] = weighted_detrend(
                d_solved[0], coverage if use_antenna_weight else None
            )
        rms_error = (
            4
            * torch.pi
            * torch.sqrt(
                torch.sum(coverage * torch.mean(torch.square(d_solved / wl), dim=0))
                / coverage_sum
            ).item()
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

        img = form_image(pos_new)
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
