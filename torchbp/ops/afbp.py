import math
from warnings import warn

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from .backproj import (
    backprojection_polar_2d,
    _prepare_backprojection_polar_2d_args,
    _prepare_backprojection_polar_2d_lanczos_args,
    _prepare_backprojection_polar_2d_knab_args,
)
from ._utils import unpack_polar_grid, parse_interp_method
from ..data import materialize as _materialize
from ..util import next_fast_len

__all__ = [
    "kC0",
    "afbp",
]

if TYPE_CHECKING:
    from ..grid import PolarGrid

kC0 = 299792458.0


def _bp_polar_shared_dealias(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float,
    z0: float,
    data_fmod: float,
    alias_fmod: float,
    att: Tensor | None,
    g: Tensor | None,
    g_extent: list | None,
    normalize: bool,
    interp_method: tuple = ("linear",),
) -> Tensor:
    """Dealiased polar backprojection with an explicit shared z0 reference.

    Same as ``backprojection_polar_2d(..., dealias=True)`` except that the
    dealias carrier reference height is passed in instead of being computed
    from the mean of ``pos``, and batched input is allowed. All afbp
    subapertures must be demodulated with the identical carrier or the
    wavenumber fusion would see spurious phase steps between them.
    ``interp_method`` is a tuple from :func:`parse_interp_method`; the
    lanczos and knab prepare functions insert their parameters after the
    z0 argument, so the dealias/z0 indices patched here are the same for
    every method.
    """
    if interp_method[0] == "lanczos":
        args = list(_prepare_backprojection_polar_2d_lanczos_args(
            data, grid, fc, r_res, pos, d0, False, interp_method[1],
            att, g, g_extent, data_fmod, alias_fmod, normalize))
        op = torch.ops.torchbp.backprojection_polar_2d_lanczos.default
    elif interp_method[0] == "knab":
        args = list(_prepare_backprojection_polar_2d_knab_args(
            data, grid, fc, r_res, pos, d0, False, interp_method[1],
            interp_method[2], att, g, g_extent, data_fmod, alias_fmod,
            normalize))
        op = torch.ops.torchbp.backprojection_polar_2d_knab.default
    else:
        args = list(_prepare_backprojection_polar_2d_args(
            data, grid, fc, r_res, pos, d0, False, att, g, g_extent,
            data_fmod, alias_fmod, normalize))
        op = torch.ops.torchbp.backprojection_polar_2d.default
    args[15] = True  # dealias
    args[16] = z0
    return op(*args)


def _dealias_carrier(
    grid: dict, fc: float, alias_fmod: float, z0: float, device
) -> Tensor:
    """Range carrier removed by the dealias option, ``exp(1j*ph)`` with
    ``ph = pi*(4*fc/c*sqrt(r^2 + z0^2) - alias_fmod*idr)``, shape [nr, 1].

    Multiplying a dealiased image by this restores the ``dealias=False``
    output of :func:`backprojection_polar_2d`. The phase is computed and
    wrapped in float64 because ``keff * d`` is on the order of 1e4..1e5.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    idr = torch.arange(nr, device=device, dtype=torch.float64)
    d = torch.sqrt((r0 + dr * idr) ** 2 + float(z0) ** 2)
    ph = torch.remainder(4.0 * fc / kC0 * d - alias_fmod * idr, 2.0).float() * torch.pi
    return torch.polar(torch.ones_like(ph), ph)[:, None]


def afbp(
    data: Tensor,
    grid: "PolarGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    nsub: int,
    d0: float = 0.0,
    dealias: bool = False,
    data_fmod: float = 0.0,
    alias_fmod: float = 0.0,
    guard_theta: int = 4,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    normalize: bool = True,
    weight_map_downsample: int = 4,
    weight_eps: float | None = None,
    _batched_fusion: bool | None = None,
    dem: Tensor | None = None,
    data_interp_method: "str | tuple" = "linear",
) -> Tensor:
    """
    Accelerated factorized backprojection. [1]_

    Single-level factorized backprojection for short apertures. The aperture
    is split into ``nsub`` subapertures which are backprojected onto a
    shared polar grid decimated ``nsub`` times in theta; because every
    subaperture image uses the same origin, each one holds a contiguous
    patch of the full image's azimuth wavenumber spectrum, aliased into the
    decimated grid band. The full-resolution image is assembled in the 2-D
    wavenumber domain: FFT of the subaperture images, placement of each
    aliased spectrum patch at its true azimuth wavenumber (the patch center
    ``K_r * x_u`` is proportional to the range wavenumber, so the placement
    is computed per range-wavenumber row), and inverse FFT. There is no
    interpolation anywhere, which makes the fusion considerably more
    accurate than the recursive :func:`ffbp` merges; the output matches
    ``backprojection_polar_2d(data, grid, ...)`` including pixel phase.

    The subaperture spectrum patch positions assume a straight track along
    the y-axis in the grid coordinate frame (``pos`` should be centered,
    see :func:`torchbp.util.center_pos`). Deviations of the track from a
    straight line are handled exactly by the subaperture backprojections
    and only degrade the fusion in proportion to the deviation over the
    aperture length. The classical algorithm assumes a slant plane image;
    here the grid is the torchbp ground plane one and the track altitude
    scales the patch positions by the ground-to-slant range ratio, which
    varies over the swath. The fusion follows it by processing the swath
    in range blocks, each with the ratio at its own center (automatic,
    exact in the flat geometry limit). For long apertures over a wide
    near-in swath prefer :func:`ffbp`, which handles altitude exactly,
    possibly with afbp as its base layer (``ffbp(..., afbp_nsub=nsub)``).

    Gradient can be calculated with respect to data. Gradient with respect
    to pos flows through the subaperture backprojections; the fusion
    spectrum placement is treated as constant (same as :func:`ffbp`).

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
        The theta axis should be oversampled with respect to the full
        aperture azimuth bandwidth; the subaperture spectrum patches only
        stay unaliased on the decimated grid when ``(theta1 - theta0) /
        ntheta <= lambda_min / (2 * L / nsub)`` where ``L`` is the aperture
        length (checked, warns when violated).
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    nsub : int
        Number of subapertures, treated as an upper bound. The subaperture
        backprojection cost drops roughly ``nsub`` times below direct
        backprojection. Larger values need a theta-oversampled grid (see
        ``grid``) and increase the relative cost of the guard band and the
        fusion. The value is lowered silently when the requested split
        cannot work: to keep at least two pulses per subaperture, and to
        keep the internal grid alpha extent within the grating-free
        extent of the pulse sampling ``lambda_min / (2 * max pulse
        spacing)`` — a subaperture of few pulses has azimuth
        grating replicas at that period which would alias through the
        fusion, and the internal extent grows with nsub through the guard
        band. A lowered value sits at the constraint boundary, where
        accuracy degrades gradually. Falls back to direct backprojection
        when no split works: the output theta extent alone reaches the
        grating replicas (a direct image of such a grid carries grating
        lobes anyway), or the decimated grid would not be smaller than
        the output.
    d0 : float
        Zero range correction.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    data_fmod : float
        Range modulation frequency applied to input data.
    alias_fmod : float
        Range modulation frequency applied to SAR image.
    guard_theta : int
        Internal guard band on each side of the theta extent in decimated
        grid cells, cropped from the output. The subaperture images treat
        theta as circular, so the response of a target near the grid edge
        wraps around with a broadband spectral skirt that the fusion cannot
        separate; the guard moves the wrap point away from the scene.
        The default suits scenes with targets at the theta edges; a scene
        with an empty margin can use less.
    att : Tensor or None
        Antenna rotation tensor [roll, pitch, yaw], shape [nsweeps, 3].
        Only used with an antenna pattern.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates,
        shape: [elevation, azimuth]. With ``normalize=False`` the output is
        the unnormalized gain-weighted accumulation, matching
        ``backprojection_polar_2d(..., normalize=False)`` (used by
        :func:`ffbp` as its base image). With ``normalize=True`` the
        accumulation is normalized with regularized illumination moment
        maps, matching the antenna-weighted :func:`ffbp` output; the
        direct kernel's exact per-pixel normalization is an aperture-wide
        quantity that cannot be reassembled from subaperture images, so
        ``backprojection_polar_2d(..., normalize=True)`` is only matched
        where the illumination is not weak.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1]. See
        :func:`backprojection_polar_2d`.
    normalize : bool
        See ``g``. Ignored when no antenna pattern is given.
    weight_map_downsample : int
        Decimation of the illumination weight maps used by the
        ``normalize=True`` antenna path, as in :func:`ffbp`. The maps are
        additionally computed from a ``1/nsub`` subset of the pulses (the
        per-pulse footprints move slowly along the aperture); computed
        from every pulse they can cost more than the backprojection
        itself.
    weight_eps : float or None
        Regularization of the antenna normalization, see :func:`ffbp`.
    data_interp_method : str or tuple
        Range interpolation method of the subaperture backprojections, see
        :func:`backprojection_polar_2d`. Gradients are only supported with
        the default "linear".

    Returns
    -------
    img : Tensor
        Polar format radar image. Shape is [nr, ntheta].

    References
    ----------
    .. [1] L. Zhang, H. -l. Li, Z. -j. Qiao and Z. -w. Xu, "A Fast BP
    Algorithm With Wavenumber Spectrum Fusion for High-Resolution Spotlight
    SAR Imaging," in IEEE Geoscience and Remote Sensing Letters, vol. 11,
    no. 9, pp. 1460-1464, Sept. 2014.
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    if dem is not None:
        raise NotImplementedError(
            "afbp does not support dem; use ffbp (with afbp_nsub=1) or "
            "backprojection_polar_2d")
    # The wavenumber-domain fusion gathers pulses with an index tensor, so
    # a lazy input is materialized whole here: afbp accepts LazyData but is
    # not memory efficient with it (inside ffbp the input is leaf-sized).
    data = _materialize(data)
    if data.dim() != 2:
        raise ValueError("data shape should be [nsweeps, samples]")
    if pos.dim() != 2 or pos.shape[0] != data.shape[0]:
        raise ValueError("pos shape should be [nsweeps, 3]")
    if guard_theta < 0:
        raise ValueError("guard_theta should be >= 0")
    data_interp_method = parse_interp_method(
        data_interp_method, name="data_interp_method")
    if g is not None and normalize:
        # The direct kernel's per-pixel sum(g)/sum(g^2) normalization is an
        # aperture-wide quantity that cannot be reassembled from
        # subaperture images. Instead the unnormalized gain-weighted
        # accumulation is normalized with decimated illumination moment
        # maps exactly like the antenna-weighted ffbp, whose output this
        # matches (not backprojection_polar_2d(..., normalize=True), which
        # differs at weakly illuminated pixels).
        img = afbp(
            data, grid, fc, r_res, pos, nsub, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            guard_theta=guard_theta, att=att, g=g, g_extent=g_extent,
            normalize=False, _batched_fusion=_batched_fusion,
            data_interp_method=data_interp_method)
        from .ffbp import _illumination_pulse_decimated, _weighted_normalize
        # The per-pulse footprints move slowly along the aperture, so the
        # maps can be computed from a pulse subset (computed from every
        # pulse they can cost more than the whole backprojection); the map
        # grid itself must stay fine to resolve the sharp footprint
        # boundaries, see _illumination_pulse_decimated.
        w1, w2 = _illumination_pulse_decimated(
            pos, att, g, g_extent, grid, weight_map_downsample, nsub)
        return _weighted_normalize(img, w1, w2, eps=weight_eps)

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nsweeps, nsamples = data.shape
    device = data.device

    # Direct backprojection when no split is possible at all: a split
    # needs at least two subapertures of at least two pulses.
    if nsub <= 1 or nsweeps < 4:
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize,
            interp_method=data_interp_method)[0]

    # Effective range wavenumber center of the image. data_fmod shifts the
    # matched filter phase the same way as in cfbp's keff.
    krc = 4.0 * math.pi * fc / kC0 - data_fmod / r_res
    # Range spectrum center of the dealiased image in cycles/m.
    nu_c = alias_fmod / (2.0 * dr) - data_fmod / (2.0 * r_res)

    # Single transfer instead of one device sync per use.
    pos_y = pos[:, 1].double().cpu()
    z0 = float(pos[:, 2].double().mean())
    # Ground-to-slant factor: with nonzero altitude the azimuth carrier of
    # a pixel at ground range r is kr * x * rd(r) with rd(r) = r / sqrt(r^2
    # + z0^2), so the patch placement depends on ground range. The fusion
    # runs in the range wavenumber domain where the rows mix all ranges;
    # rd is handled by fusing the swath in range blocks, each with the
    # factor at its center (see below). fac here is the swath center value
    # used by the split sizing and validity checks.
    r_mid = 0.5 * (r0 + r1)
    fac = r_mid / math.sqrt(r_mid**2 + z0**2)
    kr_max = abs(krc) + min(math.pi / dr, math.pi / r_res)

    # nsub is an upper bound. The internal grid alpha extent n_c * nsub *
    # dtheta must stay within the grating-free extent of the pulse
    # sampling, 2*pi / (kr * dp * fac) with dp the pulse spacing: a
    # subaperture of few pulses has azimuth grating replicas at that alpha
    # period, and once the internal extent (output plus guard and ceil
    # padding cells, whose width scales with nsub) reaches them, they
    # alias through the fusion into the image. The output extent is
    # fixed, so the constraint only bounds the guard and padding
    # contribution and with it nsub: lower nsub until the extent fits,
    # keeping at least two pulses per subaperture.
    dp = float((pos_y[1:] - pos_y[:-1]).abs().max())
    # Grating-free extent in units of dtheta.
    e_cells = 2.0 * math.pi / (kr_max * max(dp, 1e-12) * fac * dtheta)
    nsub_max = min(nsub, nsweeps // 2)
    nsub_eff = nsub_max
    while nsub_eff > 1 and (
            (-(-ntheta // nsub_eff) + 2 * guard_theta) * nsub_eff > e_cells):
        nsub_eff -= 1
    if nsub_eff <= 1:
        warn("afbp: the pulse spacing cannot support the internal grid "
             "extent for any subaperture split (the output theta extent "
             "reaches the azimuth grating replicas of the pulse sampling "
             "within the guard band); falling back to direct "
             "backprojection")
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize,
            interp_method=data_interp_method)[0]
    nsub = nsub_eff

    # Fall back to direct backprojection when the split cannot pay off: a
    # decimated grid that would not be smaller than the output grid once
    # the guard band and ceil padding are added (the subaperture
    # backprojection work scales with the decimated grid size). Happens
    # when afbp is applied to tiny grids, e.g. a deep ffbp recursion base
    # level.
    n_c_full = -(-ntheta // nsub) + 2 * guard_theta
    if n_c_full >= ntheta:
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize,
            interp_method=data_interp_method)[0]

    # Split the track into nsub contiguous chunks.
    bounds = [round(i * nsweeps / nsub) for i in range(nsub + 1)]
    m = max(bounds[i + 1] - bounds[i] for i in range(nsub))
    x_u = torch.tensor([float(pos_y[bounds[u] : bounds[u + 1]].mean())
                        for u in range(nsub)], dtype=torch.float64)

    # Validity checks. Patch width at the largest occupied range wavenumber
    # must fit the decimated grid band, and the linear phase approximation
    # of the patch placement must hold across the theta extent.
    order = torch.argsort(x_u)
    x_s = x_u[order]
    spacing = (x_s[1:] - x_s[:-1]).abs()
    l_sub = float(spacing.max()) if nsub > 1 else 0.0
    if l_sub <= 0.0:
        warn("afbp: subapertures have no along-track extent, falling back "
             "to direct backprojection")
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize,
            interp_method=data_interp_method)[0]
    if kr_max * l_sub * fac * nsub * dtheta > 2.0 * math.pi:
        warn(f"afbp: subaperture spectrum patch does not fit the decimated "
             f"theta grid band; decrease nsub or the grid theta step "
             f"(dtheta <= {2.0 * math.pi / (kr_max * l_sub * fac * nsub):.2e})")
    alpha_max = max(abs(theta0), abs(theta1))
    # Quadratic phase error bound alpha * l <= r / 4: ranges below r_qpe are
    # degraded. Grids often start much closer than the actual scene content
    # (or than the ranges where degradation matters), so only warn when a
    # meaningful part of the swath is affected.
    r_qpe = 4.0 * alpha_max * l_sub
    if r_qpe > r0 + 0.02 * (r1 - r0):
        warn(f"afbp: quadratic phase error bound alpha*l <= r/4 violated "
             f"for r < {r_qpe:.1f}; image degraded at near range; increase "
             f"nsub or use ffbp")

    # Internal decimated grid. Rounding n_c up to a fast FFT length only
    # adds guard cells; the decimation alignment holds for any n_c. The
    # extra cells are split evenly between the two guard bands. The fast
    # length is only used when it keeps the internal extent within the
    # grating-free extent (n_c_full itself fits by the nsub cap above).
    n_c = next_fast_len(n_c_full)
    if n_c * nsub > e_cells:
        n_c = n_c_full
    if n_c >= ntheta:
        # The guard and padding ate the decimation gain.
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize,
            interp_method=data_interp_method)[0]
    n_fine = n_c * nsub
    guard_lo = (n_c - (-(-ntheta // nsub))) // 2
    theta0_i = theta0 - guard_lo * nsub * dtheta
    grid_c = {
        "r": (r0, r1),
        "theta": (theta0_i, theta0_i + n_c * nsub * dtheta),
        "nr": nr,
        "ntheta": n_c,
    }

    # Padded chunks for one batched backprojection call, built with single
    # gathers instead of per-chunk copies. Rows past a chunk end repeat
    # its last pulse; the repeated data rows are zeroed, which makes the
    # padding exact. Equal chunks need no padding at all and the gather
    # reduces to a reshape.
    if nsweeps % nsub == 0:
        data_b = data.reshape(nsub, m, nsamples)
        pos_b = pos.reshape(nsub, m, 3)
        att_b = att.reshape(nsub, m, 3) if att is not None else None
    else:
        idx = (torch.tensor(bounds[:-1], device=device)[:, None]
               + torch.arange(m, device=device)[None, :])
        ends = torch.tensor(bounds[1:], device=device)[:, None]
        valid = idx < ends
        idx = torch.minimum(idx, ends - 1).reshape(-1)
        data_b = data[idx].view(nsub, m, nsamples) * valid[:, :, None]
        pos_b = pos[idx].view(nsub, m, 3)
        att_b = att[idx].view(nsub, m, 3) if att is not None else None

    # The kernel dealias path does not support gradients; with gradients
    # the carrier is instead removed with a differentiable multiply, which
    # matches the kernel dealias up to its float32 phase rounding.
    use_torch_dealias = torch.is_grad_enabled() and (
        data.requires_grad or pos.requires_grad)
    if use_torch_dealias:
        if data_interp_method[0] != "linear":
            raise ValueError(
                f"data_interp_method={data_interp_method[0]!r} does not "
                "support gradients, only \"linear\" does")
        args = _prepare_backprojection_polar_2d_args(
            data_b, grid_c, fc, r_res, pos_b, d0, False, att_b, g, g_extent,
            data_fmod, alias_fmod, normalize)
        imgs = torch.ops.torchbp.backprojection_polar_2d.default(*args)
        imgs = imgs * _dealias_carrier(grid_c, fc, alias_fmod, z0, device).conj()
    else:
        imgs = _bp_polar_shared_dealias(
            data_b, grid_c, fc, r_res, pos_b, d0, z0, data_fmod, alias_fmod,
            att_b, g, g_extent, normalize,
            interp_method=data_interp_method)

    # The guard band and the ceil padding can push internal grid columns
    # past the polar domain |theta| <= 1. The backprojection kernel
    # computes the smooth continuation of the azimuth signal there (its
    # phase stays linear in theta for a straight track), which the linear
    # spectrum placement below handles like any interior column, so the
    # columns can be kept: they carry valid guard content for output grids
    # that themselves extend past |theta| = 1 (ffbp guard band base
    # images).

    # Wavenumber-domain fusion. Azimuth FFT of every subaperture image; a
    # component exp(-1j*k*x_u*rd*alpha) of a dealiased image lands at
    # azimuth frequency -k*x_u*rd/(2*pi) cycles per unit theta. The
    # subapertures are processed in ascending x_u order so that the
    # per-bin contributors form a contiguous index run. An ascending track
    # is already in order; skip the permute copy then.
    if not torch.equal(order, torch.arange(nsub)):
        imgs = imgs[order.to(device)]
    Sa = torch.fft.fft(imgs, dim=-1)

    # Range blocks: rd(r) varies over the swath and after the range FFT
    # the rows mix all ranges, so a single factor would misplace the
    # patches of off-center ranges by up to (rd variation) * kr * |x_u|.
    # Each block is fused with the factor at its own center, sized to keep
    # the misplacement below a small fraction of the decimated band. One
    # block when z0 = 0 (or a short aperture) where rd is constant.
    r_rows = r0 + dr * torch.arange(nr, dtype=torch.float64)
    rd_rows = r_rows / torch.sqrt(r_rows**2 + z0**2)
    rd_span = float(rd_rows.max() - rd_rows.min())
    x_max = float(x_u.abs().max())
    n_blocks = 1 + int(rd_span * abs(krc) * x_max * nsub * dtheta / (0.1 * 2.0 * math.pi))
    # Short blocks lose range wavenumber resolution for the placement.
    n_blocks = max(1, min(n_blocks, nr // 32))
    # Equal-rd partition, denser at near range where rd varies fastest.
    if n_blocks > 1:
        levels = torch.linspace(float(rd_rows[0]), float(rd_rows[-1]), n_blocks + 1)[1:-1]
        bnds = torch.searchsorted(rd_rows, levels).tolist()
        row_bounds = [0]
        for b in bnds:
            b = max(b, row_bounds[-1] + 32)
            if b > nr - 32:
                break
            row_bounds.append(b)
        row_bounds.append(nr)
    else:
        row_bounds = [0, nr]

    x_c = float(0.5 * (x_s[0] + x_s[-1]))
    spacing_min = max(float(spacing.min()), 1e-9)
    cols = (torch.arange(n_fine, device=device) % n_c)[None, :]
    # Roll the region weight off smoothly between the physical patch
    # half-extent and the region edge instead of a hard cut. The hard
    # truncation of the patch leakage tails is a spectral discontinuity
    # whose image-domain ripple reaches theta regions with no true
    # content, where the antenna Wiener normalization then amplifies it
    # into visible artifacts; the taper keeps the residual localized near
    # the content that caused it. Contributions at less than 0.55 *
    # subaperture spacing keep unit weight, so the seam-sum region between
    # adjacent subapertures is unaffected.
    x_taper = 0.55 * float(spacing.max())

    # The fusion runs either as a loop over the range blocks or as batched
    # tensor ops over a zero-padded block stack. The batched form exists
    # for the GPU, where the loop is a kernel launch storm; on CPU it
    # moves several times more memory and loses. _batched_fusion is a
    # testing override.
    use_batched = (_batched_fusion if _batched_fusion is not None
                   else device.type == "cuda")
    if not use_batched:
        # Per-block loop: on CPU the padded batched path below moves several
        # times more memory (small near-range blocks pad to the shared FFT
        # length and the gathers broadcast large index tensors), while the
        # per-op overhead the batching avoids is negligible.
        #
        # As in the batched path, the placement is computed in float64 only
        # for the small 1-D tables and used in float32: the placement only
        # needs to be accurate to a small fraction of a subaperture
        # spacing, and float64 weights would also promote the spectrum
        # accumulation to complex128.
        nua = torch.fft.fftfreq(n_fine, d=dtheta, dtype=torch.float64, device=device)
        nua_f = nua.float()
        xs_f32 = x_s.float().to(device)
        # Fused kernel for the placement + gather + accumulate between the
        # range FFTs: it computes the per-element placement on the fly from
        # the 1-D tables instead of materializing several [nrf, n_fine]
        # intermediates per contribution. The tensor-op fallback is kept
        # for gradients (the op has no autograd; the spectrum gathers are
        # in the data/pos gradient path) and for non-CPU devices.
        use_fused = device.type == "cpu" and not (
            torch.is_grad_enabled() and Sa.requires_grad)
        fine = Sa.new_zeros((nr, n_fine))
        for b0, b1 in zip(row_bounds[:-1], row_bounds[1:]):
            nrb = b1 - b0
            # Pad the block range FFT to a fast length; the analysis is
            # linear per row so cropping after the inverse FFT is exact.
            nrf = next_fast_len(nrb)
            fac_b = float(0.5 * (rd_rows[b0] + rd_rows[b1 - 1]))
            Sb = torch.fft.fft(Sa[:, b0:b1, :], n=nrf, dim=-2)
            fr = torch.fft.fftfreq(nrf, d=dr, dtype=torch.float64, device=device)
            half = 1.0 / (2.0 * dr)
            # Physical range frequency offset of each row, unwrapped
            # around the modulation-shifted spectrum center.
            fr_phys = torch.remainder(fr - nu_c + half, 2.0 * half) - half
            kr = krc + 2.0 * math.pi * fr_phys  # [nrf]
            # A row with kr ~ 0 (range grid step below ~lambda/4) would
            # give an infinite band and NaN from the wrap below; such rows
            # carry no signal, park them on a harmless value.
            kr = torch.where(kr.abs() < 1e-6, torch.full_like(kr, 1e-6), kr)
            # Along-track position owning each (range wavenumber row,
            # azimuth bin) is x_eq = nua * inv_kr.
            inv_kr = (-2.0 * math.pi / (kr * fac_b)).float()  # [nrf]
            # Each subaperture patch is reconstructed over a full
            # decimated band centered on the patch: the overlapping
            # regions of adjacent subapertures are summed, which keeps the
            # patch edge transitions that a disjoint tiling would
            # truncate. Content outside every region (past the aperture
            # edges) is zero. Only the subapertures whose region covers a
            # bin contribute: a contiguous run of at most ~(region width /
            # subaperture spacing) + 1 indices independent of nsub.
            x_half = (math.pi * n_c / (n_fine * dtheta) / (kr * fac_b)).abs().float()
            kmax = min(nsub, int(math.ceil(2.0 * float(x_half.max()) / spacing_min)) + 1)
            # x_eq spans one alias band of the fine grid; recentering the
            # span on the subaperture positions makes the placement
            # consistent with the aliased carrier a direct backprojection
            # of an off-center track produces (no-op for a centered one).
            band = (2.0 * math.pi / (kr * fac_b * dtheta)).abs().float()  # [nrf]
            if use_fused:
                fine_b = torch.ops.torchbp.afbp_fuse.default(
                    Sb, nua_f, xs_f32, inv_kr, x_half, band,
                    x_c, x_taper, kmax)
            else:
                x_eq = nua_f[None, :] * inv_kr[:, None]
                bnd = band[:, None]
                x_eq = torch.remainder(x_eq - x_c + bnd / 2, bnd) + (x_c - bnd / 2)
                u0 = torch.bucketize(x_eq - x_half[:, None], xs_f32)
                rows = torch.arange(nrf, device=device)[:, None]
                fine_b = Sb.new_zeros((nrf, n_fine))
                for k in range(kmax):
                    uk = u0 + k
                    u = uk.clamp(max=nsub - 1)
                    w = _region_weight(
                        (x_eq - xs_f32[u]).abs(), x_half[:, None], x_taper)
                    w = w * (uk < nsub)
                    fine_b += Sb[u, rows, cols] * w
            fine[b0:b1] = torch.fft.ifft(fine_b, dim=-2)[:nrb]
        fine *= nsub
        return _afbp_finish(fine, grid, fc, alias_fmod, z0, dealias,
                            guard_lo * nsub, ntheta, device)

    # Cap the block size so that every block can be padded to one shared
    # FFT length: the whole per-block fusion then runs as a handful of
    # batched tensor ops instead of a kernel launch storm proportional to
    # the block count (float64 is also very slow on most GPUs, so the
    # placement tables are float32 here). Splitting a block only refines
    # the range localization of its factor, so it is always valid.
    if len(row_bounds) > 2:
        b_cap = 128
        rb = [0]
        for b0, b1 in zip(row_bounds[:-1], row_bounds[1:]):
            pieces = -(-(b1 - b0) // b_cap)
            for p in range(1, pieces + 1):
                rb.append(b0 + round(p * (b1 - b0) / pieces))
        row_bounds = rb
    sizes = [b1 - b0 for b0, b1 in zip(row_bounds[:-1], row_bounds[1:])]
    nb = len(sizes)
    nrf = next_fast_len(max(sizes))

    # Row scatter/gather map between the image rows and the zero-padded
    # block stack.
    dest_idx = torch.cat([
        torch.arange(b * nrf, b * nrf + s) for b, s in enumerate(sizes)
    ]).to(device)
    Sblk = Sa.new_zeros((nsub, nb * nrf, n_c))
    Sblk.index_copy_(1, dest_idx, Sa)
    S = torch.fft.fft(Sblk.view(nsub, nb, nrf, n_c), dim=-2)

    # Placement tables, shared by all blocks up to the per-block factor.
    # Computed in float64 and used in float32: the placement only needs to
    # be accurate to a small fraction of a subaperture spacing, and
    # float64 arithmetic is very slow on most GPUs.
    fr = torch.fft.fftfreq(nrf, d=dr, dtype=torch.float64)
    half = 1.0 / (2.0 * dr)
    # Physical range frequency offset of each row, unwrapped around the
    # modulation-shifted spectrum center.
    fr_phys = torch.remainder(fr - nu_c + half, 2.0 * half) - half
    kr = krc + 2.0 * math.pi * fr_phys  # [nrf]
    # See the kr ~ 0 note in the loop path.
    kr = torch.where(kr.abs() < 1e-6, torch.full_like(kr, 1e-6), kr)
    nua = torch.fft.fftfreq(n_fine, d=dtheta, dtype=torch.float64)
    # Along-track position owning each (range wavenumber row, azimuth bin)
    # at rd = 1, and the half width of one decimated band in the same
    # units. Scaled per block by 1 / fac_b.
    base_xeq = (-2.0 * math.pi * nua[None, :] / kr[:, None]).float().to(device)
    base_xh = (math.pi * n_c / (n_fine * dtheta) / kr).abs().float().to(device)
    inv_fac = torch.tensor(
        [1.0 / float(0.5 * (rd_rows[b0] + rd_rows[b1 - 1]))
         for b0, b1 in zip(row_bounds[:-1], row_bounds[1:])],
        dtype=torch.float32, device=device)
    xs_f32 = x_s.float().to(device)

    x_eq = base_xeq[None] * inv_fac[:, None, None]  # [nb, nrf, n_fine]
    x_half = (base_xh[None] * inv_fac[:, None])[:, :, None]  # [nb, nrf, 1]
    # Recenter the alias band on the subaperture positions (see the loop
    # path).
    base_band = (2.0 * math.pi / (kr * dtheta)).abs().float().to(device)
    band = (base_band[None] * inv_fac[:, None])[:, :, None]
    x_eq = torch.remainder(x_eq - x_c + band / 2, band) + (x_c - band / 2)
    # Each subaperture patch is reconstructed over a full decimated band
    # centered on the patch: the overlapping regions of adjacent
    # subapertures are summed, which keeps the patch edge transitions that
    # a disjoint tiling would truncate. Content outside every region (past
    # the aperture edges) is zero. Only the subapertures whose region
    # covers a bin contribute: a contiguous run of at most ~(region width
    # / subaperture spacing) + 1 indices independent of nsub.
    kmax = min(nsub, int(math.ceil(
        2.0 * float(base_xh.max()) * float(inv_fac.max()) / spacing_min)) + 1)
    u0 = torch.bucketize(x_eq - x_half, xs_f32)
    bidx = torch.arange(nb, device=device)[:, None, None]
    ridx = torch.arange(nrf, device=device)[None, :, None]
    cidx = (torch.arange(n_fine, device=device) % n_c)[None, None, :]
    fine_b = S.new_zeros((nb, nrf, n_fine))
    for k in range(kmax):
        uk = u0 + k
        u = uk.clamp(max=nsub - 1)
        w = _region_weight((x_eq - xs_f32[u]).abs(), x_half, x_taper)
        w = w * (uk < nsub)
        fine_b += S[u, bidx, ridx, cidx] * w
    fine = torch.fft.ifft(fine_b, dim=-2).reshape(nb * nrf, n_fine)
    fine = fine.index_select(0, dest_idx)
    fine *= nsub
    return _afbp_finish(fine, grid, fc, alias_fmod, z0, dealias,
                        guard_lo * nsub, ntheta, device)


def _region_weight(dist: Tensor, x_half: Tensor, x_taper: float) -> Tensor:
    """Subaperture region weight: 1 inside ``x_taper``, raised cosine roll
    to 0 at the region edge ``x_half``, 0 beyond. Falls back to a hard cut
    when the region is not wider than the taper start."""
    span = (x_half - x_taper).clamp(min=1e-9)
    t = ((dist - x_taper) / span).clamp(min=0.0, max=1.0)
    w = 0.5 * (1.0 + torch.cos(math.pi * t))
    return torch.where(x_half > x_taper, w, (dist <= x_half).to(w.dtype))


def _afbp_finish(
    fine: Tensor,
    grid: dict,
    fc: float,
    alias_fmod: float,
    z0: float,
    dealias: bool,
    gt: int,
    ntheta: int,
    device,
) -> Tensor:
    """Inverse azimuth FFT of the assembled fine spectrum, guard crop and
    optional carrier restore."""
    out = torch.fft.ifft(fine, dim=-1)
    out = out[:, gt : gt + ntheta]
    if not dealias:
        out = out * _dealias_carrier(grid, fc, alias_fmod, z0, device)
    return out
