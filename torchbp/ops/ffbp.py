import math
import torch
from torch import Tensor
from typing import Union, TYPE_CHECKING
from warnings import warn
from .afbp import afbp
from .backproj import backprojection_polar_2d, backprojection_polar_2d_tx_power, _backprojection_polar_2d_tx_power_accum, _tx_power_finish
from .polar_interp import ffbp_merge2, ffbp_merge2_poly, ffbp_merge2_poly_weighted, ffbp_tx_power_merge2, compute_knab_poly_coefs_full, select_knab_poly_degree
from ..util import center_pos
from copy import deepcopy
from ._utils import AntennaPattern, unpack_polar_grid

if TYPE_CHECKING:
    from ..grid import PolarGrid

# Recursion floor: stop splitting subapertures where the steady-state theta
# guard band would exceed this fraction of the subaperture image (see the
# min_core comment in _ffbp_impl). 1/3 measured fastest on CPU (order 6):
# stricter fractions stop profitable splits (0.1 was 2.3x slower), while
# unbounded splitting wastes ~25% on guard-dominated tiny subapertures.
_GUARD_STOP_FRAC = 1.0 / 3.0


def compute_subaperture_illumination(
    pos: Tensor,
    att: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    decimation: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Compute illumination weight maps (sum of gains and sum of squared gains) for a subaperture.

    This computes both W1 (sum of gains) and W2 (sum of squared gains) for each pixel,
    which are used to correctly weight subaperture contributions during FFBP merge.

    Needed for antenna pattern weighted FFBP.

    Parameters
    ----------
    pos : Tensor
        Platform positions for the subaperture. Shape: [nsweeps, 3].
    att : Tensor
        Antenna pointing for each sweep. Shape: [nsweeps, 3] with [roll, pitch, yaw].
    g : Tensor
        Antenna gain pattern. Shape: [g_nel, g_naz].
    g_extent : list
        Pattern extent: [g_el0, g_az0, g_el1, g_az1] in radians.
    grid : PolarGrid or dict
        Polar grid for output weight map.
    decimation : int
        Decimation factor for output (1 = full resolution, 4 = 1/16 size).

    Returns
    -------
    tuple[Tensor, Tensor]
        (w1_map, w2_map) where:
        - w1_map: Sum of gains. Shape: [nr // decimation, ntheta // decimation].
        - w2_map: Sum of squared gains. Shape: [nr // decimation, ntheta // decimation].
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    antenna = AntennaPattern(g, g_extent)
    args = antenna.to_cpp_args()[:-2]

    return torch.ops.torchbp.compute_illumination.default(
        pos, att, *args,
        r0, dr, theta0, dtheta,
        nr, ntheta, decimation
    )


def _weighted_normalize(
    A: Tensor,
    w1: Tensor,
    w2: Tensor,
    eps: float | None = None,
    rel_floor: float = 1e-3,
) -> Tensor:
    """
    Regularized (Wiener) normalization of antenna-pattern-weighted FFBP.

    The weighted merge accumulates, per pixel, the gain-weighted sum
    ``A = Sum_i g_i d_i`` and the illumination moments ``W1 = Sum_i g_i`` and
    ``W2 = Sum_i g_i^2`` (``g_i`` is the amplitude antenna gain toward the pixel on
    pulse ``i``). The SNR-optimal image is the matched filter ``A / W2`` rescaled
    by ``W1`` to unweighted-backprojection brightness, i.e. ``A * W1 / W2``.

    That divides by the illumination power ``W2``, which underflows to zero at
    swath edges / antenna nulls, so a single low-gain sample blows a pixel up.
    Regularizing the divisor fixes it::

        img = A * W1 / (W2 + lambda)

    ``W2 >> lambda`` (every illuminated target) recovers the matched filter
    unchanged; ``W2 <~ lambda`` rolls off to zero instead of amplifying noise. The
    value of ``lambda`` is not critical. It only sets how far below the signal
    the weakly-illuminated residual sits, so by default it is a small fraction
    ``rel_floor`` of the peak illumination power ``W2_max``.

    ``A`` is modified in place and returned. The normalization is applied in
    row chunks so that decimated weight maps are never materialized at full
    image resolution (memory saving is the reason to decimate them in the
    first place). Decimated maps are interpolated at the kernels' sample
    positions ``dec * i`` (not cell centers).
    """
    # Linear interpolation cannot exceed the sample max, so the decimated
    # max equals the upsampled max.
    w2max = float(w2.max())
    if w2max <= 0.0:
        # nothing illuminated. A = 0
        return A
    lam = max(float(eps) ** 2 if eps is not None else 0.0, rel_floor * w2max)

    nr, ntheta = A.shape[-2], A.shape[-1]
    nr_w, ntheta_w = w1.shape[-2], w1.shape[-1]
    full_res = nr_w == nr and ntheta_w == ntheta
    dec_r = -(-nr // nr_w)      # ceil, inverse of out = ceil(n / dec)
    dec_t = -(-ntheta // ntheta_w)

    if not full_res:
        # Theta-axis sample positions, shared by all chunks.
        xt = torch.arange(ntheta, device=A.device, dtype=torch.float32) / dec_t
        t0 = xt.long().clamp(max=ntheta_w - 1)
        t1 = torch.clamp(t0 + 1, max=ntheta_w - 1)
        ft = (xt - t0.to(xt.dtype))[None, :]
        w1 = w1.float()
        w2 = w2.float()

    # Bound the temporaries to ~chunk * ntheta floats.
    chunk = max(1, 8_000_000 // ntheta)
    for rs in range(0, nr, chunk):
        re = min(rs + chunk, nr)
        if full_res:
            # w1c/w2c are views; don't modify them in place.
            w1c = w1[..., rs:re, :]
            w2c = w2[..., rs:re, :]
            A[..., rs:re, :] *= w1c / (w2c + lam)
        else:
            xr = torch.arange(rs, re, device=A.device, dtype=torch.float32) / dec_r
            r0 = xr.long().clamp(max=nr_w - 1)
            r1 = torch.clamp(r0 + 1, max=nr_w - 1)
            fr = (xr - r0.to(xr.dtype))[:, None]

            def up(w):
                rows = w.index_select(-2, r0).lerp_(w.index_select(-2, r1), fr)
                return rows.index_select(-1, t0).lerp_(rows.index_select(-1, t1), ft)

            w1c = up(w1)
            w2c = up(w2)
            w2c += lam
            w1c /= w2c
            A[..., rs:re, :] *= w1c
    return A


def ffbp(
    data: Tensor,
    grid: "PolarGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int = 2,
    d0: float = 0.0,
    interp_method: tuple = ("knab", 6, 1.5),
    oversample_r: float = 1.4,
    oversample_theta: float = 1.4,
    grid_oversample: float = 1,
    dealias: bool = False,
    data_fmod: float = 0,
    alias_fmod: float = None,
    use_poly: bool = True,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    weight_map_downsample: int = 1,
    weight_eps: float | None = None,
    afbp_nsub: int = 1,
    guard_max_ratio: float = 0.125,
) -> Tensor:
    """
    Fast factorized backprojection.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object (recommended): ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict (legacy): ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
        Batched input is not supported.
    stages : int
        Number of recursions. This is an upper bound: the recursion also
        stops early where a further split would make the subaperture grids
        guard band dominated (guard above ~1/3 of the subaperture image,
        reached when the subaperture core drops below ~8 * (order/2 + 2)
        theta bins), where deeper splitting costs more in guard columns
        than it saves in backprojection. Large values are therefore safe.
    divisions : int
        Number of subapertures divisions per stage. Default is 2.

        Subaperture grids at every level include an automatically sized
        theta guard band so that the merge interpolation has full window
        support at the scene theta edges. The guard band can extend past
        |theta| = 1 (e.g. with a full 180 degree scene), where the
        backprojection and merge kernels evaluate the smooth continuation
        of the azimuth signal past the fold, so edge accuracy does not
        depend on the subaperture size.
    d0 : float
        Zero range correction.
    interp_method : tuple
        Interpolation method: ("knab", order, oversample) where order is the
        number of samples used and oversample is the oversampling factor.
    oversample_r : float
        Internally oversample range by this amount to avoid aliasing.
    oversample_theta : float
        Internally oversample theta by this amount to avoid aliasing.
    grid_oversample : float
        Oversample ratio of the output grid. Used only for calculating the
        alias_fmod when alias_fmod is None. 1 for critically sampled grid, 2 for
        twice oversampled grid, etc.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    data_fmod : float
        Range modulation frequency applied to input data.
    alias_fmod : float or None
        Range modulation frequency applied to SAR image.
        If None, the alias frequency is calculated automatically assuming
        that the image spectrum is at positive frequencies. grid_oversample
        should be given when automatic calculation is used.

        Note that when `dealias` is True and the `alias_fmod` is calculated
        automatically, the output will not have `alias_fmod` modulation and
        `alias_fmod` is used only internally to decrease interpolation errors.
    use_poly: bool
        Use polynomial approximation for interpolation.
    att : Tensor or None
        Antenna pointing for each sweep. Shape: [nsweeps, 3] with [roll, pitch, yaw].
        Required when g is provided for antenna pattern weighting.
    g : Tensor or None
        Antenna gain pattern. Shape: [g_nel, g_naz].
        When provided, applies SNR-optimal weighted combination of subapertures.
    g_extent : list or None
        Antenna pattern extent: [g_el0, g_az0, g_el1, g_az1] in radians.
        Required when g is provided.
    weight_map_downsample : int
        Downsample factor for weight maps relative to image grid. Default 1.
        Lower values give more accurate weights but use more memory.
        Higher values reduce memory use and is faster to calculate, but
        increases error especially if the antenna pattern is not wide.
    weight_eps : float or None
        Regularization for the antenna-pattern (Wiener) normalization
        ``img = A*W1/(W2 + weight_eps**2)``, in units of square-root illumination
        power. See :func:`_weighted_normalize`. If None (default) a small fixed
        fraction of the peak illumination power is used, which keeps illuminated
        pixels at the matched-filter value and only suppresses weakly illuminated
        ones. Exact value should not be critical.
        Ignored when no antenna pattern is given.
    afbp_nsub : int
        If greater than 1, compute the base level subaperture images with
        :func:`afbp` using this many sub-subapertures instead of direct
        backprojection, which reduces the base level cost roughly
        ``afbp_nsub`` times. This does not change the output accuracy,
        which stays dominated by the merge interpolation error; the gain
        is that ``stages`` can be reduced by about ``log2(afbp_nsub)``
        merge levels at constant total cost, which does reduce the error.
        Default is 1 (direct backprojection).
    guard_max_ratio : float
        Cap of the automatic theta guard band width, per side, as a
        fraction of the subaperture grid core theta bins (a minimum of a
        few interpolation windows is always kept). The guard band adds
        this fraction to every image in the merge tree, so it bounds the
        memory overhead to ``1 + 2 * guard_max_ratio``. When the
        subaperture origin offsets are comparable to the near range, full
        interpolation support at the scene theta edges can require a guard
        wider than the cap; a warning is emitted and edge accuracy at near
        range degrades gracefully instead of exhausting memory. Increase
        for full edge accuracy at more memory, decrease (0 gives the
        minimum window support) for less memory. Default is 0.125.

    Returns
    -------
    img : Tensor
        SAR image.
    """
    # Convert Grid object to dict for backward compatibility
    if hasattr(grid, 'to_dict'):
        grid = grid.to_dict()

    nsweeps = data.shape[0]
    device = data.device

    output_alias = alias_fmod is not None
    if alias_fmod is None:
        if grid_oversample < 1:
            warn(f"Grid is undersampled. grid_oversample={grid_oversample}")
        im_margin = max(0, grid_oversample * oversample_r - 1)
        alias_fmod = -torch.pi * (1 - im_margin / (1 + im_margin))

    # Parse interpolation method - only knab is supported
    if interp_method[0] != "knab":
        raise ValueError("interp_method should be ('knab', order, oversample)")
    if len(interp_method) != 3:
        raise ValueError("interp_method should be ('knab', order, oversample)")

    knab_order = interp_method[1]
    knab_oversample = interp_method[2]

    # Precompute polynomial coefficients once for all merge operations
    poly_degree = select_knab_poly_degree(knab_oversample, knab_order)
    poly_coefs = compute_knab_poly_coefs_full(knab_order, knab_oversample, poly_degree)

    # Validate antenna pattern parameters
    use_antenna_pattern = g is not None
    if use_antenna_pattern:
        if att is None:
            raise ValueError("att must be provided when antenna pattern g is specified")
        if g_extent is None:
            raise ValueError("g_extent must be provided when antenna pattern g is specified")
        if att.shape[0] != nsweeps:
            raise ValueError(f"att must have {nsweeps} sweeps, got {att.shape[0]}")

    if nsweeps < divisions:
        # Too few sweeps to split into subapertures
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod,
            alias_fmod=alias_fmod if (output_alias and dealias) else 0.0,
            att=att, g=g, g_extent=g_extent,
        )[0]

    # Worst (needed / cap) guard shortfall over the whole merge tree,
    # collected during the recursion so that a capped guard warns once per
    # call instead of once per tree node.
    guard_shortfall = [0.0, 0, 0]
    result = _ffbp_impl(
        data, grid, fc, r_res, pos, stages, divisions, d0, interp_method,
        oversample_r, oversample_theta, dealias, data_fmod, alias_fmod,
        output_alias, use_poly, poly_coefs,
        att, g, g_extent, weight_map_downsample,
        is_top_level=True,
        afbp_nsub=afbp_nsub,
        guard_max_ratio=guard_max_ratio,
        guard_shortfall=guard_shortfall,
    )
    # A small shortfall only truncates the window support of guard bins,
    # whose error reaches the scene attenuated by the interpolation kernel
    # tails; warn when a large part of the support is missing.
    if guard_shortfall[0] > 2.0:
        warn(f"ffbp: theta guard band capped at {guard_shortfall[2]} bins "
             f"({guard_shortfall[1]} needed for full interpolation support "
             f"at the near range edge). Edge accuracy at near range may be "
             f"reduced; increase guard_max_ratio (more memory) if it "
             f"matters.")
    # _ffbp_impl returns (img, w1_map, w2_map, weight_grid)
    # Without an antenna pattern the image is already normalized
    # With it the Wiener normalization is applied here
    img = result[0]
    if use_antenna_pattern:
        img = _weighted_normalize(img, result[1], result[2], eps=weight_eps)
    return img


def _ffbp_impl(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int,
    d0: float,
    interp_method: tuple,
    oversample_r: float,
    oversample_theta: float,
    dealias: bool,
    data_fmod: float,
    alias_fmod: float,
    output_alias: bool,
    use_poly: bool,
    poly_coefs: Tensor,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    weight_map_downsample: int = 1,
    is_top_level: bool = True,
    pos_z: list | None = None,
    afbp_nsub: int = 1,
    core_theta: tuple | None = None,
    guard_max_ratio: float = 0.125,
    pos_xy: list | None = None,
    guard_shortfall: list | None = None,
) -> Tensor:
    """Internal implementation of ffbp with precomputed polynomial coefficients.

    ``core_theta`` is the scene theta extent, constant down the recursion;
    the node grid covers it plus a guard band (possibly asymmetric) whose
    bins carry valid signal but are only consumed as interpolation window
    support by the parent's merges. The guard band may extend past
    |theta| = 1, where the backprojection and merge kernels compute the
    smooth continuation of the azimuth signal (exact for a straight
    track). Each level sizes its children's guard so that every lookup
    window of its own merges lands inside the child grids, up to a cap of
    ``guard_max_ratio`` of the child core bins per side.
    """
    nsweeps = data.shape[0]
    use_antenna_pattern = g is not None

    if pos_z is None:
        # Sweep coordinates as Python floats: the z0/new_z and guard sizing
        # scalars then never touch the device (one sync here instead of one
        # per tree node on GPU). center_pos doesn't modify z, so plain
        # slices of these lists stay valid down the recursion (x/y are only
        # used for origin differences, which centering doesn't change).
        pos_z = pos[:, 2].tolist()
    if pos_xy is None:
        pos_xy = pos[:, :2].tolist()

    # Scene (core) theta extent; the node grid is the core plus its guard
    # band.
    if core_theta is None:
        core_theta = grid["theta"]
    core_t0, core_t1 = core_theta
    theta0_g, theta1_g = grid["theta"]
    dtheta_node = (theta1_g - theta0_g) / grid["ntheta"]
    core_ntheta = round((core_t1 - core_t0) / dtheta_node)

    # Split at rounded boundaries so that no sweeps are dropped when
    # divisions does not divide nsweeps. Subaperture sizes differ by at most
    # one sweep, which the merge handles.
    bounds = [round(i * nsweeps / divisions) for i in range(divisions + 1)]

    # Children guard sizing. A merge output pixel at theta = t reads the
    # child image at tp = (d*t + dy)/rp, so the child grid must cover the
    # image of this node's own grid edges under that map, plus the
    # interpolation window. The shift tp - t has one sign per child (set by
    # the along-track origin offset dy), so the guard is one-sided per
    # child up to the window margin. It is evaluated with the exact kernel
    # formula: the linearization |dy|/r0 wildly overestimates when the
    # subaperture offsets are comparable to the near range.
    div_xy = [pos_xy[bounds[i]:bounds[i + 1]] for i in range(divisions)]
    means = [(sum(x for x, _ in s) / len(s), sum(y for _, y in s) / len(s))
             for s in div_xy]
    cx = sum(m[0] for m in means) / divisions
    cy = sum(m[1] for m in means) / divisions
    r0_min, r1_max = grid["r"]

    def _tp_shift(te: float, dy: float, dx: float, d: float) -> float:
        # Lookup shift tp - te of the merge transform, same math as the
        # kernels.
        ct = math.sqrt(max(0.0, 1.0 - te * te))
        rp2 = d * d + dx * dx + dy * dy + 2.0 * d * (dy * te + dx * ct)
        rp = math.sqrt(max(rp2, 1e-12))
        return (d * te + dy) / rp - te

    # Interpolation window half width in bins, plus one bin of rounding
    # slack.
    a_bins = interp_method[1] // 2 + 1

    imgs = []
    for d_idx in range(divisions):
        i0, i1 = bounds[d_idx], bounds[d_idx + 1]
        pos_local, origin_local = center_pos(pos[i0:i1])
        pos_z_local = pos_z[i0:i1]
        pos_xy_local = pos_xy[i0:i1]
        z0 = sum(pos_z_local) / len(pos_z_local)
        # Oversample the subaperture grid to leave aliasing margin for the
        # merges. Applies to both the recursive and the base backprojection
        # branch; deeper levels receive oversample=1 since the grid is
        # already increased.
        core_nt_child = (core_ntheta + divisions - 1) // divisions
        core_nt_child = int(oversample_theta * core_nt_child)
        dth_child = (core_t1 - core_t0) / core_nt_child

        # Worst-case signed lookup shifts at the node grid edges over the
        # swath and over the merge-chain origin variation (dy = 0 covers
        # the no-shift window; the 0.5/1.5 factors bound the intermediate
        # pairwise-mean origins of unbalanced in-node merge chains).
        dy = cy - means[d_idx][1]
        dx = cx - means[d_idx][0]
        cand = [(sy * dy, sx * dx, d)
                for sy in (0.0, 0.5, 1.5) for sx in (-1.5, 1.5)
                for d in (r0_min, r1_max)]
        s_hi = max(_tp_shift(theta1_g, *c) for c in cand)
        s_lo = min(_tp_shift(theta0_g, *c) for c in cand)

        # The useful support saturates a couple of interpolation windows
        # past the fold: beyond |theta| = 1 the child image holds only the
        # smooth continuation of the scene content near the fold, so
        # lookups further out add nothing. Without the clamp the exact
        # shift diverges when a grid edge past the fold combines with a
        # subaperture origin offset near the pixel ground range (rp -> 0).
        fold_margin = 2 * a_bins * dth_child
        hi_req = min(theta1_g + s_hi, max(theta1_g, 1.0) + fold_margin)
        lo_req = max(theta0_g + s_lo, min(theta0_g, -1.0) - fold_margin)

        # Guard bins per side: cover [lo_req, hi_req] plus the window,
        # never shrinking inside the core. Cap the guard to bound the
        # memory overhead: with subaperture offsets comparable to the near
        # range the shift can reach a large fraction of the whole theta
        # domain, which would multiply every image in the tree.
        n_lo = math.ceil((core_t0 - lo_req) / dth_child) + a_bins + 1
        n_hi = math.ceil((hi_req - core_t1) / dth_child) + a_bins + 1
        g_cap = max(4 * a_bins, int(guard_max_ratio * core_nt_child))
        if guard_shortfall is not None and max(n_lo, n_hi) > g_cap:
            ratio = max(n_lo, n_hi) / g_cap
            if ratio > guard_shortfall[0]:
                guard_shortfall[:] = [ratio, max(n_lo, n_hi), g_cap]
        n_lo = min(max(n_lo, 0), g_cap)
        n_hi = min(max(n_hi, 0), g_cap)

        grid_local = deepcopy(grid)
        grid_local["theta"] = (core_t0 - n_lo * dth_child,
                               core_t1 + n_hi * dth_child)
        grid_local["ntheta"] = core_nt_child + n_lo + n_hi
        grid_local["nr"] = int(oversample_r * grid_local["nr"])
        data_local = data[i0:i1]
        att_local = att[i0:i1] if att is not None else None

        # Minimum subaperture size: deep in the tree the guard settles to
        # its window-support fixed point of ~2*(a_bins+1) bins per side
        # (each level inherits half the parent guard extent in its own
        # bins, plus the window), while the core halves per level, so ever
        # deeper splits become guard-dominated and stop paying off. Stop
        # splitting when the next level's core would put the fixed-point
        # guard above _GUARD_STOP_FRAC of the subaperture image.
        min_core = int(4 * (a_bins + 1) * (1.0 / _GUARD_STOP_FRAC - 1.0))
        next_core = (core_nt_child + divisions - 1) // divisions
        if stages > 1 and len(data_local) >= 2 * divisions and next_core >= min_core:
            img, w1_map, w2_map, weight_grid = _ffbp_impl(
                data_local,
                grid_local,
                fc,
                r_res,
                pos_local,
                stages=stages - 1,
                divisions=divisions,
                d0=d0,
                interp_method=interp_method,
                oversample_r=1,  # Grid is already increased
                oversample_theta=1,
                dealias=True,
                data_fmod=data_fmod,
                alias_fmod=alias_fmod,
                output_alias=True,
                use_poly=use_poly,
                poly_coefs=poly_coefs,
                att=att_local,
                g=g,
                g_extent=g_extent,
                weight_map_downsample=weight_map_downsample,
                is_top_level=False,
                pos_z=pos_z_local,
                afbp_nsub=afbp_nsub,
                core_theta=core_theta,
                guard_max_ratio=guard_max_ratio,
                pos_xy=pos_xy_local,
                guard_shortfall=guard_shortfall,
            )
        else:
            # When using antenna pattern, request unnormalized output
            normalize = not use_antenna_pattern
            if afbp_nsub > 1:
                img = afbp(
                    data_local, grid_local, fc, r_res, pos_local,
                    nsub=afbp_nsub, d0=d0, dealias=True,
                    data_fmod=data_fmod, alias_fmod=alias_fmod,
                    att=att_local, g=g, g_extent=g_extent,
                    normalize=normalize
                )[None]
            else:
                img = backprojection_polar_2d(
                    data_local, grid_local, fc, r_res, pos_local, d0=d0, dealias=True,
                    data_fmod=data_fmod, alias_fmod=alias_fmod,
                    att=att_local, g=g, g_extent=g_extent, normalize=normalize
                )

            # Compute weight maps at base level for antenna pattern weighting
            w1_map = None
            w2_map = None
            weight_grid = None
            if use_antenna_pattern:
                # Compute illumination with decimation
                # CUDA kernel outputs at decimated resolution directly
                w1_map, w2_map = compute_subaperture_illumination(
                    pos_local, att_local, g, g_extent, grid_local,
                    decimation=weight_map_downsample
                )
                # Create weight grid matching output dimensions
                dec = weight_map_downsample
                out_nr = (grid_local["nr"] + dec - 1) // dec
                out_ntheta = (grid_local["ntheta"] + dec - 1) // dec
                r0, r1 = grid_local["r"]
                theta0, theta1 = grid_local["theta"]
                dr = (r1 - r0) / grid_local["nr"]
                dtheta = (theta1 - theta0) / grid_local["ntheta"]
                weight_grid = {
                    "r": (r0, r0 + dr * dec * out_nr),
                    "theta": (theta0, theta0 + dtheta * dec * out_ntheta),
                    "nr": out_nr,
                    "ntheta": out_ntheta,
                }

        imgs.append((origin_local[0], grid_local, img, z0, w1_map, w2_map, weight_grid))

    while len(imgs) > 1:
        img1 = imgs[0]
        img2 = imgs[1]
        new_origin = 0.5 * img1[0] + 0.5 * img2[0]
        new_z = 0.5 * (img1[3] + img2[3])
        alias = False
        # output_alias only applies to final merge
        out_alias = output_alias
        is_final_merge = len(imgs) == 2
        if is_final_merge:
            # Interpolate the final image to the desired grid.
            grid_polar_new = grid
            alias = not dealias
        else:
            # Union of the source extents (the per-child guard bands are
            # one-sided, so the extents differ) at half the source theta
            # step: the azimuth bandwidth doubles when the subapertures
            # merge.
            g1, g2 = img1[1], img2[1]
            t0_u = min(g1["theta"][0], g2["theta"][0])
            t1_u = max(g1["theta"][1], g2["theta"][1])
            dth_u = 0.5 * (g1["theta"][1] - g1["theta"][0]) / g1["ntheta"]
            grid_polar_new = deepcopy(g1)
            grid_polar_new["theta"] = (t0_u, t1_u)
            grid_polar_new["ntheta"] = round((t1_u - t0_u) / dth_u)
            out_alias = True

        i1 = img1[2]
        i2 = img2[2]
        dorigin1 = new_origin - img1[0]
        dorigin1[2] = -(new_z - img1[3])
        dorigin2 = new_origin - img2[0]
        dorigin2[2] = -(new_z - img2[3])

        # Get weight maps and grids
        # Tuple structure: (origin, grid, img, z0, w1_map, w2_map, weight_grid)
        w1_map1, w2_map1, wgrid1 = img1[4], img1[5], img1[6]
        w1_map2, w2_map2, wgrid2 = img2[4], img2[5], img2[6]

        if use_antenna_pattern and w1_map1 is not None and w2_map1 is not None:
            # Carry the unnormalized accumulation A and the illumination moments
            # W1, W2 up the tree. Wiener normalization is applied once in
            # ffbp(). The top-level final merge emits full-resolution weight maps
            # (decimation 1) so that normalization is exact per pixel.
            final_top = is_top_level and is_final_merge
            out_dec = 1 if final_top else weight_map_downsample

            img_sum, w1_out, w2_out, merged_weight_grid = ffbp_merge2_poly_weighted(
                i1,
                i2,
                dorigin1,
                dorigin2,
                [img1[1], img2[1]],
                fc,
                grid_polar_new,
                z0=new_z,
                order=interp_method[1],
                oversample=interp_method[2],
                alias=alias,
                alias_fmod=alias_fmod,
                output_alias=out_alias,
                poly_coefs=poly_coefs,
                w1_map0=w1_map1,
                w2_map0=w2_map1,
                weight_grid0=wgrid1,
                w1_map1=w1_map2,
                w2_map1=w2_map2,
                weight_grid1=wgrid2,
                output_weight_map=True,
                output_weight_decimation=out_dec,
            )
        else:
            # Standard merge (no antenna pattern)
            img_sum = ffbp_merge2(
                i1,
                i2,
                dorigin1,
                dorigin2,
                [img1[1], img2[1]],
                fc,
                grid_polar_new,
                z0=new_z,
                method=interp_method,
                alias=alias,
                alias_fmod=alias_fmod,
                output_alias=out_alias,
                use_poly=use_poly,
                poly_coefs=poly_coefs
            )
            w1_out = None
            w2_out = None
            merged_weight_grid = None

        imgs[0] = None
        imgs[1] = None
        del i1
        del img1
        del i2
        del img2

        merged = (new_origin, grid_polar_new, img_sum, new_z, w1_out, w2_out, merged_weight_grid)
        imgs = imgs[2:] + [merged]

    # Return different values depending on whether we're at top level or recursive
    # At top level (called from ffbp), just return the image
    # At recursive level, return (img, w1_map, w2_map, weight_grid)
    if use_antenna_pattern:
        return imgs[0][2], imgs[0][4], imgs[0][5], imgs[0][6]
    return imgs[0][2], None, None, None


def _tx_power_node_grid(
    boundary: Tensor,
    pos_local: Tensor,
    yaw: Tensor,
    g_extent: list,
    g_daz: float,
    sub_dr: float,
    sub_dpsi: float,
    margin: int,
    min_nr: int,
    min_ntheta: int,
    altitude: float,
) -> dict | None:
    """Polar grid covering the output grid as seen from a subaperture origin.

    boundary contains the output grid boundary in ground coordinates relative
    to the subaperture origin. The azimuth extent is intersected with the
    azimuth range visible through the antenna pattern given the subaperture
    yaw range. Returns None if the subaperture cannot see the output grid.

    The returned grid theta extents are the azimuth angle psi = asin(theta)
    in radians and the map is sampled uniformly in psi (theta_psi grids).
    Uniform psi resolves the antenna pattern with a constant number of cells
    per beamwidth; a grid uniform in theta = sin(psi) compresses the pattern
    features by cos(psi) towards |theta| = 1 where it would need much finer
    sampling.
    """
    dx = boundary[:, 0]
    dy = boundary[:, 1]
    if altitude > 0.0:
        r = torch.sqrt(dx * dx + dy * dy + altitude * altitude)
    else:
        r = torch.sqrt(dx * dx + dy * dy)
    r_floor = altitude if altitude > 0.0 else 0.0
    if bool((dx.min() < 0) & (dx.max() > 0) & (dy.min() < 0) & (dy.max() > 0)):
        # Origin may be inside the scene footprint: the boundary samples do
        # not bound the interior minimum.
        r_lo = r_floor
        r_hi = float(r.max())
        p_lo = -math.pi / 2
        p_hi = math.pi / 2
    else:
        th = dy / torch.clamp(r, min=1e-9)
        r_lo = float(r.min())
        r_hi = float(r.max())
        p_lo = math.asin(max(-1.0, min(1.0, float(th.min()))))
        p_hi = math.asin(max(-1.0, min(1.0, float(th.max()))))

    # Azimuth angles visible through the antenna pattern. Pad accounts for
    # pattern lookup interpolation and for the pulses being offset from the
    # subaperture origin by up to the subaperture length.
    g_el0, g_az0, g_el1, g_az1 = g_extent
    max_offset = float(torch.linalg.norm(pos_local[:, :2], dim=-1).max())
    psi_pad = 2.0 * g_daz + max_offset / max(r_lo, 1e-3)
    psi_lo = g_az0 + float(yaw.min()) - psi_pad
    psi_hi = g_az1 + float(yaw.max()) + psi_pad
    # Polar grid theta cannot represent the rear half plane.
    if psi_lo > math.pi / 2 or psi_hi < -math.pi / 2:
        return None
    p_lo = max(p_lo, psi_lo)
    p_hi = min(p_hi, psi_hi)
    if p_hi <= p_lo or r_hi <= r_lo:
        return None

    r_lo = max(r_floor, r_lo - margin * sub_dr)
    r_hi = r_hi + margin * sub_dr
    p_lo = max(-math.pi / 2, p_lo - margin * sub_dpsi)
    p_hi = min(math.pi / 2, p_hi + margin * sub_dpsi)

    nr = max(min_nr, int(math.ceil((r_hi - r_lo) / sub_dr)))
    ntheta = max(min_ntheta, int(math.ceil((p_hi - p_lo) / sub_dpsi)))
    return {"r": (r_lo, r_hi), "theta": (p_lo, p_hi), "nr": nr, "ntheta": ntheta}


def _ffbp_tx_power_impl(
    wa: Tensor,
    pos: Tensor,
    att: Tensor,
    boundary: Tensor,
    stages: int,
    divisions: int,
    g: Tensor,
    g_extent: list,
    g_daz: float,
    normalization: str | None,
    sub_dr: float,
    sub_dpsi: float,
    margin: int,
    min_nr: int,
    min_ntheta: int,
    min_nsweeps: int,
    dr_ref: float,
    h_ref: float,
    altitude: float,
    grid_out: dict | None = None,
    is_top_level: bool = True,
) -> tuple[Tensor, dict, Tensor] | None:
    """Recursive tx_power accumulator map computation.

    Returns (acc, grid, frame_offset) where frame_offset is the origin of the
    accumulator map frame relative to this node's frame, or None if no pulse
    of this node illuminates the output grid.
    """
    nsweeps = wa.shape[0]
    device = wa.device

    # Grid for the intermediate merges of this node.
    grid_node = _tx_power_node_grid(
        boundary, pos, att[:, 2], g_extent, g_daz, sub_dr, sub_dpsi,
        margin, min_nr, min_ntheta, altitude)
    if grid_node is None:
        return None

    nodes = []
    edges = torch.linspace(0, nsweeps, divisions + 1).long()
    for d_idx in range(divisions):
        i0 = int(edges[d_idx])
        i1 = int(edges[d_idx + 1])
        if i1 - i0 < 1:
            continue
        pos_local, origin_local = center_pos(pos[i0:i1])
        o = origin_local[0]
        boundary_local = boundary - o[None, :2]
        wa_local = wa[i0:i1]
        att_local = att[i0:i1]

        if stages > 1 and i1 - i0 > min_nsweeps:
            child = _ffbp_tx_power_impl(
                wa_local, pos_local, att_local, boundary_local,
                stages - 1, divisions, g, g_extent, g_daz, normalization,
                sub_dr, sub_dpsi, margin, min_nr, min_ntheta, min_nsweeps,
                dr_ref, h_ref, altitude, grid_out=None, is_top_level=False)
            if child is None:
                continue
            acc, grid_local, child_offset = child
            offset = o + child_offset
        else:
            grid_local = _tx_power_node_grid(
                boundary_local, pos_local, att_local[:, 2], g_extent, g_daz,
                sub_dr, sub_dpsi, margin, min_nr, min_ntheta, altitude)
            if grid_local is None:
                continue
            acc = _backprojection_polar_2d_tx_power_accum(
                wa_local, g, g_extent, grid_local, pos_local, att_local,
                normalization, dr_ref, h_ref, altitude, theta_psi=True)
            offset = o
        nodes.append((acc, grid_local, offset))

    if len(nodes) == 0:
        return None

    while len(nodes) > 1:
        n1 = nodes[0]
        n2 = nodes[1]
        is_final_merge = is_top_level and len(nodes) == 2
        grid_new = grid_out if is_final_merge else grid_node
        # dorigin = merged frame origin - input frame origin
        dorigin1 = -n1[2].clone()
        dorigin2 = -n2[2].clone()
        dorigin1[2] = 0.0
        dorigin2[2] = 0.0
        merged = ffbp_tx_power_merge2(
            n1[0], n2[0], dorigin1, dorigin2, [n1[1], n2[1]], grid_new,
            altitude=altitude, in_psi=True, out_psi=not is_final_merge)
        nodes = nodes[2:] + [(merged, grid_new, torch.zeros(3, device=device))]

    return nodes[0]


def backprojection_polar_2d_tx_power_ffbp(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    stages: int,
    divisions: int = 2,
    normalization: str | None = None,
    azimuth_resolution: bool = True,
    downsample_r: float = 4.0,
    downsample_theta: float = 4.0,
    min_nsweeps: int = 64,
    min_nr: int = 32,
    min_ntheta: int = 32,
    margin: int = 4,
    altitude: float = 0.0,
    beam_theta_samples: float = 32.0,
) -> Tensor:
    """
    Fast factorized version of :func:`backprojection_polar_2d_tx_power`.

    Splits the track recursively into subapertures like :func:`ffbp`,
    computes coarse per-subaperture accumulator maps and merges them
    pairwise with bilinear interpolation. Unlike the coherent FFBP the
    accumulated fields are smooth and phase-free, so the subaperture maps
    can be sampled much more coarsely than the output grid
    (``downsample_r``, ``downsample_theta``) and the azimuth resolution
    moments merge exactly (Chan's parallel variance formula).

    Each subaperture gets its own grid extents computed from the output grid
    geometry as seen from the subaperture center, intersected with the
    azimuth range visible through the antenna pattern. Subapertures on a
    long or curved track that see the scene at a different angle than the
    output grid theta range are handled correctly, and subapertures that
    cannot see the output grid at all are skipped.

    Implements the same moment-based azimuth resolution model as the direct
    function. Output matches the direct function up to interpolation errors,
    except near illumination edges where pixels with very few contributing
    pulses can differ. With ``altitude > 0`` the region just outside the
    nadir shadow zone is also approximate because the slant range to ground
    range mapping is singular at the shadow boundary and the interpolated
    fields vary too fast there.

    Parameters
    ----------
    wa : Tensor
        Weighting coefficient for amplitude of each pulse, shape: [nsweeps].
    g : Tensor
        Square-root of two-way antenna gain in spherical coordinates,
        shape: [elevation, azimuth]. (0, 0) angle is at the beam center.
    g_extent : list
        Antenna pattern extent: [g_el0, g_az0, g_el1, g_az1] in radians.
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
    r_res : float
        Range bin resolution. Unused, kept for signature parity with
        :func:`backprojection_polar_2d_tx_power`.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be
        [nsweeps, 3]. [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    stages : int
        Number of recursions.
    divisions : int
        Number of subaperture divisions per stage. Default is 2.
    normalization : str or None
        Valid choices are:
            "sigma" to divide each value by sin of incidence angle.
            "gamma" to divide each value by of tan of incidence angle.
            "beta" or None for no incidence angle normalization.
            "point" to normalize to constant reflectivity (no ground patch).
    azimuth_resolution : bool
        If True (default), also normalize for the varying azimuth resolution.
        See :func:`backprojection_polar_2d_tx_power`.
    downsample_r : float
        Subaperture map range step relative to the output grid range step.
        The accumulated fields are smooth, so values well above 1 are
        usually fine. Default is 4.
    downsample_theta : float
        Subaperture map azimuth step relative to the output grid theta step.
        The subaperture maps are sampled uniformly in azimuth angle
        psi = asin(theta), so relative to the output theta grid the maps get
        denser towards |theta| = 1 where the sin mapping compresses the
        antenna pattern features. Default is 4.
    min_nsweeps : int
        Do not recurse into subapertures with fewer pulses than this.
    min_nr : int
        Minimum number of range points in a subaperture map. Avoids large
        interpolation errors near edges of very small maps.
    min_ntheta : int
        Minimum number of theta points in a subaperture map.
    margin : int
        Extra margin in subaperture map cells around the computed extents.
    altitude : float
        If greater than zero, use slant plane grid semantics identical to
        :func:`backprojection_polar_2d_tx_power_slant` with this reference
        altitude. Default 0 uses ground plane grid.
    beam_theta_samples : float
        Minimum number of subaperture map azimuth samples across the half
        amplitude azimuth beamwidth of the antenna pattern. The interpolation
        error scales as the squared ratio of the azimuth step to the azimuth
        beamwidth, so a beam that is thin compared to the output grid theta
        step needs subaperture maps finer than ``downsample_theta`` implies.
        The azimuth step is the smaller of ``downsample_theta`` times the
        output grid theta step and the beamwidth divided by this value.
        A quarter of the same sample density is applied to the range step
        relative to the elevation beamwidth, which matters only when the
        near edge of the grid approaches nadir.

    Returns
    -------
    tx_power : Tensor
        Pseudo-polar format image of square root of power returned from each
        pixel assuming constant reflectivity, shape [nr, ntheta].
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)

    assert wa.dim() == 1, "backprojection_polar_2d_tx_power_ffbp supports only a single batch"
    nsweeps = wa.shape[0]
    assert pos.shape == (nsweeps, 3)
    assert att.shape == (nsweeps, 3)
    device = wa.device

    g_el0, g_az0, g_el1, g_az1 = g_extent
    g_daz = (g_az1 - g_az0) / g.shape[1]

    h_ref = altitude if altitude > 0 else float(pos[nsweeps // 2, 2])
    sub_dr = dr * downsample_r

    # Subaperture maps are sampled uniformly in azimuth angle psi = asin(theta)
    # rather than theta, see _tx_power_node_grid. The step is downsample_theta
    # output grid cells at scene center, additionally capped so that the
    # azimuth beamwidth stays resolved: the illumination fields vary at the
    # scale of the beamwidth, not the output grid step, and a beam that is
    # thin compared to the output theta step degrades the bilinear
    # interpolation badly (see beam_theta_samples).
    sub_dpsi = dtheta * downsample_theta
    p_az = torch.amax(g, dim=0)
    above = (p_az >= 0.5 * float(p_az.max())).nonzero()
    if len(above) > 0:
        half_width = float(above[-1] - above[0] + 1) * g_daz
        sub_dpsi = min(sub_dpsi, half_width / beam_theta_samples)

    # Same in range: an elevation pattern feature of angular size del maps to
    # a range interval del * (r^2 + h^2) / h, smallest at near range. Only
    # binding when the near edge of the grid is close to nadir. The fields
    # vary more slowly in range than in azimuth (no track-parallel geometry),
    # so a quarter of the azimuth sample density is enough.
    g_del = (g_el1 - g_el0) / g.shape[0]
    p_el = torch.amax(g, dim=1)
    above = (p_el >= 0.5 * float(p_el.max())).nonzero()
    if len(above) > 0 and h_ref > 0:
        el_half_width = float(above[-1] - above[0] + 1) * g_del
        if altitude > 0.0:
            # Slant grid: r is slant range, dr/dl = sqrt(r^2 - h^2).
            rg0 = math.sqrt(max(altitude * 0.1, r0**2 - altitude**2))
            r_scale = rg0 * r0 / altitude
        else:
            r_scale = (r0**2 + h_ref**2) / h_ref
        r_cap = 4.0 * el_half_width / beam_theta_samples * r_scale
        sub_dr = min(sub_dr, max(dr, r_cap))

    # Output grid boundary in ground coordinates. Subaperture grid extents
    # are computed from these points shifted to each subaperture frame.
    nb = 64
    rr = torch.linspace(r0, r1, nb, device=device)
    tt = torch.linspace(theta0, theta1, nb, device=device)
    r_s = torch.cat((torch.full_like(tt, r0), torch.full_like(tt, r1), rr, rr))
    t_s = torch.cat((tt, tt, torch.full_like(rr, theta0), torch.full_like(rr, theta1)))
    if altitude > 0.0:
        gx = torch.sqrt(torch.clamp(r_s**2 * (1 - t_s**2) - altitude**2, min=0.0))
    else:
        gx = r_s * torch.sqrt(torch.clamp(1 - t_s**2, min=0.0))
    gy = r_s * t_s
    boundary = torch.stack((gx, gy), dim=1)

    node = _ffbp_tx_power_impl(
        wa, pos, att, boundary, stages, divisions, g, g_extent, g_daz,
        normalization, sub_dr, sub_dpsi, margin, min_nr, min_ntheta,
        min_nsweeps, dr, h_ref, altitude, grid_out=grid, is_top_level=True)

    if node is None:
        acc = torch.zeros((4, nr, ntheta), dtype=torch.float32, device=device)
    elif node[1] is not grid or bool(torch.any(node[2] != 0)):
        # Single surviving subaperture: regrid it to the output grid.
        dorigin = -node[2].clone()
        dorigin[2] = 0.0
        acc = ffbp_tx_power_merge2(
            node[0], None, dorigin, None, [node[1], node[1]], grid,
            altitude=altitude, in_psi=True, out_psi=False)
    else:
        acc = node[0]

    # Finishing step. Matches the direct kernel epilogue.
    r_vec = r0 + dr * torch.arange(nr, dtype=torch.float32, device=device)
    if altitude > 0.0:
        Rg = torch.sqrt(torch.clamp(r_vec**2 - altitude**2, min=0.0))
    else:
        Rg = r_vec
    out = _tx_power_finish(acc, Rg[:, None], azimuth_resolution)
    if altitude > 0.0:
        # Shadow zone below nadir is zero in the direct kernel.
        t_vec = theta0 + dtheta * torch.arange(ntheta, dtype=torch.float32, device=device)
        shadow = r_vec[:, None]**2 * (1 - t_vec[None, :]**2) < altitude**2
        out = torch.where(shadow, torch.zeros_like(out), out)
    return out


