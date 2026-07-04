import torch
from torch import Tensor
from typing import Union, TYPE_CHECKING
from warnings import warn
from .backproj import backprojection_polar_2d, backprojection_polar_2d_tx_power
from .polar_interp import ffbp_merge2, ffbp_merge2_poly, ffbp_merge2_poly_weighted, compute_knab_poly_coefs_full, select_knab_poly_degree
from ..util import center_pos
from copy import deepcopy
from ._utils import AntennaPattern, unpack_polar_grid

if TYPE_CHECKING:
    from ..grid import PolarGrid


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
        Number of recursions.
    divisions : int
        Number of subapertures divisions per stage. Default is 2.
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

    result = _ffbp_impl(
        data, grid, fc, r_res, pos, stages, divisions, d0, interp_method,
        oversample_r, oversample_theta, dealias, data_fmod, alias_fmod,
        output_alias, use_poly, poly_coefs,
        att, g, g_extent, weight_map_downsample,
        is_top_level=True
    )
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
) -> Tensor:
    """Internal implementation of ffbp with precomputed polynomial coefficients."""
    nsweeps = data.shape[0]
    use_antenna_pattern = g is not None

    imgs = []
    # Split at rounded boundaries so that no sweeps are dropped when
    # divisions does not divide nsweeps. Subaperture sizes differ by at most
    # one sweep, which the merge handles.
    bounds = [round(i * nsweeps / divisions) for i in range(divisions + 1)]
    for d_idx in range(divisions):
        i0, i1 = bounds[d_idx], bounds[d_idx + 1]
        pos_local, origin_local = center_pos(pos[i0:i1])
        z0 = torch.mean(pos_local[:, 2])
        grid_local = deepcopy(grid)
        grid_local["ntheta"] = (grid["ntheta"] + divisions - 1) // divisions
        # Oversample the subaperture grid to leave interpolation margin for
        # the merges. Applies to both the recursive and the base
        # backprojection branch; deeper levels receive oversample=1 since
        # the grid is already increased.
        grid_local["nr"] = int(oversample_r * grid_local["nr"])
        grid_local["ntheta"] = int(oversample_theta * grid_local["ntheta"])
        data_local = data[i0:i1]
        att_local = att[i0:i1] if att is not None else None

        # TODO: Better edge handling for interpolation.
        # Interpolation doesn't work too well with too small image due to edges.
        # Limit the minimum image size to avoid large interpolation errors.
        if stages > 1 and len(data_local) > 128:
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
            )
        else:
            # When using antenna pattern, request unnormalized output
            normalize = not use_antenna_pattern
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
            grid_polar_new = deepcopy(img1[1])
            grid_polar_new["ntheta"] += img2[1]["ntheta"]
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


