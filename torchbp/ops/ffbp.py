import torch
from torch import Tensor
from warnings import warn
from .backproj import backprojection_polar_2d
from .polar_interp import ffbp_merge2, ffbp_merge2_poly, compute_knab_poly_coefs_full, select_knab_poly_degree
from ..util import center_pos
from copy import deepcopy

def ffbp(
    data: Tensor,
    grid: dict,
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
    use_poly: bool = True
) -> Tensor:
    """
    Fast factorized backprojection.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    grid : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
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

    Returns
    -------
    img : Tensor
        SAR image.
    """
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

    return _ffbp_impl(
        data, grid, fc, r_res, pos, stages, divisions, d0, interp_method,
        oversample_r, oversample_theta, dealias, data_fmod, alias_fmod,
        output_alias, use_poly, poly_coefs
    )


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
) -> Tensor:
    """Internal implementation of ffbp with precomputed polynomial coefficients."""
    nsweeps = data.shape[0]

    imgs = []
    n = nsweeps // divisions
    for d in range(divisions):
        pos_local, origin_local = center_pos(pos[d * n : (d + 1) * n])
        z0 = torch.mean(pos_local[:, 2])
        grid_local = deepcopy(grid)
        grid_local["ntheta"] = (grid["ntheta"] + divisions - 1) // divisions
        data_local = data[d * n : (d + 1) * n]
        # TODO: Better edge handling for interpolation.
        # Interpolation doesn't work too well with too small image due to edges.
        # Limit the minimum image size to avoid large interpolation errors.
        if stages > 1 and len(data_local) > 128:
            grid_local["nr"] = int(oversample_r * grid_local["nr"])
            grid_local["ntheta"] = int(oversample_theta * grid_local["ntheta"])
            img = _ffbp_impl(
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
            )
        else:
            img = backprojection_polar_2d(
                data_local, grid_local, fc, r_res, pos_local, d0=d0, dealias=True, data_fmod=data_fmod, alias_fmod=alias_fmod
            )[0]
        imgs.append((origin_local[0], grid_local, img, z0))
    while len(imgs) > 1:
        img1 = imgs[0]
        img2 = imgs[1]
        new_origin = 0.5 * img1[0] + 0.5 * img2[0]
        new_z = 0.5 * (img1[3] + img2[3])
        alias = False
        # output_alias only applies to final merge
        out_alias = output_alias
        if len(imgs) == 2:
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
        imgs[0] = None
        imgs[1] = None
        del i1
        del img1
        del i2
        del img2

        merged = (new_origin, grid_polar_new, img_sum, new_z)
        imgs = imgs[2:] + [merged]
    return imgs[0][2]


