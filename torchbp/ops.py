import torch
from math import pi
from torch import Tensor
from copy import deepcopy
from .util import bp_polar_range_dealias, center_pos
from warnings import warn

cart_2d_nargs = 16
polar_2d_nargs = 26
polar_interp_linear_args = 19
polar_to_cart_linear_args = 18
entropy_args = 3
abs_sum_args = 2
coherence_2d_args = 7


def entropy(img: Tensor) -> Tensor:
    """
    Calculates entropy of:

    -sum(y*log(y))

    , where y = abs(x) / sum(abs(x)).

    Uses less memory than pytorch implementation when used in optimization.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
    else:
        nbatch = 1

    norm = torch.ops.torchbp.abs_sum.default(img, nbatch)
    x = torch.ops.torchbp.entropy.default(img, norm, nbatch)
    if nbatch == 1:
        return x.squeeze(0)
    return x


def polar_interp(
    img: Tensor,
    origin_old: Tensor,
    origin_new: Tensor,
    grid_polar: dict,
    fc: float,
    rotation: float = 0,
    grid_polar_new: dict = None,
    method: str | tuple = "linear",
    alias_fmod : float = 0
) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.
    Allows choosing the interpolation method.

    Gradient calculation is only supported with "linear" method.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    origin_old : Tensor
        Origin of the img. Units in meters. [nbatch, 3] if img shape is 3D.
    origin_new: Tensor
        Origin after interpolation.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    method : str or tuple
        Interpolation method. Valid choices are:
        - "linear": Linear interpolation.
        - ("lanczos", n): Lanczos resampling. `n` is the half of kernel length.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """
    if type(method) in (list, tuple):
        method_params = method[1]
        method = method[0]
    else:
        method_params = None

    dorigin = origin_new - origin_old
    if origin_new.dim() == 2:
        z0 = origin_new[0,2].item()
    else:
        z0 = origin_new[2].item()
    if not torch.all(origin_new[...,2] == z0):
        raise ValueError("Batched interpolation with different output heights not supported")
    if dorigin.dim() == 1 and img.dim() == 3:
        dorigin = torch.tile(dorigin[None,:], (img.shape[0],1))
    if method == "linear":
        return polar_interp_linear(
            img, dorigin, grid_polar, fc, rotation, grid_polar_new, z0, alias_fmod
        )
    elif method == "lanczos":
        return polar_interp_lanczos(
            img,
            dorigin,
            grid_polar,
            fc,
            rotation,
            grid_polar_new,
            z0,
            order=method_params,
            alias_fmod=alias_fmod
        )
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")


def polar_interp_linear(
    img: Tensor,
    dorigin: Tensor,
    grid_polar: dict,
    fc: float,
    rotation: float = 0,
    grid_polar_new: dict = None,
    z0: float = 0,
    alias_fmod: float = 0,
) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.

    Gradient can be calculated with respect to img and dorigin.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    dorigin : Tensor
        Difference between the origin of the old image to the new image. Units in meters
        [nbatch, 3] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center in the new image.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert dorigin.shape == (nbatch, 3)
    else:
        nbatch = 1
        assert dorigin.shape == (3,)

    r1_0, r1_1 = grid_polar["r"]
    theta1_0, theta1_1 = grid_polar["theta"]
    ntheta1 = grid_polar["ntheta"]
    nr1 = grid_polar["nr"]
    dtheta1 = (theta1_1 - theta1_0) / ntheta1
    dr1 = (r1_1 - r1_0) / nr1

    if grid_polar_new is None:
        r3_0 = r1_0
        r3_1 = r1_1
        theta3_0 = theta1_0
        theta3_1 = theta1_1
        nr3 = nr1
        ntheta3 = 2 * ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    return torch.ops.torchbp.polar_interp_linear.default(
        img,
        dorigin,
        nbatch,
        rotation,
        fc,
        r1_0,
        dr1,
        theta1_0,
        dtheta1,
        nr1,
        ntheta1,
        r3_0,
        dr3,
        theta3_0,
        dtheta3,
        nr3,
        ntheta3,
        z0,
        alias_fmod,
    )


def polar_interp_lanczos(
    img: Tensor,
    dorigin: Tensor,
    grid_polar: dict,
    fc: float,
    rotation: float = 0,
    grid_polar_new: dict = None,
    z0: float = 0,
    order: int = 6,
    alias_fmod : float = 0
) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.

    Gradient not supported.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    dorigin : Tensor
        Difference between the origin of the old image to the new image. Units in meters
        [nbatch, 3] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center in the new image.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert dorigin.shape == (nbatch, 3)
    else:
        nbatch = 1
        assert dorigin.shape == (3,)

    r1_0, r1_1 = grid_polar["r"]
    theta1_0, theta1_1 = grid_polar["theta"]
    ntheta1 = grid_polar["ntheta"]
    nr1 = grid_polar["nr"]
    dtheta1 = (theta1_1 - theta1_0) / ntheta1
    dr1 = (r1_1 - r1_0) / nr1

    if grid_polar_new is None:
        r3_0 = r1_0
        r3_1 = r1_1
        theta3_0 = theta1_0
        theta3_1 = theta1_1
        nr3 = nr1
        ntheta3 = 2 * ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    return torch.ops.torchbp.polar_interp_lanczos.default(
        img,
        dorigin,
        nbatch,
        rotation,
        fc,
        r1_0,
        dr1,
        theta1_0,
        dtheta1,
        nr1,
        ntheta1,
        r3_0,
        dr3,
        theta3_0,
        dtheta3,
        nr3,
        ntheta3,
        z0,
        order,
        alias_fmod
    )


def ffbp_merge2_lanczos(
    img0: Tensor,
    img1: Tensor,
    dorigin0: Tensor,
    dorigin1: Tensor,
    grid_polars: list,
    fc: float,
    grid_polar_new: dict = None,
    z0: float = 0,
    order: int = 6,
    alias: bool = False,
    alias_fmod: float = 0,
    output_alias: bool = True
) -> Tensor:
    """
    Interpolate two pseudo-polar radar images to new grid and change origin
    position by `dorigin`.

    Gradient not supported.

    Parameters
    ----------
    img0 : Tensor
        2D radar image in [range, angle] format. Dimensions should
        match with grid_polars grid. Image dimension can be different for each
        element in the list.
    img1 : Tensor
        Same format as img0.
    dorigin0 : Tensor
        Difference between the origin of the old image to the new image. Units in meters.
        Shape: [3].
    dorigin1 : Tensor
        Same format as dorigin0.
    grid_polar : list of dict
        List of grid definitions for each input image. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center in the new image.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
    alias : bool
        Add back range dependent phase. Inverse of `util.bp_polar_range_dealias`.
    alias_fmod : float
        Range modulation frequency applied to input.
    output_alias : bool
        If True and `alias` is True apply `alias_fmod` to output.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    device = img0.device
    nimages = 2

    r0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    dr0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    theta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    dtheta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    Nr0 = torch.zeros(nimages, dtype=torch.int32, device=device)
    Ntheta0 = torch.zeros(nimages, dtype=torch.int32, device=device)
    for i in range(nimages):
        r1_0, r1_1 = grid_polars[i]["r"]
        theta1_0, theta1_1 = grid_polars[i]["theta"]
        ntheta1 = grid_polars[i]["ntheta"]
        nr1 = grid_polars[i]["nr"]
        dtheta1 = (theta1_1 - theta1_0) / ntheta1
        dr1 = (r1_1 - r1_0) / nr1
        r0[i] = r1_0
        dr0[i] = dr1
        theta0[i] = theta1_0
        dtheta0[i] = dtheta1
        Nr0[i] = nr1
        Ntheta0[i] = ntheta1

    if grid_polar_new is None:
        r3_0 = r1_0
        r3_1 = r1_1
        theta3_0 = theta1_0
        theta3_1 = theta1_1
        nr3 = nr1
        ntheta3 = 2 * ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    assert dorigin0.shape == (3,)
    assert dorigin1.shape == (3,)
    dorigin = torch.stack((dorigin0, dorigin1), dim=0)

    alias_mode = 0
    if alias:
        if not output_alias:
            alias_mode = 2
        else:
            alias_mode = 1

    return torch.ops.torchbp.ffbp_merge2_lanczos.default(
        img0,
        img1,
        dorigin,
        fc,
        r0,
        dr0,
        theta0,
        dtheta0,
        Nr0,
        Ntheta0,
        r3_0,
        dr3,
        theta3_0,
        dtheta3,
        nr3,
        ntheta3,
        z0,
        order,
        alias_mode,
        alias_fmod,
    )


def ffbp_merge2_knab(
    img0: Tensor,
    img1: Tensor,
    dorigin0: Tensor,
    dorigin1: Tensor,
    grid_polars: list,
    fc: float,
    grid_polar_new: dict = None,
    z0: float = 0,
    order: int = 6,
    oversample: float = 1.5,
    alias: bool = False,
    alias_fmod: float = 0,
    output_alias: bool = True
) -> Tensor:
    """
    Interpolate two pseudo-polar radar images to new grid and change origin
    position by `dorigin`. Uses truncated sinc with Knab pulse for interpolation [1]_.

    Gradient not supported.

    Parameters
    ----------
    img0 : Tensor
        2D radar image in [range, angle] format. Dimensions should
        match with grid_polars grid. Image dimension can be different for each
        element in the list.
    img1 : Tensor
        Same format as img0.
    dorigin0 : Tensor
        Difference between the origin of the old image to the new image. Units in meters.
        Shape: [3].
    dorigin1 : Tensor
        Same format as dorigin0.
    grid_polar : list of dict
        List of grid definitions for each input image. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center in the new image.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
        Even number is preferred.
    oversample : float
        Oversampling factor in the input data.
    alias : bool
        Add back range dependent phase. Inverse of `util.bp_polar_range_dealias`.
    alias_fmod : float
        Range modulation frequency applied to input.
    output_alias : bool
        If True and `alias` is True apply `alias_fmod` to output.

    References
    ----------
    .. [1] J. Knab, "The sampling window," in IEEE Transactions on Information
        Theory, vol. 29, no. 1, pp. 157-159, January 1983.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    device = img0.device
    nimages = 2

    r0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    dr0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    theta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    dtheta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
    Nr0 = torch.zeros(nimages, dtype=torch.int32, device=device)
    Ntheta0 = torch.zeros(nimages, dtype=torch.int32, device=device)
    for i in range(nimages):
        r1_0, r1_1 = grid_polars[i]["r"]
        theta1_0, theta1_1 = grid_polars[i]["theta"]
        ntheta1 = grid_polars[i]["ntheta"]
        nr1 = grid_polars[i]["nr"]
        dtheta1 = (theta1_1 - theta1_0) / ntheta1
        dr1 = (r1_1 - r1_0) / nr1
        r0[i] = r1_0
        dr0[i] = dr1
        theta0[i] = theta1_0
        dtheta0[i] = dtheta1
        Nr0[i] = nr1
        Ntheta0[i] = ntheta1

    if grid_polar_new is None:
        r3_0 = r1_0
        r3_1 = r1_1
        theta3_0 = theta1_0
        theta3_1 = theta1_1
        nr3 = nr1
        ntheta3 = 2 * ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    assert dorigin0.shape == (3,)
    assert dorigin1.shape == (3,)
    dorigin = torch.stack((dorigin0, dorigin1), dim=0)

    alias_mode = 0
    if alias:
        if not output_alias:
            alias_mode = 2
        else:
            alias_mode = 1

    return torch.ops.torchbp.ffbp_merge2_knab.default(
        img0,
        img1,
        dorigin,
        fc,
        r0,
        dr0,
        theta0,
        dtheta0,
        Nr0,
        Ntheta0,
        r3_0,
        dr3,
        theta3_0,
        dtheta3,
        nr3,
        ntheta3,
        z0,
        order,
        oversample,
        alias_mode,
        alias_fmod,
    )


def ffbp_merge2(
    img0: Tensor,
    img1: Tensor,
    dorigin0: Tensor,
    dorigin1: Tensor,
    grid_polars: list,
    fc: float,
    grid_polar_new: dict = None,
    z0: float = 0,
    method : tuple = ('lanczos', 6),
    alias: bool = False,
    alias_fmod: float = 0,
    output_alias: bool = True
) -> Tensor:
    """
    Interpolate two pseudo-polar radar images to new grid and change origin
    position by `dorigin`.

    Gradient not supported.

    Parameters
    ----------
    img0 : Tensor
        2D radar image in [range, angle] format. Dimensions should
        match with grid_polars grid. Image dimension can be different for each
        element in the list.
    img1 : Tensor
        Same format as img0.
    dorigin0 : Tensor
        Difference between the origin of the old image to the new image. Units in meters.
        Shape: [3].
    dorigin1 : Tensor
        Same format as dorigin0.
    grid_polar : list of dict
        List of grid definitions for each input image. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center in the new image.
    method : str or tuple
        Interpolation method. Valid choices are:
            - ("lanczos", n): Lanczos resampling. `n` is the number of samples used.
            - ("knab", n, v): Knab pulse resampling. `n` is the number of samples used.
              length and v is oversampling factor in the data.
    alias : bool
        Add back range dependent phase. Inverse of `util.bp_polar_range_dealias`.
    alias_fmod : float
        Range modulation frequency applied to input.
    output_alias : bool
        If True and `alias` is True apply `alias_fmod` to output.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """
    if type(method) in (list, tuple):
        method_params = method[1:]
        method = method[0]
    else:
        method_params = None

    if method == "lanczos":
        if len(method_params) != 1:
            raise ValueError("Lanczos interpolation needs sample length as argument")
        return ffbp_merge2_lanczos(
            img0,
            img1,
            dorigin0,
            dorigin1,
            grid_polars,
            fc,
            grid_polar_new,
            z0,
            order=method_params[0],
            alias=alias,
            alias_fmod=alias_fmod,
            output_alias=output_alias
        )
    elif method == "knab":
        if len(method_params) != 2:
            raise ValueError("Knab interpolation needs sample length and oversampling factor as argument")
        return ffbp_merge2_knab(
            img0,
            img1,
            dorigin0,
            dorigin1,
            grid_polars,
            fc,
            grid_polar_new,
            z0,
            order=method_params[0],
            oversample=method_params[1],
            alias=alias,
            alias_fmod=alias_fmod,
            output_alias=output_alias
        )
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")


def polar_to_cart(
    img: Tensor,
    origin: Tensor,
    grid_polar: dict,
    grid_cart: dict,
    fc: float,
    rotation: float = 0,
    alias_fmod: float = 0,
    method: str | tuple = "linear",
) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid.

    The input image should be either generated with `dealias=True` or call
    `torchbp.util.bp_polar_range_dealias` first.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    origin : Tensor
        3D antenna phase center of the old image in with respect to new image.
        Units in meters [nbatch, 3] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
            - "x": (x0, x1), tuple of min and max x-axis (range),
            - "y": (y0, y1), tuple of min and max y-axis (cross-range),
            - "nx": number of x-axis pixels.
            - "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    method : str or tuple
        Interpolation method. Valid choices are:
        - "linear": Linear interpolation.
        - ("lanczos", n): Lanczos resampling. `n` is the half of kernel length.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """
    if type(method) in (list, tuple):
        method_params = method[1]
        method = method[0]
    else:
        method_params = None

    if method == "linear":
        return polar_to_cart_linear(img, origin, grid_polar, grid_cart, fc, rotation, alias_fmod)
    elif method == "lanczos":
        return polar_to_cart_lanczos(
            img, origin, grid_polar, grid_cart, fc, rotation, alias_fmod, order=method_params
        )
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")


def polar_to_cart_linear(
    img: Tensor,
    origin: Tensor,
    grid_polar: dict,
    grid_cart: dict,
    fc: float,
    rotation: float = 0,
    alias_fmod: float = 0
) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid with linear interpolation.

    The input image should be either generated with `dealias=True` or call
    `torchbp.util.bp_polar_range_dealias` first.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    origin : Tensor
        3D antenna phase center of the old image in with respect to new image.
        Units in meters [nbatch, 3] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
            - "x": (x0, x1), tuple of min and max x-axis (range),
            - "y": (y0, y1), tuple of min and max y-axis (cross-range),
            - "nx": number of x-axis pixels.
            - "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert origin.shape == (nbatch, 3)
    else:
        nbatch = 1
        assert origin.shape == (3,)

    r0, r1 = grid_polar["r"]
    theta0, theta1 = grid_polar["theta"]
    ntheta = grid_polar["ntheta"]
    nr = grid_polar["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    x0, x1 = grid_cart["x"]
    y0, y1 = grid_cart["y"]
    nx = grid_cart["nx"]
    ny = grid_cart["ny"]
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    return torch.ops.torchbp.polar_to_cart_linear.default(
        img,
        origin,
        nbatch,
        rotation,
        fc,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        x0,
        y0,
        dx,
        dy,
        nx,
        ny,
        alias_fmod
    )


def polar_to_cart_lanczos(
    img: Tensor,
    origin: Tensor,
    grid_polar: dict,
    grid_cart: dict,
    fc: float,
    rotation: float = 0,
    alias_fmod: float = 0,
    order: int = 6,
) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid with linear interpolation.

    The input image should be either generated with `dealias=True` or call
    `torchbp.util.bp_polar_range_dealias` first.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    origin : Tensor
        3D antenna phase center of the old image in with respect to new image.
        Units in meters [nbatch, 3] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
            - "x": (x0, x1), tuple of min and max x-axis (range),
            - "y": (y0, y1), tuple of min and max y-axis (cross-range),
            - "nx": number of x-axis pixels.
            - "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
    alias_fmod : float
        Range modulation frequency applied to input.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert origin.shape == (nbatch, 3)
    else:
        nbatch = 1
        assert origin.shape == (3,)

    r0, r1 = grid_polar["r"]
    theta0, theta1 = grid_polar["theta"]
    ntheta = grid_polar["ntheta"]
    nr = grid_polar["nr"]
    dtheta = (theta1 - theta0) / ntheta
    dr = (r1 - r0) / nr

    x0, x1 = grid_cart["x"]
    y0, y1 = grid_cart["y"]
    nx = grid_cart["nx"]
    ny = grid_cart["ny"]
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    return torch.ops.torchbp.polar_to_cart_lanczos.default(
        img,
        origin,
        nbatch,
        rotation,
        fc,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        x0,
        y0,
        dx,
        dy,
        nx,
        ny,
        alias_fmod,
        order,
    )


def backprojection_polar_2d(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    dealias: bool = False,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    alias_fmod: float = 0
) -> Tensor:
    """
    2D backprojection with pseudo-polar coordinates.

    Gradient can be calculated with respect to data and pos.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nbatch, nsweeps, samples] or
        [nsweeps, samples]. If input is 3 dimensional the first dimensions is number
        of independent images to form at the same time. Whole batch is processed
        with same grid and other arguments.
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
    d0 : float
        Zero range correction.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center. Isotropic antenna is assumed if g is None.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    data_fmod : float
        Range modulation frequency applied to input data.
    alias_fmod : float
        Range modulation frequency applied to SAR image.

    Returns
    -------
    img : Tensor
        Pseudo-polar format radar image.
    """

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    nr = grid["nr"]
    ntheta = grid["ntheta"]
    dr = (r1 - r0) / nr
    dtheta = (theta1 - theta0) / ntheta

    if data.dim() == 2:
        nbatch = 1
        nsweeps = data.shape[0]
        sweep_samples = data.shape[1]
        assert pos.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)

    if att is None or g is None:
        att = None
        g = None
        g_nel = 0
        g_naz = 0
        g_daz = 0
        g_del = 0
        g_el0, g_az0, g_el1, g_az1 = 0, 0, 0, 0
    else:
        g_nel = g.shape[0]
        g_naz = g.shape[1]
        assert g.shape == torch.Size([g_nel, g_naz])
        g_el0, g_az0, g_el1, g_az1 = g_extent
        g_daz = (g_az1 - g_az0) / g_naz
        g_del = (g_el1 - g_el0) / g_nel

    z0 = 0
    if dealias:
        if nbatch != 1:
            raise ValueError("Only nbatch=1 supported with dealias")
        z0 = torch.mean(pos[..., 2])

    return torch.ops.torchbp.backprojection_polar_2d.default(
        data,
        pos,
        att,
        nbatch,
        sweep_samples,
        nsweeps,
        fc,
        r_res,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        d0,
        dealias,
        z0,
        g,
        g_az0,
        g_el0,
        g_daz,
        g_del,
        g_naz,
        g_nel,
        data_fmod,
        alias_fmod,
    )


def backprojection_polar_2d_lanczos(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    dealias: bool = False,
    order: int = 6,
    att: Tensor | None = None,
    g: Tensor = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    alias_fmod: float = 0
) -> Tensor:
    """
    2D backprojection with pseudo-polar coordinates. Interpolates input data
    using lanczos interpolation.

    Gradient not supported.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nbatch, nsweeps, samples] or
        [nsweeps, samples]. If input is 3 dimensional the first dimensions is number
        of independent images to form at the same time. Whole batch is processed
        with same grid and other arguments.
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
    d0 : float
        Zero range correction.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    data_fmod : float
        Range modulation frequency applied to input data.
    alias_fmod : float
        Range modulation frequency applied to SAR image.

    Returns
    -------
    img : Tensor
        Pseudo-polar format radar image.
    """

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    nr = grid["nr"]
    ntheta = grid["ntheta"]
    dr = (r1 - r0) / nr
    dtheta = (theta1 - theta0) / ntheta

    if data.dim() == 2:
        nbatch = 1
        nsweeps = data.shape[0]
        sweep_samples = data.shape[1]
        assert pos.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)

    if att is None or g is None:
        att = None
        g = None
        g_nel = 0
        g_naz = 0
        g_daz = 0
        g_del = 0
        g_el0, g_az0, g_el1, g_az1 = 0, 0, 0, 0
    else:
        g_nel = g.shape[0]
        g_naz = g.shape[1]
        assert g.shape == torch.Size([g_nel, g_naz])
        g_el0, g_az0, g_el1, g_az1 = g_extent
        g_daz = (g_az1 - g_az0) / g_naz
        g_del = (g_el1 - g_el0) / g_nel

    z0 = 0
    if dealias:
        z0 = torch.mean(pos[:, 2])

    return torch.ops.torchbp.backprojection_polar_2d_lanczos.default(
        data,
        pos,
        att,
        nbatch,
        sweep_samples,
        nsweeps,
        fc,
        r_res,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        d0,
        dealias,
        z0,
        order,
        g,
        g_az0,
        g_el0,
        g_daz,
        g_del,
        g_naz,
        g_nel,
        data_fmod,
        alias_fmod
    )


def backprojection_polar_2d_knab(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    dealias: bool = False,
    order: int = 4,
    oversample: float = 2,
    att: Tensor | None = None,
    g: Tensor = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    alias_fmod: float = 0
) -> Tensor:
    """
    2D backprojection with pseudo-polar coordinates. Interpolates input data
    using knab interpolation.

    Gradient not supported.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nbatch, nsweeps, samples] or
        [nsweeps, samples]. If input is 3 dimensional the first dimensions is number
        of independent images to form at the same time. Whole batch is processed
        with same grid and other arguments.
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
    d0 : float
        Zero range correction.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    order : int
        Number of nearby samples to use for interpolation of one new sample.
        Even number is preferred.
    oversample : float
        Oversampling factor in the input data.
    att : Tensor
        Antenna rotation tensor.
        [Roll, pitch, yaw]. Only yaw is used and only if beamwidth < Pi to filter
        out data outside the antenna beam.
    g : Tensor
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    data_fmod : float
        Range modulation frequency applied to input data.
    alias_fmod : float
        Range modulation frequency applied to SAR image.

    Returns
    -------
    img : Tensor
        Pseudo-polar format radar image.
    """

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    nr = grid["nr"]
    ntheta = grid["ntheta"]
    dr = (r1 - r0) / nr
    dtheta = (theta1 - theta0) / ntheta

    if data.dim() == 2:
        nbatch = 1
        nsweeps = data.shape[0]
        sweep_samples = data.shape[1]
        assert pos.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)

    if att is None or g is None:
        att = None
        g = None
        g_nel = 0
        g_naz = 0
        g_daz = 0
        g_del = 0
        g_el0, g_az0, g_el1, g_az1 = 0, 0, 0, 0
    else:
        g_nel = g.shape[0]
        g_naz = g.shape[1]
        assert g.shape == torch.Size([g_nel, g_naz])
        g_el0, g_az0, g_el1, g_az1 = g_extent
        g_daz = (g_az1 - g_az0) / g_naz
        g_del = (g_el1 - g_el0) / g_nel

    z0 = 0
    if dealias:
        z0 = torch.mean(pos[:, 2])

    return torch.ops.torchbp.backprojection_polar_2d_knab.default(
        data,
        pos,
        att,
        nbatch,
        sweep_samples,
        nsweeps,
        fc,
        r_res,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        d0,
        dealias,
        z0,
        order,
        oversample,
        g,
        g_az0,
        g_el0,
        g_daz,
        g_del,
        g_naz,
        g_nel,
        data_fmod,
        alias_fmod
    )


def backprojection_cart_2d(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    beamwidth: float = pi,
    data_fmod: float = 0
) -> Tensor:
    """
    2D backprojection with cartesian coordinates.

    Gradient can be calculated with respect to data and pos.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nbatch, nsweeps, samples] or
        [nsweeps, samples]. If input is 3 dimensional the first dimensions is number
        of independent images to form at the same time. Whole batch is processed
        with same grid and other arguments.
    grid : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
            - "x": (x0, x1), tuple of min and max x-axis (range),
            - "y": (y0, y1), tuple of min and max y-axis (cross-range),
            - "nx": number of x-axis pixels.
            - "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    beamwidth : float
        Beamwidth of the antenna in radians. Points outside the beam are not calculated.
    d0 : float
        Zero range correction.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    img : Tensor
        Cartesian format radar image.
    """

    x0, x1 = grid["x"]
    y0, y1 = grid["y"]
    nx = grid["nx"]
    ny = grid["ny"]
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    if data.dim() == 2:
        nbatch = 1
        nsweeps = data.shape[0]
        sweep_samples = data.shape[1]
        assert pos.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)

    return torch.ops.torchbp.backprojection_cart_2d.default(
        data,
        pos,
        nbatch,
        sweep_samples,
        nsweeps,
        fc,
        r_res,
        x0,
        dx,
        y0,
        dy,
        nx,
        ny,
        beamwidth,
        d0,
        data_fmod
    )


def projection_cart_2d(
    img: Tensor,
    pos: Tensor,
    grid: dict,
    fc: float,
    fs:float,
    gamma: float,
    sweep_samples: int,
    d0: float = 0.0,
    dem: Tensor | None = None,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    use_rvp: bool = True,
    normalization: str = "beta",
    vel: Tensor | None = None,
) -> Tensor:
    """
    Calculate FMCW radar data for each radar position in `pos` when measuring
    the scene in `img`.

    Parameters
    ----------
    img : Tensor
        SAR image in Cartesian coordinates. Shape [nx, ny] or [nbatch, nx, ny].
    grid : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
            - "x": (x0, x1), tuple of min and max x-axis (range),
            - "y": (y0, y1), tuple of min and max y-axis (cross-range),
            - "nx": number of x-axis pixels.
            - "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    gamma : float
        Distance to IF frequency conversion factor. For FMCW radar: BW / tsweep.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    d0 : float
        Zero range correction.
    dem : Tensor or None
        Digital elevation map. Should have shape: [nx, ny].
        Set to zero if None.
    att : Tensor or None
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    g : Tensor or None
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center. Isotropic antenna is assumed if g is None.
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1].
        g_el0, g_el1 are grx and gtx elevation axis start and end values. Units
        in radians. -pi/2 + +pi/2 if including data over the whole sphere.
        g_az0, g_az1 are grx and gtx azimuth axis start and end values. Units in
        radians. -pi to +pi if including data over the whole sphere.
    use_rvp : bool
        True to add residual video phase term.
    normalization : str
        Surface reflectivity definition to use. Valid choices are "sigma" or "gamma".
        "sigma": No look angle dependency (unphysical).
        "gamma": Multiply the reflectivity be cross-sectional area of the patch
        (more realistic).
    vel : Tensor or None
        Velocity tensor in m/s. Shape should match with pos.

    Returns
    -------
    data : Tensor
        FMCW radar data at each position. Shape [nbatch, nsweeps, nsamples].
    """

    x0, x1 = grid["x"]
    y0, y1 = grid["y"]
    nx = grid["nx"]
    ny = grid["ny"]
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    if img.dim() == 2:
        nbatch = 1
        nsweeps = pos.shape[0]
        if img.shape[0] != nx:
            raise ValueError("grid and img have different number of points in x")
        if img.shape[1] != ny:
            raise ValueError("grid and img have different number of points in y")
        if list(pos.shape) != [nsweeps, 3]:
            raise ValueError(f"Invalid pos shape {pos.shape}, expected {[nsweeps, 3]}")
    else:
        nbatch = img.shape[0]
        nsweeps = pos.shape[1]
        if img.shape[1] != nx:
            raise ValueError("grid and img have different number of points in x")
        if img.shape[2] != ny:
            raise ValueError("grid and img have different number of points in y")

        if list(pos.shape) != [nbatch, nsweeps, 3]:
            raise ValueError(f"Invalid pos shape {pos.shape}, expected {[nbatch, nsweeps, 3]}")

    if normalization == "sigma":
        norm = 0
    elif normalization == "gamma":
        norm = 1
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if dem is not None and list(dem.shape) != [nx, ny]:
        raise ValueError("img and dem shapes are different")

    if att is None or g is None:
        att = None
        g = None
        g_nel = 0
        g_naz = 0
        g_daz = 0
        g_del = 0
        g_el0, g_az0, g_el1, g_az1 = 0, 0, 0, 0
    else:
        g_nel = g.shape[0]
        g_naz = g.shape[1]
        assert g.shape == torch.Size([g_nel, g_naz])
        g_el0, g_az0, g_el1, g_az1 = g_extent
        g_daz = (g_az1 - g_az0) / g_naz
        g_del = (g_el1 - g_el0) / g_nel

    if vel is not None:
        if vel.shape != pos.shape:
            raise ValueError(f"vel shape {vel.shape} doesn't match with pos shape {pos.shape}")

    return torch.ops.torchbp.projection_cart_2d.default(
        img,
        dem,
        pos,
        vel,
        att,
        nbatch,
        sweep_samples,
        nsweeps,
        fc,
        fs,
        gamma,
        x0,
        dx,
        y0,
        dy,
        nx,
        ny,
        d0,
        g,
        g_az0,
        g_el0,
        g_daz,
        g_del,
        g_naz,
        g_nel,
        use_rvp,
        norm
    )


def gpga_backprojection_2d_core(
    target_pos: Tensor,
    data: Tensor,
    pos: Tensor,
    fc: float,
    r_res: float,
    d0: float = 0.0,
    interp_method: str = "linear",
    data_fmod: float = 0
) -> Tensor:
    """
    Generalized phase gradient autofocus.

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
    d0 : float
        Zero range correction.
    interp_method : str
        Interpolation method
        "linear": linear interpolation.
        ("lanczos", N): Lanczos interpolation with order N.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    data_out : Tensor
        Values from input data used in backprojection of each target in
        target_pos tensor. Shape is [ntargets, nsweeps].
    """
    nsweeps = data.shape[0]
    ntargets = target_pos.shape[0]
    sweep_samples = data.shape[1]
    assert target_pos.shape == (ntargets, 3)
    assert pos.shape == (nsweeps, 3)

    if type(interp_method) in (list, tuple):
        method_params = interp_method[1]
        interp_method = interp_method[0]
    else:
        method_params = None
    if interp_method == "linear":
        return torch.ops.torchbp.gpga_backprojection_2d.default(
            target_pos, data, pos, sweep_samples, nsweeps, fc, r_res, ntargets, d0, data_fmod
        )
    elif interp_method == "lanczos":
        return torch.ops.torchbp.gpga_backprojection_2d_lanczos.default(
            target_pos,
            data,
            pos,
            sweep_samples,
            nsweeps,
            fc,
            r_res,
            ntargets,
            d0,
            method_params,
            data_fmod
        )
    else:
        raise ValueError(f"Unknown interp_method f{interp_method}")


def cfar_2d(
    img: Tensor, Navg: tuple, Nguard: tuple, threshold: float, peaks_only: bool = False
) -> Tensor:
    """
    Constant False Alaram Rate detection for 2D image.

    Parameters
    ----------
    img : Tensor
        Absolute valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).
    Nguard : tuple
        Number of guard cells in 2D (N1, N0).
        Both dimensions need to be less or same than the same entry in Navg.
    threshold : float
        Threshold for detection.
    peaks_only : bool
        Reject pixels that are not higher than their immediate neighbors.
        Defaults is False.

    Returns
    -------
    detection : Tensor
        Detections tensor same shape as the input image. For each pixel 0 for no
        detection, positive value is the SNR of detection at that position.
    """
    if img.dim() == 3:
        nbatch = img.shape[0]
        N0 = img.shape[1]
        N1 = img.shape[2]
    elif img.dim() == 2:
        nbatch = 1
        N0 = img.shape[0]
        N1 = img.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if threshold <= 0:
        raise ValueError("Threshold should be positive")
    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if len(Nguard) != 2:
        raise ValueError("Nguard dimension should be 2")
    if Nguard[0] > Navg[0]:
        raise ValueError("Nguard[0] > Navg[0]")
    if Nguard[1] > Navg[1]:
        raise ValueError("Nguard[1] > Navg[1]")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")
    if Nguard[0] < 0:
        raise ValueError("Nguard[0] < 0")
    if Nguard[1] < 0:
        raise ValueError("Nguard[1] < 0")

    return torch.ops.torchbp.cfar_2d.default(
        img,
        nbatch,
        N0,
        N1,
        Navg[0],
        Navg[1],
        Nguard[0],
        Nguard[1],
        threshold,
        peaks_only,
    )


def coherence_2d(img0: Tensor, img1: Tensor, Navg: tuple) -> Tensor:
    """
    Coherence of two complex images over moving window `Navg`.

    Parameters
    ----------
    img0 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    img1 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).

    Returns
    -------
    out : Tensor
        Real valued coherence image with same shape as input calculated over the
        moving window.
    """
    if img0.shape != img1.shape:
        raise ValueError(f"img0.shape != img1.shape. {img0.shape} != {img1.shape}")
    if img0.dim() == 3:
        nbatch = img0.shape[0]
        N0 = img0.shape[1]
        N1 = img0.shape[2]
    elif img0.dim() == 2:
        nbatch = 1
        N0 = img0.shape[0]
        N1 = img0.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img0.shape}")

    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")

    return torch.ops.torchbp.coherence_2d.default(
        img0,
        img1,
        nbatch,
        N0,
        N1,
        Navg[0],
        Navg[1],
    )


def power_coherence_2d(
    img0: Tensor, img1: Tensor, Navg: tuple, corr_output: bool = True
) -> Tensor:
    """
    Coherence of two complex images over moving window `Navg`. Calculated from
    squared absolute value of the images. [1]_

    Parameters
    ----------
    img0 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    img1 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).
    corr_output : bool
        Return ordinary correlation coefficient by calculating sqrt(2*v-1) for
        all output values if v > 0.5 and else 0.

    References
    ----------
    .. [1] A. M. Guarnieri and C. Prati, "SAR interferometry: a "Quick and
        dirty" coherence estimator for data browsing," in IEEE Transactions on
        Geoscience and Remote Sensing, vol. 35, no. 3, pp. 660-669, May 1997.

    Returns
    -------
    out : Tensor
        Real valued coherence image with same shape as input calculated over the
        moving window.
    """
    if img0.shape != img1.shape:
        raise ValueError(f"img0.shape != img1.shape. {img0.shape} != {img1.shape}")
    if img0.dim() == 3:
        nbatch = img0.shape[0]
        N0 = img0.shape[1]
        N1 = img0.shape[2]
    elif img0.dim() == 2:
        nbatch = 1
        N0 = img0.shape[0]
        N1 = img0.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img0.shape}")

    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")

    return torch.ops.torchbp.power_coherence_2d.default(
        img0, img1, nbatch, N0, N1, Navg[0], Navg[1], corr_output
    )


def backprojection_polar_2d_tx_power(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: dict,
    r_res: float,
    pos: Tensor,
    att: Tensor,
    normalization: str | None = None,
) -> Tensor:
    """
    Calculate square root of transmitted power to image plane. Can be used to
    correct for antenna pattern and distance effect on the radar image.

    Parameters
    ----------
    wa : Tensor
        Weighting coefficient for amplitude of each pulse. Should include window
        function and transmit power variation if known, shape: [nsweeps] or
        [nbatch, nsweeps].
    g : Tensor
        Square-root of two-way antenna gain in spherical coordinates, shape: [elevation, azimuth].
        If TX antenna equals RX antenna, then this should be just antenna gain.
        (0, 0) angle is at the beam center.
    g_az0 : float
        grx and gtx azimuth axis starting value. Units in radians. -pi if
        including data over the whole sphere.
    g_el0 : float
        grx and gtx elevation axis starting value. Units in radians. -pi/2 if
        including data over the whole sphere.
    g_az1 : float
        grx and gtx azimuth axis end value. Units in radians. +pi if
        including data over the whole sphere.
    g_el1 : float
        grx and gtx elevation axis end value. Units in radians. +pi/2 if
        including data over the whole sphere.
    grid : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
            - "r": (r0, r1), tuple of min and max range,
            - "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
            - "nr": nr, number of range bins.
            - "ntheta": number of angle bins.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    normalization : str or None
        Valid choices are:
            "sigma" to divide each value by sin of incidence angle.
            "gamma" to divide each value by of tan of incidence angle.
            "beta" or None for no incidence angle normalization.
            "point" to normalize to constant reflectivity (no ground patch).

    Returns
    -------
    tx_power : Tensor
        Pseudo-polar format image of square root of power returned from each
        pixel assuming constant reflectivity.
    """

    r0, r1 = grid["r"]
    theta0, theta1 = grid["theta"]
    nr = grid["nr"]
    ntheta = grid["ntheta"]
    dr = (r1 - r0) / nr
    dtheta = (theta1 - theta0) / ntheta

    if wa.dim() == 1:
        nbatch = 1
        nsweeps = wa.shape[0]
        assert pos.shape == (nsweeps, 3)
        assert att.shape == (nsweeps, 3)
    else:
        nbatch = wa.shape[0]
        nsweeps = wa.shape[1]
        assert pos.shape == (nbatch, nsweeps, 3)
        assert att.shape == (nbatch, nsweeps, 3)

    g_nel = g.shape[0]
    g_naz = g.shape[1]
    g_el0, g_az0, g_el1, g_az1 = g_extent
    g_daz = (g_az1 - g_az0) / g_naz
    g_del = (g_el1 - g_el0) / g_nel

    if normalization == "beta" or normalization is None:
        norm = 0
    elif normalization == "sigma":
        norm = 1
    elif normalization == "gamma":
        norm = 2
    elif normalization == "point":
        norm = 3
    else:
        raise ValueError(f"Invalid normalization {normalization}.")

    return torch.ops.torchbp.backprojection_polar_2d_tx_power.default(
        wa,
        pos,
        att,
        g,
        nbatch,
        g_az0,
        g_el0,
        g_daz,
        g_del,
        g_naz,
        g_nel,
        nsweeps,
        r_res,
        r0,
        dr,
        theta0,
        dtheta,
        nr,
        ntheta,
        norm,
    )


def lee_filter(img: Tensor, wx: int, wy: int, cu: float) -> Tensor:
    """
    Lee filter for speckle noise reduction.

    Parameters
    ----------
    img : Tensor
        Complex or real SAR image.
    wx : int
        Window size in the first dimension.
    wy : int
        Window size in the second dimension.
    cu : float
        Coefficient of variance of the noise-free image.

    Returns
    -------
    img : Tensor
        Filtered input image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        Nx = img.shape[1]
        Ny = img.shape[2]
    elif img.dim() == 2:
        nbatch = 1
        Nx = img.shape[0]
        Ny = img.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    # Half-window size in C++
    wx = wx // 2
    wy = wy // 2

    return torch.ops.torchbp.lee_filter.default(img, nbatch, Nx, Ny, wx, wy, cu)


def ffbp(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int = 2,
    d0: float = 0.0,
    interp_method: str | tuple = ("lanczos", 6),
    oversample_r: float = 1.4,
    oversample_theta: float = 1.4,
    grid_oversample: float = 1,
    dealias: bool = False,
    data_fmod: float = 0,
    alias_fmod: float = None,
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
    interp_method : str or tuple
        Interpolation method. See `polar_interp` function.
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

    try:
        if interp_method[0] not in ["lanczos", "knab"]:
            raise ValueError(
                "interp_method should be ('lanczos', N) or ('knab', N, v)"
            )
    except IndexError:
        raise ValueError("interp_method should be ('lanczos', N) or ('knab', N, v)")
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
            img = ffbp(
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
            output_alias=out_alias
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


def multilook_polar(sar_img: Tensor, kernel: tuple, grid_polar: dict) -> (Tensor, dict):
    sar_img = torch.nn.functional.avg_pool2d(
        sar_img.real, kernel, stride=None
    ) + 1j * torch.nn.functional.avg_pool2d(sar_img.imag, kernel, stride=None)
    grid_out = {
        "r": grid_polar["r"],
        "theta": grid_polar["theta"],
        "nr": sar_img.shape[-2],
        "ntheta": sar_img.shape[-1],
    }
    return sar_img, grid_out


def subpixel_correlation_op(im_m: Tensor, im_s: Tensor) -> (Tensor, Tensor, Tensor):
    if im_m.dim() == 3:
        nbatch = im_m.shape[0]
        Nx = im_m.shape[1]
        Ny = im_m.shape[2]
    elif im_m.dim() == 2:
        nbatch = 1
        Nx = im_m.shape[0]
        Ny = im_m.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {im_m.shape}")

    if im_m.shape != im_s.shape:
        raise ValueError(f"Image shapes are different {im_m.shape} != {im_s.shape}")

    mean_m = torch.mean(im_m, dim=(-2,-1))
    mean_s = torch.mean(im_s, dim=(-2,-1))
    return torch.ops.torchbp.subpixel_correlation.default(
            im_m,
            im_s,
            mean_m,
            mean_s,
            nbatch,
            Nx,
            Ny)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torchbp::polar_interp_linear")
def _fake_polar_interp_linear(
    img: Tensor,
    dorigin: Tensor,
    rotation: float,
    fc: float,
    r0: float,
    dr0: float,
    theta0: float,
    dtheta0: float,
    Nr0: float,
    Ntheta0: float,
    r1: float,
    dr1: float,
    theta1: float,
    dtheta1: float,
    Nr1: float,
    Ntheta1: float,
    z1: float,
    alias_fmod: float,
) -> Tensor:
    torch._check(dorigin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((Nr1, Ntheta1), dtype=torch.complex64, device=img.device)


@torch.library.register_fake("torchbp::polar_interp_linear_grad")
def _fake_polar_interp_linear_grad(
    grad: Tensor,
    img: Tensor,
    dorigin: Tensor,
    rotation: float,
    fc: float,
    r0: float,
    dr0: float,
    theta0: float,
    dtheta0: float,
    Nr0: float,
    Ntheta0: float,
    r1: float,
    dr1: float,
    theta1: float,
    dtheta1: float,
    Nr1: float,
    Ntheta1: float,
    z1: float,
    alias_fmod: float,
) -> Tensor:
    torch._check(dorigin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    ret = []
    if img.requires_grad:
        ret.append(
            torch.empty((Nr1, Ntheta1), dtype=torch.complex64, device=img.device)
        )
    else:
        ret.append(None)
    if dorigin.requires_grad:
        ret.append(torch.empty((2,), dtype=torch.float, device=img.device))
    else:
        ret.append(None)
    return ret


@torch.library.register_fake("torchbp::polar_to_cart_linear")
def _fake_polar_to_cart_linear(
    img: Tensor,
    dorigin: Tensor,
    nbatch: int,
    rotation: float,
    fc: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    nr: int,
    ntheta: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> Tensor:
    torch._check(dorigin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((Nx, Ny), dtype=torch.complex64, device=img.device)


@torch.library.register_fake("torchbp::polar_to_cart_linear_grad")
def _fake_polar_interp_linear_grad(
    grad: Tensor,
    img: Tensor,
    dorigin: Tensor,
    rotation: float,
    fc: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    Nr: float,
    Ntheta: float,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    Nx: float,
    Ny: float,
) -> Tensor:
    torch._check(dorigin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    ret = []
    if img.requires_grad:
        ret.append(torch.empty_like(img))
    else:
        ret.append(None)
    if dorigin.requires_grad:
        ret.append(torch.empty_like(dorigin))
    else:
        ret.append(None)
    return ret


@torch.library.register_fake("torchbp::backprojection_polar_2d")
def _fake_polar_2d(
    data: Tensor,
    pos: Tensor,
    nbatch: int,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    r_res: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    Nr: int,
    Ntheta: int,
    d0: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64 or data.dtype == torch.complex32)
    return torch.empty((nbatch, Nr, Ntheta), dtype=torch.complex64, device=data.device)


@torch.library.register_fake("torchbp::backprojection_polar_2d_grad")
def _fake_polar_2d_grad(
    grad: Tensor,
    data: Tensor,
    pos: Tensor,
    nbatch: int,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    r_res: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    Nr: int,
    Ntheta: int,
    d0: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64 or data.dtype == torch.complex32)
    torch._check(grad.dtype == torch.complex64)
    ret = []
    if data.requires_grad:
        ret.append(torch.empty_like(data))
    else:
        ret.append(None)
    if pos.requires_grad:
        ret.append(torch.empty_like(pos))
    else:
        ret.append(None)
    return ret


@torch.library.register_fake("torchbp::backprojection_cart_2d")
def _fake_cart_2d(
    data: Tensor,
    pos: Tensor,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    r_res: float,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    Nx: int,
    Ny: int,
    beamwidth: float,
    d0: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64)
    return torch.empty((Nx, Ny), dtype=torch.complex64, device=data.device)


@torch.library.register_fake("torchbp::backprojection_cart_2d_grad")
def _fake_cart_2d_grad(
    grad: Tensor,
    data: Tensor,
    pos: Tensor,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    r_res: float,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    Nx: int,
    Ny: int,
    beamwidth: float,
    d0: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64)
    torch._check(grad.dtype == torch.complex64)
    ret = []
    if data.requires_grad:
        ret.append(torch.empty_like(data))
    else:
        ret.append(None)
    if pos.requires_grad:
        ret.append(torch.empty_like(pos))
    else:
        ret.append(None)
    return ret


def _setup_context_polar_2d(ctx, inputs, output):
    data, pos, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i <= 1:
                continue
            raise NotImplementedError("Only data and pos gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(data, pos)


def _backward_polar_2d(ctx, grad):
    data = ctx.saved_tensors[0]
    pos = ctx.saved_tensors[1]
    if ctx.saved[13]:
        raise ValueError("dealias gradient not supported")
    ret = torch.ops.torchbp.backprojection_polar_2d_grad.default(
        grad, data, pos, *ctx.saved
    )
    grads = [None] * polar_2d_nargs
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)


def _backward_cart_2d(ctx, grad):
    data = ctx.saved_tensors[0]
    pos = ctx.saved_tensors[1]
    ret = torch.ops.torchbp.backprojection_cart_2d_grad.default(
        grad, data, pos, *ctx.saved
    )
    grads = [None] * cart_2d_nargs
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)


def _setup_context_cart_2d(ctx, inputs, output):
    data, pos, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i <= 1:
                continue
            raise NotImplementedError("Only data and pos gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(data, pos)


def _backward_polar_interp_linear(ctx, grad):
    img = ctx.saved_tensors[0]
    dorigin = ctx.saved_tensors[1]
    ret = torch.ops.torchbp.polar_interp_linear_grad.default(
        grad, img, dorigin, *ctx.saved
    )
    grads = [None] * polar_interp_linear_args
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)


def _setup_context_polar_interp_linear(ctx, inputs, output):
    img, dorigin, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i <= 1:
                continue
            raise NotImplementedError("Only img and dorigin gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(img, dorigin)


def _backward_polar_to_cart_linear(ctx, grad):
    img = ctx.saved_tensors[0]
    origin = ctx.saved_tensors[1]
    ret = torch.ops.torchbp.polar_to_cart_linear_grad.default(
        grad, img, origin, *ctx.saved
    )
    grads = [None] * polar_to_cart_linear_args
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)


def _setup_context_polar_to_cart_linear(ctx, inputs, output):
    img, origin, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i <= 1:
                continue
            raise NotImplementedError("Only img gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(img, origin)


def _backward_entropy(ctx, grad):
    data, norm = ctx.saved_tensors
    ret = torch.ops.torchbp.entropy_grad.default(data, norm, grad, *ctx.saved)
    grads = [None] * entropy_args
    grads[: len(ret)] = ret
    return tuple(grads)


def _setup_context_entropy(ctx, inputs, output):
    data, norm, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data, norm)


def _backward_abs_sum(ctx, grad):
    data = ctx.saved_tensors[0]
    ret = torch.ops.torchbp.abs_sum_grad.default(data, grad, *ctx.saved)
    grads = [None] * abs_sum_args
    grads[0] = ret
    return tuple(grads)


def _setup_context_abs_sum(ctx, inputs, output):
    data, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data)


def _backward_coherence_2d(ctx, grad):
    img0, img1 = ctx.saved_tensors
    ret = torch.ops.torchbp.coherence_2d_grad.default(grad, img0, img1, *ctx.saved)
    grads = [None] * coherence_2d_args
    grads[: len(ret)] = ret
    return tuple(grads)


def _setup_context_coherence_2d(ctx, inputs, output):
    img0, img1, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(img0, img1)


torch.library.register_autograd(
    "torchbp::backprojection_polar_2d",
    _backward_polar_2d,
    setup_context=_setup_context_polar_2d,
)
torch.library.register_autograd(
    "torchbp::backprojection_cart_2d",
    _backward_cart_2d,
    setup_context=_setup_context_cart_2d,
)
torch.library.register_autograd(
    "torchbp::polar_interp_linear",
    _backward_polar_interp_linear,
    setup_context=_setup_context_polar_interp_linear,
)
torch.library.register_autograd(
    "torchbp::polar_to_cart_linear",
    _backward_polar_to_cart_linear,
    setup_context=_setup_context_polar_to_cart_linear,
)
torch.library.register_autograd(
    "torchbp::entropy", _backward_entropy, setup_context=_setup_context_entropy
)
torch.library.register_autograd(
    "torchbp::abs_sum", _backward_abs_sum, setup_context=_setup_context_abs_sum
)
torch.library.register_autograd(
    "torchbp::coherence_2d",
    _backward_coherence_2d,
    setup_context=_setup_context_coherence_2d,
)
