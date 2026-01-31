import torch
from torch import Tensor
import numpy as np
from ._utils import unpack_polar_grid, unpack_cartesian_grid, get_batch_dims_img

polar_interp_linear_args = 19
polar_to_cart_linear_args = 18

def select_knab_poly_degree(oversample: float, order: int) -> int:
    """
    Select minimum polynomial degree for full Knab kernel polynomial approximation.

    Based on empirical testing with MSE threshold = 0.1 dB.

    Parameters
    ----------
    oversample : float
        Oversampling ratio.
    order : int
        Kernel order (number of samples used in interpolation).

    Returns
    -------
    int
        Polynomial degree.
    """
    if order <= 2:
        return 6
    elif order <= 4:
        return 6
    elif order <= 6:
        if oversample < 1.29:
            return 6
        elif oversample < 1.98:
            return 8
        else:
            return 10
    else:  # order >= 8
        if oversample < 1.25:
            return 8
        elif oversample < 1.66:
            return 10
        elif oversample < 2.76:
            return 12
        else:
            return 14


def compute_knab_poly_coefs_full(order, oversample, poly_degree=None):
    if poly_degree is None:
        poly_degree = select_knab_poly_degree(oversample, order)

    a = order / 2.0
    v = 1.0 - 1.0 / oversample

    # Chebyshev nodes in t ∈ [0, 1]
    n = max(3 * order, 3 * poly_degree)
    k = np.arange(n)
    t = 0.5 * (1 + np.cos((2*k + 1) * np.pi / (2*n)))

    # x from t
    x = a * np.sqrt(t)

    # True kernel
    cosh_num = np.cosh(np.pi * v * a * np.sqrt(1 - t))
    cosh_den = np.cosh(np.pi * v * a)
    y = np.sinc(x) * (cosh_num / cosh_den)

    # Build Vandermonde for w(t) = 1 + c1 t + c2 t^2 + ...
    y_target = y - 1.0
    A = t[:, None] ** np.arange(1, poly_degree + 1)

    # Enforce w(1) = 0 as hard constraint
    row = np.ones((1, poly_degree))
    A = np.vstack([A, row])
    y_target = np.concatenate([y_target, [-1.0]])

    coefs, *_ = np.linalg.lstsq(A, y_target, rcond=None)
    return torch.from_numpy(coefs).float()


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


def _prepare_polar_interp_linear_args(
    img: Tensor,
    dorigin: Tensor,
    grid_polar: dict,
    fc: float,
    rotation: float = 0,
    grid_polar_new: dict = None,
    z0: float = 0,
    alias_fmod: float = 0,
) -> tuple:
    """Prepare arguments for C++ polar_interp_linear operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by polar_interp_linear and for testing.
    """
    nbatch = get_batch_dims_img(img, dorigin)
    r1_0, r1_1, theta1_0, theta1_1, nr1, ntheta1, dr1, dtheta1 = unpack_polar_grid(grid_polar)

    if grid_polar_new is None:
        r3_0, r3_1 = r1_0, r1_1
        theta3_0, theta3_1 = theta1_0, theta1_1
        nr3 = nr1
        ntheta3 = 2 * ntheta1
        dr3 = dr1
        dtheta3 = dtheta1 / 2
    else:
        r3_0, r3_1, theta3_0, theta3_1, nr3, ntheta3, dr3, dtheta3 = unpack_polar_grid(grid_polar_new)

    return (img, dorigin, nbatch, rotation, fc, r1_0, dr1, theta1_0, dtheta1,
            nr1, ntheta1, r3_0, dr3, theta3_0, dtheta3, nr3, ntheta3, z0, alias_fmod)


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
    cpp_args = _prepare_polar_interp_linear_args(
        img, dorigin, grid_polar, fc, rotation, grid_polar_new, z0, alias_fmod
    )
    return torch.ops.torchbp.polar_interp_linear.default(*cpp_args)


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
            alias_mode = 1
        else:
            alias_mode = 2
    elif not output_alias:
        alias_mode = 3

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
            alias_mode = 1
        else:
            alias_mode = 2
    elif not output_alias:
        alias_mode = 3

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


def ffbp_merge2_poly(
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
    output_alias: bool = True,
    poly_degree: int = None,
    poly_coefs: Tensor = None
) -> Tensor:
    """
    Interpolate two pseudo-polar radar images to new grid and change origin
    position by `dorigin`. Uses polynomial approximation for interpolation.

    This function uses polynomial approximation of the Knab kernel,
    avoiding expensive exp and sqrt operations.

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
        Even number is preferred. Must be in [2, 8].
    oversample : float
        Oversampling factor in the input data.
    alias : bool
        Add back range dependent phase. Inverse of `util.bp_polar_range_dealias`.
    alias_fmod : float
        Range modulation frequency applied to input.
    output_alias : bool
        If True and `alias` is True apply `alias_fmod` to output.
    poly_degree : int, optional
        Degree of polynomial approximation for the full Knab kernel.
        The polynomial approximates sinc(x)*window(x) as a single polynomial in x²,
        eliminating expensive sinpif and division operations.
        If None (default), automatically selected based on oversample:
        - oversample < 1.5: degree 10
        - oversample >= 1.5: degree 12
    poly_coefs : Tensor, optional
        Precomputed polynomial coefficients. If provided, poly_degree is ignored.
        Use compute_knab_poly_coefs_full() to compute these coefficients.
        Precomputing is useful when calling this function multiple times with
        the same order and oversample parameters.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    device = img0.device
    nimages = 2

    if order > 8:
        raise ValueError(f"Polynomial interpolation knab order must be <= 8, got {order}")

    # Use precomputed coefficients if provided, otherwise compute them
    if poly_coefs is None:
        # Auto-select polynomial degree if not specified
        if poly_degree is None:
            poly_degree = select_knab_poly_degree(oversample, order)

        # Compute polynomial coefficients for full Knab kernel (sinc * window as single polynomial)
        # This eliminates sinpif and division from the CUDA kernel
        poly_coefs = compute_knab_poly_coefs_full(order, oversample, poly_degree).to(device)
    else:
        poly_coefs = poly_coefs.to(device)

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
            alias_mode = 1
        else:
            alias_mode = 2
    elif not output_alias:
        alias_mode = 3

    return torch.ops.torchbp.ffbp_merge2_poly.default(
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
        poly_coefs,
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
    method : tuple = ('knab', 6, 1.5),
    alias: bool = False,
    alias_fmod: float = 0,
    output_alias: bool = True,
    use_poly: bool = True,
    poly_coefs: Tensor = None
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
    method : tuple
        Interpolation method: ("knab", n, v) where n is the number of samples used
        and v is the oversampling factor in the data.
    alias : bool
        Add back range dependent phase. Inverse of `util.bp_polar_range_dealias`.
    alias_fmod : float
        Range modulation frequency applied to input.
    output_alias : bool
        If True and `alias` is True apply `alias_fmod` to output.
    use_poly : bool
        If True (default), use polynomial-approximation kernel for knab interpolation.
    poly_coefs : Tensor, optional
        Precomputed polynomial coefficients.
        Use compute_knab_poly_coefs_full() to compute these for Knab kernels.

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

    if method == "knab":
        if len(method_params) != 2:
            raise ValueError("Knab interpolation needs sample length and oversampling factor as argument")
        order = method_params[0]
        knab_func = ffbp_merge2_poly if (use_poly and order <= 8) else ffbp_merge2_knab
        kwargs = dict(
            order=order,
            oversample=method_params[1],
            alias=alias,
            alias_fmod=alias_fmod,
            output_alias=output_alias
        )
        if use_poly and poly_coefs is not None:
            kwargs['poly_coefs'] = poly_coefs
        return knab_func(
            img0,
            img1,
            dorigin0,
            dorigin1,
            grid_polars,
            fc,
            grid_polar_new,
            z0,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown interp_method: {method}")


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
        raise ValueError(f"Unknown method: {method}")


def _prepare_polar_to_cart_linear_args(
    img: Tensor,
    origin: Tensor,
    grid_polar: dict,
    grid_cart: dict,
    fc: float,
    rotation: float = 0,
    alias_fmod: float = 0
) -> tuple:
    """Prepare arguments for C++ polar_to_cart_linear operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by polar_to_cart_linear and for testing.
    """
    nbatch = get_batch_dims_img(img, origin)
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid_polar)
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid_cart)

    return (img, origin, nbatch, rotation, fc, r0, dr, theta0, dtheta,
            nr, ntheta, x0, y0, dx, dy, nx, ny, alias_fmod)


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
    cpp_args = _prepare_polar_to_cart_linear_args(
        img, origin, grid_polar, grid_cart, fc, rotation, alias_fmod
    )
    return torch.ops.torchbp.polar_to_cart_linear.default(*cpp_args)


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


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torchbp::polar_interp_linear")
def _fake_polar_interp_linear(
    img: Tensor,
    dorigin: Tensor,
    nbatch: int,
    rotation: float,
    fc: float,
    r0: float,
    dr0: float,
    theta0: float,
    dtheta0: float,
    Nr0: int,
    Ntheta0: int,
    r1: float,
    dr1: float,
    theta1: float,
    dtheta1: float,
    Nr1: int,
    Ntheta1: int,
    z1: float,
    alias_fmod: float,
) -> Tensor:
    torch._check(dorigin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((nbatch, Nr1, Ntheta1), dtype=torch.complex64, device=img.device)


@torch.library.register_fake("torchbp::polar_interp_linear_grad")
def _fake_polar_interp_linear_grad(
    grad: Tensor,
    img: Tensor,
    dorigin: Tensor,
    nbatch: int,
    rotation: float,
    fc: float,
    r0: float,
    dr0: float,
    theta0: float,
    dtheta0: float,
    Nr0: int,
    Ntheta0: int,
    r1: float,
    dr1: float,
    theta1: float,
    dtheta1: float,
    Nr1: int,
    Ntheta1: int,
    z1: float,
    alias_fmod: float,
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


@torch.library.register_fake("torchbp::polar_to_cart_linear")
def _fake_polar_to_cart_linear(
    img: Tensor,
    origin: Tensor,
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
    alias_fmod: float,
) -> Tensor:
    torch._check(origin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((nbatch, nx, ny), dtype=torch.complex64, device=img.device)


@torch.library.register_fake("torchbp::polar_to_cart_linear_grad")
def _fake_polar_to_cart_linear_grad(
    grad: Tensor,
    img: Tensor,
    origin: Tensor,
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
    alias_fmod: float,
) -> Tensor:
    torch._check(origin.dtype == torch.float32)
    torch._check(img.dtype == torch.complex64)
    ret = []
    if img.requires_grad:
        ret.append(torch.empty_like(img))
    else:
        ret.append(None)
    if origin.requires_grad:
        ret.append(torch.empty_like(origin))
    else:
        ret.append(None)
    return ret


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
