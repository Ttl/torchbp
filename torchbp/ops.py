import torch
from math import pi
from torch import Tensor
from copy import deepcopy
from .util import bp_polar_range_dealias

cart_2d_nargs = 18
polar_2d_nargs = 19
polar_interp_linear_args = 18
polar_to_cart_linear_args = 18
polar_to_cart_bicubic_args = 21
entropy_args = 3
abs_sum_args = 2

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
    ----------
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

def polar_interp(img: Tensor, dorigin: Tensor, grid_polar: dict,
        fc: float, rotation: float=0,
        grid_polar_new: dict=None, z0: float=0,
        method : str | tuple="linear") -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.
    Allows choosing the interpolation method.

    Gradient calculation is only supported with "linear" method.

    Note: Z-axis interpolation likely incorrect.

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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center.
    method : str or tuple
        Interpolation method. Valid choices are:
        - "linear": Linear interpolation.
        - "cubic": Cubic interpolation.
        - ("lanczos", n): Lanczos resampling. `n` is the half of kernel length.

    Returns
    ----------
    out : Tensor
        Interpolated radar image.
    """
    if type(method) in (list, tuple):
        method_params = method[1]
        method = method[0]
    else:
        method_params = None

    if method == "linear":
        return polar_interp_linear(img, dorigin, grid_polar, fc, rotation,
                grid_polar_new, z0)
    elif method == "cubic":
        return polar_interp_bicubic(img, dorigin, grid_polar, fc, rotation,
                grid_polar_new, z0)
    elif method == "lanczos":
        return polar_interp_lanczos(img, dorigin, grid_polar, fc, rotation,
                grid_polar_new, z0, order=method_params)
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")

def polar_interp_linear(img: Tensor, dorigin: Tensor, grid_polar: dict,
        fc: float, rotation: float=0,
        grid_polar_new: dict=None, z0: float=0) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.

    Gradient can be calculated with respect to img and dorigin.

    Note: Z-axis interpolation likely incorrect.

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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center.

    Returns
    ----------
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
        ntheta3 = 2*ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    return torch.ops.torchbp.polar_interp_linear.default( img, dorigin, nbatch,
            rotation, fc, r1_0, dr1, theta1_0, dtheta1, nr1, ntheta1, r3_0, dr3,
            theta3_0, dtheta3, nr3, ntheta3, z0)

def polar_interp_bicubic(img: Tensor, dorigin: Tensor, grid_polar: dict,
        fc: float, rotation: float=0,
        grid_polar_new: dict=None, z0: float=0) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.

    Gradient not supported.

    Note: Z-axis interpolation likely incorrect.

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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center.

    Returns
    ----------
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
        ntheta3 = 2*ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    img_gx, img_gy = torch.gradient(img, dim=(-2,-1), edge_order=2)
    img_gxy = torch.gradient(img_gx, dim=-1)[0]

    return torch.ops.torchbp.polar_interp_bicubic.default(img, img_gx, img_gy,
            img_gxy, dorigin, nbatch, rotation, fc, r1_0, dr1, theta1_0,
            dtheta1, nr1, ntheta1, r3_0, dr3, theta3_0, dtheta3, nr3, ntheta3,
            z0)

def polar_interp_lanczos(img: Tensor, dorigin: Tensor, grid_polar: dict,
        fc: float, rotation: float=0,
        grid_polar_new: dict=None, z0: float=0, order: int=4) -> Tensor:
    """
    Interpolate pseudo-polar radar image to new grid and change origin position by `dorigin`.

    Gradient not supported.

    Note: Z-axis interpolation likely incorrect.

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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Angle rotation to apply in radians.
    grid_polar_new : dict, optional
        Grid definition of the new image.
        If None uses the same grid as input, but with double the angle points.
    z0 : float
        Height of the antenna phase center.

    Returns
    ----------
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
        ntheta3 = 2*ntheta1
    else:
        r3_0, r3_1 = grid_polar_new["r"]
        theta3_0, theta3_1 = grid_polar_new["theta"]
        ntheta3 = grid_polar_new["ntheta"]
        nr3 = grid_polar_new["nr"]
    dtheta3 = (theta3_1 - theta3_0) / ntheta3
    dr3 = (r3_1 - r3_0) / nr3

    return torch.ops.torchbp.polar_interp_lanczos.default( img, dorigin, nbatch,
            rotation, fc, r1_0, dr1, theta1_0, dtheta1, nr1, ntheta1, r3_0, dr3,
            theta3_0, dtheta3, nr3, ntheta3, z0, order)

def polar_to_cart_linear(img: Tensor, dorigin: Tensor, grid_polar: dict,
        grid_cart: dict, fc: float, rotation: float=0, polar_interp: bool=False) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid with linear interpolation.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    dorigin : Tensor
        2D origin of the old image in with respect to new image. Units in meters
        [nbatch, 2] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
        "x": (x0, x1), tuple of min and max x-axis (range),
        "y": (y0, y1), tuple of min and max y-axis (cross-range),
        "nx": number of x-axis pixels.
        "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    polar_interp : bool
        Interpolate in polar coordinates.

    Returns
    ----------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert dorigin.shape == (nbatch, 2)
    else:
        nbatch = 1
        assert dorigin.shape == (2,)

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
            img, dorigin, nbatch, rotation, fc, r0, dr, theta0, dtheta, nr, ntheta,
            x0, y0, dx, dy, nx, ny, polar_interp)

def polar_to_cart_bicubic(img: Tensor, dorigin: Tensor, grid_polar: dict,
        grid_cart: dict, fc: float, rotation: float=0, polar_interp: bool=False) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid with bicubic interpolation.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    dorigin : Tensor
        2D origin of the old image in with respect to new image. Units in meters
        [nbatch, 2] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
        "x": (x0, x1), tuple of min and max x-axis (range),
        "y": (y0, y1), tuple of min and max y-axis (cross-range),
        "nx": number of x-axis pixels.
        "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    polar_interp : bool
        Interpolate in polar coordinates.

    Returns
    ----------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert dorigin.shape == (nbatch, 2)
    else:
        nbatch = 1
        assert dorigin.shape == (2,)

    img_gx, img_gy = torch.gradient(img, dim=(-2,-1), edge_order=2)
    img_gxy = torch.gradient(img_gx, dim=-1)[0]

    return _polar_to_cart_bicubic(img, img_gx, img_gy, img_gxy,
        dorigin, grid_polar, grid_cart, fc, rotation, polar_interp)

def _polar_to_cart_bicubic(img: Tensor, img_gx: Tensor, img_gy: Tensor, img_gxy:
        Tensor, dorigin: Tensor, grid_polar: dict, grid_cart: dict, fc: float,
        rotation: float=0, polar_interp: bool=False) -> Tensor:
    """
    Interpolate polar radar image to cartesian grid.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.
    img_gx : Tensor
        X-axis gradient of img.
    img_gy : Tensor
        Y-axis gradient of img.
    img_gxy : Tensor
        XY-axis gradient of img.
    dorigin : Tensor
        2D origin of the old image in with respect to new image. Units in meters
        [nbatch, 2] if img shape is 3D.
    grid_polar : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    grid_cart : dict
        Grid definition. Dictionary with keys "x", "y", "nx", "ny".
        "x": (x0, x1), tuple of min and max x-axis (range),
        "y": (y0, y1), tuple of min and max y-axis (cross-range),
        "nx": number of x-axis pixels.
        "ny": number of y-axis pixels.
    fc : float
        RF center frequency in Hz.
    rotation : float
        Polar origin rotation angle.
    polar_interp : bool
        Interpolate in polar coordinates.

    Returns
    ----------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
        assert dorigin.shape == (nbatch, 2)
    else:
        nbatch = 1
        assert dorigin.shape == (2,)

    assert img.shape == img_gx.shape
    assert img.shape == img_gy.shape
    assert img.shape == img_gxy.shape

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

    return torch.ops.torchbp.polar_to_cart_bicubic.default(img, img_gx,
            img_gy, img_gxy, dorigin, nbatch, rotation, fc, r0, dr, theta0,
            dtheta, nr, ntheta, x0, y0, dx, dy, nx, ny, polar_interp)

def backprojection_polar_2d(data: Tensor, grid: dict,
        fc: float, r_res: float,
        pos: Tensor, vel: Tensor, att: Tensor,
        d0: float=0.0, ant_tx_dy: float=0.0,
        dealias: bool=False) -> Tensor:
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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    vel : Tensor
        Velocity of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3]. Unused.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only the yaw is used at the moment.
    d0 : float
        Zero range correction.
    ant_tx_dy : float
        RX antenna Y-position (along the track) distance from TX antenna.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.

    Returns
    ----------
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
        assert vel.shape == (nsweeps, 3)
        assert att.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)
        assert vel.shape == (nbatch, nsweeps, 3)
        assert att.shape == (nbatch, nsweeps, 3)

    z0 = 0
    if dealias:
        z0 = torch.mean(pos[:,2])

    return torch.ops.torchbp.backprojection_polar_2d.default(
            data, pos, vel, att,
            nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta,
            d0, ant_tx_dy, dealias, z0)

def backprojection_polar_2d_lanczos(data: Tensor, grid: dict,
        fc: float, r_res: float,
        pos: Tensor, vel: Tensor, att: Tensor,
        d0: float=0.0, ant_tx_dy: float=0.0,
        dealias: bool=False, order: int=4) -> Tensor:
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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    vel : Tensor
        Velocity of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3]. Unused.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only the yaw is used at the moment.
    d0 : float
        Zero range correction.
    ant_tx_dy : float
        RX antenna Y-position (along the track) distance from TX antenna.
    dealias : bool
        If True removes the range spectrum aliasing. Equivalent to applying
        `torchbp.util.bp_polar_range_dealias` on the SAR image.
        Default is False.
    order : int
        Lanczos interpolation order. The default is 4.

    Returns
    ----------
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
        assert vel.shape == (nsweeps, 3)
        assert att.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)
        assert vel.shape == (nbatch, nsweeps, 3)
        assert att.shape == (nbatch, nsweeps, 3)

    z0 = 0
    if dealias:
        z0 = torch.mean(pos[:,2])

    return torch.ops.torchbp.backprojection_polar_2d_lanczos.default(
            data, pos, vel, att,
            nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta,
            d0, ant_tx_dy, dealias, z0, order)

def backprojection_cart_2d(data: Tensor, grid: dict,
        fc: float, r_res: float,
        pos: Tensor, vel: Tensor, att: Tensor,
        d0: float=0.0, ant_tx_dy: float=0.0, beamwidth: float=pi) -> Tensor:
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
        "x": (x0, x1), tuple of min and max range,
        "y": (y0, y1), tuple of min and max along-track coordinates.
        "nx": number of X-axis bins.
        "ny": number of Y-axis bins.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    vel : Tensor
        Velocity of the platform at each data point. Shape should be [nsweeps, 3]. Unused.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3].
        [Roll, pitch, yaw]. Only the yaw is used at the moment.
    beamwidth : float
        Beamwidth of the antenna in radians. Points outside the beam are not calculated.
    d0 : float
        Zero range correction.
    ant_tx_dy : float
        RX antenna Y-position (along the track) distance from TX antenna.

    Returns
    ----------
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
        assert vel.shape == (nsweeps, 3)
        assert att.shape == (nsweeps, 3)
    else:
        nbatch = data.shape[0]
        nsweeps = data.shape[1]
        sweep_samples = data.shape[2]
        assert pos.shape == (nbatch, nsweeps, 3)
        assert vel.shape == (nbatch, nsweeps, 3)
        assert att.shape == (nbatch, nsweeps, 3)

    return torch.ops.torchbp.backprojection_cart_2d.default(
            data, pos, vel, att,
            nbatch, sweep_samples, nsweeps, fc, r_res,
            x0, dx, y0, dy, nx, ny,
            beamwidth, d0, ant_tx_dy)

def gpga_backprojection_2d_core(target_pos: Tensor, data: Tensor, pos: Tensor, vel: Tensor, att: Tensor, fc: float, r_res: float, d0: float=0.0, ant_tx_dy: float=0.0) -> Tensor:
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
    vel : Tensor
        Velocity of the platform at each data point. Shape should be [nsweeps, 3].
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3].
        [Roll, pitch, yaw]. Only the yaw is used at the moment.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    d0 : float
        Zero range correction.
    ant_tx_dy : float
        RX antenna Y-position (along the track) distance from TX antenna.

    Returns
    ----------
    data_out : Tensor
        Values from input data used in backprojection of each target in
        target_pos tensor. Shape is [ntargets, nsweeps].
    """
    nsweeps = data.shape[0]
    ntargets = target_pos.shape[0]
    sweep_samples = data.shape[1]
    assert target_pos.shape == (ntargets, 3)
    assert pos.shape == (nsweeps, 3)
    assert vel.shape == (nsweeps, 3)
    assert att.shape == (nsweeps, 3)

    return torch.ops.torchbp.gpga_backprojection_2d.default(
            target_pos, data, pos, vel, att,
            sweep_samples, nsweeps, fc, r_res,
            ntargets,
            d0, ant_tx_dy)

def cfar_2d(img: Tensor, Navg: tuple, Nguard: tuple, threshold: float, peaks_only: bool=False) -> Tensor:
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
    ----------
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

    return torch.ops.torchbp.cfar_2d.default(img, nbatch, N0, N1, Navg[0], Navg[1],
            Nguard[0], Nguard[1], threshold, peaks_only)

def backprojection_polar_2d_tx_power(wa: Tensor, gtx: Tensor, grx: Tensor,
        g_az0: float, g_el0: float, g_az1: float, g_el1: float, grid: dict,
        r_res: float, pos: Tensor, att: Tensor, sin_look_angle: bool=False) -> Tensor:
    """
    Calculate transmitted power to image plane. Can be used to correct for
    antenna pattern and distance effect on the radar image.

    Parameters
    ----------
    wa : Tensor
        Weighting coefficient for each pulse. Should include window function and
        transmit power variation if known, shape: [nsweeps] or [nbatch, nsweeps].
    gtx : Tensor
        Transmit antenna gain in spherical coordinates, shape: [elevation, azimuth].
        Should have same dimensions as grx.
        (0, 0) angle is at the beam center.
    grx : Tensor
        Receive antenna gain in spherical coordinates, shape: [elevation, azimuth].
        Should have same dimensions as gtx.
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
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only roll and yaw are used at the moment.
    sin_look_angle : bool
        Multiply pixel value by sin of look angle.

    Returns
    ----------
    img : Tensor
        Pseudo-polar format radar image.
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

    g_nel = gtx.shape[0]
    g_naz = gtx.shape[1]
    assert gtx.shape == grx.shape == torch.Size([g_nel, g_naz])
    g_daz = (g_az1 - g_az0) / g_naz
    g_del = (g_el1 - g_el0) / g_nel

    return torch.ops.torchbp.backprojection_polar_2d_tx_power.default(
            wa, pos, att, gtx, grx, nbatch,
            g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            nsweeps, r_res,
            r0, dr, theta0, dtheta, nr, ntheta, sin_look_angle)

def ffbp(data: Tensor, grid: dict, fc: float, r_res: float, pos: Tensor, vel:
        Tensor, att: Tensor, stages: int, divisions: int=2, d0: float=0.0,
        ant_tx_dy: float=0.0, interp_method: str | tuple=("lanczos", 3)) -> Tensor:
    """
    Fast factorized backprojection.

    Large Z-axis movements likely not handled correctly.

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    grid : dict
        Grid definition. Dictionary with keys "r", "theta", "nr", "ntheta".
        "r": (r0, r1), tuple of min and max range,
        "theta": (theta0, theta1), sin of min and max angle. (-1, 1) for 180 degree view.
        "nr": nr, number of range bins.
        "ntheta": number of angle bins.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
    vel : Tensor
        Velocity of the platform at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3]. Unused.
    att : Tensor
        Euler angles of the radar antenna at each data point. Shape should be [nsweeps, 3] or [nbatch, nsweeps, 3].
        [Roll, pitch, yaw]. Only the yaw is used at the moment.
    stages : int
        Number of recursions.
    divisions : int
        Number of subapertures divisions per stage. Default is 2.
    d0 : float
        Zero range correction.
    ant_tx_dy : float
        RX antenna Y-position (along the track) distance from TX antenna.
    interp_method : str or tuple
        Interpolation method. See `polar_interp` function.
    """
    nsweeps = data.shape[0]
    device = data.device

    imgs = []
    n = nsweeps // divisions
    for d in range(divisions):
        origin_local = torch.tensor([torch.mean(pos[d*n:(d+1)*n,0]),
            torch.mean(pos[d*n:(d+1)*n,1]), 0], device=device,
            dtype=torch.float32)[None,:]
        pos_local = pos[d*n:(d+1)*n] - origin_local
        z0 = torch.mean(pos_local[:,2])
        vel_local = vel[d*n:(d+1)*n]
        att_local = att[d*n:(d+1)*n]
        grid_local = deepcopy(grid)
        grid_local["ntheta"] = (grid["ntheta"] + divisions - 1) // divisions
        data_local = data[d*n:(d+1)*n]
        if stages > 1 and len(data_local) > 4*divisions:
            img = ffbp(data_local, grid_local, fc, r_res, pos_local, vel_local, att_local,
                    stages=stages-1, divisions=divisions, d0=d0, ant_tx_dy=ant_tx_dy)
        else:
            img = backprojection_polar_2d(data_local, grid_local, fc, r_res,
                    pos_local, vel_local, att_local, d0=d0,
                    ant_tx_dy=ant_tx_dy)[0]
        imgs.append((origin_local[0], grid_local, img, z0))
    while len(imgs) > 1:
        img1 = imgs[0]
        img2 = imgs[1]
        new_origin = 0.5 * img1[0] + 0.5 * img2[0]
        if len(imgs) == 2:
            # Interpolate the final image to the desired grid.
            grid_polar_new = grid
        else:
            grid_polar_new = deepcopy(img1[1])
            grid_polar_new["ntheta"] += img2[1]["ntheta"]
        i1 = img1[2]
        i2 = img2[2]
        img_interpolated1 = polar_interp(i1, new_origin - img1[0], img1[1], fc,
                0, grid_polar_new, z0=img1[3], method=interp_method).squeeze()
        img_interpolated2 = polar_interp(i2, new_origin - img2[0], img2[1], fc,
                0, grid_polar_new, z0=img2[3], method=interp_method).squeeze()
        img_sum = img_interpolated1 + img_interpolated2
        merged = (new_origin, grid_polar_new, img_sum, img1[3])
        imgs = imgs[2:] + [merged]
    return imgs[0][2]

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torchbp::polar_interp_linear")
def _fake_polar_interp_linear(img: Tensor, dorigin: Tensor, rotation: float,
        fc: float, r0: float, dr0: float, theta0: float, dtheta0: float,
        Nr0: float, Ntheta0: float, r1: float, dr1: float, theta1: float,
        dtheta1: float, Nr1: float, Ntheta1: float) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((Nr1, Ntheta1), dtype=torch.complex64, device=img.device)

@torch.library.register_fake("torchbp::polar_interp_linear_grad")
def _fake_polar_interp_linear_grad(grad: Tensor, img: Tensor, dorigin: Tensor, rotation: float,
        fc: float, r0: float, dr0: float, theta0: float, dtheta0: float,
        Nr0: float, Ntheta0: float, r1: float, dr1: float, theta1: float,
        dtheta1: float, Nr1: float, Ntheta1: float) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
    torch._check(img.dtype == torch.complex64)
    ret = []
    if img.requires_grad:
        ret.append(torch.empty((Nr1, Ntheta1), dtype=torch.complex64, device=img.device))
    else:
        ret.append(None)
    if dorigin.requires_grad:
        ret.append(torch.empty((2,), dtype=torch.float, device=img.device))
    else:
        ret.append(None)
    return ret

@torch.library.register_fake("torchbp::polar_to_cart_linear")
def _fake_polar_to_cart_linear(img: Tensor, dorigin: Tensor, nbatch: int, rotation: float,
        fc: float, r0: float, dr: float, theta0: float, dtheta: float, nr: int, ntheta: int,
        x0: float, y0: float, dx: float, dy: float, nx: int, ny: int) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((Nx, Ny), dtype=torch.complex64, device=img.device)

@torch.library.register_fake("torchbp::polar_to_cart_linear_grad")
def _fake_polar_interp_linear_grad(grad: Tensor, img: Tensor, dorigin: Tensor, rotation: float,
        fc: float, r0: float, dr: float, theta0: float, dtheta: float,
        Nr: float, Ntheta: float, x0: float, dx: float, y0: float,
        dy: float, Nx: float, Ny: float) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
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

@torch.library.register_fake("torchbp::polar_to_cart_bicubic")
def _fake_polar_to_cart_bicubic(img: Tensor, img_gx: Tensor, img_gy: Tensor,
        img_gxy: Tensor, dorigin: Tensor, nbatch: int, rotation: float, fc:
        float, r0: float, dr: float, theta0: float, dtheta: float, nr: int,
        ntheta: int, x0: float, y0: float, dx: float, dy: float, nx: int, ny:
        int) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
    torch._check(img.dtype == torch.complex64)
    return torch.empty((Nx, Ny), dtype=torch.complex64, device=img.device)

@torch.library.register_fake("torchbp::polar_to_cart_bicubic_grad")
def _fake_polar_interp_bicubic_grad(grad: Tensor, img: Tensor, img_gx: Tensor,
        img_gy: Tensor, img_gxy: Tensor, dorigin: Tensor, rotation: float, fc:
        float, r0: float, dr: float, theta0: float, dtheta: float, Nr: float,
        Ntheta: float, x0: float, dx: float, y0: float, dy: float, Nx: float,
        Ny: float) -> Tensor:
    torch._check(dorigin.dtype == torch.float)
    torch._check(img.dtype == torch.complex64)
    torch._check(img_gx.dtype == torch.complex64)
    torch._check(img_gy.dtype == torch.complex64)
    torch._check(img_gxy.dtype == torch.complex64)
    ret = []
    if img.requires_grad:
        ret.append(torch.empty_like(img))
        ret.append(torch.empty_like(img_gx))
        ret.append(torch.empty_like(img_gy))
        ret.append(torch.empty_like(img_gxy))
    else:
        ret.append(None)
        ret.append(None)
        ret.append(None)
        ret.append(None)
    if dorigin.requires_grad:
        ret.append(torch.empty_like(dorigin))
    else:
        ret.append(None)
    return ret

@torch.library.register_fake("torchbp::backprojection_polar_2d")
def _fake_cart_2d(data: Tensor, pos: Tensor, vel: Tensor, att: Tensor,
        nbatch: int, sweep_samples: int, nsweeps: int, fc: float, r_res: float,
        r0: float, dr: float, theta0: float, dtheta: float, Nr: int, Ntheta: int,
        d0: float, ant_tx_dy: float):
    torch._check(pos.dtype == torch.float)
    torch._check(vel.dtype == torch.float)
    torch._check(att.dtype == torch.float)
    torch._check(data.dtype == torch.complex64 or data.dtype == torch.complex32)
    return torch.empty((nbatch, Nr, Ntheta), dtype=torch.complex64, device=data.device)

@torch.library.register_fake("torchbp::backprojection_polar_2d_grad")
def _fake_cart_2d_grad(grad: Tensor, data: Tensor, pos: Tensor, vel: Tensor, att: Tensor,
        nbatch: int, sweep_samples: int, nsweeps: int, fc: float, r_res: float,
        r0: float, dr: float, theta0: float, dtheta: float, Nr: int, Ntheta: int,
        d0: float, ant_tx_dy: float):
    torch._check(pos.dtype == torch.float)
    torch._check(vel.dtype == torch.float)
    torch._check(att.dtype == torch.float)
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
def _fake_cart_2d(data: Tensor, pos: Tensor, vel: Tensor, att: Tensor,
        sweep_samples: int, nsweeps: int, fc: float, r_res: float,
        x0: float, dx: float, y0: float, dy: float, Nx: int, Ny: int,
        beamwidth: float, d0: float, ant_tx_dy: float):
    torch._check(pos.dtype == torch.float)
    torch._check(vel.dtype == torch.float)
    torch._check(att.dtype == torch.float)
    torch._check(data.dtype == torch.complex64)
    return torch.empty((Nx, Ny), dtype=torch.complex64, device=data.device)

@torch.library.register_fake("torchbp::backprojection_cart_2d_grad")
def _fake_cart_2d_grad(grad: Tensor, data: Tensor, pos: Tensor, vel: Tensor, att: Tensor,
        sweep_samples: int, nsweeps: int, fc: float, r_res: float,
        x0: float, dx: float, y0: float, dy: float, Nx: int, Ny: int,
        beamwidth: float, d0: float, ant_tx_dy: float):
    torch._check(pos.dtype == torch.float)
    torch._check(vel.dtype == torch.float)
    torch._check(att.dtype == torch.float)
    torch._check(data.dtype == torch.complex64)
    torch._check(grad.dtype == torch.complex64)
    return torch.empty_like(pos)

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
    ret = torch.ops.torchbp.backprojection_polar_2d_grad.default(
            grad,
            data, pos, *ctx.saved)
    grads = [None] * polar_2d_nargs
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)

def _backward_cart_2d(ctx, grad):
    data = ctx.saved_tensors[0]
    pos  = ctx.saved_tensors[1]
    ret = torch.ops.torchbp.backprojection_cart_2d_grad.default(
            grad, data, pos, *ctx.saved)
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
            grad, img, dorigin, *ctx.saved)
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
    dorigin = ctx.saved_tensors[1]
    if ctx.saved[-1]:
        raise NotImplementedError("polar_interp gradient not supported")
    ret = torch.ops.torchbp.polar_to_cart_linear_grad.default(
            grad, img, dorigin, *ctx.saved[:-1])
    grads = [None] * polar_to_cart_linear_args
    grads[0] = ret[0]
    grads[1] = ret[1]
    return tuple(grads)

def _setup_context_polar_to_cart_linear(ctx, inputs, output):
    img, dorigin, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i <= 1:
                continue
            raise NotImplementedError("Only img and dorigin gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(img, dorigin)

def _backward_polar_to_cart_bicubic(ctx, grad):
    img, img_gx, img_gy, img_gxy, dorigin = ctx.saved_tensors
    if ctx.saved[-1]:
        raise NotImplementedError("polar_interp gradient not supported")
    ret = torch.ops.torchbp.polar_to_cart_bicubic_grad.default(
            grad, img, img_gx, img_gy, img_gxy, dorigin, *ctx.saved[:-1])
    grads = [None] * polar_to_cart_bicubic_args
    grads[:len(ret)] = ret
    return tuple(grads)

def _setup_context_polar_to_cart_bicubic(ctx, inputs, output):
    img, img_gx, img_gy, img_gxy, dorigin, *rest = inputs
    for i in range(len(ctx.needs_input_grad)):
        if ctx.needs_input_grad[i]:
            if i == 4:
                raise NotImplementedError("dorigin grad not supported")
            if i <= 4:
                continue
            raise NotImplementedError("Only img, img_gx, img_gy, img_gxy and dorigin gradient supported")
    ctx.saved = rest
    ctx.save_for_backward(img, img_gx, img_gy, img_gxy, dorigin)

def _backward_entropy(ctx, grad):
    data, norm = ctx.saved_tensors
    ret = torch.ops.torchbp.entropy_grad.default(
            data, norm, grad, *ctx.saved)
    grads = [None] * entropy_args
    grads[:len(ret)] = ret
    return tuple(grads)

def _setup_context_entropy(ctx, inputs, output):
    data, norm, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data, norm)

def _backward_abs_sum(ctx, grad):
    data = ctx.saved_tensors[0]
    ret = torch.ops.torchbp.abs_sum_grad.default(
            data, grad, *ctx.saved)
    grads = [None] * abs_sum_args
    grads[0] = ret
    return tuple(grads)

def _setup_context_abs_sum(ctx, inputs, output):
    data, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data)

torch.library.register_autograd(
    "torchbp::backprojection_polar_2d", _backward_polar_2d, setup_context=_setup_context_polar_2d)
torch.library.register_autograd(
    "torchbp::backprojection_cart_2d", _backward_cart_2d, setup_context=_setup_context_cart_2d)
torch.library.register_autograd(
    "torchbp::polar_interp_linear", _backward_polar_interp_linear, setup_context=_setup_context_polar_interp_linear)
torch.library.register_autograd(
    "torchbp::polar_to_cart_linear", _backward_polar_to_cart_linear, setup_context=_setup_context_polar_to_cart_linear)
torch.library.register_autograd(
    "torchbp::polar_to_cart_bicubic", _backward_polar_to_cart_bicubic, setup_context=_setup_context_polar_to_cart_bicubic)
torch.library.register_autograd(
    "torchbp::entropy", _backward_entropy, setup_context=_setup_context_entropy)
torch.library.register_autograd(
    "torchbp::abs_sum", _backward_abs_sum, setup_context=_setup_context_abs_sum)
