import torch
from torch import Tensor
from typing import Union, TYPE_CHECKING
from ._utils import unpack_polar_grid, unpack_cartesian_grid, get_batch_dims, AntennaPattern

if TYPE_CHECKING:
    from ..grid import PolarGrid, CartesianGrid

cart_2d_nargs = 16
polar_2d_nargs = 26

def _prepare_backprojection_polar_2d_args(
    data: Tensor,
    grid: "PolarGrid | dict",
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
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d and for testing.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

    z0 = 0
    if dealias:
        if nbatch != 1:
            raise ValueError("Only nbatch=1 supported with dealias")
        z0 = torch.mean(pos[..., 2])

    return (data, pos, att, nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta, d0, dealias, z0,
            *antenna.to_cpp_args(),
            data_fmod, alias_fmod)


def _prepare_backprojection_polar_2d_lanczos_args(
    data: Tensor,
    grid: "PolarGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    dealias: bool = False,
    order: int = 6,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    alias_fmod: float = 0
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d_lanczos operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d_lanczos and for testing.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

    z0 = 0
    if dealias:
        if nbatch != 1:
            raise ValueError("Only nbatch=1 supported with dealias")
        z0 = torch.mean(pos[..., 2])

    return (data, pos, att, nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta, d0, dealias, z0, order,
            *antenna.to_cpp_args(),
            data_fmod, alias_fmod)


def _prepare_backprojection_polar_2d_knab_args(
    data: Tensor,
    grid: "PolarGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    dealias: bool = False,
    order: int = 6,
    oversample: float = 1.5,
    att: Tensor | None = None,
    g: Tensor | None = None,
    g_extent: list | None = None,
    data_fmod: float = 0,
    alias_fmod: float = 0
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d_knab operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d_knab and for testing.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

    z0 = 0
    if dealias:
        if nbatch != 1:
            raise ValueError("Only nbatch=1 supported with dealias")
        z0 = torch.mean(pos[..., 2])

    return (data, pos, att, nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta, d0, dealias, z0, order, oversample,
            *antenna.to_cpp_args(),
            data_fmod, alias_fmod)


def backprojection_polar_2d(
    data: Tensor,
    grid: "PolarGrid | dict",
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
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
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
    cpp_args = _prepare_backprojection_polar_2d_args(
        data, grid, fc, r_res, pos, d0, dealias, att, g, g_extent,
        data_fmod, alias_fmod
    )
    return torch.ops.torchbp.backprojection_polar_2d.default(*cpp_args)


def backprojection_polar_2d_lanczos(
    data: Tensor,
    grid: "PolarGrid | dict",
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
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
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

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

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
        *antenna.to_cpp_args(),
        data_fmod,
        alias_fmod
    )


def backprojection_polar_2d_knab(
    data: Tensor,
    grid: "PolarGrid | dict",
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
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
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

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

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
        *antenna.to_cpp_args(),
        data_fmod,
        alias_fmod
    )


def _prepare_backprojection_cart_2d_args(
    data: Tensor,
    grid: "CartesianGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    beamwidth: float = torch.pi,
    data_fmod: float = 0
) -> tuple:
    """Prepare arguments for C++ backprojection_cart_2d operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_cart_2d and for testing.
    """
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)

    return (data, pos, nbatch, sweep_samples, nsweeps, fc, r_res,
            x0, dx, y0, dy, nx, ny, beamwidth, d0, data_fmod)


def backprojection_cart_2d(
    data: Tensor,
    grid: "CartesianGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    d0: float = 0.0,
    beamwidth: float = torch.pi,
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
    grid : CartesianGrid or dict
        Cartesian grid definition. Can be:

        - CartesianGrid object: ``CartesianGrid(x_range=(-50, 50), y_range=(-50, 50), nx=200, ny=200)``
        - dict: ``{"x": (x0, x1), "y": (y0, y1), "nx": nx, "ny": ny}``
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
    cpp_args = _prepare_backprojection_cart_2d_args(
        data, grid, fc, r_res, pos, d0, beamwidth, data_fmod
    )
    return torch.ops.torchbp.backprojection_cart_2d.default(*cpp_args)


def _prepare_projection_cart_2d_args(
    img: Tensor,
    pos: Tensor,
    grid: "CartesianGrid | dict",
    fc: float,
    fs: float,
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
) -> tuple:
    """Prepare arguments for C++ projection_cart_2d operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by projection_cart_2d and for testing.
    """
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)

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

    antenna = AntennaPattern(g, g_extent)

    if vel is not None:
        if vel.shape != pos.shape:
            raise ValueError(f"vel shape {vel.shape} doesn't match with pos shape {pos.shape}")
    else:
        vel = torch.zeros_like(pos)

    if dem is None:
        dem = torch.zeros((nx, ny), device=img.device, dtype=torch.float32)

    return (img, dem, pos, vel, att, nbatch, sweep_samples, nsweeps,
            fc, fs, gamma, x0, dx, y0, dy, nx, ny, d0,
            *antenna.to_cpp_args(),
            use_rvp, norm)


def projection_cart_2d(
    img: Tensor,
    pos: Tensor,
    grid: "CartesianGrid | dict",
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
    grid : CartesianGrid or dict
        Cartesian grid definition. Can be:

        - CartesianGrid object: ``CartesianGrid(x_range=(-50, 50), y_range=(-50, 50), nx=200, ny=200)``
        - dict: ``{"x": (x0, x1), "y": (y0, y1), "nx": nx, "ny": ny}``
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
    cpp_args = _prepare_projection_cart_2d_args(
        img, pos, grid, fc, fs, gamma, sweep_samples, d0,
        dem, att, g, g_extent, use_rvp, normalization, vel
    )
    return torch.ops.torchbp.projection_cart_2d.default(*cpp_args)


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


def _prepare_backprojection_polar_2d_tx_power_args(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    normalization: str | None = None,
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d_tx_power operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d_tx_power and for testing.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)

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

    return (wa, pos, att, g, nbatch, g_az0, g_el0, g_daz, g_del,
            g_naz, g_nel, nsweeps, r_res, r0, dr, theta0, dtheta,
            nr, ntheta, norm)


def backprojection_polar_2d_tx_power(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
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
    grid : PolarGrid or dict
        Polar grid definition. Can be:

        - PolarGrid object: ``PolarGrid(r_range=(50, 100), theta_range=(-1, 1), nr=200, ntheta=400)``
        - dict: ``{"r": (r0, r1), "theta": (theta0, theta1), "nr": nr, "ntheta": ntheta}``

        where ``theta`` represents sin of angle (-1, 1 for 180 degree view).
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
    cpp_args = _prepare_backprojection_polar_2d_tx_power_args(
        wa, g, g_extent, grid, r_res, pos, att, normalization
    )
    return torch.ops.torchbp.backprojection_polar_2d_tx_power.default(*cpp_args)


@torch.library.register_fake("torchbp::backprojection_polar_2d")
def _fake_polar_2d(
    data: Tensor,
    pos: Tensor,
    att: Tensor,
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
    dealias: bool,
    z0: float,
    g: Tensor,
    g_az0: float,
    g_el0: float,
    g_daz: float,
    g_del: float,
    g_naz: int,
    g_nel: int,
    data_fmod: float,
    alias_fmod: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64 or data.dtype == torch.complex32)
    return torch.empty((nbatch, Nr, Ntheta), dtype=torch.complex64, device=data.device)


@torch.library.register_fake("torchbp::backprojection_polar_2d_grad")
def _fake_polar_2d_grad(
    grad: Tensor,
    data: Tensor,
    pos: Tensor,
    att: Tensor,
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
    dealias: bool,
    z0: float,
    g: Tensor,
    g_az0: float,
    g_el0: float,
    g_daz: float,
    g_del: float,
    g_naz: int,
    g_nel: int,
    data_fmod: float,
    alias_fmod: float,
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
    nbatch: int,
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
    data_fmod: float,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64)
    return torch.empty((nbatch, Nx, Ny), dtype=torch.complex64, device=data.device)


@torch.library.register_fake("torchbp::backprojection_cart_2d_grad")
def _fake_cart_2d_grad(
    grad: Tensor,
    data: Tensor,
    pos: Tensor,
    nbatch: int,
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
    data_fmod: float,
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
