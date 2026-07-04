import torch
from torch import Tensor
from typing import Union, TYPE_CHECKING
from ._utils import unpack_polar_grid, unpack_cartesian_grid, get_batch_dims, AntennaPattern

if TYPE_CHECKING:
    from ..grid import PolarGrid, CartesianGrid

cart_2d_nargs = 16
polar_2d_nargs = 27

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
    alias_fmod: float = 0,
    normalize: bool = True
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d and for testing.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nbatch, nsweeps, sweep_samples = get_batch_dims(data, pos)
    antenna = AntennaPattern(g, g_extent)

    # Validate and normalize att shape when antenna pattern is used
    if g is not None:
        if att is None:
            raise ValueError("att must be provided when antenna pattern g is specified")
        # Accept [nsweeps, 3] for nbatch=1
        expected_shape = (nbatch, nsweeps, 3)
        if nbatch == 1 and att.ndim == 2 and att.shape == (nsweeps, 3):
            expected_shape = (nsweeps, 3)
        if att.shape != expected_shape:
            raise ValueError(f"att must have shape {expected_shape} when g is provided, got {att.shape}")

    z0 = 0
    if dealias:
        if nbatch != 1:
            raise ValueError("Only nbatch=1 supported with dealias")
        z0 = torch.mean(pos[..., 2])

    return (data, pos, att, nbatch, sweep_samples, nsweeps, fc, r_res,
            r0, dr, theta0, dtheta, nr, ntheta, d0, dealias, z0,
            *antenna.to_cpp_args(),
            data_fmod, alias_fmod, normalize)


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
    alias_fmod: float = 0,
    normalize: bool = True
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d_lanczos operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d_lanczos and for testing.
    """
    # Reuse linear preparation and insert order parameter
    base_args = _prepare_backprojection_polar_2d_args(
        data, grid, fc, r_res, pos, d0, dealias, att, g, g_extent, data_fmod, alias_fmod, normalize
    )
    # Insert order after z0 (index 17)
    return base_args[:17] + (order,) + base_args[17:]


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
    alias_fmod: float = 0,
    normalize: bool = True
) -> tuple:
    """Prepare arguments for C++ backprojection_polar_2d_knab operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by backprojection_polar_2d_knab and for testing.
    """
    # Reuse linear preparation and insert order and oversample parameters
    base_args = _prepare_backprojection_polar_2d_args(
        data, grid, fc, r_res, pos, d0, dealias, att, g, g_extent, data_fmod, alias_fmod, normalize
    )
    # Insert order and oversample after z0 (index 17)
    return base_args[:17] + (order, oversample) + base_args[17:]


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
    alias_fmod: float = 0,
    normalize: bool = True
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
    normalize : bool
        If True (default), apply W1/W2 normalization when antenna pattern is used.
        The antenna pattern should be nonzero everywhere to avoid noise
        amplification at pixels illuminated only near pattern nulls.
        Set to False for FFBP to output unnormalized accumulation.

    Returns
    -------
    img : Tensor
        Pseudo-polar format radar image.
    """
    cpp_args = _prepare_backprojection_polar_2d_args(
        data, grid, fc, r_res, pos, d0, dealias, att, g, g_extent,
        data_fmod, alias_fmod, normalize
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
    cpp_args = _prepare_backprojection_polar_2d_lanczos_args(
        data, grid, fc, r_res, pos, d0, dealias, order, att, g, g_extent,
        data_fmod, alias_fmod
    )
    return torch.ops.torchbp.backprojection_polar_2d_lanczos.default(*cpp_args)


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
    g : Tensor or None
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
    cpp_args = _prepare_backprojection_polar_2d_knab_args(
        data, grid, fc, r_res, pos, d0, dealias, order, oversample, att, g, g_extent,
        data_fmod, alias_fmod
    )
    return torch.ops.torchbp.backprojection_polar_2d_knab.default(*cpp_args)


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
        (0, 0) angle is at the beam center.
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


def _prepare_projection_cart_2d_nufft_args(
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
) -> tuple:
    """Prepare arguments for C++ projection_cart_2d_nufft operator."""
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

    if dem is None:
        dem = torch.zeros((nx, ny), device=img.device, dtype=torch.float32)

    return (img, dem, pos, att, nbatch, sweep_samples, nsweeps,
            fc, fs, gamma, x0, dx, y0, dy, nx, ny, d0,
            *antenna.to_cpp_args(),
            use_rvp, norm)


def projection_cart_2d_nufft(
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
) -> Tensor:
    """
    NUFFT-based forward projection. Equivalent to :func:`projection_cart_2d`
    without velocity (vel=None) but uses a Type-1 NUFFT for O(N log M) cost
    instead of O(N·M).

    Parameters match :func:`projection_cart_2d` except ``vel`` is not accepted.
    """
    cpp_args = _prepare_projection_cart_2d_nufft_args(
        img, pos, grid, fc, fs, gamma, sweep_samples, d0,
        dem, att, g, g_extent, use_rvp, normalization
    )
    return torch.ops.torchbp.projection_cart_2d_nufft.default(*cpp_args)


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


def blocksvd_alpha(
    img: Tensor,
    data: Tensor,
    pos: Tensor,
    blocks: Tensor,
    fc: float,
    r_res: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    d0: float = 0.0,
    data_fmod: float = 0.0,
) -> Tensor:
    """
    Per-block inner product of a master image against per-sweep slave
    backprojection footprints, used by
    :func:`torchbp.autofocus.insar_rme_blocksvd`.

    For each block ``b`` and sweep ``m``::

        alpha[b, m] = Sum_pix conj(img[pix]) * data[m, r_idx(pix, m)]
                      * exp(j k R(pix, m))

    where the sum runs over the block's pixel rectangle on the polar
    grid. Pixels are at ``r = r0 + dr * i``, ``theta = theta0 + dtheta * j``
    on the z=0 plane, matching :func:`backprojection_polar_2d`. Linear
    range interpolation; samples outside the data range window
    contribute zero. Equivalent to ``conj(img_patch) @ B`` with ``B``
    from :func:`gpga_backprojection_2d_core` over the block's pixels,
    without materializing ``B``.

    Parameters
    ----------
    img : Tensor [nr, ntheta]
        Complex master image on the polar grid, with any per-pixel
        weighting (e.g. coherence) already applied.
    data : Tensor [nsweeps, nsamples]
        Range-compressed slave data.
    pos : Tensor [nsweeps, 3]
        Slave platform positions.
    blocks : Tensor [nblocks, 6]
        Integer block definitions ``(r_idx0, r_idx1, theta_idx0,
        theta_idx1, sweep_lo, sweep_hi)``. Pixel rectangles are
        half-open index ranges into ``img``; ``alpha[b, m]`` is zero for
        sweeps outside ``[sweep_lo, sweep_hi)``.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
    r0, dr, theta0, dtheta : float
        Polar grid definition.
    d0 : float
        Zero range correction.
    data_fmod : float
        Range modulation frequency applied to input data.

    Returns
    -------
    alpha : Tensor [nblocks, nsweeps]
        Complex per-block per-sweep inner products.
    """
    nsweeps = data.shape[0]
    sweep_samples = data.shape[1]
    nblocks = blocks.shape[0]
    ntheta = img.shape[1]
    assert img.dim() == 2
    assert pos.shape == (nsweeps, 3)
    assert blocks.shape == (nblocks, 6)
    blocks = blocks.to(torch.int32).contiguous()
    return torch.ops.torchbp.blocksvd_alpha.default(
        img, data, pos, blocks, sweep_samples, nsweeps, nblocks, ntheta,
        fc, r_res, r0, dr, theta0, dtheta, d0, data_fmod
    )


def _prepare_backprojection_polar_2d_tx_power_args(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    normalization: str | None = None,
    azimuth_resolution: bool = True,
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
            nr, ntheta, norm, int(azimuth_resolution))


def backprojection_polar_2d_tx_power(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    normalization: str | None = None,
    azimuth_resolution: bool = True,
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
        Range bin resolution in data (meters). Currently unused by the
        computation and kept for API compatibility; the nadir resolution
        floor uses the image grid range spacing instead.
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
    azimuth_resolution : bool
        If True (default), also normalize for the varying azimuth resolution.
        The gain-weighted angular spread of the line of sight is
        measured per pixel and folded into the returned power so that dividing a
        SAR image by `tx_power` removes the residual azimuth brightness slope
        (caused by the resolution cell growing toward the swath edges) in
        addition to the antenna pattern and range falloff.
        See :func:`backprojection_polar_2d_resolution` for the standalone map.
        Pixels with fewer than two contributing sweeps have no measurable
        azimuth aperture and are set to inf.
        Set to False to get the pure antenna/range illumination.

    Returns
    -------
    tx_power : Tensor
        Pseudo-polar format image of square root of power returned from each
        pixel assuming constant reflectivity.
    """
    cpp_args = _prepare_backprojection_polar_2d_tx_power_args(
        wa, g, g_extent, grid, r_res, pos, att, normalization, azimuth_resolution
    )
    return torch.ops.torchbp.backprojection_polar_2d_tx_power.default(*cpp_args)


def backprojection_polar_2d_tx_power_slant(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    altitude: float,
    normalization: str | None = None,
    azimuth_resolution: bool = True,
) -> Tensor:
    """
    Slant-range variant of :func:`backprojection_polar_2d_tx_power`.

    For slant-range BP images (BP origin at sensor altitude, pos z ≈ 0),
    the standard tx_power kernel gets wrong elevation angles because
    pos_z ≈ 0.  This variant maps each polar pixel (r, θ) to its ground
    position ``(sqrt(r²cos²θ − H²), r·sinθ)`` and uses the supplied
    altitude *H* for all elevation / distance / normalization calculations.

    Parameters
    ----------
    wa, g, g_extent, grid, r_res, pos, att, normalization, azimuth_resolution
        Same as :func:`backprojection_polar_2d_tx_power`.
    altitude : float
        Sensor altitude above ground (metres).  Must be > 0.

    Returns
    -------
    tx_power : Tensor
        Same as :func:`backprojection_polar_2d_tx_power`.
    """
    cpp_args = _prepare_backprojection_polar_2d_tx_power_args(
        wa, g, g_extent, grid, r_res, pos, att, normalization, azimuth_resolution
    )
    return torch.ops.torchbp.backprojection_polar_2d_tx_power_slant.default(
        *cpp_args, altitude
    )


def backprojection_polar_2d_resolution(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "PolarGrid | dict",
    fc: float,
    pos: Tensor,
    att: Tensor,
    altitude: float = 0.0,
    sweeps_chunk: int = 64,
) -> Tensor:
    """
    Estimate the angular (azimuth) resolution of every polar pixel from the
    trajectory and antenna pattern.

    Parameters
    ----------
    wa : Tensor
        Per-sweep amplitude weighting (window and/or transmit power), the same
        ``wa`` passed to :func:`backprojection_polar_2d_tx_power`,
        shape: [nsweeps] or [nbatch, nsweeps].
    g : Tensor
        Square-root of two-way antenna gain, shape: [elevation, azimuth].
        See :func:`backprojection_polar_2d_tx_power`.
    g_extent : list
        ``[el0, az0, el1, az1]`` angular extent of ``g`` in radians.
    grid : PolarGrid or dict
        Polar grid definition, ``theta`` is the sine of the azimuth angle.
    fc : float
        Radar center frequency (Hz). Used for the wavelength ``c/fc``.
    pos : Tensor
        Platform position per sweep, shape: [nsweeps, 3] or [nbatch, nsweeps, 3].
    att : Tensor
        Antenna Euler angles [roll, pitch, yaw] per sweep, same shape as ``pos``.
        Roll and yaw are used (pitch is ignored), matching
        :func:`backprojection_polar_2d_tx_power`.
    altitude : float
        If > 0 the grid is treated as slant-range (BP origin at sensor altitude,
        ``pos`` z ~ 0) and pixels are mapped to the ground using this altitude,
        matching :func:`backprojection_polar_2d_tx_power_slant`. If 0 (default)
        the grid is ground-range and the per-sweep ``pos`` z is used.
    sweeps_chunk : int
        Number of sweeps processed at once. Trades memory for speed, does not
        affect the result.

    Returns
    -------
    resolution : Tensor
        Pseudo-polar map of the estimated angular azimuth resolution (radians),
        shape [nr, ntheta] or [nbatch, nr, ntheta]. Pixels never illuminated by
        any sweep are ``inf``.
    """
    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)

    batched = wa.dim() == 2
    if not batched:
        wa = wa[None]
        pos = pos[None]
        att = att[None]
    nbatch, nsweeps = wa.shape
    assert pos.shape == (nbatch, nsweeps, 3)
    assert att.shape == (nbatch, nsweeps, 3)

    device = g.device
    dtype = g.dtype
    wl = 299792458.0 / fc

    g_el0, g_az0, g_el1, g_az1 = g_extent
    g_nel, g_naz = g.shape
    g_del = (g_el1 - g_el0) / g_nel
    g_daz = (g_az1 - g_az0) / g_naz

    r = (r0 + dr * torch.arange(nr, device=device, dtype=dtype))[:, None]
    theta = (theta0 + dtheta * torch.arange(ntheta, device=device, dtype=dtype))[None, :]
    cos2 = 1.0 - theta * theta

    # Pixel ground position, mirroring the tx_power kernel.
    if altitude > 0.0:
        r2cos2 = r * r * cos2
        H2 = altitude * altitude
        valid_pixel = r2cos2 >= H2
        px_base = torch.sqrt(torch.clamp(r2cos2 - H2, min=0.0))
        py_base = r * theta
        h_all = torch.full((nbatch, 1, 1), float(altitude), device=device, dtype=dtype)
    else:
        valid_pixel = torch.ones((nr, ntheta), dtype=torch.bool, device=device)
        px_base = r * torch.sqrt(torch.clamp(cos2, min=0.0))
        py_base = r * theta
        h_all = pos[:, :, 2]  # per sweep, filled in the loop

    # Weighted moments of the line-of-sight azimuth angle.
    W = torch.zeros((nbatch, nr, ntheta), device=device, dtype=dtype)
    M1 = torch.zeros_like(W)
    M2 = torch.zeros_like(W)

    for s in range(0, nsweeps, sweeps_chunk):
        e = min(s + sweeps_chunk, nsweeps)
        # [nbatch, nchunk, 1, 1] broadcast against [nr, ntheta].
        pos_x = pos[:, s:e, 0, None, None]
        pos_y = pos[:, s:e, 1, None, None]
        roll = att[:, s:e, 0, None, None]
        yaw = att[:, s:e, 2, None, None]
        wa_c = wa[:, s:e, None, None]

        px = px_base - pos_x
        py = py_base - pos_y
        if altitude > 0.0:
            h = h_all[:, :, :, None]  # broadcast scalar altitude
        else:
            h = pos[:, s:e, 2, None, None]
        d = torch.sqrt(px * px + py * py + h * h)

        look = torch.asin(torch.clamp(-h / d, min=-1.0, max=1.0))
        el = look - roll
        psi = torch.atan2(py, px)  # ground-frame LOS azimuth drives cross-range
        az = psi - yaw             # antenna-frame azimuth selects the gain

        el_idx = (el - g_el0) / g_del
        az_idx = (az - g_az0) / g_daz
        e0 = torch.floor(el_idx).long()
        a0 = torch.floor(az_idx).long()
        ef = el_idx - e0
        af = az_idx - a0
        inb = (e0 >= 0) & (e0 + 1 < g_nel) & (a0 >= 0) & (a0 + 1 < g_naz)
        e0c = e0.clamp(0, g_nel - 2)
        a0c = a0.clamp(0, g_naz - 2)
        gf = g.reshape(-1)
        i00 = e0c * g_naz + a0c
        gi = (gf[i00] * (1 - ef) * (1 - af)
              + gf[i00 + 1] * (1 - ef) * af
              + gf[i00 + g_naz] * ef * (1 - af)
              + gf[i00 + g_naz + 1] * ef * af)
        gi = torch.where(inb, gi, torch.zeros_like(gi))

        w = gi * gi * wa_c * wa_c / (d * d * d)
        W += w.sum(dim=1)
        M1 += (w * psi).sum(dim=1)
        M2 += (w * psi * psi).sum(dim=1)

    mean = M1 / W
    var = torch.clamp(M2 / W - mean * mean, min=0.0)
    dpsi = (12.0 ** 0.5) * torch.sqrt(var)
    # Angular resolution: cross-range resolution lambda/(2*dpsi) divided by the
    # pixel ground range. The range dependence of the cross-range resolution is
    # already handled by backprojection_polar_2d_tx_power (1/d^3 weighting), so
    # only this range-independent angular part remains as an azimuth residual.
    r_ground = torch.sqrt(torch.clamp(px_base * px_base + py_base * py_base, min=0.0))
    resolution = wl / (2.0 * dpsi * r_ground[None])
    # Pixels with no illumination (or a single contributing sweep) have no
    # measurable aperture; report them as infinite resolution.
    resolution = torch.where((W > 0) & valid_pixel[None] & (dpsi > 0) & (r_ground[None] > 0),
                             resolution, torch.full_like(resolution, float("inf")))

    if not batched:
        resolution = resolution[0]
    return resolution


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
    normalize: bool,
):
    torch._check(pos.dtype == torch.float32)
    torch._check(data.dtype == torch.complex64 or data.dtype == torch.complex32)
    return torch.empty((nbatch, Nr, Ntheta), dtype=torch.complex64, device=data.device)


@torch.library.register_fake("torchbp::blocksvd_alpha")
def _fake_blocksvd_alpha(
    img: Tensor,
    data: Tensor,
    pos: Tensor,
    blocks: Tensor,
    sweep_samples: int,
    nsweeps: int,
    nblocks: int,
    Ntheta: int,
    fc: float,
    r_res: float,
    r0: float,
    dr: float,
    theta0: float,
    dtheta: float,
    d0: float,
    data_fmod: float,
):
    torch._check(img.dtype == torch.complex64)
    torch._check(data.dtype == torch.complex64)
    torch._check(pos.dtype == torch.float32)
    torch._check(blocks.dtype == torch.int32)
    return torch.empty((nblocks, nsweeps), dtype=torch.complex64, device=data.device)


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
    normalize: bool,
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


@torch.library.register_fake("torchbp::projection_cart_2d")
def _fake_projection_cart_2d(
    img: Tensor,
    dem: Tensor,
    pos: Tensor,
    vel: Tensor,
    att: Tensor,
    nbatch: int,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    fs: float,
    gamma: float,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    Nx: int,
    Ny: int,
    d0: float,
    g: Tensor,
    g_az0: float,
    g_el0: float,
    g_daz: float,
    g_del: float,
    g_naz: int,
    g_nel: int,
    use_rvp: int,
    normalization: int,
):
    torch._check(img.dtype == torch.complex64)
    torch._check(pos.dtype == torch.float32)
    return torch.empty((nbatch, nsweeps, sweep_samples), dtype=torch.complex64, device=img.device)


@torch.library.register_fake("torchbp::projection_cart_2d_nufft")
def _fake_projection_cart_2d_nufft(
    img: Tensor,
    dem: Tensor,
    pos: Tensor,
    att: Tensor,
    nbatch: int,
    sweep_samples: int,
    nsweeps: int,
    fc: float,
    fs: float,
    gamma: float,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    Nx: int,
    Ny: int,
    d0: float,
    g: Tensor,
    g_az0: float,
    g_el0: float,
    g_daz: float,
    g_del: float,
    g_naz: int,
    g_nel: int,
    use_rvp: int,
    normalization: int,
):
    torch._check(img.dtype == torch.complex64)
    torch._check(pos.dtype == torch.float32)
    return torch.empty((nbatch, nsweeps, sweep_samples), dtype=torch.complex64, device=img.device)


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
    if ctx.saved[15] is not None:
        raise ValueError("gradient with antenna pattern g not supported")
    if data.dtype == torch.complex32:
        raise NotImplementedError("complex32 gradient not supported, use complex64 data")
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
