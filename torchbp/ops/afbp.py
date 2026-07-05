import math
from warnings import warn

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from .backproj import (
    backprojection_polar_2d,
    _prepare_backprojection_polar_2d_args,
)
from ._utils import unpack_polar_grid

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
) -> Tensor:
    """Dealiased polar backprojection with an explicit shared z0 reference.

    Same as ``backprojection_polar_2d(..., dealias=True)`` except that the
    dealias carrier reference height is passed in instead of being computed
    from the mean of ``pos``, and batched input is allowed. All afbp
    subapertures must be demodulated with the identical carrier or the
    wavenumber fusion would see spurious phase steps between them.
    """
    args = list(_prepare_backprojection_polar_2d_args(
        data, grid, fc, r_res, pos, d0, False, att, g, g_extent,
        data_fmod, alias_fmod, normalize))
    args[15] = True  # dealias
    args[16] = z0
    return torch.ops.torchbp.backprojection_polar_2d.default(*args)


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
        Number of subapertures. The subaperture backprojection cost drops
        roughly ``nsub`` times below direct backprojection. Larger values
        need a theta-oversampled grid (see ``grid``) and increase the
        relative cost of the guard band and the fusion. Falls back to
        direct backprojection when the split cannot pay off (fewer than
        two pulses per subaperture, or ``ntheta / nsub + 2 * guard_theta
        >= ntheta`` so that the decimated grid would not be smaller than
        the output).
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
        shape: [elevation, azimuth]. Requires ``normalize=False``: the
        direct kernel normalization is a per-pixel scale over the full
        aperture which cannot be reproduced from per-subaperture images.
        The unnormalized output matches ``backprojection_polar_2d(...,
        normalize=False)`` and can be normalized afterwards with
        illumination maps (see :func:`ffbp`).
    g_extent : list or None
        List of [g_el0, g_az0, g_el1, g_az1]. See
        :func:`backprojection_polar_2d`.
    normalize : bool
        See ``g``. Ignored when no antenna pattern is given.

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
    if data.dim() != 2:
        raise ValueError("data shape should be [nsweeps, samples]")
    if pos.dim() != 2 or pos.shape[0] != data.shape[0]:
        raise ValueError("pos shape should be [nsweeps, 3]")
    if g is not None and normalize:
        raise NotImplementedError(
            "afbp with an antenna pattern requires normalize=False; the "
            "aperture-wide kernel normalization is not factorizable")
    if guard_theta < 0:
        raise ValueError("guard_theta should be >= 0")

    r0, r1, theta0, theta1, nr, ntheta, dr, dtheta = unpack_polar_grid(grid)
    nsweeps, nsamples = data.shape
    device = data.device

    # Fall back to direct backprojection when the split cannot pay off:
    # too few pulses, or a decimated grid that would not be smaller than
    # the output grid once the guard band and ceil padding are added (the
    # subaperture backprojection work scales with the decimated grid
    # size). The latter happens when afbp is applied to tiny grids, e.g.
    # a deep ffbp recursion base level.
    n_c_full = -(-ntheta // nsub) + 2 * guard_theta
    if nsub <= 1 or nsweeps < 2 * nsub or n_c_full >= ntheta:
        return backprojection_polar_2d(
            data, grid, fc, r_res, pos, d0=d0, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod,
            att=att, g=g, g_extent=g_extent, normalize=normalize)[0]

    # Effective range wavenumber center of the image. data_fmod shifts the
    # matched filter phase the same way as in cfbp's keff.
    krc = 4.0 * math.pi * fc / kC0 - data_fmod / r_res
    # Range spectrum center of the dealiased image in cycles/m.
    nu_c = alias_fmod / (2.0 * dr) - data_fmod / (2.0 * r_res)

    # Internal decimated grid with the theta guard band. The fine grid the
    # fusion assembles has n_c * nsub cells starting guard_theta * nsub
    # cells below theta0.
    n_c = n_c_full
    n_fine = n_c * nsub
    theta0_i = theta0 - guard_theta * nsub * dtheta
    grid_c = {
        "r": (r0, r1),
        "theta": (theta0_i, theta0_i + n_c * nsub * dtheta),
        "nr": nr,
        "ntheta": n_c,
    }

    # Split the track into nsub contiguous chunks, padded to equal length
    # for one batched backprojection call. Zero data rows contribute
    # nothing, so padding with repeated positions is exact.
    bounds = [round(i * nsweeps / nsub) for i in range(nsub + 1)]
    m = max(bounds[i + 1] - bounds[i] for i in range(nsub))
    data_b = data.new_zeros((nsub, m, nsamples))
    pos_b = pos.new_zeros((nsub, m, 3))
    att_b = att.new_zeros((nsub, m, 3)) if att is not None else None
    x_u = torch.zeros(nsub, dtype=torch.float64)
    for u in range(nsub):
        i0, i1 = bounds[u], bounds[u + 1]
        data_b[u, : i1 - i0] = data[i0:i1]
        pos_b[u, : i1 - i0] = pos[i0:i1]
        pos_b[u, i1 - i0 :] = pos[i1 - 1]
        if att is not None:
            att_b[u, : i1 - i0] = att[i0:i1]
            att_b[u, i1 - i0 :] = att[i1 - 1]
        x_u[u] = float(pos[i0:i1, 1].double().mean())

    z0 = float(pos[:, 2].double().mean())
    # Ground-to-slant factor: with nonzero altitude the azimuth carrier of
    # a pixel at ground range r is kr * x * rd(r) with rd(r) = r / sqrt(r^2
    # + z0^2), so the patch placement depends on ground range. The fusion
    # runs in the range wavenumber domain where the rows mix all ranges;
    # rd is handled by fusing the swath in range blocks, each with the
    # factor at its center (see below). fac here is the swath center value
    # used by the validity checks.
    r_mid = 0.5 * (r0 + r1)
    fac = r_mid / math.sqrt(r_mid**2 + z0**2)

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
            att=att, g=g, g_extent=g_extent, normalize=normalize)[0]
    kr_max = abs(krc) + min(math.pi / dr, math.pi / r_res)
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

    # The kernel dealias path does not support gradients; with gradients
    # the carrier is instead removed with a differentiable multiply, which
    # matches the kernel dealias up to its float32 phase rounding.
    use_torch_dealias = torch.is_grad_enabled() and (
        data.requires_grad or pos.requires_grad)
    if use_torch_dealias:
        args = _prepare_backprojection_polar_2d_args(
            data_b, grid_c, fc, r_res, pos_b, d0, False, att_b, g, g_extent,
            data_fmod, alias_fmod, normalize)
        imgs = torch.ops.torchbp.backprojection_polar_2d.default(*args)
        imgs = imgs * _dealias_carrier(grid_c, fc, alias_fmod, z0, device).conj()
    else:
        imgs = _bp_polar_shared_dealias(
            data_b, grid_c, fc, r_res, pos_b, d0, z0, data_fmod, alias_fmod,
            att_b, g, g_extent, normalize)

    # The guard band and the ceil padding can push internal grid columns
    # past the polar domain |theta| <= 1, where the backprojection kernel
    # returns NaN. Zero them: they carry no scene content and zeros act as
    # a clean guard buffer. nan_to_num catches the domain-boundary columns
    # where the kernel's float32 theta rounding may still cross 1.
    th_c = theta0_i + (nsub * dtheta) * torch.arange(n_c, dtype=torch.float64)
    outside = th_c.abs() > 1.0
    if bool(outside.any()):
        imgs = imgs * (~outside).to(device=device, dtype=imgs.real.dtype)
        imgs = torch.nan_to_num(imgs)

    # Wavenumber-domain fusion. Azimuth FFT of every subaperture image; a
    # component exp(-1j*k*x_u*rd*alpha) of a dealiased image lands at
    # azimuth frequency -k*x_u*rd/(2*pi) cycles per unit theta.
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

    nua = torch.fft.fftfreq(n_fine, d=dtheta, dtype=torch.float64, device=device)
    cols = torch.arange(n_fine, device=device) % n_c
    fine = Sa.new_zeros((nr, n_fine))
    for b0, b1 in zip(row_bounds[:-1], row_bounds[1:]):
        nrb = b1 - b0
        fac_b = float(0.5 * (rd_rows[b0] + rd_rows[b1 - 1]))
        Sb = torch.fft.fft(Sa[:, b0:b1, :], dim=-2)
        fr = torch.fft.fftfreq(nrb, d=dr, dtype=torch.float64, device=device)
        half = 1.0 / (2.0 * dr)
        # Physical range frequency offset of each row, unwrapped around
        # the modulation-shifted spectrum center.
        fr_phys = torch.remainder(fr - nu_c + half, 2.0 * half) - half
        kr = krc + 2.0 * math.pi * fr_phys  # [nrb]
        # Along-track position owning each (range wavenumber row, azimuth
        # bin).
        x_eq = -2.0 * math.pi * nua[None, :] / (kr[:, None] * fac_b)  # [nrb, n_fine]
        # Each subaperture patch is reconstructed over a full decimated
        # band centered on the patch: the overlapping regions of adjacent
        # subapertures are summed, which keeps the patch edge transitions
        # that a disjoint tiling would truncate. Content outside every
        # region (past the aperture edges) is zero.
        x_half = math.pi * n_c / (n_fine * dtheta) / (kr * fac_b)  # [nrb]
        fine_b = Sb.new_zeros((nrb, n_fine))
        for u in range(nsub):
            masku = (x_eq - float(x_u[u])).abs() <= x_half[:, None].abs()
            fine_b += Sb[u, :, cols] * masku
        fine[b0:b1] = torch.fft.ifft(fine_b, dim=-2)
    fine *= nsub

    out = torch.fft.ifft(fine, dim=-1)
    gt = guard_theta * nsub
    out = out[:, gt : gt + ntheta]
    if not dealias:
        out = out * _dealias_carrier(grid, fc, alias_fmod, z0, device)
    return out
