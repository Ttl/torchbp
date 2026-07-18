import math
import functools

import torch
from torch import Tensor
from typing import TYPE_CHECKING
from copy import deepcopy

from .backproj import backprojection_cart_2d, _backprojection_cart_2d_tx_power_accum, _tx_power_finish
from ..data import materialize as _materialize
from ..util import center_pos
from ._utils import unpack_cartesian_grid, parse_interp_method

if TYPE_CHECKING:
    from ..grid import CartesianGrid

kC0 = 299792458.0


def _fft_resample_dim(img: Tensor, n_new: int, dim: int) -> Tensor:
    """
    Exact FFT resampling of a uniformly sampled dimension.

    Samples are assumed to lie at ``v0 + i*(v1 - v0)/n`` for ``i in [0, n)``,
    which is the Cartesian grid convention, so the resampled points land
    exactly on the grid of the new size over the same extent. Upsampling
    zero-pads the spectrum (even-length Nyquist bin is split between the
    positive and negative frequencies), downsampling truncates it (Nyquist
    bins folded together).
    """
    n_old = img.shape[dim]
    if n_new == n_old:
        return img
    X = torch.fft.fft(img, dim=dim)
    shape = list(X.shape)
    shape[dim] = n_new
    Y = X.new_zeros(shape)

    def sl(t, a, b):
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(a, b)
        return t[tuple(idx)]

    if n_new > n_old:
        h = n_old // 2
        if n_old % 2 == 0:
            sl(Y, 0, h).copy_(sl(X, 0, h))
            sl(Y, h, h + 1).copy_(0.5 * sl(X, h, h + 1))
            sl(Y, n_new - h, n_new - h + 1).copy_(0.5 * sl(X, h, h + 1))
            sl(Y, n_new - h + 1, n_new).copy_(sl(X, h + 1, n_old))
        else:
            sl(Y, 0, h + 1).copy_(sl(X, 0, h + 1))
            sl(Y, n_new - h, n_new).copy_(sl(X, h + 1, n_old))
    else:
        h = n_new // 2
        if n_new % 2 == 0:
            sl(Y, 0, h).copy_(sl(X, 0, h))
            sl(Y, h, h + 1).copy_(sl(X, h, h + 1) + sl(X, n_old - h, n_old - h + 1))
            sl(Y, h + 1, n_new).copy_(sl(X, n_old - n_new + h + 1, n_old))
        else:
            sl(Y, 0, h + 1).copy_(sl(X, 0, h + 1))
            sl(Y, h + 1, n_new).copy_(sl(X, n_old - h, n_old))
    return torch.fft.ifft(Y, dim=dim) * (n_new / n_old)


@functools.lru_cache(maxsize=64)
def _merge_weight_table(ny_c: int, ny_new: int, order: int, v: float) -> tuple[Tensor, Tensor]:
    """
    Knab windowed-sinc weights for resampling ny_c -> ny_new samples over a
    shared extent with the ``v0 + i*L/n`` sample convention.

    Output index j interpolates the child at fractional index
    ``u = j*ny_c/ny_new``. The start index is clamped to keep all taps in
    bounds; out-of-window taps get zero weight, so clamping is exactly window
    truncation at the edges. Returns (w [ny_new, order_eff] float32,
    idx [ny_new] int32) on CPU.
    """
    if ny_c == ny_new:
        w = torch.ones((ny_new, 1), dtype=torch.float32)
        idx = torch.arange(ny_new, dtype=torch.int32)
        return w, idx
    order_eff = min(order, ny_c)
    a = order_eff / 2
    j = torch.arange(ny_new, dtype=torch.float64)
    u = j * ny_c / ny_new
    idx = torch.clamp(torch.floor(u).long() - order_eff // 2 + 1, 0, ny_c - order_eff)
    t = torch.arange(order_eff, dtype=torch.float64)
    x = u[:, None] - (idx[:, None].double() + t[None, :])
    arg = torch.clamp(1.0 - (x / a) ** 2, min=0.0)
    w = torch.sinc(x) * torch.cosh(math.pi * v * a * torch.sqrt(arg)) / math.cosh(math.pi * v * a)
    w = torch.where(x.abs() < a, w, torch.zeros_like(w))
    return w.float(), idx.to(torch.int32)


@functools.lru_cache(maxsize=64)
def _merge_weight_table_dev(ny_c: int, ny_new: int, order: int, v: float, device: str) -> tuple[Tensor, Tensor]:
    w, idx = _merge_weight_table(ny_c, ny_new, order, v)
    return w.to(device), idx.to(device)


def cfbp_merge2(
    img0: Tensor,
    img1: Tensor,
    w0: Tensor,
    idx0: Tensor,
    w1: Tensor,
    idx1: Tensor,
    dx: float,
    dy: float,
    ox0: float,
    oy0: float,
    z0: float,
    ox1: float,
    oy1: float,
    z1: float,
    oxp: float,
    oyp: float,
    zp: float,
    ref_phase: float,
) -> Tensor:
    """
    Merge two demodulated cfbp subaperture images into their parent image.

    Both children and the output share the same x and y extents; the y
    interpolation from child sampling to output sampling uses the
    precomputed per-output-row weight tables (see
    :func:`_merge_weight_table`). ``ox*, oy*`` are the output grid start
    coordinates relative to each subaperture center (child 0, child 1,
    parent) and ``z*`` the corresponding center heights. Each interpolated
    child is re-referenced from its own demodulation carrier to the parent
    carrier with phase ``pi * ref_phase * (d_child - d_parent)`` and the two
    are summed.
    """
    nbatch, nx, ny0 = img0.shape
    ny1 = img1.shape[-1]
    nyout = idx0.shape[0]
    return torch.ops.torchbp.cfbp_merge2.default(
        img0, img1, w0, idx0, w1, idx1,
        nbatch, nx, ny0, ny1, nyout, w0.shape[-1], w1.shape[-1],
        dx, dy, ox0, oy0, z0, ox1, oy1, z1, oxp, oyp, zp, ref_phase,
    )


@torch.library.register_fake("torchbp::cfbp_merge2")
def _fake_cfbp_merge2(
    img0: Tensor,
    img1: Tensor,
    w0: Tensor,
    idx0: Tensor,
    w1: Tensor,
    idx1: Tensor,
    nbatch: int,
    Nx: int,
    Ny0: int,
    Ny1: int,
    Nyout: int,
    order0: int,
    order1: int,
    dx: float,
    dy: float,
    ox0: float,
    oy0: float,
    z0: float,
    ox1: float,
    oy1: float,
    z1: float,
    oxp: float,
    oyp: float,
    zp: float,
    ref_phase: float,
):
    torch._check(img0.dtype == torch.complex64)
    torch._check(img1.dtype == torch.complex64)
    torch._check(w0.dtype == torch.float32)
    torch._check(w1.dtype == torch.float32)
    torch._check(idx0.dtype == torch.int32)
    torch._check(idx1.dtype == torch.int32)
    return torch.empty((nbatch, Nx, Nyout), dtype=torch.complex64, device=img0.device)


def _grid_axes(grid: dict, origin: Tensor, device) -> tuple[Tensor, Tensor]:
    """Pixel coordinate axes centered on origin, in float64."""
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)
    x = x0 - float(origin[0]) + dx * torch.arange(nx, device=device, dtype=torch.float64)
    y = y0 - float(origin[1]) + dy * torch.arange(ny, device=device, dtype=torch.float64)
    return x, y


def _grid_distance(grid: dict, origin: Tensor, z: float, device) -> Tensor:
    """Distance from (origin[0], origin[1], z) to each grid pixel. [nx, ny] float32."""
    x, y = _grid_axes(grid, origin, device)
    x = x.float()
    y = y.float()
    return torch.sqrt(x[:, None] ** 2 + y[None, :] ** 2 + float(z) ** 2)


def _carrier_ref(grid: dict, origin: Tensor, z: float, keff: float, sign: float, device) -> Tensor:
    """
    Carrier reference exp(sign * 1j * pi * keff * d(x, y)) on the grid.

    The phase argument is large (keff * d is on the order of 1e4..1e5), so it
    is computed and wrapped in float64 before converting to float32.
    """
    x, y = _grid_axes(grid, origin, device)
    d = torch.sqrt(x[:, None] ** 2 + y[None, :] ** 2 + float(z) ** 2)
    ph = torch.remainder(sign * keff * d, 2.0).float() * torch.pi
    return torch.polar(torch.ones_like(ph), ph)


def cfbp(
    data: Tensor,
    grid: "CartesianGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int = 2,
    d0: float = 0.0,
    oversample_y: float = 1.4,
    guard_y: float = 0.05,
    beamwidth: float = torch.pi,
    data_fmod: float = 0,
    interp_method: "tuple | str" = ("knab", 8, 1.4),
) -> Tensor:
    """
    Cartesian factorized backprojection. [1]_

    Factorized backprojection directly on a Cartesian grid. The aperture is
    recursively split into subapertures which are backprojected with
    :func:`backprojection_cart_2d` onto the full output grid extent, coarsely
    sampled in the cross-range (y) dimension. Subaperture images are
    demodulated with the carrier referenced to the subaperture center, which
    makes them bandlimited so that merging only needs upsampling along y and
    a phase re-reference, done in a single fused kernel (see
    ``interp_method``). The output matches ``backprojection_cart_2d(data,
    grid, ...)`` up to interpolation error.

    Antenna pattern normalization is not implemented.

    Gradient can be calculated with respect to data. Gradient with respect to
    pos flows only through the base-level backprojections; the merge phase
    references are treated as constants (same as :func:`ffbp`).

    Parameters
    ----------
    data : Tensor
        Range compressed input data. Shape should be [nsweeps, samples].
    grid : CartesianGrid or dict
        Cartesian grid definition. Can be:

        - CartesianGrid object: ``CartesianGrid(x_range=(-50, 50), y_range=(-50, 50), nx=200, ny=200)``
        - dict: ``{"x": (x0, x1), "y": (y0, y1), "nx": nx, "ny": ny}``

        The grid should be oversampled for good interpolation accuracy.
    fc : float
        RF center frequency in Hz.
    r_res : float
        Range bin resolution in data (meters).
        For FMCW radar: c/(2*bw*oversample), where c is speed of light, bw is sweep bandwidth,
        and oversample is FFT oversampling factor.
    pos : Tensor
        Position of the platform at each data point. Shape should be [nsweeps, 3].
    stages : int
        Number of recursions.
    divisions : int
        Number of subaperture divisions per stage. Default is 2.
    d0 : float
        Zero range correction.
    oversample_y : float
        Internally oversample y by this amount to avoid aliasing.
        The x dimension needs no internal oversampling since subaperture
        images are never resampled in x.
    guard_y : float
        Internal guard band on each side of the y extent as a fraction of ny,
        cropped from the output. Absorbs the y edge effects of the merge
        interpolation (periodic wrap-around for the fft method, window
        truncation for knab).
    beamwidth : float
        Beamwidth of the antenna in radians. Passed to
        :func:`backprojection_cart_2d`.
    data_fmod : float
        Range modulation frequency applied to input data.
    interp_method : tuple or str
        Merge interpolation method. ``("knab", order, oversample)`` uses the
        compiled merge kernel with an exactly evaluated Knab windowed-sinc
        interpolator of ``order`` taps designed for signals occupying
        ``1/oversample`` of the sampling rate; ``oversample`` should not
        exceed ``oversample_y``. ``"fft"`` uses exact FFT resampling in
        pure torch, which is slower but has no interpolation error and
        supports gradient calculation through the merge. The knab method
        falls back to the fft path when the merged images require gradient.

    Returns
    -------
    img : Tensor
        Cartesian format radar image. Shape is [1, nx, ny].

    Notes
    -----
    The subaperture images are sampled in y at ``oversample_y * ny /
    divisions**stage`` points. This covers the azimuth bandwidth, which
    shrinks with the subaperture length, but the demodulated images also
    contain the range envelope bandwidth projected onto y, approximately
    ``sin(theta_max) * B``, where ``theta_max`` is the largest angle of the
    scene from boresight and ``B = 1/(os * r_res)`` with ``os`` the range FFT
    oversampling factor of the data. This term does not shrink when the
    aperture is split, so scenes with large angular extent combined with
    small range oversampling need a larger ``oversample_y`` or fewer
    ``stages`` to avoid aliasing in the base images. Polar :func:`ffbp` does
    not have this limitation and is a better choice for very wide-angle
    imaging.

    References
    ----------
    .. [1] Q. Dong, G. -C. Sun, Z. Yang, L. Guo and M. Xing, "Cartesian
    Factorized Backprojection Algorithm for High-Resolution Spotlight SAR
    Imaging," in IEEE Sensors Journal, vol. 18, no. 3, pp. 1160-1168, 1 Feb.1,
    2018.
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    data = _materialize(data)
    if data.dim() != 2:
        raise ValueError("data shape should be [nsweeps, samples]")
    if pos.dim() != 2 or pos.shape[0] != data.shape[0]:
        raise ValueError("pos shape should be [nsweeps, 3]")
    interp_method = parse_interp_method(interp_method, allowed=("knab", "fft"))
    if interp_method[0] == "knab" and interp_method[2] <= 1:
        raise ValueError("interp_method oversample should be > 1")

    # Carrier spatial frequency of the image in units of pi. Includes the
    # data_fmod contribution to the range phase applied by the kernel.
    keff = 4.0 * fc / kC0 - data_fmod / (torch.pi * r_res)

    n_guard = int(round(guard_y * grid["ny"]))
    grid_impl = grid
    if n_guard > 0:
        y0, y1 = grid["y"]
        dy = (y1 - y0) / grid["ny"]
        grid_impl = dict(
            grid,
            y=(y0 - n_guard * dy, y1 + n_guard * dy),
            ny=grid["ny"] + 2 * n_guard,
        )

    img, origin, z0 = _cfbp_impl(
        data, grid_impl, fc, r_res, pos, stages, divisions, d0,
        oversample_y, beamwidth, data_fmod, keff, interp_method
    )
    if n_guard > 0:
        img = img[..., n_guard : n_guard + grid["ny"]]
    # Restore the carrier. Demodulation and remodulation phases telescope so
    # that the output matches direct backprojection.
    return img * _carrier_ref(grid, origin, z0, keff, 1.0, data.device)


def _k_bucket(k: int) -> int:
    """Round k up to the nearest value of form 2**m or 3*2**m.

    Bounds the number of distinct block densities (about two per octave)
    while inflating the internal grid by at most 33%.
    """
    if k <= 1:
        return 1
    m = 1
    while True:
        if k <= m:
            return m
        if k <= 3 * m // 2:
            return 3 * m // 2
        m *= 2


def cfbp_adaptive_blocks(
    grid: "CartesianGrid | dict",
    pos: Tensor,
    fc: float,
    r_res: float,
    stages: int,
    divisions: int = 2,
    oversample_y: float = 1.4,
    data_oversample: float = 2.0,
    merge_cost: float = 8.0,
) -> list:
    """
    Compute the range-adaptive y-density blocks used by :func:`cfbp_adaptive`.

    For every output row (ground range x) the demodulated subaperture image
    y-bandwidth is estimated at each recursion level as::

        B(l, rho) = (2*fc/c) * l / sqrt((l/2)**2 + rho**2) + sin_max / (data_oversample * r_res)

    where ``l`` is the subaperture length at that level, ``rho = sqrt(x**2 +
    h**2)`` is the closest slant distance from the row to the track line, and
    ``sin_max`` is the largest angle of arrival at the row. The first term is
    the subaperture angular extent seen from the pixel (saturates for long
    subapertures at close range), the second is the range envelope bandwidth
    projected onto y. For a given stage count s, the required integer density
    multiplier k(s) is the smallest one for which every level of the
    recursion satisfies its Nyquist limit, rounded up to the 2**m / 3*2**m
    bucket grid.

    The stage count is also chosen per row, by minimizing the estimated cost
    per output sample ``k(s) * (nsweeps / divisions**s + merge_cost * s)``
    (base backprojection plus merge passes): deep recursion pays off at far
    range where the angular term is small, while near range rows saturate
    and prefer shallow recursion. ``stages`` is the maximum allowed.
    ``merge_cost`` is the assumed cost of one level of merging relative to
    one backprojection operation. Consecutive rows with equal (k, s) are
    grouped into blocks.

    Returns
    -------
    blocks : list of (row_start, row_end, k, s)
        Half-open output row ranges with their density multiplier and stage
        count, in ascending row order covering the full grid.
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)
    Ly = y1 - y0

    uy = pos[:, 1].double().cpu()
    L = float(uy.max() - uy.min())
    h = abs(float(pos[:, 2].double().mean()))
    nsweeps = pos.shape[0]

    # Maximum recursion depth, mirrors the sweep count guard in _cfbp_impl.
    s_max = 1
    n = nsweeps
    while s_max < stages and n // divisions > 128:
        n //= divisions
        s_max += 1

    kc = 2.0 * fc / kC0  # cycles/m per unit sin(theta)
    b_env = 1.0 / (r_res * data_oversample)

    x = x0 + dx * torch.arange(nx, dtype=torch.float64)
    rho = torch.sqrt(x**2 + h**2)
    # Largest |y_pixel - y_pulse| of the row determines the worst angle of
    # arrival for the envelope projection.
    dy_max = max(y1 - float(uy.min()), float(uy.max()) - y0, 0.0)
    sin_max = dy_max / torch.sqrt(dy_max**2 + rho**2)

    # For each row pick the (k, s) with the lowest estimated cost per output
    # sample: nsweeps/div**s base backprojection operations plus the merge
    # passes of each level.
    req = torch.zeros(nx, dtype=torch.float64)
    best_score = torch.full((nx,), torch.inf, dtype=torch.float64)
    best_k = torch.ones(nx, dtype=torch.long)
    best_s = torch.ones(nx, dtype=torch.long)
    for s in range(1, s_max + 1):
        ell = L / divisions**s
        dsin = ell / torch.sqrt((ell / 2) ** 2 + rho**2)
        bw = kc * dsin + sin_max * b_env
        req = torch.maximum(req, divisions**s * bw)
        k_s = torch.ceil(req * Ly / (oversample_y * ny)).clamp(min=1)
        k_s = torch.tensor([_k_bucket(int(k)) for k in k_s], dtype=torch.float64)
        score = k_s * (nsweeps / divisions**s + merge_cost * s)
        better = score < best_score
        best_score = torch.where(better, score, best_score)
        best_k = torch.where(better, k_s.long(), best_k)
        best_s = torch.where(better, torch.full_like(best_s, s), best_s)

    blocks = []
    start = 0
    for i in range(1, nx + 1):
        if i == nx or (best_k[i] != best_k[start] or best_s[i] != best_s[start]):
            blocks.append((start, i, int(best_k[start]), int(best_s[start])))
            start = i
    return blocks


def cfbp_adaptive(
    data: Tensor,
    grid: "CartesianGrid | dict",
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int = 2,
    d0: float = 0.0,
    oversample_y: float = 1.4,
    guard_y: float = 0.05,
    beamwidth: float = torch.pi,
    data_fmod: float = 0,
    data_oversample: float = 2.0,
    merge_cost: float = 8.0,
    interp_method: "tuple | str" = ("knab", 8, 1.4),
) -> Tensor:
    """
    Cartesian factorized backprojection with range-adaptive y-density.

    Plain :func:`cfbp` on a subsampled output grid fails when the aperture is
    long compared to the closest scene range: the demodulated subaperture
    images then have more y-bandwidth than the proportionally decimated grids
    can hold at near range, at every level of the recursion. Because the
    Cartesian scheme never resamples x, the scene can be split into ground
    range blocks with zero seam error and each block run at its own internal
    y-density: block rows at ground range x get an integer density multiplier
    k(x) and stage count s(x) (see :func:`cfbp_adaptive_blocks`), the cfbp
    tree runs on a grid with ``k * ny`` y-samples, and the output is
    decimated by taking every k-th sample, which lands exactly on the
    requested grid positions. The result therefore matches
    ``backprojection_cart_2d`` on the output grid to normal cfbp accuracy at
    all ranges, while far ranges keep the full factorization speedup (deeper
    recursion is chosen there, shallower near-in where the angular extent
    saturates). ``stages`` is the maximum recursion depth.

    Near range blocks saturate geometrically (the angular extent of a
    subaperture seen from a pixel is bounded), so k stays finite for scenes
    starting near zero ground range as long as the platform altitude is
    nonzero.

    Parameters are as in :func:`cfbp`, plus:

    Parameters
    ----------
    data_oversample : float
        Range FFT oversampling factor of the range compressed data, i.e. the
        fraction ``1/data_oversample`` of the sample bandwidth occupied by
        the data spectrum. Used to size the projected range envelope
        bandwidth term of the internal grids. Default is 2.
    merge_cost : float
        Assumed cost of one level of merging relative to one backprojection
        operation, used when selecting the per-block stage count. See
        :func:`cfbp_adaptive_blocks`.

    Returns
    -------
    img : Tensor
        Cartesian format radar image. Shape is [1, nx, ny].
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    data = _materialize(data)
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)

    blocks = cfbp_adaptive_blocks(
        grid, pos, fc, r_res, stages, divisions, oversample_y, data_oversample,
        merge_cost
    )

    outs = []
    for i0, i1, k, s in blocks:
        grid_b = {
            "x": (x0 + i0 * dx, x0 + i1 * dx),
            "y": (y0, y1),
            "nx": i1 - i0,
            "ny": k * ny,
        }
        img_b = cfbp(
            data,
            grid_b,
            fc,
            r_res,
            pos,
            s,
            divisions=divisions,
            d0=d0,
            oversample_y=oversample_y,
            guard_y=guard_y,
            beamwidth=beamwidth,
            data_fmod=data_fmod,
            interp_method=interp_method,
        )
        # Internal samples lie at y0 + i*Ly/(k*ny); every k-th one is exactly
        # an output grid position, so decimation is pure subsampling.
        outs.append(img_b[..., ::k])
    return torch.cat(outs, dim=-2)


def _cfbp_impl(
    data: Tensor,
    grid: dict,
    fc: float,
    r_res: float,
    pos: Tensor,
    stages: int,
    divisions: int,
    d0: float,
    oversample_y: float,
    beamwidth: float,
    data_fmod: float,
    keff: float,
    interp_method: tuple,
) -> tuple[Tensor, Tensor, float]:
    """Recursive cfbp. Returns (demodulated image [1, nx, ny], origin [3], z0)."""
    device = data.device
    nsweeps = data.shape[0]

    imgs = []
    bounds = [round(i * nsweeps / divisions) for i in range(divisions + 1)]
    for d_idx in range(divisions):
        data_local = data[bounds[d_idx] : bounds[d_idx + 1]]
        pos_chunk = pos[bounds[d_idx] : bounds[d_idx + 1]]
        grid_local = deepcopy(grid)
        grid_local["ny"] = (grid["ny"] + divisions - 1) // divisions

        if stages > 1 and len(data_local) > 128:
            grid_local["ny"] = int(oversample_y * grid_local["ny"])
            img, origin, z0 = _cfbp_impl(
                data_local, grid_local, fc, r_res, pos_chunk,
                stages=stages - 1,
                divisions=divisions,
                d0=d0,
                oversample_y=1.0,  # Grid is already increased
                beamwidth=beamwidth,
                data_fmod=data_fmod,
                keff=keff,
                interp_method=interp_method,
            )
        else:
            pos_local, origin_row = center_pos(pos_chunk)
            origin = origin_row[0]
            z0 = float(torch.mean(pos_local[:, 2]))
            grid_shift = deepcopy(grid_local)
            ox, oy = float(origin[0]), float(origin[1])
            grid_shift["x"] = (grid_local["x"][0] - ox, grid_local["x"][1] - ox)
            grid_shift["y"] = (grid_local["y"][0] - oy, grid_local["y"][1] - oy)
            img = backprojection_cart_2d(
                data_local, grid_shift, fc, r_res, pos_local,
                d0=d0, beamwidth=beamwidth, data_fmod=data_fmod
            )
            img = img * _carrier_ref(grid_local, origin, z0, keff, -1.0, device)

        imgs.append((origin, grid_local, img, z0))

    while len(imgs) > 1:
        origin1, grid1, img1, z1 = imgs[0]
        origin2, grid2, img2, z2 = imgs[1]
        new_origin = 0.5 * (origin1 + origin2)
        new_z = 0.5 * (z1 + z2)
        if len(imgs) == 2:
            # Final merge outputs the requested grid.
            grid_new = deepcopy(grid)
        else:
            grid_new = deepcopy(grid1)
            grid_new["ny"] = grid1["ny"] + grid2["ny"]
            grid_new["nx"] = max(grid1["nx"], grid2["nx"])

        use_kernel = (
            interp_method[0] == "knab"
            and grid1["nx"] == grid_new["nx"]
            and grid2["nx"] == grid_new["nx"]
            and grid1["ny"] <= grid_new["ny"]
            and grid2["ny"] <= grid_new["ny"]
            and not (torch.is_grad_enabled() and (img1.requires_grad or img2.requires_grad))
        )
        if use_kernel:
            order, oversample = interp_method[1], interp_method[2]
            v = round(1.0 - 1.0 / oversample, 6)
            ny_new = grid_new["ny"]
            dev = str(device)
            w1, i1 = _merge_weight_table_dev(grid1["ny"], ny_new, order, v, dev)
            w2, i2 = _merge_weight_table_dev(grid2["ny"], ny_new, order, v, dev)
            x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid_new)
            img_sum = cfbp_merge2(
                img1, img2, w1, i1, w2, i2, dx, dy,
                x0 - float(origin1[0]), y0 - float(origin1[1]), z1,
                x0 - float(origin2[0]), y0 - float(origin2[1]), z2,
                x0 - float(new_origin[0]), y0 - float(new_origin[1]), new_z,
                keff,
            )
        else:
            d_parent = _grid_distance(grid_new, new_origin, new_z, device)

            img_sum = None
            for origin_c, grid_c, img_c, z_c in ((origin1, grid1, img1, z1), (origin2, grid2, img2, z2)):
                if grid_c["ny"] != grid_new["ny"]:
                    img_c = _fft_resample_dim(img_c, grid_new["ny"], dim=-1)
                if grid_c["nx"] != grid_new["nx"]:
                    img_c = _fft_resample_dim(img_c, grid_new["nx"], dim=-2)
                # Re-reference the demodulation carrier to the merged
                # subaperture center. The phase difference is small so float32
                # is accurate enough here.
                ph = (torch.pi * keff) * (_grid_distance(grid_new, origin_c, z_c, device) - d_parent)
                img_c = img_c * torch.polar(torch.ones_like(ph), ph)
                img_sum = img_c if img_sum is None else img_sum + img_c

        imgs[0] = None
        imgs[1] = None
        del img1, img2
        imgs = imgs[2:] + [(new_origin, grid_new, img_sum, new_z)]

    return imgs[0][2], imgs[0][0], imgs[0][3]


def cart_tx_power_merge2(
    acc0: Tensor,
    acc1: Tensor | None,
    grid0: "CartesianGrid | dict",
    grid1: "CartesianGrid | dict | None",
    grid_new: "CartesianGrid | dict",
) -> Tensor:
    """
    Merge two Cartesian tx_power accumulator maps onto a new Cartesian grid.

    Bilinearly interpolates the 4-channel accumulator maps (S, W, P1, M2, see
    :func:`torchbp.ops.backproj._backprojection_cart_2d_tx_power_accum`) from
    each input grid onto the output grid and combines them. All grids are in
    absolute world coordinates, so an output pixel maps straight into each
    input's local index; no origin shift is needed. The psi moments are
    combined with Chan's parallel variance formula so the merge is exact up to
    interpolation. ``acc1 = None`` interpolates only ``acc0`` (single-input
    regrid). Used internally by
    :func:`~torchbp.ops.backprojection_cart_2d_tx_power_cfbp`.
    """
    device = acc0.device
    if hasattr(grid_new, "to_dict"):
        grid_new = grid_new.to_dict()
    if hasattr(grid0, "to_dict"):
        grid0 = grid0.to_dict()
    gx0_0, _, gy0_0, _, nx0, ny0, gdx0, gdy0 = unpack_cartesian_grid(grid0)
    if acc1 is None:
        acc1 = torch.empty(0, dtype=torch.float32, device=device)
        gx0_1 = gdx1 = gy0_1 = gdy1 = 0.0
        nx1 = ny1 = 0
    else:
        if hasattr(grid1, "to_dict"):
            grid1 = grid1.to_dict()
        gx0_1, _, gy0_1, _, nx1, ny1, gdx1, gdy1 = unpack_cartesian_grid(grid1)
    ox0, _, oy0, _, onx, ony, odx, ody = unpack_cartesian_grid(grid_new)
    return torch.ops.torchbp.cart_tx_power_merge2.default(
        acc0, acc1,
        gx0_0, gdx0, gy0_0, gdy0, nx0, ny0,
        gx0_1, gdx1, gy0_1, gdy1, nx1, ny1,
        ox0, odx, oy0, ody, onx, ony,
    )


@torch.library.register_fake("torchbp::cart_tx_power_merge2")
def _fake_cart_tx_power_merge2(
    acc0: Tensor,
    acc1: Tensor,
    x0_0: float, dx_0: float, y0_0: float, dy_0: float, Nx_0: int, Ny_0: int,
    x0_1: float, dx_1: float, y0_1: float, dy_1: float, Nx_1: int, Ny_1: int,
    x1: float, dx1: float, y1: float, dy1: float, Nx1: int, Ny1: int,
):
    torch._check(acc0.dtype == torch.float32)
    return torch.empty((4, Nx1, Ny1), dtype=torch.float32, device=acc0.device)


def _tx_power_cart_node_grid(
    boundary: Tensor,
    pos: Tensor,
    yaw: Tensor,
    g_extent: list,
    g_daz: float,
    sub_dx: float,
    sub_dy: float,
    margin: int,
    min_nx: int,
    min_ny: int,
) -> dict | None:
    """Absolute Cartesian grid covering the region a subaperture illuminates.

    ``boundary`` is the output-grid perimeter in absolute ground coordinates.
    The azimuth extent visible through the antenna pattern (given the yaw
    range) selects the boundary points the subaperture can see; their bounding
    box, expanded by ``margin`` cells, is the node grid. The margin extends
    slightly *beyond* the output boundary (not clipped to it) so the pairwise
    bilinear merge has valid interior samples for the output edge pixels;
    clipping to the output extent would leave the outermost output rows/columns
    with no interpolation neighbours and drop them. Range/elevation is not
    windowed (kept to the full output extent) since unilluminated pixels merge
    as zero; the azimuth window is what shrinks the maps on long tracks.
    Returns None if the subaperture cannot see the output grid. Cannot
    represent scenes in the rear half plane (|azimuth| > pi/2), like the polar
    :func:`_tx_power_node_grid`.
    """
    bx = boundary[:, 0]
    by = boundary[:, 1]
    cen = pos[:, :2].mean(dim=0)
    dxb = bx - cen[0]
    dyb = by - cen[1]
    r = torch.sqrt(dxb * dxb + dyb * dyb)
    psi = torch.atan2(dyb, dxb)
    r_lo = float(torch.clamp(r.min(), min=1e-3))

    g_el0, g_az0, g_el1, g_az1 = g_extent
    # Pad for pattern lookup interpolation and for the pulses being offset from
    # the aperture centroid by up to the subaperture length (as ffbp).
    max_offset = float(torch.linalg.norm(pos[:, :2] - cen[None, :], dim=-1).max())
    psi_pad = 2.0 * g_daz + max_offset / r_lo
    psi_lo = g_az0 + float(yaw.min()) - psi_pad
    psi_hi = g_az1 + float(yaw.max()) + psi_pad
    visible = (psi >= psi_lo) & (psi <= psi_hi)
    if not bool(visible.any()):
        return None
    vbx = bx[visible]
    vby = by[visible]
    # Expand by margin cells beyond the visible boundary (not clipped to the
    # output extent) so output edge pixels have interior merge neighbours.
    x_lo = float(vbx.min()) - margin * sub_dx
    x_hi = float(vbx.max()) + margin * sub_dx
    y_lo = float(vby.min()) - margin * sub_dy
    y_hi = float(vby.max()) + margin * sub_dy
    if x_hi <= x_lo or y_hi <= y_lo:
        return None
    nx = max(min_nx, int(math.ceil((x_hi - x_lo) / sub_dx)))
    ny = max(min_ny, int(math.ceil((y_hi - y_lo) / sub_dy)))
    return {"x": (x_lo, x_hi), "y": (y_lo, y_hi), "nx": nx, "ny": ny}


def _cfbp_tx_power_impl(
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
    sub_dx: float,
    sub_dy: float,
    margin: int,
    min_nx: int,
    min_ny: int,
    min_nsweeps: int,
    dx_ref: float,
    h_ref: float,
    grid_out: dict | None = None,
    is_top_level: bool = True,
) -> tuple[Tensor, dict] | None:
    """Recursive Cartesian tx_power accumulator map computation.

    Returns (acc, grid) with the 4-channel accumulator on grid, or None if no
    pulse of this node illuminates the output grid. All grids are absolute; no
    frame offset is tracked (tx_power is phase free).
    """
    nsweeps = wa.shape[0]

    grid_node = _tx_power_cart_node_grid(
        boundary, pos, att[:, 2], g_extent, g_daz, sub_dx, sub_dy,
        margin, min_nx, min_ny)
    if grid_node is None:
        return None

    nodes = []
    edges = torch.linspace(0, nsweeps, divisions + 1).long()
    for d_idx in range(divisions):
        i0 = int(edges[d_idx])
        i1 = int(edges[d_idx + 1])
        if i1 - i0 < 1:
            continue
        wa_local = wa[i0:i1]
        pos_local = pos[i0:i1]
        att_local = att[i0:i1]

        if stages > 1 and i1 - i0 > min_nsweeps:
            child = _cfbp_tx_power_impl(
                wa_local, pos_local, att_local, boundary, stages - 1, divisions,
                g, g_extent, g_daz, normalization, sub_dx, sub_dy, margin,
                min_nx, min_ny, min_nsweeps, dx_ref, h_ref,
                grid_out=None, is_top_level=False)
            if child is None:
                continue
            acc, grid_local = child
        else:
            grid_local = _tx_power_cart_node_grid(
                boundary, pos_local, att_local[:, 2], g_extent, g_daz,
                sub_dx, sub_dy, margin, min_nx, min_ny)
            if grid_local is None:
                continue
            acc = _backprojection_cart_2d_tx_power_accum(
                wa_local, g, g_extent, grid_local, pos_local, att_local,
                normalization, dx_ref, h_ref)
        nodes.append((acc, grid_local))

    if len(nodes) == 0:
        return None

    while len(nodes) > 1:
        n1 = nodes[0]
        n2 = nodes[1]
        is_final_merge = is_top_level and len(nodes) == 2
        grid_new = grid_out if is_final_merge else grid_node
        merged = cart_tx_power_merge2(n1[0], n2[0], n1[1], n2[1], grid_new)
        nodes = nodes[2:] + [(merged, grid_new)]

    return nodes[0]


def backprojection_cart_2d_tx_power_cfbp(
    wa: Tensor,
    g: Tensor,
    g_extent: list,
    grid: "CartesianGrid | dict",
    r_res: float,
    pos: Tensor,
    att: Tensor,
    stages: int,
    divisions: int = 2,
    normalization: str | None = None,
    azimuth_resolution: bool = True,
    downsample_x: float = 4.0,
    downsample_y: float = 4.0,
    min_nsweeps: int = 64,
    min_nx: int = 32,
    min_ny: int = 32,
    margin: int = 4,
    beam_theta_samples: float = 32.0,
) -> Tensor:
    """
    Fast factorized (CFBP style) version of
    :func:`backprojection_cart_2d_tx_power`.

    Splits the track recursively into subapertures like :func:`cfbp`, computes
    coarse per-subaperture accumulator maps over the ground region each
    illuminates, and merges them pairwise with bilinear interpolation. Like the
    polar :func:`backprojection_polar_2d_tx_power_ffbp` the accumulated fields
    are smooth and phase-free, so the subaperture maps can be sampled much more
    coarsely than the output grid (``downsample_x``, ``downsample_y``) and the
    azimuth resolution moments merge exactly (Chan's parallel variance
    formula). Unlike the polar version everything is in absolute Cartesian
    coordinates: no origin shift and no carrier phase.

    Output matches :func:`backprojection_cart_2d_tx_power` up to interpolation
    error, except near illumination edges where pixels with very few
    contributing pulses can differ. Ground-plane grid only (z = 0, per-sweep
    ``pos`` z altitude).

    Parameters
    ----------
    wa : Tensor
        Amplitude weighting of each pulse, shape [nsweeps].
    g : Tensor
        Square-root of two-way antenna gain, shape [elevation, azimuth].
    g_extent : list
        Antenna pattern extent [g_el0, g_az0, g_el1, g_az1] in radians.
    grid : CartesianGrid or dict
        Cartesian grid ``{"x": (x0, x1), "y": (y0, y1), "nx": nx, "ny": ny}``.
        x is range, y is along-track / azimuth.
    r_res : float
        Range bin resolution. Unused, kept for signature parity with
        :func:`backprojection_cart_2d_tx_power`.
    pos : Tensor
        Platform position at each pulse, shape [nsweeps, 3].
    att : Tensor
        Euler angles [roll, pitch, yaw] at each pulse, shape [nsweeps, 3].
        Pitch rotates the antenna pattern about its boresight
        (the along-track attitude angle for a side-looking antenna).
    stages : int
        Number of recursions.
    divisions : int
        Subaperture divisions per stage. Default 2.
    normalization : str or None
        "sigma", "gamma", "beta"/None or "point"; see
        :func:`backprojection_cart_2d_tx_power`.
    azimuth_resolution : bool
        If True (default), also normalize for the varying azimuth resolution.
    downsample_x, downsample_y : float
        Subaperture map step relative to the output grid step in x (range) and
        y (azimuth). The fields are smooth, so values well above 1 are usually
        fine. Default 4.
    min_nsweeps : int
        Do not recurse into subapertures with fewer pulses than this.
    min_nx, min_ny : int
        Minimum subaperture map size. Avoids large edge interpolation errors.
    margin : int
        Extra margin in subaperture map cells around the computed extents.
    beam_theta_samples : float
        Minimum number of subaperture map samples across the antenna half
        amplitude azimuth beamwidth (and a quarter of that across the elevation
        beamwidth in range). Caps ``downsample_y``/``downsample_x`` so a beam
        thin compared to the output grid step stays resolved.

    Returns
    -------
    tx_power : Tensor
        Cartesian image of square root of power returned from each pixel
        assuming constant reflectivity, shape [nx, ny].
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    x0, x1, y0, y1, nx, ny, dx, dy = unpack_cartesian_grid(grid)

    assert wa.dim() == 1, "backprojection_cart_2d_tx_power_cfbp supports only a single batch"
    nsweeps = wa.shape[0]
    assert pos.shape == (nsweeps, 3)
    assert att.shape == (nsweeps, 3)
    device = wa.device

    g_el0, g_az0, g_el1, g_az1 = g_extent
    g_daz = (g_az1 - g_az0) / g.shape[1]
    g_del = (g_el1 - g_el0) / g.shape[0]

    h_ref = float(pos[nsweeps // 2, 2])
    dx_ref = dx

    sub_dx = dx * downsample_x
    sub_dy = dy * downsample_y

    # Cap the subaperture map steps so the antenna illumination features stay
    # resolved. The illumination varies at the antenna beam scale, not the
    # output grid scale, and a beam edge undersampled by the coarse maps shows
    # up as a bilinear reconstruction lattice ("staircase") in the merged
    # result. Two ground features matter: the azimuth beamwidth ACROSS the line
    # of sight (cross-range) and the elevation beamwidth ALONG it (range). For
    # a SQUINTED beam the line of sight is at an angle psi_c to the grid axes,
    # so each feature projects onto BOTH x and y and both steps must resolve
    # it; a broadside (psi_c = 0) beam recovers the plain azimuth->y,
    # elevation->x split. Near range is used because the same angular beamwidth
    # maps to the smallest, sharpest ground feature there.
    px_mid = float(pos[nsweeps // 2, 0])
    py_mid = float(pos[nsweeps // 2, 1])
    horiz_near = min(math.hypot(xx - px_mid, yy - py_mid)
                     for xx in (x0, x1) for yy in (y0, y1))
    r_near = min(math.hypot(xx, yy) for xx in (x0, x1) for yy in (y0, y1))
    # Beam centre ground azimuth = antenna azimuth peak + mean yaw. The cross
    # line-of-sight direction is (-sin psi_c, cos psi_c), the along direction
    # (cos psi_c, sin psi_c); a step resolves a feature when its projection on
    # the feature normal is below the ground feature width / beam_theta_samples.
    mean_yaw = float(att[:, 2].mean())
    az_peak = g_az0 + (int(torch.argmax(torch.amax(g, dim=0))) + 0.5) * g_daz
    psi_c = az_peak + mean_yaw
    cpsi = abs(math.cos(psi_c))
    spsi = abs(math.sin(psi_c))
    eps = 1e-3

    p_az = torch.amax(g, dim=0)
    above = (p_az >= 0.5 * float(p_az.max())).nonzero()
    if len(above) > 0 and horiz_near > 0:
        half_width_az = float(above[-1] - above[0] + 1) * g_daz
        L_az = half_width_az * horiz_near
        sub_dy = min(sub_dy, max(dy, L_az / (beam_theta_samples * (cpsi + eps))))
        sub_dx = min(sub_dx, max(dx, L_az / (beam_theta_samples * (spsi + eps))))

    # Elevation beamwidth mapped to ground range. Only binding when the near
    # edge of the grid approaches nadir. The fields vary more slowly in range,
    # so a quarter of the azimuth sample density.
    p_el = torch.amax(g, dim=1)
    above = (p_el >= 0.5 * float(p_el.max())).nonzero()
    if len(above) > 0 and h_ref > 0:
        el_half_width = float(above[-1] - above[0] + 1) * g_del
        L_el = el_half_width * (r_near ** 2 + h_ref ** 2) / h_ref
        bts_el = beam_theta_samples / 4.0
        sub_dx = min(sub_dx, max(dx, L_el / (bts_el * (cpsi + eps))))
        sub_dy = min(sub_dy, max(dy, L_el / (bts_el * (spsi + eps))))

    # Output grid perimeter in absolute ground coordinates. Subaperture grid
    # extents are the visible subset of these points.
    nb = 64
    xs = torch.linspace(x0, x1, nb, device=device)
    ys = torch.linspace(y0, y1, nb, device=device)
    bx = torch.cat((torch.full_like(ys, x0), torch.full_like(ys, x1), xs, xs))
    by = torch.cat((ys, ys, torch.full_like(xs, y0), torch.full_like(xs, y1)))
    boundary = torch.stack((bx, by), dim=1)

    node = _cfbp_tx_power_impl(
        wa, pos, att, boundary, stages, divisions, g, g_extent, g_daz,
        normalization, sub_dx, sub_dy, margin, min_nx, min_ny, min_nsweeps,
        dx_ref, h_ref, grid_out=grid, is_top_level=True)

    if node is None:
        acc = torch.zeros((4, nx, ny), dtype=torch.float32, device=device)
    elif node[1] is not grid:
        # Single surviving subaperture: regrid it to the output grid.
        acc = cart_tx_power_merge2(node[0], None, node[1], None, grid)
    else:
        acc = node[0]

    # Finishing step. Matches the direct kernel epilogue (ground plane).
    x_vec = x0 + dx * torch.arange(nx, dtype=torch.float32, device=device)
    y_vec = y0 + dy * torch.arange(ny, dtype=torch.float32, device=device)
    Rg = torch.sqrt(x_vec[:, None] ** 2 + y_vec[None, :] ** 2)
    return _tx_power_finish(acc, Rg, azimuth_resolution)
