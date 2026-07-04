import torch
from torch import Tensor
from typing import TYPE_CHECKING
from copy import deepcopy

from .backproj import backprojection_cart_2d
from ..util import center_pos
from ._utils import unpack_cartesian_grid

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
) -> Tensor:
    """
    Cartesian factorized backprojection.

    Factorized backprojection directly on a Cartesian grid. The aperture is
    recursively split into subapertures which are backprojected with
    :func:`backprojection_cart_2d` onto the full output grid extent, coarsely
    sampled in the cross-range (y) dimension. Subaperture images are
    demodulated with the carrier referenced to the subaperture center, which
    makes them bandlimited so that merging only needs exact FFT upsampling
    along y and a phase re-reference. The output matches
    ``backprojection_cart_2d(data, grid, ...)`` up to interpolation error.

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
        cropped from the output. Absorbs the periodic wrap-around of the FFT
        interpolation at the y edges of the image.
    beamwidth : float
        Beamwidth of the antenna in radians. Passed to
        :func:`backprojection_cart_2d`.
    data_fmod : float
        Range modulation frequency applied to input data.

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
    """
    if hasattr(grid, "to_dict"):
        grid = grid.to_dict()
    if data.dim() != 2:
        raise ValueError("data shape should be [nsweeps, samples]")
    if pos.dim() != 2 or pos.shape[0] != data.shape[0]:
        raise ValueError("pos shape should be [nsweeps, 3]")

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
        oversample_y, beamwidth, data_fmod, keff
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
            data, grid_b, fc, r_res, pos, s, divisions, d0,
            oversample_y, guard_y, beamwidth, data_fmod
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
