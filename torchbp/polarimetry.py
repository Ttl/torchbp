import torch
from torch import Tensor
from math import sin, cos


def correlation_matrix(
    sar_img: Tensor,
    weight: Tensor | None = None,
    pol_order: list = ["VV", "VH", "HV", "HH"],
    output_order: list = ["HH", "HV", "VH", "VV"],
    device=None,
    dtype=None,
) -> Tensor:
    """
    Calculate polarimetric correlation matrix.

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [C, M, N], where C is the number of polarizations.
    weight : Tensor or None
        Weight for correlation matrix calculation, should have shape [M, N].
    pol_order : list
        Order of polarizations in the SAR image.
    output_order : list
        Order of polarizations in the correlation matrix.
    device : str
        Pytorch device.
    dtype : torch.dtype
        Correlation matrix dtype. Normally either `torch.complex64` or `torch.complex128`.

    Returns
    ----------
    c : Tensor
        Correlation matrix.
    """
    if device is None:
        device = sar_img.device
    if dtype is None:
        dtype = sar_img.dtype

    permutation = []
    ch = sar_img.shape[0]
    if ch != len(pol_order) != len(output_order):
        raise ValueError(f"SAR image has {ch} channels which doesn't match pol_order and output_order parameters.")

    for i in range(ch):
        permutation.append(pol_order.index(output_order[i]))

    c = torch.zeros((ch, ch), dtype=dtype, device=device)
    if weight is not None:
        if weight.shape != sar_img.shape[1:]:
            raise ValueError(f"Invalid weight shape {weight.shape}, expected {sar_img.shape[1:]}")
        weight_mean = torch.mean(weight).item()
    for i in range(ch):
        for j in range(ch):
            v = sar_img[permutation[i]] * sar_img[permutation[j]].conj()
            if weight is not None:
                v *= weight / weight_mean
                v *= weight / weight_mean
            if device == torch.device("cpu"):
                c[i, j] = torch.mean(torch.nan_to_num(v)).to(device=device, dtype=dtype)
            else:
                c[i, j] = torch.nanmean(v).to(device=device, dtype=dtype)
    return c


def k_alpha_cal(
    sar_img: Tensor,
    weight: Tensor | None = None,
    alpha: complex = None,
    k: complex = 1,
    pol_order: list = ["VV", "VH", "HV", "HH"],
    corner_hh_vv: complex | None = None,
) -> Tensor:
    """
    Polarimetric calibration assuming zero crosstalk.

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [4, M, N].
    weight : Tensor or None
        Weight for correlation matrix calculation, should have shape [M, N].
    alpha : complex or None
      `sqrt(RXVV * TXHH / (RXHH * TXVV))`. Estimated from the data if `alpha` is None.
    k : complex
        RXHH/RXVV calibration factor.
    pol_order : list
        Order of polarizations in the SAR image.
    corner_hh_vv : complex or None
        Measured HH/VV ratio of corner reflector.
        Used to solve for k if not None.

    Returns
    ----------
    Minv : Tensor
        Normalized polarimetric calibration matrix.
    """
    order = ["HH", "HV", "VH", "VV"]
    permutation = []
    for i in range(4):
        permutation.append(pol_order.index(order[i]))

    if alpha is None:
        # Correlation matrix
        c = correlation_matrix(sar_img, weight=weight, pol_order=pol_order, output_order=order)

        # Derivation assumes this polarization ordering
        hv = 1
        vh = 2

        alpha = torch.abs(c[vh, vh] / c[hv, hv]) ** 0.25 * torch.exp(
            1j * torch.angle(c[vh, hv]) / 2
        )
        alpha = alpha.item()

    if corner_hh_vv is not None:
        k_guess = (corner_hh_vv / alpha**2) ** 0.5
        if abs(k - k_guess) < abs(k + k_guess):
            k = k_guess
        else:
            k = -k_guess

    M = torch.tensor(
        [
            [k * alpha, 0, 0, 0],
            [0, 1 / alpha, 0, 0],
            [0, 0, alpha, 0],
            [0, 0, 0, 1 / (k * alpha)],
        ],
        device=sar_img.device,
    )

    # Permutate to original channel order
    M = M[permutation, :][:, permutation]

    Minv = torch.linalg.inv(M).to(dtype=torch.complex64, device=sar_img.device)
    return Minv


def ainsworth(
    sar_img: Tensor,
    weight: Tensor | None = None,
    k: complex = 1,
    pol_order: list = ["VV", "VH", "HV", "HH"],
    max_iters: int = 50,
    epsilon: float = 1e-6,
    corner_hh_vv: complex | None = None,
) -> Tensor:
    """
    Solve for polarimetric calibration from fully-polarimetric data. [1]_

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [4, M, N].
    weight : Tensor or None
        Weight for correlation matrix calculation, should have shape [M, N].
    k : complex
        HH/VV calibration factor.
    pol_order : list
        Order of polarizations in the SAR image.
    max_iters : int
        Maximum number of optimization iterations.
    epsilon: float
        Optimization termination threshold.
    corner_hh_vv : complex or None
        Measured HH/VV ratio of corner reflector.
        Used to solve for k if not None.

    References
    ----------
    .. [1] T. L. Ainsworth, L. Ferro-Famil and Jong-Sen Lee, "Orientation angle
    preserving a posteriori polarimetric SAR calibration," in IEEE Transactions
    on Geoscience and Remote Sensing, vol. 44, no. 4, pp. 994-1003, April 2006.

    Returns
    ----------
    Minv : Tensor
        Normalized polarimetric calibration matrix.
    """
    order = ["HH", "HV", "VH", "VV"]
    permutation = []
    for i in range(4):
        permutation.append(pol_order.index(order[i]))

    # Correlation matrix
    c = correlation_matrix(sar_img, weight=weight, pol_order=pol_order, output_order=order)

    # Derivation assumes this polarization ordering
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    alpha = torch.abs(c[vh, vh] / c[hv, hv]) ** 0.25 * torch.exp(
        1j * torch.angle(c[vh, hv]) / 2
    )
    alpha = alpha.to(dtype=c.dtype)

    if corner_hh_vv is not None:
        k_guess = (corner_hh_vv / alpha**2) ** 0.5
        if abs(k - k_guess) < abs(k + k_guess):
            k = k_guess
        else:
            k = -k_guess

    k = k * torch.ones_like(alpha)
    uvwz = torch.zeros(4, dtype=c.dtype, device=c.device)

    sigma = torch.tensor(
        [
            [
                c[hh, hh] / (torch.abs(k) ** 2 * torch.abs(alpha) ** 2),
                c[hh, hv] * alpha.conj() / (k * alpha),
                c[hh, vh] / (k * torch.abs(alpha) ** 2),
                c[hh, vv] * k.conj() * alpha.conj() / (k * alpha),
            ],
            [
                c[hv, hh] * alpha / (k.conj() * alpha.conj()),
                c[hv, hv] * torch.abs(alpha) ** 2,
                c[hv, vh] * alpha / alpha.conj(),
                c[hv, vv] * k.conj() * torch.abs(alpha) ** 2,
            ],
            [
                c[vh, hh] / (k.conj() * torch.abs(alpha) ** 2),
                c[vh, hv] * alpha.conj() / alpha,
                c[vh, vh] / torch.abs(alpha) ** 2,
                c[vh, vv] * k.conj() * alpha.conj() / alpha,
            ],
            [
                c[vv, hh] * k * alpha / (k.conj() * alpha.conj()),
                c[vv, hv] * k * torch.abs(alpha) ** 2,
                c[vv, vh] * k * alpha / alpha.conj(),
                c[vv, vv] * torch.abs(k) ** 2 * torch.abs(alpha) ** 2,
            ],
        ],
        device=c.device,
    )

    for i in range(max_iters):
        A = 0.5 * (sigma[hv, hh] + sigma[vh, hh])
        B = 0.5 * (sigma[hv, vv] + sigma[vh, vv])

        X = torch.tensor(
            [
                [sigma[hv, hh] - A],
                [sigma[vh, hh] - A],
                [sigma[hv, vv] - B],
                [sigma[vh, vv] - B],
            ],
            device=c.device,
        )

        zeta = torch.tensor(
            [
                [0, 0, sigma[vv, hh], sigma[hh, hh]],
                [sigma[hh, hh], sigma[vv, hh], 0, 0],
                [0, 0, sigma[vv, vv], sigma[hh, vv]],
                [sigma[hh, vv], sigma[vv, vv], 0, 0],
            ],
            device=c.device,
        )

        tau = torch.tensor(
            [
                [0, sigma[hv, hv], sigma[hv, vh], 0],
                [0, sigma[vh, hv], sigma[vh, vh], 0],
                [sigma[hv, hv], 0, 0, sigma[hv, vh]],
                [sigma[vh, hv], 0, 0, sigma[vh, vh]],
            ],
            device=c.device,
        )

        Xr = torch.cat([X.real, X.imag], dim=0)

        zeta_tau = torch.cat(
            [
                torch.cat([torch.real(zeta + tau), -torch.imag(zeta - tau)], dim=1),
                torch.cat([torch.imag(zeta + tau), torch.real(zeta - tau)], dim=1),
            ],
            dim=0,
        )

        delta = torch.linalg.solve(zeta_tau, Xr)
        delta = delta[:4] + 1j * delta[4:]

        uvwz = uvwz + delta.squeeze()
        u = uvwz[0]
        v = uvwz[1]
        w = uvwz[2]
        z = uvwz[3]

        crosstalk = torch.tensor(
            [[1, v, w, v * w], [z, 1, w * z, w], [u, u * v, 1, v], [u * z, u, z, 1]],
            device=c.device,
        )

        # sigma2 = torch.linalg.inv(crosstalk) @ sigma @ torch.linalg.inv(crosstalk.conj().T)
        sigma2 = torch.linalg.solve(
            crosstalk, torch.linalg.solve(crosstalk.conj(), sigma.T).T
        )

        alpha2 = torch.abs(sigma2[vh, vh] / sigma2[hv, hv]) ** 0.25 * torch.exp(
            1j * torch.angle(sigma2[vh, hv]) / 2
        )

        a1 = torch.diag(
            torch.tensor([alpha, 1 / alpha, alpha, 1 / alpha], device=c.device)
        )
        a2 = torch.diag(
            torch.tensor([alpha2, 1 / alpha2, alpha2, 1 / alpha2], device=c.device)
        )

        M = a1 @ crosstalk @ a2

        alpha = alpha * alpha2

        # c = M @ sigma @ M.conj().T, solve for sigma
        sigma = torch.linalg.solve(M, torch.linalg.solve(M.conj(), c.T).T)
        # sigma = torch.linalg.inv(M) @ c @ torch.linalg.inv(M.conj().transpose())

        if torch.abs(alpha2 - 1) < epsilon:
            break

    u = uvwz[0]
    v = uvwz[1]
    w = uvwz[2]
    z = uvwz[3]

    if corner_hh_vv is not None:
        k_guess = ((-corner_hh_vv + v * w) / (alpha**2 * (corner_hh_vv * u * z - 1))) ** 0.5

        if abs(k - k_guess) < abs(k + k_guess):
            k = k_guess
        else:
            k = -k_guess
    print("k", torch.abs(k), torch.angle(k))

    M = torch.tensor(
        [
            [k * alpha, v / alpha, w * alpha, v * w / (k * alpha)],
            [z * k * alpha, 1 / alpha, w * z * alpha, w / (k * alpha)],
            [u * k * alpha, u * v / alpha, alpha, v / (k * alpha)],
            [u * z * k * alpha, u / alpha, z * alpha, 1 / (k * alpha)],
        ],
        device=c.device,
    )

    # Normalize M
    M = M / torch.linalg.det(M) ** 0.25
    # Permutate to original channel order
    M = M[permutation, :][:, permutation]

    Minv = torch.linalg.inv(M).to(dtype=torch.complex64, device=sar_img.device)
    return Minv


def apply_cal(sar_img: Tensor, cal: Tensor) -> Tensor:
    """
    Apply polarimetric calibration matrix to SAR image.

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [4, M, N].
    cal : Tensor
        4x4 polarimetric calibration correction matrix.

    Returns
    ----------
    caled : Tensor
        Calibrated SAR image.
    """

    s = sar_img.shape
    if len(s) != 3:
        raise ValueError("sar_img should be 3D")
    if s[0] != 4:
        raise ValueError("sar_img should have 4 polarimetric channels")
    img_reshaped = sar_img.view(4, -1)
    caled = cal @ img_reshaped
    caled = caled.view(*s)
    return caled


def pol_antenna_rotation(sar_img: Tensor, theta: float, pol_order: list = ["VV", "VH", "HV", "HH"]) -> Tensor:
    """
    Calculate polarimetric SAR image with antenna rotated by angle theta
    around the antenna axis.

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [3, M, N] or [4, M, N].
    theta : float
        Rotation angle.
    pol_order : list
        Order of polarizations in the SAR image.
        Use "HV" for cross-polarized channel if image has three polarizations.

    References
    ----------
    .. [1] K. Sarabandi and F. T. Ulaby, "A convenient technique for
    polarimetric calibration of single-antenna radar systems," in IEEE
    Transactions on Geoscience and Remote Sensing, vol. 28, no. 6, pp.
    1022-1033, Nov. 1990.

    Returns
    ----------
    sar_img : Tensor
        SAR image after rotation.
    """
    if theta == 0:
        return sar_img

    p = []
    ch = sar_img.shape[0]
    if ch not in [3, 4]:
        raise ValueError(f"SAR image has {ch} channels which doesn't match pol_order.")

    cal_order = ["HH", "HV", "VH", "VV"]
    if ch == 3:
        cal_order = ["HH", "HV", "VV"]
    for i in range(ch):
        p.append(pol_order.index(cal_order[i]))

    shh = sar_img[p[0]]
    shv = sar_img[p[1]]
    svv = sar_img[p[-1]]
    if ch == 3:
        svh = shv
    else:
        svh = sar_img[p[2]]

    cost = cos(theta)
    sint = sin(theta)
    out_img = torch.empty_like(sar_img)
    # HH
    out_img[p[0]] = shh * cost**2 - (shv + svh)*cost*sint + svv*sint**2
    # HV
    out_img[p[1]] = shv * cost**2  + (shh - svv)*cost*sint - svh*sint**2
    if ch == 4:
        # VH
        out_img[p[2]] = svh * cost**2  + (shh - svv)*cost*sint - shv*sint**2
    # VV
    out_img[p[-1]] = svv * cost**2 - (shv + svh)*cost*sint + shh*sint**2
    return out_img
