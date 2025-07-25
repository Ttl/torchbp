import torch
from torch import Tensor


def ainsworth(
    sar_img: Tensor, k: complex = 1, pol_order: list = ["VV", "VH", "HV", "HH"],
    max_iters: int = 50, epsilon: float = 1e-6) -> Tensor:
    """
    Solve for polarimetric calibration from fully-polarimetric data. [1]_

    Parameters
    ----------
    sar_img : Tensor
        Input SAR image. Shape should be [4, M, N].
    k : complex
        HH/VV calibration factor.
    pol_order : list
        Order of polarizations in the SAR image.
    max_iters : int
        Maximum number of optimization iterations.
    epsilon: float
        Optimization termination threshold.

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
    c = torch.zeros((4, 4), dtype=torch.complex128, device="cpu")
    for i in range(4):
        for j in range(4):
            v = sar_img[permutation[i]] * sar_img[permutation[j]].conj()
            c[i, j] = torch.nanmean(v).to(device="cpu", dtype=torch.complex128)

    # Derivation assumes this polarization ordering
    hh = 0
    hv = 1
    vh = 2
    vv = 3

    alpha = torch.abs(c[vh, vh] / c[hv, hv]) ** 0.25 * torch.exp(
        1j * torch.angle(c[vh, hv]) / 2
    )
    alpha = alpha.to(torch.complex128)

    k = torch.ones_like(alpha)
    uvwz = torch.zeros(4, dtype=torch.complex128, device=c.device)
    epsilon = 1e-6

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

        # sigma2 = torch.linalg.inv(crosstalk) @ sigma @ torch.linalg.inv(crosstalk.conj())
        sigma2 = torch.linalg.solve(
            crosstalk, torch.linalg.solve(crosstalk.conj().T, sigma.T).T
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

        # c = M @ sigma @ M.conj(), solve for sigma
        sigma = torch.linalg.solve(M, torch.linalg.solve(M.conj().T, c.T).T)
        # sigma = torch.linalg.inv(M) @ c @ torch.linalg.inv(M.conj())

        if torch.abs(alpha2 - 1) < epsilon:
            break

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
        4x4 polarimetric calibration matrix.

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
