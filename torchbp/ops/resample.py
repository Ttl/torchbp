import torch
from torch import Tensor


def resample_2d_lanczos(
    img: Tensor,
    shift_r: Tensor,
    shift_az: Tensor,
    order: int = 6,
) -> Tensor:
    """Resample a 2D image using Lanczos interpolation with a per-pixel shift field.

    Each output pixel ``(i, j)`` reads from the input at
    ``(i + shift_r[i, j], j + shift_az[i, j])`` using a Lanczos kernel
    of the given order.

    Parameters
    ----------
    img : Tensor
        Input image. Shape ``[Nr, Naz]`` or ``[nbatch, Nr, Naz]``.
        Supports complex64 and float32.
    shift_r : Tensor float32 ``[Nr, Naz]``
        Per-pixel shift in the first (range) dimension.
    shift_az : Tensor float32 ``[Nr, Naz]``
        Per-pixel shift in the second (azimuth) dimension.
    order : int
        Lanczos kernel order (2–8). Default 6.

    Returns
    -------
    Tensor
        Resampled image, same shape and dtype as *img*.
    """
    squeeze = False
    if img.dim() == 2:
        img = img.unsqueeze(0)
        squeeze = True

    nbatch = img.shape[0]
    Nr = img.shape[1]
    Naz = img.shape[2]

    out = torch.ops.torchbp.resample_2d_lanczos.default(
        img, shift_r, shift_az, nbatch, Nr, Naz, order)

    if squeeze:
        out = out.squeeze(0)
    return out


def resample_2d_knab(
    img: Tensor,
    shift_r: Tensor,
    shift_az: Tensor,
    order: int = 6,
    oversample: float = 1.5,
) -> Tensor:
    """Resample a 2D image using Knab interpolation with a per-pixel shift field.

    The Knab kernel uses a Kaiser-Bessel window matched to the signal
    oversampling ratio, giving better accuracy than Lanczos when the
    oversampling is known.

    Each output pixel ``(i, j)`` reads from the input at
    ``(i + shift_r[i, j], j + shift_az[i, j])``.

    Parameters
    ----------
    img : Tensor
        Input image. Shape ``[Nr, Naz]`` or ``[nbatch, Nr, Naz]``.
        Supports complex64 and float32.
    shift_r : Tensor float32 ``[Nr, Naz]``
        Per-pixel shift in the first (range) dimension.
    shift_az : Tensor float32 ``[Nr, Naz]``
        Per-pixel shift in the second (azimuth) dimension.
    order : int
        Kernel order (2–8). Default 6.
    oversample : float
        Signal oversampling ratio. Default 1.5.

    Returns
    -------
    Tensor
        Resampled image, same shape and dtype as *img*.
    """
    squeeze = False
    if img.dim() == 2:
        img = img.unsqueeze(0)
        squeeze = True

    nbatch = img.shape[0]
    Nr = img.shape[1]
    Naz = img.shape[2]

    out = torch.ops.torchbp.resample_2d_knab.default(
        img, shift_r, shift_az, nbatch, Nr, Naz, order, oversample)

    if squeeze:
        out = out.squeeze(0)
    return out


@torch.library.register_fake("torchbp::resample_2d_lanczos")
def _fake_resample_2d_lanczos(
    img: Tensor,
    shift_r: Tensor,
    shift_az: Tensor,
    nbatch: int,
    Nr: int,
    Naz: int,
    order: int,
) -> Tensor:
    torch._check(img.dtype == torch.complex64 or img.dtype == torch.float32)
    return torch.empty((nbatch, Nr, Naz), dtype=img.dtype, device=img.device)


@torch.library.register_fake("torchbp::resample_2d_knab")
def _fake_resample_2d_knab(
    img: Tensor,
    shift_r: Tensor,
    shift_az: Tensor,
    nbatch: int,
    Nr: int,
    Naz: int,
    order: int,
    oversample: float,
) -> Tensor:
    torch._check(img.dtype == torch.complex64 or img.dtype == torch.float32)
    return torch.empty((nbatch, Nr, Naz), dtype=img.dtype, device=img.device)
