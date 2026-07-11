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


def _resample_1d(op, x: Tensor, num: int, axis: int, *extra) -> Tensor:
    """Shared driver for the 1D resample ops. Moves *axis* to the last
    position, folds the rest into a batch dimension, calls *op*, and restores
    the original layout."""
    num = int(num)
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")

    x_moved = x.movedim(axis, -1)
    moved_shape = x_moved.shape
    N = moved_shape[-1]

    x2d = x_moved.reshape(-1, N)
    nbatch = x2d.shape[0]

    out = op(x2d, nbatch, N, num, *extra)

    out = out.reshape(*moved_shape[:-1], num)
    return out.movedim(-1, axis)


def resample_1d_lanczos(
    x: Tensor,
    num: int,
    axis: int = -1,
    order: int = 6,
) -> Tensor:
    """Resample a signal to *num* samples using Lanczos interpolation.

    Changes the number of samples along *axis* to *num*, similar to
    ``scipy.signal.resample(x, num)`` but implemented as a zero-phase
    windowed-sinc interpolation in the sample domain. ``num == N`` is the
    identity, ``num > N`` upsamples (fractional interpolation), and ``num < N``
    decimates. When decimating, the kernel is lowpassed to the output Nyquist
    frequency to suppress aliasing.

    Output element ``k`` reads the input at continuous position ``k * N / num``,
    where ``N`` is the input length along *axis*. Any number of leading/trailing
    dimensions is supported; they are processed independently.

    Parameters
    ----------
    x : Tensor
        Input signal. Any shape. Supports complex64 and float32.
    num : int
        Number of output samples along *axis*. Must be >= 1.
    axis : int
        Axis to resample along. Default -1.
    order : int
        Lanczos kernel order (2–8). Default 6.

    Returns
    -------
    Tensor
        Resampled signal, same dtype as *x*, with length *num* along *axis*.
    """
    return _resample_1d(
        torch.ops.torchbp.resample_1d_lanczos.default, x, num, axis, order)


def resample_1d_knab(
    x: Tensor,
    num: int,
    axis: int = -1,
    order: int = 6,
    oversample: float = 1.5,
) -> Tensor:
    """Resample a signal to *num* samples using Knab interpolation.

    Like :func:`resample_1d_lanczos` but uses a Knab kernel with a Kaiser-Bessel
    window matched to the signal oversampling ratio, giving better accuracy than
    Lanczos when the oversampling is known. See :func:`resample_1d_lanczos` for
    the *num*, *axis*, and decimation semantics.

    Parameters
    ----------
    x : Tensor
        Input signal. Any shape. Supports complex64 and float32.
    num : int
        Number of output samples along *axis*. Must be >= 1.
    axis : int
        Axis to resample along. Default -1.
    order : int
        Kernel order (2–8). Default 6.
    oversample : float
        Signal oversampling ratio. Default 1.5.

    Returns
    -------
    Tensor
        Resampled signal, same dtype as *x*, with length *num* along *axis*.
    """
    return _resample_1d(
        torch.ops.torchbp.resample_1d_knab.default, x, num, axis, order, oversample)


@torch.library.register_fake("torchbp::resample_1d_lanczos")
def _fake_resample_1d_lanczos(
    img: Tensor,
    nbatch: int,
    N: int,
    M: int,
    order: int,
) -> Tensor:
    torch._check(img.dtype == torch.complex64 or img.dtype == torch.float32)
    return torch.empty((nbatch, M), dtype=img.dtype, device=img.device)


@torch.library.register_fake("torchbp::resample_1d_knab")
def _fake_resample_1d_knab(
    img: Tensor,
    nbatch: int,
    N: int,
    M: int,
    order: int,
    oversample: float,
) -> Tensor:
    torch._check(img.dtype == torch.complex64 or img.dtype == torch.float32)
    return torch.empty((nbatch, M), dtype=img.dtype, device=img.device)


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
