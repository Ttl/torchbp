import torch
from torch import Tensor

coherence_2d_args = 7

def _prepare_coherence_2d_args(img0: Tensor, img1: Tensor, Navg: tuple) -> tuple:
    """Prepare arguments for C++ coherence_2d operator.

    Returns tuple of arguments matching C++ operator signature.
    Used internally by coherence_2d and for testing.
    """
    if img0.shape != img1.shape:
        raise ValueError(f"img0.shape != img1.shape. {img0.shape} != {img1.shape}")
    if img0.dim() == 3:
        nbatch = img0.shape[0]
        N0 = img0.shape[1]
        N1 = img0.shape[2]
    elif img0.dim() == 2:
        nbatch = 1
        N0 = img0.shape[0]
        N1 = img0.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img0.shape}")

    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")

    return (img0, img1, nbatch, N0, N1, Navg[0], Navg[1])


def coherence_2d(img0: Tensor, img1: Tensor, Navg: tuple) -> Tensor:
    """
    Coherence of two complex images over moving window `Navg`.

    Parameters
    ----------
    img0 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    img1 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).

    Returns
    -------
    out : Tensor
        Real valued coherence image with same shape as input calculated over the
        moving window.
    """
    cpp_args = _prepare_coherence_2d_args(img0, img1, Navg)
    return torch.ops.torchbp.coherence_2d.default(*cpp_args)


def power_coherence_2d(
    img0: Tensor, img1: Tensor, Navg: tuple, corr_output: bool = True
) -> Tensor:
    """
    Coherence of two complex images over moving window `Navg`. Calculated from
    squared absolute value of the images. [1]_

    Parameters
    ----------
    img0 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    img1 : Tensor
        Complex valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).
    corr_output : bool
        Return ordinary correlation coefficient by calculating sqrt(2*v-1) for
        all output values if v > 0.5 and else 0.

    References
    ----------
    .. [1] A. M. Guarnieri and C. Prati, "SAR interferometry: a "Quick and
        dirty" coherence estimator for data browsing," in IEEE Transactions on
        Geoscience and Remote Sensing, vol. 35, no. 3, pp. 660-669, May 1997.

    Returns
    -------
    out : Tensor
        Real valued coherence image with same shape as input calculated over the
        moving window.
    """
    if img0.shape != img1.shape:
        raise ValueError(f"img0.shape != img1.shape. {img0.shape} != {img1.shape}")
    if img0.dim() == 3:
        nbatch = img0.shape[0]
        N0 = img0.shape[1]
        N1 = img0.shape[2]
    elif img0.dim() == 2:
        nbatch = 1
        N0 = img0.shape[0]
        N1 = img0.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img0.shape}")

    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")

    return torch.ops.torchbp.power_coherence_2d.default(
        img0, img1, nbatch, N0, N1, Navg[0], Navg[1], corr_output
    )


def _backward_coherence_2d(ctx, grad):
    img0, img1 = ctx.saved_tensors
    ret = torch.ops.torchbp.coherence_2d_grad.default(grad, img0, img1, *ctx.saved)
    grads = [None] * coherence_2d_args
    grads[: len(ret)] = ret
    return tuple(grads)


def _setup_context_coherence_2d(ctx, inputs, output):
    img0, img1, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(img0, img1)


@torch.library.register_fake("torchbp::coherence_2d")
def _fake_coherence_2d(
    img0: Tensor,
    img1: Tensor,
    nbatch: int,
    N0: int,
    N1: int,
    w0: int,
    w1: int,
) -> Tensor:
    torch._check(img0.dtype == torch.complex64)
    torch._check(img1.dtype == torch.complex64)
    if nbatch == 1:
        return torch.empty((N0, N1), dtype=torch.float32, device=img0.device)
    else:
        return torch.empty((nbatch, N0, N1), dtype=torch.float32, device=img0.device)


@torch.library.register_fake("torchbp::coherence_2d_grad")
def _fake_coherence_2d_grad(
    grad: Tensor,
    img0: Tensor,
    img1: Tensor,
    nbatch: int,
    N0: int,
    N1: int,
    w0: int,
    w1: int,
) -> Tensor:
    torch._check(img0.dtype == torch.complex64)
    torch._check(img1.dtype == torch.complex64)
    ret = []
    if img0.requires_grad:
        ret.append(torch.empty_like(img0))
    else:
        ret.append(None)
    if img1.requires_grad:
        ret.append(torch.empty_like(img1))
    else:
        ret.append(None)
    return ret


torch.library.register_autograd(
    "torchbp::coherence_2d",
    _backward_coherence_2d,
    setup_context=_setup_context_coherence_2d,
)
