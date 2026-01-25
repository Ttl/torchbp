import torch
from torch import Tensor

coherence_2d_args = 7

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

    return torch.ops.torchbp.coherence_2d.default(
        img0,
        img1,
        nbatch,
        N0,
        N1,
        Navg[0],
        Navg[1],
    )


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


torch.library.register_autograd(
    "torchbp::coherence_2d",
    _backward_coherence_2d,
    setup_context=_setup_context_coherence_2d,
)
