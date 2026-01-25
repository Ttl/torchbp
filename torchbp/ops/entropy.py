import torch
from torch import Tensor

entropy_args = 3
abs_sum_args = 2


def entropy(img: Tensor) -> Tensor:
    """
    Calculates entropy of:

    -sum(y*log(y))

    , where y = abs(x) / sum(abs(x)).

    Uses less memory than pytorch implementation when used in optimization.

    Parameters
    ----------
    img : Tensor
        2D radar image in [range, angle] format. Dimensions should match with grid_polar grid.
        [nbatch, range, angle] if interpolating multiple images at the same time.

    Returns
    -------
    out : Tensor
        Interpolated radar image.
    """

    if img.dim() == 3:
        nbatch = img.shape[0]
    else:
        nbatch = 1

    norm = torch.ops.torchbp.abs_sum.default(img, nbatch)
    x = torch.ops.torchbp.entropy.default(img, norm, nbatch)
    if nbatch == 1:
        return x.squeeze(0)
    return x


def _backward_entropy(ctx, grad):
    data, norm = ctx.saved_tensors
    ret = torch.ops.torchbp.entropy_grad.default(data, norm, grad, *ctx.saved)
    grads = [None] * entropy_args
    grads[: len(ret)] = ret
    return tuple(grads)


def _setup_context_entropy(ctx, inputs, output):
    data, norm, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data, norm)


def _backward_abs_sum(ctx, grad):
    data = ctx.saved_tensors[0]
    ret = torch.ops.torchbp.abs_sum_grad.default(data, grad, *ctx.saved)
    grads = [None] * abs_sum_args
    grads[0] = ret
    return tuple(grads)


def _setup_context_abs_sum(ctx, inputs, output):
    data, *rest = inputs
    ctx.saved = rest
    ctx.save_for_backward(data)


torch.library.register_autograd(
    "torchbp::entropy", _backward_entropy, setup_context=_setup_context_entropy
)
torch.library.register_autograd(
    "torchbp::abs_sum", _backward_abs_sum, setup_context=_setup_context_abs_sum
)
