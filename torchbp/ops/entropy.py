import torch
from torch import Tensor

entropy_args = 3
abs_sum_args = 2


def _prepare_entropy_args(img: Tensor) -> tuple:
    """Prepare arguments for C++ entropy and abs_sum operators.

    Returns tuple of (img, nbatch) for abs_sum call and (img, norm, nbatch) for entropy call.
    Used internally by entropy and for testing.
    """
    if img.dim() == 3:
        nbatch = img.shape[0]
    else:
        nbatch = 1

    return (img, nbatch)


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
    img_arg, nbatch = _prepare_entropy_args(img)
    norm = torch.ops.torchbp.abs_sum.default(img_arg, nbatch)
    x = torch.ops.torchbp.entropy.default(img_arg, norm, nbatch)
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


@torch.library.register_fake("torchbp::abs_sum")
def _fake_abs_sum(img: Tensor, nbatch: int) -> Tensor:
    torch._check(img.dtype == torch.complex64)
    return torch.empty((nbatch, 1), dtype=torch.float32, device=img.device)


@torch.library.register_fake("torchbp::abs_sum_grad")
def _fake_abs_sum_grad(data: Tensor, grad: Tensor, nbatch: int) -> Tensor:
    torch._check(data.dtype == torch.complex64)
    if data.requires_grad:
        return torch.empty_like(data)
    else:
        return None


@torch.library.register_fake("torchbp::entropy")
def _fake_entropy(img: Tensor, norm: Tensor, nbatch: int) -> Tensor:
    torch._check(img.dtype == torch.complex64)
    torch._check(norm.dtype == torch.float32)
    return torch.empty((nbatch,), dtype=torch.float32, device=img.device)


@torch.library.register_fake("torchbp::entropy_grad")
def _fake_entropy_grad(data: Tensor, norm: Tensor, grad: Tensor, nbatch: int) -> Tensor:
    torch._check(data.dtype == torch.complex64)
    ret = []
    if data.requires_grad:
        ret.append(torch.empty_like(data))
    else:
        ret.append(None)
    return ret


torch.library.register_autograd(
    "torchbp::entropy", _backward_entropy, setup_context=_setup_context_entropy
)
torch.library.register_autograd(
    "torchbp::abs_sum", _backward_abs_sum, setup_context=_setup_context_abs_sum
)
