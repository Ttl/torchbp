#!/usr/bin/env python
import torch
import torchbp
import time
import numpy as np
import torch.utils.benchmark as benchmark


def main(device):
    nbatch = 1
    nx = 1024
    ny = 1024
    nsweeps = 16 if device == "cpu" else 1024
    nsamples = 1024
    data_dtype = torch.complex64

    fc = 6e9
    r_res = 0.5

    grid_cart = {"x": (10, 400), "y": (-200, 200), "nx": nx, "ny": ny}

    data = torch.randn((nbatch, nsweeps, nsamples), dtype=data_dtype, device=device)

    pos = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, :, 1] = (
        0.25
        * 3e8
        / fc
        * (torch.arange(nsweeps, dtype=torch.float32, device=device) - nsweeps / 2)
    )

    pos.requires_grad = True

    backprojs = nbatch * nx * ny * nsweeps

    iterations = 10

    tf = benchmark.Timer(
        stmt="torchbp.ops.backprojection_cart_2d(data, grid_cart, fc, r_res, pos)",
        setup="import torchbp",
        globals={
            "data": data,
            "grid_cart": grid_cart,
            "fc": fc,
            "r_res": r_res,
            "pos": pos,
        },
    )

    tb = benchmark.Timer(
        stmt="torch.mean(torch.abs(torchbp.ops.backprojection_cart_2d(data, grid_cart, fc, r_res, pos))).backward()",
        setup="import torchbp; ",
        globals={
            "data": data,
            "grid_cart": grid_cart,
            "fc": fc,
            "r_res": r_res,
            "pos": pos,
        },
    )

    f = tf.timeit(iterations).median
    print(f"Device {device}, Forward: {backprojs / f:.3g} backprojections/s")
    b = tb.timeit(iterations).median
    print(f"Device {device}, Backward: {backprojs / (b - f):.3g} backprojections/s")


if __name__ == "__main__":
    if torch.cuda.is_available():
        main("cuda")
    main("cpu")
