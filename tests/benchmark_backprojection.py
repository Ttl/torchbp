#!/usr/bin/env python
import torch
import torchbp
import time
import numpy as np
import torch.utils.benchmark as benchmark


def main(device):
    nbatch = 1
    nr = 1024
    ntheta = 1024
    nsweeps = 16 if device == "cpu" else 1024
    nsamples = 1024
    data_dtype = torch.complex64

    fc = 6e9
    r_res = 0.5

    grid_polar = {"r": (10, 500), "theta": (-1, 1), "nr": nr, "ntheta": ntheta}

    data = torch.randn((nbatch, nsweeps, nsamples), dtype=data_dtype, device=device)

    pos = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, :, 1] = (
        0.25
        * 3e8
        / fc
        * (torch.arange(nsweeps, dtype=torch.float32, device=device) - nsweeps / 2)
    )

    pos.requires_grad = True

    backprojs = nbatch * nr * ntheta * nsweeps

    iterations = 10

    tf = benchmark.Timer(
        stmt="torchbp.ops.backprojection_polar_2d(data, grid_polar, fc, r_res, pos)",
        setup="import torchbp",
        globals={
            "data": data,
            "grid_polar": grid_polar,
            "fc": fc,
            "r_res": r_res,
            "pos": pos,
        },
    )

    tb = benchmark.Timer(
        stmt="torch.mean(torch.abs(torchbp.ops.backprojection_polar_2d(data, grid_polar, fc, r_res, pos))).backward()",
        setup="import torchbp; ",
        globals={
            "data": data,
            "grid_polar": grid_polar,
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
    main("cuda")
    main("cpu")
