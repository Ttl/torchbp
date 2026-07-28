#!/usr/bin/env python
import torch
import torchbp
import torch.utils.benchmark as benchmark


def main(device, compare_direct=True):
    nbatch = 1
    nx = 512
    ny = 512
    nsweeps = 32 if device == "cpu" else 256
    sweep_samples = 512

    fc = 6e9
    bw = 100e6
    tsweep = 100e-6
    fs = sweep_samples / tsweep
    gamma = bw / tsweep

    grid_cart = {"x": (10, 200), "y": (-100, 100), "nx": nx, "ny": ny}

    img = torch.randn((nbatch, nx, ny), dtype=torch.complex64, device=device)

    pos = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, :, 1] = (
        0.25
        * 3e8
        / fc
        * (torch.arange(nsweeps, dtype=torch.float32, device=device) - nsweeps / 2)
    )
    pos[:, :, 2] = 100.0

    # One projection = one (pixel, sweep) pair.
    projs = nbatch * nx * ny * nsweeps

    iterations = 10

    globals_dict = {
        "img": img,
        "grid_cart": grid_cart,
        "fc": fc,
        "fs": fs,
        "gamma": gamma,
        "sweep_samples": sweep_samples,
        "pos": pos,
    }

    tn = benchmark.Timer(
        stmt="torchbp.ops.projection_cart_2d_nufft(img, pos, grid_cart, fc, fs, gamma, sweep_samples, normalization='gamma')",
        setup="import torchbp",
        globals=globals_dict,
    )

    n = tn.timeit(iterations).median
    print(f"Device {device}, NUFFT: {n*1e3:.4g} ms / call, {projs / n:.3g} pixel-sweeps/s")

    # Antenna-pattern path (asin/atan2 + bilinear gain lookup per pixel).
    g = torch.ones((64, 64), dtype=torch.float32, device=device)
    g_extent = [-1.5, -3.1, 1.5, 3.1]
    att = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
    globals_ant = dict(globals_dict, g=g, g_extent=g_extent, att=att)

    ta = benchmark.Timer(
        stmt="torchbp.ops.projection_cart_2d_nufft(img, pos, grid_cart, fc, fs, gamma, sweep_samples, normalization='gamma', g=g, g_extent=g_extent, att=att)",
        setup="import torchbp",
        globals=globals_ant,
    )

    a = ta.timeit(iterations).median
    print(f"Device {device}, NUFFT antenna: {a*1e3:.4g} ms / call, {projs / a:.3g} pixel-sweeps/s")

    if compare_direct:
        td = benchmark.Timer(
            stmt="torchbp.ops.projection_cart_2d(img, pos, grid_cart, fc, fs, gamma, sweep_samples, normalization='gamma')",
            setup="import torchbp",
            globals=globals_dict,
        )
        d = td.timeit(iterations).median
        print(f"Device {device}, Direct: {d*1e3:.4g} ms / call, {projs / d:.3g} pixel-sweeps/s")
        print(f"Device {device}, NUFFT speedup over direct: {d / n:.3g}x")


if __name__ == "__main__":
    if torch.cuda.is_available():
        main("cuda")
    main("cpu")
