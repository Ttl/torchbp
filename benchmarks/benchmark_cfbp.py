#!/usr/bin/env python
import torch
import torchbp
import torch.utils.benchmark as benchmark


def main(device):
    if device == "cpu":
        n = 512
        nsweeps = 512
        stages = 2
    else:
        n = 1024
        nsweeps = 1024
        stages = 3
    nsamples = 2048
    fc = 6e9
    r_res = 0.5

    grid_cart = {"x": (300, 800), "y": (-125, 125), "nx": n, "ny": n}
    # Polar grid covering the same scene for the ffbp comparison.
    grid_polar = {"r": (300, 810), "theta": (-0.39, 0.39), "nr": n, "ntheta": n}

    # Bandlimit the random data to a quarter of the sample bandwidth,
    # emulating range compression with 4x FFT oversampling. Unwindowed
    # full-bandwidth data would alias in y in the subaperture images at this
    # scene angular extent (see the cfbp docstring).
    data = torch.randn((nsweeps, nsamples), dtype=torch.complex64, device=device)
    dataf = torch.fft.fft(data, dim=-1)
    dataf[:, nsamples // 8 : -nsamples // 8] = 0
    data = torch.fft.ifft(dataf, dim=-1)

    pos = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, 1] = (
        0.25
        * 3e8
        / fc
        * (torch.arange(nsweeps, dtype=torch.float32, device=device) - nsweeps / 2)
    )
    pos[:, 2] = 50.0
    origin = torch.zeros(3, dtype=torch.float32, device=device)

    backprojs = n * n * nsweeps

    iterations = 10

    g = {
        "data": data,
        "grid_cart": grid_cart,
        "grid_polar": grid_polar,
        "fc": fc,
        "r_res": r_res,
        "pos": pos,
        "origin": origin,
        "stages": stages,
    }

    t_direct = benchmark.Timer(
        stmt="torchbp.ops.backprojection_cart_2d(data, grid_cart, fc, r_res, pos)",
        setup="import torchbp",
        globals=g,
    )
    t_cfbp = benchmark.Timer(
        stmt="torchbp.ops.cfbp(data, grid_cart, fc, r_res, pos, stages=stages)",
        setup="import torchbp",
        globals=g,
    )
    t_ffbp = benchmark.Timer(
        stmt="torchbp.ops.polar_to_cart(torchbp.ops.ffbp(data, grid_polar, fc,"
        " r_res, pos, stages=stages, dealias=True), origin, grid_polar,"
        " grid_cart, fc)",
        setup="import torchbp",
        globals=g,
    )

    ref = torchbp.ops.backprojection_cart_2d(data, grid_cart, fc, r_res, pos)
    out = torchbp.ops.cfbp(data, grid_cart, fc, r_res, pos, stages=stages)
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f"Device {device}, cfbp stages={stages} relative error: {rel:.2e}")

    d = t_direct.timeit(iterations).median
    print(f"Device {device}, Direct: {d:.3f} s, {backprojs / d:.3g} backprojections/s")
    c = t_cfbp.timeit(iterations).median
    print(f"Device {device}, CFBP: {c:.3f} s, {d / c:.2f}x over direct")
    f = t_ffbp.timeit(iterations).median
    print(f"Device {device}, FFBP + polar_to_cart: {f:.3f} s, {d / f:.2f}x over direct")


if __name__ == "__main__":
    if torch.cuda.is_available():
        main("cuda")
    main("cpu")
