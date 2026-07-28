#!/usr/bin/env python
import math

import torch
import torch.utils.benchmark as benchmark
import torchbp

kC0 = 299792458.0


def point_target_metrics(img, grid):
    """IRW (-3 dB width in theta bins) and PSLR (dB) of the brightest target."""
    p = img.abs()
    idx = int(torch.argmax(p))
    ir, it = idx // p.shape[1], idx % p.shape[1]
    cut = p[ir, :]
    peak = float(cut[it])
    # Contiguous -3 dB run around the peak.
    lo = it
    while lo - 1 >= 0 and cut[lo - 1] >= peak * 10 ** (-3 / 20):
        lo -= 1
    hi = it
    while hi + 1 < len(cut) and cut[hi + 1] >= peak * 10 ** (-3 / 20):
        hi += 1
    irw = hi - lo + 1
    # First null then max sidelobe.
    n = hi
    while n + 1 < len(cut) and cut[n + 1] < cut[n]:
        n += 1
    m = lo
    while m - 1 >= 0 and cut[m - 1] < cut[m]:
        m -= 1
    side = max(
        float(cut[n:].max()) if n < len(cut) else 0.0,
        float(cut[: m + 1].max()) if m >= 0 else 0.0,
    )
    pslr = 20 * math.log10(side / peak + 1e-12)
    return irw, pslr


def main(device):
    if device == "cpu":
        n = 512
        nsweeps = 1024
    else:
        n = 1024
        nsweeps = 2048
    nsamples = 2 * n
    fc = 6e9
    lam = kC0 / fc
    r_res = 0.5

    grid = {"r": (100.0, 300.0), "theta": (-0.3, 0.3), "nr": n, "ntheta": n}

    pos = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, 1] = lam / 4 * (torch.arange(nsweeps, dtype=torch.float32, device=device) - nsweeps / 2)
    pos[:, 2] = 40.0

    # Bandlimited random scene for timing and error, emulating range
    # compression with 2x FFT oversampling.
    data = torch.randn((nsweeps, nsamples), dtype=torch.complex64, device=device)
    dataf = torch.fft.fft(data, dim=-1)
    dataf[:, nsamples // 4 : -nsamples // 4] = 0
    data = torch.fft.ifft(dataf, dim=-1)

    g = {
        "data": data,
        "grid": grid,
        "fc": fc,
        "r_res": r_res,
        "pos": pos,
    }
    iterations = 10
    backprojs = n * n * nsweeps

    def timeit(stmt):
        return benchmark.Timer(stmt=stmt, setup="import torchbp", globals=g).timeit(iterations).median

    ref = torchbp.ops.backprojection_polar_2d(data, grid, fc, r_res, pos, dealias=True)[0]

    def rel_err(out):
        return ((out - ref).abs().max() / ref.abs().max()).item()

    d = timeit("torchbp.ops.backprojection_polar_2d(data, grid, fc, r_res, pos, dealias=True)")
    print(f"Device {device}, Direct: {d:.3f} s, {backprojs / d:.3g} backprojections/s")

    for nsub in (4, 8, 16):
        g["nsub"] = nsub
        out = torchbp.ops.afbp(data, grid, fc, r_res, pos, nsub=nsub, dealias=True)
        t = timeit("torchbp.ops.afbp(data, grid, fc, r_res, pos, nsub=nsub, dealias=True)")
        print(f"Device {device}, AFBP nsub={nsub}: {t:.3f} s, {d / t:.2f}x over direct, "
              f"max err {rel_err(out):.2e}")

    for stages, nsub in ((2, 1), (2, 8), (3, 1), (3, 8)):
        g["stages"] = stages
        g["nsub"] = nsub
        out = torchbp.ops.ffbp(data, grid, fc, r_res, pos, stages=stages,
                               dealias=True, grid_oversample=2.0, afbp_nsub=nsub)
        t = timeit("torchbp.ops.ffbp(data, grid, fc, r_res, pos, stages=stages,"
                   " dealias=True, grid_oversample=2.0, afbp_nsub=nsub)")
        name = f"FFBP stages={stages}" + (f" + afbp_nsub={nsub}" if nsub > 1 else "")
        print(f"Device {device}, {name}: {t:.3f} s, {d / t:.2f}x over direct, "
              f"max err {rel_err(out):.2e}")

    # Point target quality metrics on a zoomed grid oversampled well past
    # the full aperture azimuth resolution.
    gr = math.sqrt(200.0**2 - float(pos[0, 2]) ** 2)
    tx, ty = gr * math.sqrt(1 - 0.1**2), gr * 0.1
    d_t = torch.sqrt((pos[:, 0] - tx) ** 2 + (pos[:, 1] - ty) ** 2 + pos[:, 2] ** 2).double()
    i = torch.arange(nsamples, dtype=torch.float64, device=device)
    env = torch.special.sinc((i[None, :] * r_res - d_t[:, None]) / (2 * r_res))
    data_pt = (env * torch.exp(-1j * 4 * math.pi * fc / kC0 * d_t)[:, None]).to(torch.complex64)
    grid_zoom = {"r": (180.0, 220.0), "theta": (0.1 - 0.04, 0.1 + 0.04), "nr": 256,
                 "ntheta": 1024}
    ref_pt = torchbp.ops.backprojection_polar_2d(data_pt, grid_zoom, fc, r_res, pos, dealias=True)[0]
    afbp_pt = torchbp.ops.afbp(data_pt, grid_zoom, fc, r_res, pos, nsub=8, dealias=True)
    ffbp_pt = torchbp.ops.ffbp(data_pt, grid_zoom, fc, r_res, pos, stages=3,
                               dealias=True, grid_oversample=2.0)
    lam = kC0 / fc
    L = float(pos[:, 1].max() - pos[:, 1].min())
    dtheta_z = 0.08 / 1024
    print(f"Device {device}, theoretical IRW {lam / (2 * L) / dtheta_z:.1f} bins")
    for name, img in (("direct", ref_pt), ("afbp8", afbp_pt), ("ffbp3", ffbp_pt)):
        irw, pslr = point_target_metrics(img, grid_zoom)
        print(f"Device {device}, point target {name}: IRW {irw} bins, PSLR {pslr:.1f} dB")


if __name__ == "__main__":
    if torch.cuda.is_available():
        main("cuda")
    main("cpu")
