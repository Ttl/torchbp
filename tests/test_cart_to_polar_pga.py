#!/usr/bin/env python
"""
Test cart_to_polar with phase gradient autofocus on a Cartesian image.

cart_to_polar is the inverse of polar_to_cart. Its main use is applying
algorithms that assume a polar grid, such as PGA, to Cartesian format
images. The test simulates point targets, applies a per-sweep phase error
to the data and checks that the full pipeline recovers the focus:

    cfbp (Cartesian image) -> cart_to_polar -> pga -> polar_to_cart

Also checks that the cart -> polar -> cart round trip reproduces the
original image, since PGA is useless if the resampling itself destroys
the image.

Run standalone for metrics: python tests/test_cart_to_polar_pga.py
"""
import unittest

import torch
import torchbp
from torchbp.autofocus import pga
from torchbp.util import entropy
from numpy import hamming

C0 = 299792458.0

_scene_cache = {}


def make_scene(device):
    """Simulate point targets and form reference and defocused cfbp images."""
    if device in _scene_cache:
        return _scene_cache[device]

    fc = 6e9
    bw = 200e6
    tsweep = 100e-6
    fs = 3e6
    oversample = 2
    nsamples = int(fs * tsweep)
    nsweeps = 1024

    # Straight track along y at lambda/4 spacing: L = 12.8 m.
    pos = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, 1] = 0.25 * C0 / fc * (torch.arange(nsweeps, device=device) - nsweeps / 2)
    pos[:, 2] = 30.0

    grid_cart = {"x": (60.0, 100.0), "y": (-20.0, 20.0), "nx": 512, "ny": 512}

    tx = [63.0, 68.0, 74.0, 81.0, 87.0, 93.0, 97.0, 78.0, 85.0, 70.0]
    ty = [3.0, -12.0, 8.0, -4.0, 15.0, -16.0, 5.0, 12.0, -8.0, -2.0]
    targets = list(zip(tx, ty))
    target_pos = torch.tensor([[x, y, 0.0] for x, y in zip(tx, ty)],
                              dtype=torch.float32, device=device)
    target_rcs = torch.ones((len(tx), 1), dtype=torch.float32, device=device)

    raw = torchbp.util.generate_fmcw_data(target_pos, target_rcs, pos, fc, bw, tsweep, fs)
    w = torch.tensor(hamming(raw.shape[-1])[None, :], dtype=torch.float32, device=device)
    data = torch.fft.ifft(raw * w, dim=-1, n=nsamples * oversample)
    r_res = C0 / (2 * bw * oversample)

    # Smooth per-sweep phase error, several radians peak to peak. Low order
    # so that it is spatially invariant over the scene and PGA can solve it.
    s = torch.arange(nsweeps, device=device) / nsweeps
    phase_error = (3.0 * torch.sin(2 * torch.pi * 1.7 * s + 0.5)
                   + 2.0 * torch.cos(2 * torch.pi * 3.1 * s)
                   + 5.0 * (s - 0.5) ** 2)
    data_err = data * torch.exp(1j * phase_error)[:, None]

    img_ref = torchbp.ops.cfbp(data, grid_cart, fc, r_res, pos, stages=4)[0]
    img_err = torchbp.ops.cfbp(data_err, grid_cart, fc, r_res, pos, stages=4)[0]

    # Polar grid around the aperture center covering the full Cartesian grid.
    # dtheta must oversample the aperture azimuth bandwidth lambda / (2 * L)
    # and dr the data range envelope bandwidth (dr < r_res).
    origin = torch.mean(pos, axis=0)
    grid_polar = {"r": (58.0, 104.0), "theta": (-0.35, 0.35), "nr": 384, "ntheta": 512}

    scene = {
        "fc": fc,
        "grid_cart": grid_cart,
        "grid_polar": grid_polar,
        "origin": origin,
        "img_ref": img_ref,
        "img_err": img_err,
        "targets": targets,
    }
    _scene_cache[device] = scene
    return scene


def interior_mask(grid_cart, device, margin=2.0):
    """Mask of Cartesian pixels at least margin meters from the grid edge."""
    x0, x1 = grid_cart["x"]
    y0, y1 = grid_cart["y"]
    nx, ny = grid_cart["nx"], grid_cart["ny"]
    x = x0 + (x1 - x0) / nx * torch.arange(nx, device=device)
    y = y0 + (y1 - y0) / ny * torch.arange(ny, device=device)
    mx = (x > x0 + margin) & (x < x1 - margin)
    my = (y > y0 + margin) & (y < y1 - margin)
    return mx[:, None] & my[None, :]


def run_round_trip(device, method=("lanczos", 8)):
    scene = make_scene(device)
    img_polar = torchbp.ops.cart_to_polar(
        scene["img_ref"], scene["origin"], scene["grid_cart"],
        scene["grid_polar"], scene["fc"], method=method)
    img_back = torchbp.ops.polar_to_cart(
        img_polar[0], scene["origin"], scene["grid_polar"],
        scene["grid_cart"], scene["fc"], method=method)[0]
    mask = interior_mask(scene["grid_cart"], device)
    err = (img_back - scene["img_ref"]).abs()[mask].max()
    rel_err = (err / scene["img_ref"].abs().max()).item()
    return rel_err


def target_energy_fraction(img, scene, half=8):
    """Fraction of image energy within a box around each true target position.

    Robust focus metric for a sparse point target scene: unlike entropy it is
    insensitive to the low level noise floor and to the small target position
    shift left by the detrended PGA phase estimate.
    """
    grid_cart = scene["grid_cart"]
    x0, x1 = grid_cart["x"]
    y0, y1 = grid_cart["y"]
    dx = (x1 - x0) / grid_cart["nx"]
    dy = (y1 - y0) / grid_cart["ny"]
    total = (img.abs() ** 2).sum()
    s = 0.0
    for x, y in scene["targets"]:
        ix = int((x - x0) / dx)
        iy = int((y - y0) / dy)
        s += (img[ix - half:ix + half + 1, iy - half:iy + half + 1].abs() ** 2).sum()
    return (s / total).item()


def run_pga(device, method=("lanczos", 8)):
    scene = make_scene(device)

    def round_trip(img, autofocus):
        img_polar = torchbp.ops.cart_to_polar(
            img, scene["origin"], scene["grid_cart"],
            scene["grid_polar"], scene["fc"], method=method)[0]
        if autofocus:
            img_polar, phi = pga(img_polar)
        return torchbp.ops.polar_to_cart(
            img_polar, scene["origin"], scene["grid_polar"],
            scene["grid_cart"], scene["fc"], method=method)[0]

    # Round-tripped reference and defocused images as baselines so that the
    # comparison is not biased by the interpolation error of the
    # cart -> polar -> cart resampling itself.
    img_ref_rt = round_trip(scene["img_ref"], autofocus=False)
    img_err_rt = round_trip(scene["img_err"], autofocus=False)
    img_pga = round_trip(scene["img_err"], autofocus=True)

    metrics = {
        "focus_ref": target_energy_fraction(img_ref_rt, scene),
        "focus_err": target_energy_fraction(img_err_rt, scene),
        "focus_pga": target_energy_fraction(img_pga, scene),
        "peak_ref": img_ref_rt.abs().max().item(),
        "peak_err": img_err_rt.abs().max().item(),
        "peak_pga": img_pga.abs().max().item(),
        "entropy_ref": entropy(img_ref_rt).item(),
        "entropy_err": entropy(img_err_rt).item(),
        "entropy_pga": entropy(img_pga).item(),
    }
    return metrics


def check_round_trip(device):
    rel_err = run_round_trip(device)
    assert rel_err < 0.05, f"cart -> polar -> cart round trip error too large: {rel_err}"


def check_pga(device):
    m = run_pga(device)
    # Phase error must defocus the image significantly for the test to be
    # meaningful.
    assert m["peak_err"] < 0.6 * m["peak_ref"], (
        f"phase error defocuses too little: {m}")
    assert m["focus_err"] < 0.7 * m["focus_ref"], (
        f"phase error defocuses too little: {m}")
    # PGA through cart_to_polar/polar_to_cart must recover most of the focus.
    assert m["peak_pga"] > 0.8 * m["peak_ref"], f"PGA failed to refocus: {m}"
    assert m["focus_pga"] > 0.85 * m["focus_ref"], f"PGA failed to refocus: {m}"


def test_round_trip_cpu():
    check_round_trip("cpu")


def test_pga_cpu():
    check_pga("cpu")


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
def test_round_trip_cuda():
    check_round_trip("cuda")


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
def test_pga_cuda():
    check_pga("cuda")


if __name__ == "__main__":
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        print(f"=== Device: {device} ===")
        for method in ("linear", ("lanczos", 8)):
            rel_err = run_round_trip(device, method=method)
            print(f"round trip max error ({method}): {rel_err:.2e}")
        m = run_pga(device)
        print(f"target energy fraction: ref {m['focus_ref']:.3f}, "
              f"defocused {m['focus_err']:.3f}, pga {m['focus_pga']:.3f}")
        print(f"peak:    ref {m['peak_ref']:.4f}, defocused {m['peak_err']:.4f}, "
              f"pga {m['peak_pga']:.4f}")
        print(f"entropy: ref {m['entropy_ref']:.3f}, defocused {m['entropy_err']:.3f}, "
              f"pga {m['entropy_pga']:.3f}")
