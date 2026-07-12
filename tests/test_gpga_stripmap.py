#!/usr/bin/env python
"""
Test GPGA autofocus in a stripmap-like scenario.

The scene setup follows docs/source/examples/antenna_normalization.ipynb:
an FMCW radar flies a straight track along y looking broadside (+x) with a
narrow azimuth beam, and the raw data is simulated from a Cartesian scene
with projection_cart_2d_nufft using the antenna pattern. Unlike the
spotlight tests in test_torchbp.py, the track is much longer than the
azimuth beam footprint, so each target is illuminated for only about half
of the aperture and targets continuously enter and leave the beam. Point
targets are spaced along the track so that at least one target is inside
the beam at every sweep.

Out-of-beam sweeps of a target's range ring contain unrelated clutter at
full scene amplitude, which corrupts an unweighted phase gradient
estimate. The tests check that the antenna-weighted GPGA path (att, g,
g_extent used to gate and weight the per-target phase histories and the
TDE position solve) recovers an injected position error in this scenario.

The scene also includes an azimuth-extended clutter arc ("building wall")
at constant polar range to exercise the isolation target screening.

Run standalone for metrics: python tests/test_gpga_stripmap.py
"""
import unittest

import numpy as np
import torch
import torchbp
from torchbp.autofocus import (
    gpga,
    gpga_tde,
    pga_estimator,
    _antenna_weights,
    _select_targets,
)
from torchbp.util import detrend
from numpy import hamming

C0 = 299792458.0

_scene_cache = {}


def make_scene(device="cpu"):
    """Simulate a stripmap pass over point targets with a narrow beam."""
    if device in _scene_cache:
        return _scene_cache[device]

    torch.manual_seed(0)
    np.random.seed(0)

    fc = 6e9
    bw = 200e6
    tsweep = 100e-6
    fs = 5e6
    oversample = 2
    nsamples = int(fs * tsweep)
    nsweeps = 1024
    wl = C0 / fc
    altitude = 30.0

    # Straight track along y at lambda/2 spacing: L = 25.6 m.
    pos = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
    pos[:, 1] = 0.5 * wl * (torch.arange(nsweeps, device=device) - nsweeps / 2)
    pos[:, 2] = altitude

    # Narrow azimuth beam: the two-way amplitude drops to 0.2 of peak at
    # about +-5 deg, an ~13 m footprint at mid swath, half the track
    # length. Wide elevation beam rolled to mid swath.
    naz, nel = 128, 64
    az = np.linspace(-np.pi / 6, np.pi / 6, naz)
    el = np.linspace(-np.pi / 3, np.pi / 3, nel)
    az_bw = np.deg2rad(4.0)
    el_bw = np.deg2rad(25.0)
    gain = np.exp(-((el[:, None] / el_bw) ** 2)) * np.exp(
        -((az[None, :] / az_bw) ** 2)
    )
    g = torch.tensor(gain, dtype=torch.float32, device=device)
    g_extent = [el[0], az[0], el[-1], az[-1]]

    r_mid = 75.0
    att = torch.zeros_like(pos)
    att[:, 0] = -np.arctan2(altitude, r_mid)

    # Point targets spaced 5 m along the track, less than the beam
    # footprint, so at least one target is in the beam at every sweep.
    ty = np.arange(-17.5, 18.0, 5.0)
    tx = np.array([62.0, 80.0, 70.0, 88.0, 65.0, 84.0, 74.0, 91.0])
    targets = torch.tensor(
        np.stack([tx, ty, np.zeros_like(tx)], axis=1),
        dtype=torch.float32, device=device,
    )

    # Cartesian scene for the forward projection: unit point targets, a
    # weak uniform clutter floor, and an azimuth-extended clutter arc at
    # constant polar range (a "building wall") for the isolation screen.
    grid_proj = {"x": (45.0, 105.0), "y": (-25.0, 25.0), "nx": 512, "ny": 512}
    dx = (grid_proj["x"][1] - grid_proj["x"][0]) / grid_proj["nx"]
    dy = (grid_proj["y"][1] - grid_proj["y"][0]) / grid_proj["ny"]
    scene = 0.004 * torch.randn(
        (grid_proj["nx"], grid_proj["ny"]), dtype=torch.complex64, device=device
    )

    def scene_idx(x, y):
        i = int(round((x - grid_proj["x"][0]) / dx))
        j = int(round((y - grid_proj["y"][0]) / dy))
        return i, j

    wall_r = 85.0
    for th in np.arange(-0.12, 0.12, 0.4 * min(dx, dy) / wall_r):
        i, j = scene_idx(wall_r * np.cos(th), wall_r * np.sin(th))
        scene[i, j] = 0.35 * np.exp(2j * np.pi * np.random.rand())
    for k in range(len(tx)):
        i, j = scene_idx(tx[k], ty[k])
        scene[i, j] = 1.0
        # Snap the target to the pixel center actually simulated.
        targets[k, 0] = grid_proj["x"][0] + dx * i
        targets[k, 1] = grid_proj["y"][0] + dy * j

    # Smooth zero-mean range (x) position error, rms about lambda/2.
    u = torch.arange(nsweeps, dtype=torch.float32, device=device) / nsweeps
    dx_err = 0.03 * torch.sin(2 * torch.pi * 2.5 * u) + 0.015 * torch.sin(
        2 * torch.pi * 5.5 * u + 1.0
    )
    pos_true = pos.clone()
    pos_true[:, 0] += dx_err

    # Forward projection with the antenna pattern and range compression,
    # as in the antenna_normalization example.
    data_fmod = -torch.pi * (1 - (oversample - 1) / oversample)
    n = nsamples * oversample
    r_res = C0 / (2 * bw * oversample)
    raw = torchbp.ops.projection_cart_2d_nufft(
        scene, pos_true, grid_proj, fc, fs, bw / tsweep, nsamples,
        att=att, g=g, g_extent=g_extent, use_rvp=False, normalization="sigma",
    )[0]
    w = torch.tensor(
        hamming(raw.shape[-1])[None, :], dtype=torch.float32, device=device
    )
    data = torch.fft.ifft(raw * w, dim=-1, n=n)
    data = data * torch.exp(
        1j * data_fmod * torch.arange(data.shape[-1], device=device)
    )[None, :]

    grid_polar = {"r": (50.0, 100.0), "theta": (-0.42, 0.42),
                  "nr": 96, "ntheta": 512}

    scene_data = {
        "fc": fc, "r_res": r_res, "data_fmod": data_fmod,
        "grid_polar": grid_polar, "data": data, "pos": pos,
        "pos_true": pos_true, "dx_err": dx_err, "targets": targets,
        "att": att, "g": g, "g_extent": g_extent, "wall_r": wall_r,
    }
    _scene_cache[device] = scene_data
    return scene_data


def form_image(s, pos):
    return torchbp.ops.backprojection_polar_2d(
        s["data"], s["grid_polar"], s["fc"], s["r_res"], pos,
        data_fmod=s["data_fmod"], att=s["att"], g=s["g"],
        g_extent=s["g_extent"],
    )[0]


def sharpness(img):
    """Inverse participation ratio of intensity: higher when focused."""
    p = img.abs() ** 2
    return ((p**2).sum() / (p.sum() ** 2)).item()


def target_peaks_db(s, img):
    """Peak image amplitude near each known target position, in dB."""
    r0, r1 = s["grid_polar"]["r"]
    t0, t1 = s["grid_polar"]["theta"]
    nr, ntheta = s["grid_polar"]["nr"], s["grid_polar"]["ntheta"]
    peaks = []
    for t in s["targets"]:
        r = torch.sqrt(t[0] ** 2 + t[1] ** 2).item()
        th = (t[1] / torch.sqrt(t[0] ** 2 + t[1] ** 2)).item()
        i = int(round((r - r0) / ((r1 - r0) / nr)))
        j = int(round((th - t0) / ((t1 - t0) / ntheta)))
        win = img[max(0, i - 4): i + 5, max(0, j - 8): j + 9]
        peaks.append(20 * torch.log10(win.abs().max() + 1e-12).item())
    return np.array(peaks)


def residual_rms(dx_err, solved_dx):
    """RMS of the detrended difference (mean and trend are unobservable)."""
    return detrend(dx_err - solved_dx).pow(2).mean().sqrt().item()


class TestWeightedPgaEstimator(unittest.TestCase):
    """Weighted estimator on synthetic dwells with out-of-beam corruption."""

    def _dwell_data(self):
        torch.manual_seed(1)
        nt, ns = 6, 512
        phi = 3.0 * torch.sin(
            2 * torch.pi * 3 * torch.arange(ns, dtype=torch.float32) / ns
        )
        weight = torch.zeros((nt, ns))
        g = torch.zeros((nt, ns), dtype=torch.complex64)
        # Staggered overlapping dwells covering every sweep.
        starts = torch.linspace(0, ns - 160, nt).long()
        for t in range(nt):
            weight[t, starts[t]: starts[t] + 160] = 1.0
        # In-beam: clean phase error. Out-of-beam: unrelated clutter with
        # comparable amplitude, as in stripmap.
        clutter = torch.randn(nt, ns, dtype=torch.complex64)
        g = weight * torch.exp(1j * phi)[None, :] + (1 - weight) * clutter
        return g, weight, phi

    def test_weight_recovers_phase_unweighted_fails(self):
        g, weight, phi = self._dwell_data()
        for est in ("pd", "wls"):
            phi_w = pga_estimator(g, est, weight=weight)
            err_w = detrend(phi - phi_w).pow(2).mean().sqrt().item()
            phi_uw = pga_estimator(g, est)
            err_uw = detrend(phi - phi_uw).pow(2).mean().sqrt().item()
            self.assertLess(err_w, 0.2, f"{est}: weighted rms {err_w}")
            self.assertGreater(
                err_uw, 3 * err_w,
                f"{est}: unweighted rms {err_uw} vs weighted {err_w}",
            )


class TestSelectTargets(unittest.TestCase):
    """Isolation screen rejects azimuth-extended clutter (building wall)."""

    def test_wall_rejected_point_kept(self):
        torch.manual_seed(2)
        img = 0.01 * torch.randn(64, 256, dtype=torch.complex64)
        # Point target in row 10.
        img[10, 60] = 1.0
        # Wall: rows 40-44 filled with strong clutter across azimuth.
        img[40:45, 80:200] = 0.6 * torch.exp(
            2j * torch.pi * torch.rand(5, 120)
        )
        rows, cols = _select_targets(img, target_threshold_db=20,
                                     isolation_db=6.0)
        self.assertIn(10, rows.tolist())
        for r in range(40, 45):
            self.assertNotIn(r, rows.tolist())
        # Without the screen the wall rows are picked.
        rows_uw, _ = _select_targets(img, target_threshold_db=20,
                                     isolation_db=0.0)
        self.assertTrue(any(r in rows_uw.tolist() for r in range(40, 45)))


class TestStripmapGpga(unittest.TestCase):
    """End-to-end stripmap autofocus on simulated beam-limited data."""

    def test_antenna_weights_match_data_envelope(self):
        # The Python antenna weight model must match the amplitude
        # envelope the C++ forward projection kernel produced.
        s = make_scene()
        t = s["targets"][3][None, :]  # central target
        td = torchbp.ops.gpga_backprojection_2d_core(
            t, s["data"], s["pos_true"], s["fc"], s["r_res"],
            data_fmod=s["data_fmod"],
        )[0]
        env = torch.abs(td)
        env = torch.nn.functional.avg_pool1d(
            env[None, None], 21, stride=1, padding=10
        )[0, 0]
        w = _antenna_weights(
            t, s["pos_true"], s["att"], s["g"], s["g_extent"]
        )[0]
        corr = torch.corrcoef(torch.stack([env, w]))[0, 1].item()
        self.assertGreater(corr, 0.9)
        # The target must actually leave the beam (stripmap, not spotlight).
        wn = w / w.max()
        self.assertLess((wn > 0.2).float().mean().item(), 0.8)

    def test_tde_focuses_stripmap(self):
        s = make_scene()
        img_true = form_image(s, s["pos_true"])

        img_focus, pos_new = gpga_tde(
            None, s["data"], s["pos"], s["fc"], s["r_res"], s["grid_polar"],
            azimuth_divisions=4, range_divisions=2, estimate_z=False,
            max_iters=8, att=s["att"], g=s["g"], g_extent=s["g_extent"],
            data_fmod=s["data_fmod"],
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(pos_new).all())

        # The solved x correction must track the injected error.
        resid = residual_rms(s["dx_err"], pos_new[:, 0] - s["pos"][:, 0])
        rms = s["dx_err"].pow(2).mean().sqrt().item()
        self.assertLess(resid, 0.4 * rms)

        # Unobservable along-track direction must not accumulate noise.
        dy = (pos_new[:, 1] - s["pos"][:, 1]).pow(2).mean().sqrt().item()
        self.assertLess(dy, 0.2 * rms)

        # Focused target peaks should approach the true-position image.
        loss = target_peaks_db(s, img_true) - target_peaks_db(s, img_focus)
        self.assertLess(np.mean(loss), 2.5)

    def test_gpga_focuses_stripmap(self):
        # Phase-only GPGA with the wls estimator; pd drifts at target
        # handovers with discontinuous illumination.
        s = make_scene()
        img_true = form_image(s, s["pos_true"])
        img_focus, phi = gpga(
            None, s["data"], s["pos"], s["fc"], s["r_res"], s["grid_polar"],
            max_iters=8, estimator="wls", att=s["att"], g=s["g"],
            g_extent=s["g_extent"], data_fmod=s["data_fmod"],
        )
        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(phi).all())

        d = phi * C0 / (4 * torch.pi * s["fc"])
        resid = residual_rms(s["dx_err"], d)
        rms = s["dx_err"].pow(2).mean().sqrt().item()
        self.assertLess(resid, 0.5 * rms)

        loss = target_peaks_db(s, img_true) - target_peaks_db(s, img_focus)
        self.assertLess(np.mean(loss), 2.5)


def main():
    s = make_scene()
    rms = s["dx_err"].pow(2).mean().sqrt().item()
    print(f"Injected x error rms: {1e3 * rms:.1f} mm")

    img_blur = form_image(s, s["pos"])
    img_true = form_image(s, s["pos_true"])
    print(f"Sharpness blurred: {sharpness(img_blur):.3e}")
    print(f"Sharpness true pos: {sharpness(img_true):.3e}")

    runs = {}
    print("\n--- gpga_tde with antenna weighting ---")
    img_aw, pos_aw = gpga_tde(
        None, s["data"], s["pos"], s["fc"], s["r_res"], s["grid_polar"],
        azimuth_divisions=4, range_divisions=2, estimate_z=False,
        max_iters=8, att=s["att"], g=s["g"], g_extent=s["g_extent"],
        data_fmod=s["data_fmod"], verbose=True,
    )
    runs["tde antenna-weighted"] = (img_aw, pos_aw[:, 0] - s["pos"][:, 0])

    print("\n--- gpga_tde without antenna information ---")
    img_uw, pos_uw = gpga_tde(
        None, s["data"], s["pos"], s["fc"], s["r_res"], s["grid_polar"],
        azimuth_divisions=4, range_divisions=2, estimate_z=False,
        max_iters=8, data_fmod=s["data_fmod"], verbose=True,
    )
    runs["tde unweighted"] = (img_uw, pos_uw[:, 0] - s["pos"][:, 0])

    print("\n--- gpga (phase only, wls) with antenna weighting ---")
    img_g, phi_g = gpga(
        None, s["data"], s["pos"], s["fc"], s["r_res"], s["grid_polar"],
        max_iters=8, estimator="wls", att=s["att"], g=s["g"],
        g_extent=s["g_extent"], data_fmod=s["data_fmod"],
    )
    runs["gpga antenna-weighted"] = (
        img_g, phi_g * C0 / (4 * torch.pi * s["fc"])
    )

    peaks_true = target_peaks_db(s, img_true)
    print(f"\n{'run':>24} {'sharpness':>10} {'resid mm':>9} {'peak loss dB':>12}")
    print(f"{'blurred':>24} {sharpness(img_blur):>10.3e} {1e3 * rms:>9.1f}"
          f" {np.mean(peaks_true - target_peaks_db(s, img_blur)):>12.1f}")
    for name, (img, d) in runs.items():
        resid = residual_rms(s["dx_err"], d)
        loss = np.mean(peaks_true - target_peaks_db(s, img))
        print(f"{name:>24} {sharpness(img):>10.3e} {1e3 * resid:>9.1f}"
              f" {loss:>12.1f}")


if __name__ == "__main__":
    main()
