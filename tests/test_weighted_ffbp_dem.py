#!/usr/bin/env python
"""Antenna-pattern-weighted FFBP with a DEM: correctness against direct
backprojection, exact invariances, and polarimetric channel balance.

Covers the combination the polarimetric processing pipeline uses
(ffbp + att/g/g_extent + dem), which the rest of the test suite only
exercises separately.

These tests pin the DEM-aware illumination maps: compute_illumination
must reference the elevation look angle to the pixel at the DEM height
(sin_l = (z - pos_z) / d), matching the gains backprojection applies with
the same DEM. With flat-earth (z=0) look angles the W1/W2 maps sample the
elevation pattern at the wrong angle over terrain and the weighted merge
normalization A * W1 / W2 picks up a pattern-dependent — with per-channel
patterns, channel-dependent — amplitude bias that scales with terrain
height (rel err was 0.6%/3.6%/10.7% at h = 2/9/18 m constant DEM before
the fix, ~4e-4 after).
"""
import unittest

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase

import torchbp

C0 = 299792458.0


def _terrain(r, t):
    return 8.0 + 5.0 * torch.sin(2 * torch.pi * (r - 60.0) / 45.0) + 3.0 * t


def _beam(el_width, az_width, az_offset, device):
    nel, naz = 32, 64
    el = torch.linspace(-1.2, 1.2, nel, device=device)
    az = torch.linspace(-1.0, 1.0, naz, device=device)
    gain = torch.exp(-((el[:, None]) / el_width) ** 2) * torch.exp(
        -(((az[None, :] - az_offset)) / az_width) ** 2)
    g = gain.to(torch.float32)
    g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]
    return g, g_extent


class TestWeightedFfbpDem(TestCase):
    fc = 6e9
    bw = 200e6
    tsweep = 100e-6
    fs = 10e6
    oversample = 2
    alt = 30.0
    nsweeps = 384
    grid = {"r": (60.0, 100.0), "theta": (-0.3, 0.3), "nr": 320, "ntheta": 512}

    @property
    def r_res(self):
        return C0 / (2 * self.bw * self.oversample)

    def _scene(self, device, el_width=0.8, az_width=0.25, az_offset=0.0,
               use_dem=True):
        """Point targets on terrain, raw data simulated with the antenna
        pattern applied, matching pattern/attitude returned for imaging."""
        grid = self.grid
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        nr, ntheta = grid["nr"], grid["ntheta"]
        rr = r0 + (r1 - r0) / nr * torch.arange(nr, device=device)
        tt = t0 + (t1 - t0) / ntheta * torch.arange(ntheta, device=device)
        if use_dem:
            dem = _terrain(rr[:, None], tt[None, :]).float()
        else:
            dem = None

        tr = torch.tensor([70.0, 80.0, 90.0], device=device)
        tth = torch.tensor([-0.15, 0.0, 0.12], device=device)
        tz = _terrain(tr, tth) if use_dem else torch.zeros_like(tr)
        targets = torch.stack(
            [tr * torch.sqrt(1 - tth**2), tr * tth, tz], dim=-1)

        pos = torch.zeros(self.nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-5, 5, self.nsweeps, device=device)
        pos[:, 2] = self.alt

        g, g_extent = _beam(el_width, az_width, az_offset, device)
        att = torch.zeros(self.nsweeps, 3, device=device)
        att[:, 0] = -float(np.arcsin(self.alt / 80.0))

        rcs = torch.ones((targets.shape[0], 1), device=device)
        raw = torchbp.util.generate_fmcw_data(
            targets, rcs, pos, self.fc, self.bw, self.tsweep, self.fs,
            g=g, g_extent=g_extent, att=att, rvp=False)
        nsamples = int(self.fs * self.tsweep)
        w = torch.hamming_window(nsamples, periodic=False, device=device)
        data = torch.fft.ifft(raw * w[None, :], dim=-1,
                              n=nsamples * self.oversample)
        return data, pos, att, g, g_extent, dem, tr, tth

    def _target_pixels(self, tr, tth):
        grid = self.grid
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        dr = (r1 - r0) / grid["nr"]
        dt = (t1 - t0) / grid["ntheta"]
        return [(int(round((tr[k].item() - r0) / dr)),
                 int(round((tth[k].item() - t0) / dt)))
                for k in range(tr.shape[0])]

    @staticmethod
    def _peak_pixel(img, ir, it, win=8):
        a = torch.abs(img)
        w = a[max(0, ir - win):ir + win + 1, max(0, it - win):it + win + 1]
        peak = int(torch.argmax(w))
        pr = peak // w.shape[1] + max(0, ir - win)
        pt = peak % w.shape[1] + max(0, it - win)
        return pr, pt

    # ------------------------------------------------------------------
    # Exact invariance: a constant DEM h must equal the flat-earth result
    # with the platform lowered by h. Distances, look angles (and so the
    # antenna weighting) are identical, so this holds to interpolation
    # rounding through the entire pipeline.
    # ------------------------------------------------------------------

    def test_bp_gain_constant_dem_equals_shifted_pos(self):
        device = "cpu"
        torch.manual_seed(3)
        data = torch.randn(self.nsweeps, 500, dtype=torch.complex64)
        pos = torch.zeros(self.nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-5, 5, self.nsweeps)
        pos[:, 2] = self.alt
        g, g_extent = _beam(0.8, 0.25, 0.0, device)
        att = torch.zeros(self.nsweeps, 3)
        att[:, 0] = -float(np.arcsin(self.alt / 80.0))

        h = 9.0
        dem = torch.full((self.grid["nr"], self.grid["ntheta"]), h)
        res = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos,
            att=att, g=g, g_extent=g_extent, dem=dem)[0]
        pos_shift = pos.clone()
        pos_shift[:, 2] -= h
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos_shift,
            att=att, g=g, g_extent=g_extent)[0]
        rel = (torch.linalg.norm(res - ref) / torch.linalg.norm(ref)).item()
        self.assertLess(rel, 1e-3)

    def test_weighted_ffbp_constant_dem_equals_shifted_pos(self):
        device = "cpu"
        torch.manual_seed(3)
        data = torch.randn(self.nsweeps, 500, dtype=torch.complex64)
        pos = torch.zeros(self.nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-5, 5, self.nsweeps)
        pos[:, 2] = self.alt
        g, g_extent = _beam(0.8, 0.25, 0.0, device)
        att = torch.zeros(self.nsweeps, 3)
        att[:, 0] = -float(np.arcsin(self.alt / 80.0))
        kw = dict(stages=3, dealias=True, att=att, g=g, g_extent=g_extent)

        h = 9.0
        dem = torch.full((self.grid["nr"], self.grid["ntheta"]), h)
        res = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                               dem=dem, **kw)
        pos_shift = pos.clone()
        pos_shift[:, 2] -= h
        ref = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res,
                               pos_shift, **kw)
        rel = (torch.linalg.norm(res - ref) / torch.linalg.norm(ref)).item()
        self.assertLess(rel, 1e-3)

    # ------------------------------------------------------------------
    # Weighted FFBP with DEM against direct backprojection with the same
    # pattern and DEM.
    # ------------------------------------------------------------------

    def test_weighted_ffbp_dem_matches_direct_bp(self):
        device = "cpu"
        data, pos, att, g, g_extent, dem, tr, tth = self._scene(device)
        img_bp = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos,
            att=att, g=g, g_extent=g_extent, dem=dem, dealias=True)[0]
        img_ffbp = torchbp.ops.ffbp(
            data, self.grid, self.fc, self.r_res, pos, stages=3,
            dealias=True, att=att, g=g, g_extent=g_extent, dem=dem)
        rel = (torch.linalg.norm(img_ffbp - img_bp)
               / torch.linalg.norm(img_bp)).item()

        # Baseline: the same comparison without pattern weighting. Pattern
        # weighting must not add significant additional error on top of the
        # merge interpolation error.
        img_bp_nog = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, dem=dem,
            dealias=True)[0]
        img_ffbp_nog = torchbp.ops.ffbp(
            data, self.grid, self.fc, self.r_res, pos, stages=3,
            dealias=True, dem=dem)
        rel_nog = (torch.linalg.norm(img_ffbp_nog - img_bp_nog)
                   / torch.linalg.norm(img_bp_nog)).item()
        self.assertLess(rel, 0.25)
        self.assertLess(rel, rel_nog + 0.05)

        # Peaks land on the true target ground positions.
        for (ir, it) in self._target_pixels(tr, tth):
            pr, pt = self._peak_pixel(img_ffbp, ir, it)
            self.assertLessEqual(abs(pr - ir), 2)
            self.assertLessEqual(abs(pt - it), 2)

        # Per-target amplitude agrees with direct BP: the weighted
        # normalization must reproduce backprojection brightness.
        for (ir, it) in self._target_pixels(tr, tth):
            pr, pt = self._peak_pixel(img_bp, ir, it)
            a_bp = img_bp[pr, pt].abs().item()
            a_ff = img_ffbp[pr, pt].abs().item()
            err_db = abs(20 * np.log10(a_ff / a_bp))
            self.assertLess(err_db, 1.0)

    # ------------------------------------------------------------------
    # Polarimetric channel balance: two channels of the same scene with
    # different antenna patterns. The complex channel ratio at the target
    # peaks must agree between weighted FFBP and direct backprojection:
    # a channel-dependent gain or normalization error shifts this ratio.
    # ------------------------------------------------------------------

    def test_channel_balance_matches_direct_bp(self):
        device = "cpu"
        beams = [
            dict(el_width=0.8, az_width=0.25, az_offset=0.0),
            dict(el_width=0.6, az_width=0.15, az_offset=0.05),
        ]
        imgs_bp, imgs_ffbp = [], []
        tr = tth = None
        for b in beams:
            data, pos, att, g, g_extent, dem, tr, tth = self._scene(
                device, **b)
            imgs_bp.append(torchbp.ops.backprojection_polar_2d(
                data, self.grid, self.fc, self.r_res, pos,
                att=att, g=g, g_extent=g_extent, dem=dem, dealias=True)[0])
            imgs_ffbp.append(torchbp.ops.ffbp(
                data, self.grid, self.fc, self.r_res, pos, stages=3,
                dealias=True, att=att, g=g, g_extent=g_extent, dem=dem))

        for (ir, it) in self._target_pixels(tr, tth):
            # Same pixel for all four images: peak of the first BP channel.
            pr, pt = self._peak_pixel(imgs_bp[0], ir, it)
            ratio_bp = (imgs_bp[0][pr, pt] / imgs_bp[1][pr, pt])
            ratio_ff = (imgs_ffbp[0][pr, pt] / imgs_ffbp[1][pr, pt])
            bal_db = 20 * np.log10(abs(ratio_ff) / abs(ratio_bp))
            phase_deg = np.angle(
                (ratio_ff / ratio_bp).item(), deg=True)
            self.assertLess(abs(bal_db), 0.5,
                            f"channel balance off by {bal_db:.2f} dB at "
                            f"target pixel ({pr}, {pt})")
            self.assertLess(abs(phase_deg), 5.0,
                            f"channel phase off by {phase_deg:.2f} deg at "
                            f"target pixel ({pr}, {pt})")


if __name__ == "__main__":
    unittest.main()
