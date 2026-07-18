#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import unittest.mock
import torchbp
from torch import Tensor
from conftest import requires_cuda


class TestFfbpDem(TestCase):
    """DEM support through the full ffbp merge tree."""

    def _random_scene(self, device):
        torch.manual_seed(11)
        nsweeps = 64
        sweep_samples = 512
        grid = {"r": (30.0, 60.0), "theta": (-0.4, 0.4), "nr": 96, "ntheta": 128}
        data = torch.randn(nsweeps, sweep_samples, device=device,
                           dtype=torch.complex64)
        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-2, 2, nsweeps, device=device)
        pos[:, 2] = 25.0
        return data, grid, pos

    def _test_constant_dem_equals_shifted_pos(self, device):
        # A constant DEM h through the whole merge tree (DEM-referenced base
        # dealias and merge carriers, per-node resampling) must equal flat
        # ffbp with the platform lowered by h: every distance and carrier is
        # identical.
        data, grid, pos = self._random_scene(device)
        h = 9.0
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        res = torchbp.ops.ffbp(data, grid, 6e9, 0.15, pos, stages=3, dem=dem)
        pos_shift = pos.clone()
        pos_shift[:, 2] -= h
        ref = torchbp.ops.ffbp(data, grid, 6e9, 0.15, pos_shift, stages=3)
        self.assertLess(
            (torch.linalg.norm(res - ref) / torch.linalg.norm(ref)).item(),
            1e-3)

    def test_constant_dem_equals_shifted_pos_cpu(self):
        self._test_constant_dem_equals_shifted_pos("cpu")

    @requires_cuda
    def test_constant_dem_equals_shifted_pos_cuda(self):
        self._test_constant_dem_equals_shifted_pos("cuda")

    def _terrain_scene(self, device):
        # Point targets on a smooth terrain surface. The theta grid must
        # oversample the aperture azimuth bandwidth lambda / (2 * L) for the
        # ffbp merges: L = 10 m at 6 GHz gives 2.5e-3 in sin(theta), the
        # grid steps 1.17e-3.
        c0 = 299792458.0
        fc = 6e9
        bw = 200e6
        tsweep = 100e-6
        fs = 10e6
        nsamples = int(fs * tsweep)
        oversample = 2
        r_res = c0 / (2 * bw * oversample)
        nsweeps = 384
        alt = 30.0

        grid = {"r": (60.0, 100.0), "theta": (-0.3, 0.3), "nr": 320, "ntheta": 512}

        def terrain(r, t):
            return 8.0 + 5.0 * torch.sin(2 * torch.pi * (r - 60.0) / 45.0) \
                + 3.0 * t

        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        nr, ntheta = grid["nr"], grid["ntheta"]
        rr = r0 + (r1 - r0) / nr * torch.arange(nr, device=device)
        tt = t0 + (t1 - t0) / ntheta * torch.arange(ntheta, device=device)
        dem = terrain(rr[:, None], tt[None, :]).float()

        tr = torch.tensor([70.0, 80.0, 90.0], device=device)
        tth = torch.tensor([-0.15, 0.0, 0.12], device=device)
        tz = terrain(tr, tth)
        targets = torch.stack(
            [tr * torch.sqrt(1 - tth**2), tr * tth, tz], dim=-1)

        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-5, 5, nsweeps, device=device)
        pos[:, 2] = alt

        rcs = torch.ones((targets.shape[0], 1), device=device)
        raw = torchbp.util.generate_fmcw_data(
            targets, rcs, pos, fc, bw, tsweep, fs, rvp=False)
        w = torch.hamming_window(nsamples, periodic=False, device=device)
        data = torch.fft.ifft(raw * w[None, :], dim=-1, n=nsamples * oversample)
        return data, grid, fc, r_res, pos, dem, tr, tth

    def _test_ffbp_dem_matches_direct_bp(self, device):
        data, grid, fc, r_res, pos, dem, tr, tth = self._terrain_scene(device)
        img_bp = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos, dem=dem)[0]
        img_ffbp = torchbp.ops.ffbp(
            data, grid, fc, r_res, pos, stages=3, dem=dem)
        rel = (torch.linalg.norm(img_ffbp - img_bp)
               / torch.linalg.norm(img_bp)).item()
        # The merge interpolation error on a sparse point scene dominates
        # the relative L2, so the meaningful check is that the DEM does not
        # degrade the ffbp accuracy relative to the flat-earth case.
        img_bp_flat = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos)[0]
        img_ffbp_flat = torchbp.ops.ffbp(
            data, grid, fc, r_res, pos, stages=3)
        rel_flat = (torch.linalg.norm(img_ffbp_flat - img_bp_flat)
                    / torch.linalg.norm(img_bp_flat)).item()
        self.assertLess(rel, 0.25)
        self.assertLess(rel, rel_flat + 0.02)

        # Peaks land on the true target ground positions.
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        dr = (r1 - r0) / grid["nr"]
        dt = (t1 - t0) / grid["ntheta"]
        a = torch.abs(img_ffbp)
        for k in range(tr.shape[0]):
            ir = int(round(((tr[k].item() - r0) / dr)))
            it = int(round(((tth[k].item() - t0) / dt)))
            win = a[max(0, ir - 8):ir + 9, max(0, it - 8):it + 9]
            peak = torch.argmax(win)
            pr = peak.item() // win.shape[1] + max(0, ir - 8)
            pt = peak.item() % win.shape[1] + max(0, it - 8)
            self.assertLessEqual(abs(pr - ir), 2)
            self.assertLessEqual(abs(pt - it), 2)

    def test_ffbp_dem_matches_direct_bp_cpu(self):
        self._test_ffbp_dem_matches_direct_bp("cpu")

    @requires_cuda
    def test_ffbp_dem_matches_direct_bp_cuda(self):
        self._test_ffbp_dem_matches_direct_bp("cuda")

    def test_afbp_base_matches_direct_base(self):
        # ffbp with the afbp base level and a DEM must match ffbp with the
        # direct backprojection base level.
        data, grid, fc, r_res, pos, dem, tr, tth = self._terrain_scene("cpu")
        img1 = torchbp.ops.ffbp(data, grid, fc, r_res, pos, stages=3,
                                dem=dem)
        img2 = torchbp.ops.ffbp(data, grid, fc, r_res, pos, stages=3,
                                dem=dem, afbp_nsub=4)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 1e-2)


class TestFFBPAntennaPattern(TestCase):
    """End-to-end antenna-pattern-weighted FFBP must match between CPU and CUDA."""

    @staticmethod
    def _make_inputs(device):
        torch.manual_seed(0)
        nsweeps = 64
        sweep_samples = 128
        fc = 6e9
        r_res = 0.15
        grid = {"r": (4, 8), "theta": (-0.3, 0.3), "nr": 64, "ntheta": 128}

        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = (
            torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps, device=device)
            * 0.25 * 3e8 / fc
        )
        pos[:, 2] = 2.0   # low altitude so the scene is within sweep range

        data = torch.randn(nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        nel, naz = 16, 24
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-2.0, 2.0, naz, device=device)
        gain = torch.exp(-(el[:, None] / 0.8) ** 2) * torch.exp(-(az[None, :] / 0.25) ** 2)
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]

        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        att[:, 0] = -float(np.arcsin(2.0 / 6.0))   # roll: point elevation beam at scene
        att[:, 2] = torch.linspace(-0.3, 0.3, nsweeps, device=device)
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    @staticmethod
    def _make_narrow_beam_inputs(device):
        """Narrow azimuth beam scanned across a wide swath. Many pixels are barely
        illuminated, so the W1/W2 normalization divides by near-zero illumination
        and (without regularization)."""
        torch.manual_seed(0)
        nsweeps = 128
        sweep_samples = 128
        fc = 6e9
        r_res = 0.15
        grid = {"r": (5, 12), "theta": (-0.5, 0.5), "nr": 96, "ntheta": 192}

        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = (
            torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps, device=device)
            * 0.25 * 3e8 / fc
        )
        pos[:, 2] = 3.0

        data = torch.randn(nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        nel, naz = 16, 64
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-1.5, 1.5, naz, device=device)
        # Very narrow azimuth beam
        gain = torch.exp(-(el[:, None] / 0.6) ** 2) * torch.exp(-(az[None, :] / 0.04) ** 2)
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]

        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        att[:, 0] = -float(np.arcsin(3.0 / 8.0))
        att[:, 2] = torch.linspace(-0.45, 0.45, nsweeps, device=device) # scan
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    def test_weighted_ffbp_no_blowup_downsample1(self):
        """Weighted FFBP at downsample=1 must stay bounded at swath
        edges instead of producing a large spike from dividing by near-zero
        illumination."""
        a = self._make_narrow_beam_inputs("cpu")
        kw = dict(stages=3, divisions=2, dealias=True,
                  att=a["att"], g=a["g"], g_extent=a["g_extent"])
        img = torchbp.ops.ffbp(a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
                               weight_map_downsample=1, **kw)
        self.assertTrue(torch.isfinite(img).all())
        # Output must be of the same order as the unweighted backprojection.
        ref = torchbp.ops.backprojection_polar_2d(
            a["data"], a["grid"], a["fc"], a["r_res"], a["pos"], dealias=True)
        self.assertLess(float(img.abs().max()), 50.0 * float(ref.abs().max()))

    @requires_cuda
    def test_weighted_ffbp_cpu_gpu(self):
        a = self._make_inputs("cuda")

        def to_cpu(v):
            return v.cpu() if isinstance(v, torch.Tensor) else v

        a_cpu = {k: to_cpu(v) for k, v in a.items()}
        kw = dict(stages=2, divisions=2, dealias=True)
        out_gpu = torchbp.ops.ffbp(
            a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
            att=a["att"], g=a["g"], g_extent=a["g_extent"], **kw,
        )
        out_cpu = torchbp.ops.ffbp(
            a_cpu["data"], a_cpu["grid"], a_cpu["fc"], a_cpu["r_res"], a_cpu["pos"],
            att=a_cpu["att"], g=a_cpu["g"], g_extent=a_cpu["g_extent"], **kw,
        )
        # Poly-Knab merge differs at the ULP level between CPU and CUDA and is
        # amplified by the weighted normalization (cf. TestFFBPMerge2PolyWeighted).
        torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=3e-2, rtol=5e-2)


class TestFfbpDataInterpMethod(TestCase):
    """data_interp_method plumbing to the base level backprojections."""

    fc = 6e9
    r_res = 0.5
    nsamples = 256
    nsweeps = 64
    grid = {"r": (100.0, 150.0), "theta": (-0.2, 0.2), "nr": 64, "ntheta": 64}

    def _scene(self):
        # Point targets with a 2x oversampled sinc envelope, like the afbp
        # tests: smooth enough that both linear and knab interpolation
        # follow it and the ffbp merge error dominates the comparison.
        c0 = 299792458.0
        lam = c0 / self.fc
        pos = torch.zeros((self.nsweeps, 3))
        pos[:, 1] = lam / 4 * (torch.arange(self.nsweeps) - self.nsweeps / 2)
        targets = [(125.0, 0.0), (110.0, 0.15), (140.0, -0.12)]
        data = torch.zeros((self.nsweeps, self.nsamples), dtype=torch.complex64)
        i = torch.arange(self.nsamples, dtype=torch.float64)
        for r, th in targets:
            tx = r * np.sqrt(1 - th * th)
            ty = r * th
            d = torch.sqrt((pos[:, 0].double() - tx) ** 2
                           + (pos[:, 1].double() - ty) ** 2)
            env = torch.special.sinc((i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        return data, pos

    def test_few_sweeps_matches_direct_exactly(self):
        # nsweeps < divisions takes the direct backprojection early path,
        # which must forward the method verbatim.
        data, pos = self._scene()
        data, pos = data[:1], pos[:1]
        method = ("knab", 6, 2.0)
        out = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                               stages=3, data_interp_method=method)
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, interp_method=method)[0]
        self.assertEqual((out - ref).abs().max().item(), 0.0)

    def test_leaf_calls_get_method(self):
        # Spy on the leaf backprojections: every one must receive the
        # method (ffbp-vs-direct accuracy itself is merge-limited and
        # method-independent, so an end-to-end comparison cannot see the
        # plumbing).
        import sys
        import torchbp.ops.ffbp
        ffbp_mod = sys.modules["torchbp.ops.ffbp"]
        data, pos = self._scene()
        method = ("knab", 6, 2.0)
        seen = []
        real_bp = ffbp_mod.backprojection_polar_2d

        def spy(*args, **kwargs):
            seen.append(kwargs.get("interp_method"))
            return real_bp(*args, **kwargs)

        with unittest.mock.patch.object(
                ffbp_mod, "backprojection_polar_2d", side_effect=spy):
            out = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                   stages=2, data_interp_method=method)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(m == method for m in seen))
        # ffbp output must track the leaf method: identical run with linear
        # leaves differs.
        out_lin = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                   stages=2)
        self.assertGreater((out - out_lin).abs().max().item(), 0.0)

    def test_afbp_base_gets_method(self):
        import sys
        import torchbp.ops.ffbp
        ffbp_mod = sys.modules["torchbp.ops.ffbp"]
        data, pos = self._scene()
        method = ("knab", 6, 2.0)
        seen = []
        real_afbp = ffbp_mod.afbp

        def spy(*args, **kwargs):
            seen.append(kwargs.get("data_interp_method"))
            return real_afbp(*args, **kwargs)

        with unittest.mock.patch.object(ffbp_mod, "afbp", side_effect=spy):
            out = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                   stages=2, afbp_nsub=2,
                                   data_interp_method=method)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(m == method for m in seen))

    def test_invalid_method(self):
        data, pos = self._scene()
        with self.assertRaises(ValueError):
            torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                             stages=2, data_interp_method=("knab", 6))


class TestFFBPRemainderSweeps(TestCase):
    """ffbp must use every sweep when divisions does not divide nsweeps.

    Regression test: the subaperture split used to truncate to
    nsweeps // divisions * divisions sweeps at every recursion level,
    silently dropping the remainder.
    """

    def _last_sweep_energy(self, nsweeps, stages, divisions=2):
        grid = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 32, "ntheta": 32}
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, nsweeps)
        pos[:, 2] = 30.0
        # Only the last sweep carries signal. If it is dropped the image is
        # exactly zero.
        data = torch.zeros(nsweeps, 64, dtype=torch.complex64)
        data[-1, 50] = 1.0
        img = torchbp.ops.ffbp(
            data, grid, 6e9, 2.0, pos, stages=stages, divisions=divisions
        )
        return img.abs().sum().item()

    def test_odd_nsweeps(self):
        for stages in (1, 2, 3):
            self.assertGreater(self._last_sweep_energy(513, stages), 0.0)

    def test_divisions3(self):
        self.assertGreater(self._last_sweep_energy(1000, 2, divisions=3), 0.0)


class TestFfbpFrameOrigin(TestCase):
    """ffbp must reference the output grid to the coordinate frame origin
    like backprojection_polar_2d, not to the mean of the input positions.
    """

    def _scene(self, device):
        c0 = 299792458.0
        fc = 6e9
        bw = 200e6
        tsweep = 100e-6
        fs = 10e6
        oversample = 2
        r_res = c0 / (2 * bw * oversample)
        nsweeps = 192
        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-4, 4, nsweeps, device=device)
        pos[:, 2] = 25.0
        targets = torch.tensor([[80.0, 0.0, 0.0], [70.0, -8.0, 0.0],
                                [95.0, 10.0, 0.0]], device=device)
        rcs = torch.ones((targets.shape[0], 1), device=device,
                         dtype=torch.complex64)
        raw = torchbp.util.generate_fmcw_data(
            targets, rcs, pos, fc, bw, tsweep, fs, rvp=False)
        nsamples = raw.shape[-1]
        w = torch.hamming_window(nsamples, periodic=False, device=device)
        data = torch.fft.ifft(raw * w[None, :], dim=-1,
                              n=nsamples * oversample)
        grid = {"r": (60.0, 110.0), "theta": (-0.3, 0.3),
                "nr": 256, "ntheta": 256}
        return data, grid, fc, r_res, pos

    def _test_nonzero_mean_pos_matches_bp(self, device, divisions):
        data, grid, fc, r_res, pos = self._scene(device)
        # In-plane baseline-scale offset, like the InSAR reform case.
        off = torch.tensor([0.13, 0.585, 0.0], device=device)

        def rel(a, b):
            return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()

        def run(p):
            ref = torchbp.ops.backprojection_polar_2d(
                data, grid, fc, r_res, p, dealias=True)[0]
            out = torchbp.ops.ffbp(
                data, grid, fc, r_res, p, stages=4, divisions=divisions,
                dealias=True, grid_oversample=2.0)
            return out, ref

        out0, ref0 = run(pos)
        out_off, ref_off = run(pos + off)
        # Offset accuracy must match the zero-mean accuracy (merge
        # interpolation error only).
        self.assertLess(rel(out_off, ref_off), rel(out0, ref0) + 0.01)
        # And ffbp must not be translation invariant: the offset displaces
        # the scene and changes the phase reference.
        self.assertGreater(rel(out_off, out0), 0.5)

    def test_nonzero_mean_pos_matches_bp_cpu(self):
        self._test_nonzero_mean_pos_matches_bp("cpu", divisions=2)

    def test_nonzero_mean_pos_matches_bp_divisions3_cpu(self):
        # Odd image count exercises the trailing-image reduction path with
        # the frame-origin final merge.
        self._test_nonzero_mean_pos_matches_bp("cpu", divisions=3)

    @requires_cuda
    def test_nonzero_mean_pos_matches_bp_cuda(self):
        self._test_nonzero_mean_pos_matches_bp("cuda", divisions=2)


class TestComputeIllumination(TestCase):
    """Test antenna pattern illumination map computation (compute_illumination op)."""

    def _make_inputs(self, device):
        torch.manual_seed(0)
        nsweeps = 8
        pos = torch.randn(nsweeps, 3, device=device, dtype=torch.float32)
        att = torch.zeros(nsweeps, 3, device=device, dtype=torch.float32)
        g = torch.ones(16, 16, device=device, dtype=torch.float32)
        g_extent = [-0.5, -1.0, 0.5, 1.0]
        grid = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 50, "ntheta": 40}
        return pos, att, g, g_extent, grid

    def _test_basic(self, device):
        from torchbp.ops import compute_subaperture_illumination

        pos, att, g, g_extent, grid = self._make_inputs(device)
        w1_map, w2_map = compute_subaperture_illumination(
            pos, att, g, g_extent, grid, decimation=1
        )
        # Output shape matches the (undecimated) polar grid
        self.assertEqual(tuple(w1_map.shape), (grid["nr"], grid["ntheta"]))
        self.assertEqual(tuple(w2_map.shape), (grid["nr"], grid["ntheta"]))
        for m in (w1_map, w2_map):
            self.assertFalse(torch.isnan(m).any())
            self.assertFalse(torch.isinf(m).any())
        # With a uniform unit gain pattern the sum of squared gains never exceeds
        # the sum of gains.
        self.assertTrue(torch.all(w2_map <= w1_map + 1e-4))

    def test_basic_cpu(self):
        self._test_basic("cpu")

    @requires_cuda
    def test_basic_cuda(self):
        self._test_basic("cuda")

    @requires_cuda
    def test_cpu_and_gpu(self):
        from torchbp.ops import compute_subaperture_illumination

        pos, att, g, g_extent, grid = self._make_inputs("cuda")
        w1_gpu, w2_gpu = compute_subaperture_illumination(
            pos, att, g, g_extent, grid, decimation=1
        )
        w1_cpu, w2_cpu = compute_subaperture_illumination(
            pos.cpu(), att.cpu(), g.cpu(), g_extent, grid, decimation=1
        )
        torch.testing.assert_close(w1_cpu, w1_gpu.cpu(), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(w2_cpu, w2_gpu.cpu(), atol=1e-4, rtol=1e-3)


class TestFFBPTxPower(TestCase):
    """backprojection_polar_2d_tx_power_ffbp must match the direct backprojection_polar_2d_tx_power."""

    @staticmethod
    def _antenna(device, az_width=0.25, el_width=0.8):
        nel, naz = 16, 24
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-2.0, 2.0, naz, device=device)
        gain = torch.exp(-(el[:, None] / el_width) ** 2) * torch.exp(
            -(az[None, :] / az_width) ** 2
        )
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]
        return g, g_extent

    @staticmethod
    def _straight_track(device, nsweeps=128, span=32.0, alt=20.0, r_center=60.0):
        torch.manual_seed(0)
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = torch.linspace(-span / 2, span / 2, nsweeps, device=device)
        pos[:, 2] = alt
        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        d = float(np.sqrt(r_center**2 + alt**2))
        att[:, 0] = -float(np.arcsin(alt / d))
        wa = (torch.hann_window(nsweeps, device=device) + 0.1).to(torch.float32)
        return wa, pos, att

    def _assert_matches(self, out, ref, q95=0.03, emax=0.1, margin=4, min_ratio=None):
        self.assertEqual(out.shape, ref.shape)
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
        # Infinite (unilluminated) pixels should agree except for isolated
        # interpolation edge cases.
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.99)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        self.assertGreater(int(finite.sum()), 0)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 100)
        ratio = out_c[mask] / ref_c[mask]
        err = (ratio - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), q95)
        self.assertLess(float(err.max()), emax)
        if min_ratio is not None:
            self.assertGreater(float(ratio.min()), min_ratio)

    def _run_both(self, wa, g, g_extent, grid, pos, att, device="cpu", altitude=0.0,
                  **kw):
        kw.setdefault("stages", 3)
        kw.setdefault("downsample_r", 1.0)
        kw.setdefault("downsample_theta", 1.0)
        kw.setdefault("min_nsweeps", 16)
        if altitude > 0.0:
            ref = torchbp.ops.backprojection_polar_2d_tx_power_slant(
                wa, g, g_extent, grid, 0.15, pos, att, altitude=altitude,
                normalization=kw.get("normalization"),
                azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        else:
            ref = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, grid, 0.15, pos, att,
                normalization=kw.get("normalization"),
                azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        out = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, grid, 0.15, pos, att, altitude=altitude, **kw)
        return out, ref

    def test_straight_track(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma")
        self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_long_track_narrow_theta(self):
        """Subapertures at the track ends see the scene at local theta far
        outside the narrow output theta range. A shared theta grid would lose
        their contributions entirely."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.8)
        wa, pos, att = self._straight_track(
            device, nsweeps=256, span=300.0, alt=20.0, r_center=120.0)
        grid = {"r": (100, 140), "theta": (-0.05, 0.05), "nr": 64, "ntheta": 64}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", stages=4)
        # min_ratio catches a systematic deficit from dropped subapertures.
        self._assert_matches(out, ref, q95=0.03, emax=0.1, min_ratio=0.9)

    def test_curved_track(self):
        device = "cpu"
        torch.manual_seed(0)
        nsweeps = 256
        g, g_extent = self._antenna(device, az_width=0.5)
        R = 150.0
        alpha = torch.linspace(-0.5, 0.5, nsweeps, device=device)
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 0] = R - R * torch.cos(alpha)
        pos[:, 1] = R * torch.sin(alpha)
        pos[:, 2] = 20.0
        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        # Yaw tracks the scene center so every pulse illuminates the scene.
        att[:, 2] = torch.atan2(-pos[:, 1], 100.0 - pos[:, 0])
        att[:, 0] = -float(np.arcsin(20.0 / np.sqrt(100.0**2 + 20.0**2)))
        wa = (torch.hann_window(nsweeps, device=device) + 0.1).to(torch.float32)
        grid = {"r": (80, 120), "theta": (-0.3, 0.3), "nr": 64, "ntheta": 128}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", stages=4)
        self._assert_matches(out, ref, q95=0.04, emax=0.15)

    def test_thin_beam_deep_stages(self):
        """A beam that is thin compared to the output grid theta step used to
        poison the psi moments with NaN (0/0 in the Welford update when the
        pulse weight underflows to zero) and undersample the illumination
        rolloff in the subaperture maps. Both showed up as heavy azimuth
        errors that grew with the number of stages."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.05)
        wa, pos, att = self._straight_track(
            device, nsweeps=512, span=100.0, alt=40.0, r_center=150.0)
        grid = {"r": (80, 220), "theta": (-0.6, 0.6), "nr": 128, "ntheta": 256}
        out, ref = self._run_both(
            wa, g, g_extent, grid, pos, att, normalization="sigma",
            stages=100, downsample_r=4.0, downsample_theta=4.0,
            min_nsweeps=32)
        self.assertFalse(bool(torch.isnan(out).any()))
        out_c = out[4:-4, 4:-4]
        ref_c = ref[4:-4, 4:-4]
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.98)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 1000)
        err = (out_c[mask] / ref_c[mask] - 1).abs()
        self.assertLess(float(err.median()), 0.01)
        # Isolated illumination edge pixels can still be off, quantile only.
        self.assertLess(float(torch.quantile(err, 0.95)), 0.05)

    def test_zero_gain_no_nan(self):
        """Antenna patterns with exact zeros inside the table (measured or
        rotated patterns) and windows with zero endpoints must not produce
        NaN moments in either the direct kernel or the accumulator maps."""
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum

        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.1)
        g = torch.where(g > 1e-3 * g.max(), g, torch.zeros_like(g))
        wa, pos, att = self._straight_track(device, nsweeps=128)
        wa = torch.hann_window(128).to(torch.float32)  # exact zero endpoints
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 64, "ntheta": 128}
        acc = _backprojection_polar_2d_tx_power_accum(
            wa, g, g_extent, grid, pos, att, "sigma", (80 - 40) / 64, 20.0)
        self.assertTrue(bool(torch.isfinite(acc).all()))
        ref = torchbp.ops.backprojection_polar_2d_tx_power(
            wa, g, g_extent, grid, 0.15, pos, att, normalization="sigma")[0]
        self.assertFalse(bool(torch.isnan(ref).any()))
        # Illuminated pixels (W > 0 needs at least two pulses for a nonzero
        # aperture) must be finite in the direct output too.
        self.assertGreater(int((torch.isfinite(ref) & (acc[1] > 0)).sum()), 1000)

    def test_slant(self):
        device = "cpu"
        alt = 30.0
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, alt=0.0, r_center=60.0)
        att[:, 0] = -float(np.arcsin(alt / 60.0))
        grid = {"r": (25, 90), "theta": (-0.5, 0.5), "nr": 96, "ntheta": 128}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", altitude=alt)
        # Shadow zone below nadir must be exactly zero in both.
        r_vec = 25 + (90 - 25) / 96 * torch.arange(96)
        t_vec = -0.5 + 1.0 / 128 * torch.arange(128)
        rg2 = r_vec[:, None] ** 2 * (1 - t_vec[None, :] ** 2) - alt**2
        shadow = rg2 < 0
        self.assertGreater(int(shadow.sum()), 0)
        self.assertTrue((ref[shadow] == 0).all())
        self.assertTrue((out[shadow] == 0).all())
        # The slant range to ground range mapping is singular at the shadow
        # boundary and the accumulator fields cannot be interpolated
        # accurately there. Exclude the steep fold zone (ground range less
        # than half the slant range) in addition to the shadow zone itself.
        steep = rg2 < (r_vec[:, None] / 2) ** 2
        out = torch.where(steep, torch.inf, out)
        ref = torch.where(steep, torch.inf, ref)
        self._assert_matches(out, ref, q95=0.04, emax=0.15)

    def test_normalization_variants(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, nsweeps=64)
        grid = {"r": (40, 80), "theta": (-0.4, 0.4), "nr": 48, "ntheta": 64}
        for normalization in ["beta", "sigma", "gamma", "point"]:
            out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                      normalization=normalization, stages=2)
            self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_azimuth_resolution_false(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", azimuth_resolution=False)
        self.assertTrue(torch.isfinite(out).all())
        self._assert_matches(out, ref, q95=0.02, emax=0.05)

    def test_downsampled_subaperture_maps(self):
        """Coarse subaperture maps must stay reasonably accurate."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma",
                                  downsample_r=4.0, downsample_theta=4.0)
        self._assert_matches(out, ref, q95=0.05, emax=0.2)

    def _merge_exactness(self, device):
        """Merging two half track accumulator maps on the same grid must equal
        the full track accumulator map (Chan's formula is exact)."""
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum
        from torchbp.ops import ffbp_tx_power_merge2

        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, nsweeps=64)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 48, "ntheta": 64}
        kw = dict(normalization="sigma", dr_ref=(80 - 40) / 48, h_ref=20.0)
        full = _backprojection_polar_2d_tx_power_accum(
            wa, g, g_extent, grid, pos, att, **kw)
        acc0 = _backprojection_polar_2d_tx_power_accum(
            wa[:32], g, g_extent, grid, pos[:32], att[:32], **kw)
        acc1 = _backprojection_polar_2d_tx_power_accum(
            wa[32:], g, g_extent, grid, pos[32:], att[32:], **kw)
        zero = torch.zeros(3, device=device)
        merged = ffbp_tx_power_merge2(
            acc0, acc1, zero, zero.clone(), [grid, grid], grid)
        # The first and last rows and columns are on the edge of the bilinear
        # interpolation support of the merge kernel where one ulp of rounding
        # in the coordinate computation flips the bounds check.
        torch.testing.assert_close(
            merged[:, 1:-1, 1:-1], full[:, 1:-1, 1:-1], atol=1e-5, rtol=1e-2)

    def test_merge_exactness_cpu(self):
        self._merge_exactness("cpu")

    @requires_cuda
    def test_merge_exactness_cuda(self):
        self._merge_exactness("cuda")

    @requires_cuda
    def test_accum_cpu_cuda(self):
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum

        for altitude in (0.0, 30.0):
            g, g_extent = self._antenna("cuda", az_width=0.4)
            wa, pos, att = self._straight_track(
                "cuda", nsweeps=64, alt=0.0 if altitude > 0 else 20.0)
            att[:, 0] = -float(np.arcsin(max(altitude, 20.0) / 65.0))
            grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 48, "ntheta": 64}
            kw = dict(normalization="sigma", dr_ref=(80 - 40) / 48,
                      h_ref=altitude if altitude > 0 else 20.0, altitude=altitude)
            out_gpu = _backprojection_polar_2d_tx_power_accum(
                wa, g, g_extent, grid, pos, att, **kw)
            out_cpu = _backprojection_polar_2d_tx_power_accum(
                wa.cpu(), g.cpu(), g_extent, grid, pos.cpu(), att.cpu(), **kw)
            torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-3)

    @requires_cuda
    def test_ffbp_tx_power_cpu_cuda(self):
        g, g_extent = self._antenna("cuda", az_width=0.4)
        wa, pos, att = self._straight_track("cuda")
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        kw = dict(stages=3, downsample_r=1.0, downsample_theta=1.0,
                  min_nsweeps=16, normalization="sigma")
        out_gpu = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        out_cpu = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa.cpu(), g.cpu(), g_extent, grid, 0.15, pos.cpu(), att.cpu(), **kw)
        finite = torch.isfinite(out_cpu) & torch.isfinite(out_gpu.cpu())
        self.assertGreater(float(finite.float().mean()), 0.95)
        torch.testing.assert_close(
            out_cpu[finite], out_gpu.cpu()[finite], atol=1e-3, rtol=1e-2)

    def _opcheck(self, device):
        wa = torch.rand(4, device=device) + 0.5
        pos = torch.randn(4, 3, device=device)
        pos[:, 0] -= 5.0
        att = torch.zeros(4, 3, device=device)
        g = torch.rand(8, 8, device=device)
        opcheck(
            torch.ops.torchbp.backprojection_polar_2d_tx_power_accum,
            (wa, pos, att, g, -1.0, -0.5, 2.0 / 8, 1.0 / 8, 8, 8, 4,
             1.0, 0.5, -0.9, 0.2, 8, 8, 1, 0.5, 10.0, 0.0, 0, None),
            test_utils=["test_schema"],
        )
        acc0 = torch.rand(4, 8, 8, device=device)
        acc1 = torch.rand(4, 8, 8, device=device)
        dorigin = torch.zeros(2, 3, device=device)
        r0 = torch.ones(2, device=device)
        dr0 = torch.full((2,), 0.5, device=device)
        theta0 = torch.full((2,), -0.9, device=device)
        dtheta0 = torch.full((2,), 0.2, device=device)
        nr0 = torch.full((2,), 8, dtype=torch.int32, device=device)
        ntheta0 = torch.full((2,), 8, dtype=torch.int32, device=device)
        opcheck(
            torch.ops.torchbp.ffbp_tx_power_merge2,
            (acc0, acc1, dorigin, r0, dr0, theta0, dtheta0, nr0, ntheta0,
             1.0, 0.5, -0.9, 0.2, 8, 8, 0.0, 0, 0),
            test_utils=["test_schema"],
        )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @requires_cuda
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def test_matches_polar_at_coincident_ground_points(self):
        """The Cartesian tx_power kernel maps a pixel's ground position
        directly from the grid; the polar kernel maps (r, theta) to the same
        ground point. Evaluated at coincident ground positions with the
        azimuth-resolution normalization off, they must agree."""
        import math
        from torchbp.ops.backproj import (
            backprojection_cart_2d_tx_power,
            backprojection_polar_2d_tx_power,
        )

        torch.manual_seed(1)
        nsweeps = 32
        wa = torch.rand(nsweeps)
        g = torch.rand(16, 16)
        g_extent = [-1.4, -1.2, 1.4, 1.2]
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-20, 20, nsweeps)
        pos[:, 2] = 100.0
        att = torch.zeros(nsweeps, 3)

        r0, r1, nr = 150.0, 260.0, 40
        t0, t1, nt = -0.3, 0.3, 40
        pgrid = {"r": (r0, r1), "theta": (t0, t1), "nr": nr, "ntheta": nt}
        dr = (r1 - r0) / nr
        dt = (t1 - t0) / nt
        ir, it = 20, 25
        r = r0 + ir * dr
        t = t0 + it * dt
        x = r * math.sqrt(1 - t * t)
        y = r * t
        cgrid = {"x": (x, x + 1.0), "y": (y, y + 1.0), "nx": 1, "ny": 1}

        for norm in ["beta", "sigma", "gamma", "point"]:
            pol = backprojection_polar_2d_tx_power(
                wa, g, g_extent, pgrid, 0.15, pos, att,
                normalization=norm, azimuth_resolution=False)[0]
            cart = backprojection_cart_2d_tx_power(
                wa, g, g_extent, cgrid, 0.15, pos, att,
                normalization=norm, azimuth_resolution=False)[0, 0, 0]
            torch.testing.assert_close(
                cart, pol[ir, it], atol=1e-6, rtol=1e-5)

    @staticmethod
    def sample_inputs(device):
        # Track at 20 m altitude looking at ~60 m slant range: ground range
        # ~57 m, so the grid brackets the illuminated patch with some
        # unilluminated (inf) margin.
        g, g_extent = TestFFBPTxPower._antenna(device, az_width=0.4)
        wa, pos, att = TestFFBPTxPower._straight_track(device, nsweeps=64)
        grid = {"x": (40.0, 80.0), "y": (-30.0, 30.0), "nx": 48, "ny": 64}
        return [dict(wa=wa, g=g, g_extent=g_extent, grid=grid, r_res=0.15,
                     pos=pos, att=att, normalization="sigma")]

    @requires_cuda
    def test_cpu_cuda_match(self):
        from torchbp.ops.backproj import backprojection_cart_2d_tx_power

        args = self.sample_inputs("cpu")[0]
        out_cpu = backprojection_cart_2d_tx_power(**args)
        gargs = {k: (v.cuda() if torch.is_tensor(v) else v)
                 for k, v in args.items()}
        out_cuda = backprojection_cart_2d_tx_power(**gargs).cpu()
        fin = torch.isfinite(out_cpu) & torch.isfinite(out_cuda)
        torch.testing.assert_close(out_cpu[fin], out_cuda[fin],
                                   atol=1e-6, rtol=1e-3)
        self.assertTrue(
            (torch.isinf(out_cpu) == torch.isinf(out_cuda)).all())


class TestFFBPLongBaseline(TestCase):
    """On a long baseline the subapertures away from the merge origin see the
    far/near scene at a shifted range (largest toward |theta| = 1). Their
    grids must extend in range so the merge lookups stay inside them,
    otherwise edge targets at high |theta| lose the contribution of the
    offset subapertures and dim. See the range guard band in _ffbp_impl."""

    fc = 6e9
    r_res = 0.5
    nsamples = 600
    nsweeps = 1024

    def _scene(self, grid, targets, device="cpu"):
        c0 = 299792458.0
        lam = c0 / self.fc
        pos = torch.zeros((self.nsweeps, 3), device=device)
        # Aperture ~25 m: offset of the top-level subapertures from the merge
        # origin is a several-meter fraction of the ~100 m range, enough to
        # push rp off the grid by many range bins at |theta| ~ 0.8.
        pos[:, 1] = (lam / 4) * (torch.arange(self.nsweeps, device=device)
                                 - self.nsweeps / 2)
        data = torch.zeros((self.nsweeps, self.nsamples),
                           dtype=torch.complex64, device=device)
        i = torch.arange(self.nsamples, dtype=torch.float64, device=device)
        for r, th in targets:
            tx = r * np.sqrt(1 - th * th)
            ty = r * th
            d = torch.sqrt((pos[:, 0].double() - tx) ** 2
                           + (pos[:, 1].double() - ty) ** 2
                           + pos[:, 2].double() ** 2)
            env = torch.special.sinc(
                (i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        return data, pos

    def _peaks(self, ref, out, grid, targets):
        (r0, r1), (t0, t1) = grid["r"], grid["theta"]
        dr = (r1 - r0) / grid["nr"]
        dth = (t1 - t0) / grid["ntheta"]
        ratios = []
        for r, th in targets:
            ri = int(round((r - r0) / dr))
            ti = int(round((th - t0) / dth))
            a = ref[max(0, ri - 3):ri + 4, max(0, ti - 3):ti + 4].abs().max()
            b = out[max(0, ri - 3):ri + 4, max(0, ti - 3):ti + 4].abs().max()
            ratios.append((b / a).item())
        return ratios

    def test_edge_targets_recovered_cpu(self):
        # Targets pinned to the range edges at high |theta|, where an offset
        # subaperture's rp leaves the [r0, r1] grid without the range guard.
        c0 = 299792458.0
        lam = c0 / self.fc
        aperture = (lam / 4) * (self.nsweeps - 1)
        r0f, r1f, th0, th1 = 100.0, 180.0, -0.85, 0.85
        # Sample azimuth finely enough for the aperture (else the scene
        # aliases regardless of the merge geometry).
        ntheta = int(np.ceil((th1 - th0) / (lam / (2 * aperture))))
        grid = {"r": (r0f, r1f), "theta": (th0, th1), "nr": 200,
                "ntheta": ntheta}
        targets = [(140.0, 0.0), (178.0, 0.82), (178.0, -0.82),
                   (102.0, 0.82), (102.0, -0.82)]
        data, pos = self._scene(grid, targets)
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos)[0]
        out = torchbp.ops.ffbp(
            data, grid, self.fc, self.r_res, pos, stages=6, grid_oversample=2.0)
        ratios = self._peaks(ref, out, grid, targets)
        # Broadside (no range shift) is a control; the edge targets are the
        # ones the range guard rescues. Without the guard they sit near 0.58.
        for (r, th), ratio in zip(targets, ratios):
            self.assertGreater(ratio, 0.9,
                               f"target r={r} th={th} dimmed to {ratio:.3f}")

    def test_odd_divisions_cpu(self):
        # The pairwise merge chain must land on the node centroid and merge
        # adjacent subapertures for ANY divisions, not just powers of two.
        # Odd/non-power-of-two divisions previously drifted the output frame
        # (0.5/0.5 origin averaging) and merged non-adjacent subapertures
        # (gapped aperture -> azimuth aliasing), collapsing targets (e.g.
        # divisions=3 broadside to 0.13, divisions=5 to 0.31). The result must
        # not depend on divisions: every count should match divisions=2.
        r0f, r1f, th0, th1 = 100.0, 180.0, -0.3, 0.3
        grid = {"r": (r0f, r1f), "theta": (th0, th1), "nr": 200, "ntheta": 256}
        # Broadside (th=0) has no range shift, so it isolates the frame-drift
        # and merge-order bugs from the range-guard geometry.
        targets = [(140.0, 0.0), (120.0, 0.2), (170.0, -0.22), (110.0, 0.25)]
        data, pos = self._scene(grid, targets)
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos)[0]
        base = self._peaks(ref, torchbp.ops.ffbp(
            data, grid, self.fc, self.r_res, pos, stages=5, divisions=2,
            grid_oversample=2.0), grid, targets)
        for divisions in (3, 4, 5, 7):
            out = torchbp.ops.ffbp(
                data, grid, self.fc, self.r_res, pos, stages=5,
                divisions=divisions, grid_oversample=2.0)
            ratios = self._peaks(ref, out, grid, targets)
            for (r, th), ratio, base_ratio in zip(targets, ratios, base):
                self.assertAlmostEqual(
                    ratio, base_ratio, delta=0.05,
                    msg=f"divisions={divisions} target r={r} th={th}: peak "
                        f"{ratio:.3f} vs divisions=2 {base_ratio:.3f}")


class TestTxPowerPitch(TestCase):
    """Pitch in the tx_power kernels: the along-track attitude angle rotates
    the antenna pattern about its boresight (side-looking case). Per-sweep
    kernel pitch must be equivalent to zero pitch with the pattern
    pre-rotated about [1, 0, 0] (the rotate_antenna_pattern compensation the
    processing pipeline applies with the mean pitch)."""

    el_width = 0.8
    az_width = 0.25
    # Fine pattern sampling: both paths bilinearly interpolate the pattern at
    # different coordinates (rotated lookup vs pre-rotated table), so the
    # comparison error is the pattern interpolation error, quadratic in the
    # step-to-beamwidth ratio.
    nel, naz = 128, 512

    @classmethod
    def _gauss(cls, el, az):
        return np.exp(-((el / cls.el_width) ** 2) - (az / cls.az_width) ** 2)

    @classmethod
    def _patterns(cls, pitch):
        """Anisotropic Gaussian pattern and the same pattern rotated by
        `pitch` about the boresight, both sampled on the same grid. The
        rotated pattern is evaluated analytically: for each grid direction
        u(el, az) (x = cos(el)cos(az), y = cos(el)sin(az), z = sin(el)) take
        the original gain at the angles of R_x(pitch) @ u, which is what
        rotate_antenna_pattern computes without its interpolation error.

        The tables are sampled on the kernel's cell-start convention
        (sample k at angle0 + k * (angle1 - angle0) / n): the two paths
        rotate in different places (lookup coordinates vs table values), so
        any angle-to-index misregistration would not cancel between them."""
        el0, el1 = -1.2, 1.2
        az0, az1 = -2.0, 2.0
        el = el0 + (el1 - el0) / cls.nel * np.arange(cls.nel)
        az = az0 + (az1 - az0) / cls.naz * np.arange(cls.naz)
        EL, AZ = np.meshgrid(el, az, indexing="ij")
        g = cls._gauss(EL, AZ)
        ux = np.cos(EL) * np.cos(AZ)
        uy = np.cos(EL) * np.sin(AZ)
        uz = np.sin(EL)
        cp, sp = np.cos(pitch), np.sin(pitch)
        uyr = cp * uy - sp * uz
        uzr = sp * uy + cp * uz
        el_r = np.arcsin(np.clip(uzr, -1.0, 1.0))
        az_r = np.arctan2(uyr, ux)
        g_rot = cls._gauss(el_r, az_r)
        g_extent = [el0, az0, el1, az1]
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        return to_t(g), to_t(g_rot), g_extent

    @staticmethod
    def _track(pitch, nsweeps=128, alt=20.0, r_center=60.0, span=32.0):
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32)
        pos[:, 1] = torch.linspace(-span / 2, span / 2, nsweeps)
        pos[:, 2] = alt
        att = torch.zeros([nsweeps, 3], dtype=torch.float32)
        d = float(np.sqrt(r_center**2 + alt**2))
        att[:, 0] = -float(np.arcsin(alt / d))
        att[:, 1] = pitch
        wa = (torch.hann_window(nsweeps) + 0.1).to(torch.float32)
        return wa, pos, att

    def _rel_err(self, out, ref, margin=4):
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
        finite = torch.isfinite(out_c) & torch.isfinite(ref_c)
        mask = finite & (ref_c > 1e-3 * float(ref_c[finite].max()))
        self.assertGreater(int(mask.sum()), 1000)
        return (out_c[mask] / ref_c[mask] - 1).abs()

    def test_constant_pitch_equals_rotated_pattern(self):
        pitch = 0.2
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        g, g_rot, g_extent = self._patterns(pitch)
        wa, pos, att = self._track(pitch)
        att0 = att.clone()
        att0[:, 1] = 0.0
        for norm in [None, "sigma"]:
            out = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, grid, 0.15, pos, att,
                normalization=norm)[0]
            ref = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g_rot, g_extent, grid, 0.15, pos, att0,
                normalization=norm)[0]
            err = self._rel_err(out, ref)
            self.assertLess(float(torch.quantile(err, 0.95)), 5e-3)
            self.assertLess(float(err.max()), 2e-2)
            # Sanity: the pitch must actually matter for the anisotropic
            # pattern, or the equivalence above is vacuous.
            ignored = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, grid, 0.15, pos, att0,
                normalization=norm)[0]
            diff = self._rel_err(ignored, ref)
            self.assertGreater(float(torch.quantile(diff, 0.95)), 0.05)

    def test_ffbp_matches_direct_with_pitch(self):
        pitch = 0.2
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        g, _, g_extent = self._patterns(pitch)
        wa, pos, att = self._track(pitch)
        ref = torchbp.ops.backprojection_polar_2d_tx_power(
            wa, g, g_extent, grid, 0.15, pos, att, normalization="sigma")[0]
        out = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, grid, 0.15, pos, att, stages=3,
            downsample_r=1.0, downsample_theta=1.0, min_nsweeps=16,
            normalization="sigma")
        err = self._rel_err(out, ref)
        self.assertLess(float(torch.quantile(err, 0.95)), 0.03)


if __name__ == "__main__":
    unittest.main()
