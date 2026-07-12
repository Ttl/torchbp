#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp


class TestCartTxPowerCFBP(TestCase):
    """backprojection_cart_2d_tx_power_cfbp must match the direct
    backprojection_cart_2d_tx_power."""

    @staticmethod
    def _antenna(device, az_width=0.4, el_width=0.8):
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
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = torch.linspace(-span / 2, span / 2, nsweeps, device=device)
        pos[:, 2] = alt
        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        d = float(np.sqrt(r_center**2 + alt**2))
        att[:, 0] = -float(np.arcsin(alt / d))
        wa = (torch.hann_window(nsweeps, device=device) + 0.1).to(torch.float32)
        return wa, pos, att

    def _assert_matches(self, out, ref, q95=0.03, emax=0.1, margin=4):
        self.assertEqual(out.shape, ref.shape)
        # No illuminated pixel of the direct result may be dropped (turned to
        # inf) by the factorization, including the image border. Guards against
        # the merge losing edge rows/columns for want of interpolation margin.
        missing = torch.isfinite(ref) & ~torch.isfinite(out)
        self.assertEqual(int(missing.sum()), 0)
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
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

    def _run_both(self, wa, g, g_extent, grid, pos, att, **kw):
        kw.setdefault("stages", 3)
        kw.setdefault("downsample_x", 1.0)
        kw.setdefault("downsample_y", 1.0)
        kw.setdefault("min_nsweeps", 16)
        ref = torchbp.ops.backprojection_cart_2d_tx_power(
            wa, g, g_extent, grid, 0.15, pos, att,
            normalization=kw.get("normalization"),
            azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        out = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        return out, ref

    def test_straight_track(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        for norm in [None, "sigma", "gamma", "point"]:
            out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                      normalization=norm)
            self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_no_azimuth_resolution(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", azimuth_resolution=False)
        self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_downsampled_deep_stages(self):
        """Coarse subaperture maps (downsample 4) merged over many stages must
        stay accurate and free of NaN."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(
            device, nsweeps=512, span=60.0, alt=20.0, r_center=60.0)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 256, "ny": 512}
        out, ref = self._run_both(
            wa, g, g_extent, grid, pos, att, normalization="sigma",
            stages=100, downsample_x=4.0, downsample_y=4.0, min_nsweeps=32)
        self.assertFalse(bool(torch.isnan(out).any()))
        # No illuminated direct pixel dropped, even at the image border.
        self.assertEqual(int((torch.isfinite(ref) & ~torch.isfinite(out)).sum()), 0)
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

    def _opcheck(self, device):
        wa = torch.rand(4, device=device) + 0.5
        pos = torch.randn(4, 3, device=device)
        pos[:, 2] = 20.0
        att = torch.zeros(4, 3, device=device)
        g = torch.rand(8, 8, device=device)
        opcheck(
            torch.ops.torchbp.backprojection_cart_2d_tx_power_accum,
            (wa, pos, att, g, -1.0, -0.5, 2.0 / 8, 1.0 / 8, 8, 8, 4,
             45.0, 0.25, -25.0, 0.2, 8, 8, 1, 0.25, 20.0),
            test_utils=["test_schema"],
        )
        acc0 = torch.rand(4, 8, 8, device=device)
        acc1 = torch.rand(4, 8, 8, device=device)
        opcheck(
            torch.ops.torchbp.cart_tx_power_merge2,
            (acc0, acc1,
             45.0, 0.25, -25.0, 0.2, 8, 8,
             45.0, 0.25, -25.0, 0.2, 8, 8,
             45.0, 0.25, -25.0, 0.2, 8, 8),
            test_utils=["test_schema"],
        )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_match(self):
        g, g_extent = self._antenna("cuda", az_width=0.4)
        wa, pos, att = self._straight_track("cuda")
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        kw = dict(stages=3, downsample_x=1.0, downsample_y=1.0,
                  min_nsweeps=16, normalization="sigma")
        out_gpu = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        out_cpu = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa.cpu(), g.cpu(), g_extent, grid, 0.15, pos.cpu(), att.cpu(), **kw)
        finite = torch.isfinite(out_cpu) & torch.isfinite(out_gpu.cpu())
        self.assertGreater(float(finite.float().mean()), 0.95)
        torch.testing.assert_close(
            out_cpu[finite], out_gpu.cpu()[finite], atol=1e-3, rtol=1e-2)


class TestCFBP(TestCase):
    """Cartesian factorized backprojection against direct backprojection."""

    fc = 6e9
    r_res = 0.3
    grid = {"x": (60.0, 160.0), "y": (-25.0, 25.0), "nx": 256, "ny": 512}
    sweep_samples = 1024

    def _make_data(self, targets, amps, pos, d0=0.0, data_fmod=0.0):
        """Point responses consistent with the backprojection phase model."""
        c0 = 299792458.0
        data = torch.zeros(
            pos.shape[0], self.sweep_samples, dtype=torch.complex64
        )
        m_idx = torch.arange(pos.shape[0])
        for t, a in zip(targets, amps):
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = (d + d0) / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += a * w[valid] * phase[valid]
        if data_fmod != 0.0:
            mod = torch.exp(1j * data_fmod * torch.arange(self.sweep_samples))
            data = data * mod[None, :]
        return data

    def _scene(self, nsweeps=512):
        torch.manual_seed(5)
        ntargets = 12
        x = 70.0 + 80.0 * torch.rand(ntargets)
        y = -20.0 + 40.0 * torch.rand(ntargets)
        targets = torch.stack([x, y, torch.zeros_like(x)], dim=1)
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, nsweeps)
        pos[:, 2] = 20.0
        return targets, amps, pos

    def _compare(self, device, stages, nsweeps=512, grid=None, d0=0.0,
                 data_fmod=0.0, divisions=2, tol=0.05,
                 interp_method=("knab", 8, 1.4)):
        if grid is None:
            grid = self.grid
        targets, amps, pos = self._scene(nsweeps)
        data = self._make_data(targets, amps, pos, d0=d0, data_fmod=data_fmod)
        data = data.to(device)
        pos = pos.to(device)
        ref = torchbp.ops.backprojection_cart_2d(
            data, grid, self.fc, self.r_res, pos, d0=d0, data_fmod=data_fmod
        )
        out = torchbp.ops.cfbp(
            data, grid, self.fc, self.r_res, pos, stages=stages,
            divisions=divisions, d0=d0, data_fmod=data_fmod,
            interp_method=interp_method
        )
        self.assertEqual(out.shape, ref.shape)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, tol, f"cfbp relative error {rel:.2e} exceeds {tol}")
        return out

    def test_stages1_matches_direct(self):
        self._compare("cpu", stages=1, tol=0.03)

    def test_stages2_matches_direct(self):
        self._compare("cpu", stages=2)

    def test_stages3_matches_direct(self):
        # Enough sweeps that all three stages actually recurse.
        self._compare("cpu", stages=3, nsweeps=1040)

    def test_odd_nsweeps(self):
        # divisions does not divide nsweeps: no sweep may be dropped.
        self._compare("cpu", stages=2, nsweeps=511)

    def test_odd_ny(self):
        self._compare("cpu", stages=2, grid=dict(self.grid, ny=255))
        self._compare("cpu", stages=2, grid=dict(self.grid, ny=250))

    def test_divisions3(self):
        self._compare("cpu", stages=2, nsweeps=1040, divisions=3)

    def test_d0(self):
        self._compare("cpu", stages=2, d0=-0.5)

    def test_data_fmod(self):
        self._compare("cpu", stages=2, data_fmod=-torch.pi / 2)

    def test_grid_object(self):
        from torchbp.grid import CartesianGrid
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        grid_obj = CartesianGrid(
            x_range=self.grid["x"], y_range=self.grid["y"],
            nx=self.grid["nx"], ny=self.grid["ny"],
        )
        out_dict = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2
        )
        out_obj = torchbp.ops.cfbp(
            data, grid_obj, self.fc, self.r_res, pos, stages=2
        )
        torch.testing.assert_close(out_dict, out_obj)

    def test_data_gradient(self):
        targets, amps, pos = self._scene(256)
        data = self._make_data(targets, amps, pos).requires_grad_(True)
        out = torchbp.ops.cfbp(data, self.grid, self.fc, self.r_res, pos, stages=2)
        out.abs().mean().backward()
        self.assertTrue(torch.isfinite(data.grad).all())
        self.assertGreater(data.grad.abs().sum().item(), 0)

    def test_interp_method_fft(self):
        self._compare("cpu", stages=2, interp_method="fft")

    def test_knab_matches_fft(self):
        # The merge kernel should differ from the exact FFT merge only by
        # interpolation error.
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        out_fft = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2,
            interp_method="fft"
        )
        out_knab = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2,
            interp_method=("knab", 8, 1.4)
        )
        rel = ((out_knab - out_fft).abs().max() / out_fft.abs().max()).item()
        self.assertLess(rel, 1e-2)

    def test_guard_y0(self):
        # Without a guard band the y edges differ (FFT wrap-around vs window
        # truncation); the interior should still match direct backprojection.
        grid = self.grid
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        ref = torchbp.ops.backprojection_cart_2d(
            data, grid, self.fc, self.r_res, pos
        )
        out = torchbp.ops.cfbp(
            data, grid, self.fc, self.r_res, pos, stages=2, guard_y=0.0
        )
        m = int(0.05 * grid["ny"])
        rel = ((out - ref)[..., m:-m].abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 0.05)

    def _merge2_args(self, device):
        from torchbp.ops.cfbp import _merge_weight_table

        torch.manual_seed(2)
        nb, nx, ny0, ny1, nyout = 2, 16, 100, 130, 260
        img0 = torch.randn(nb, nx, ny0, dtype=torch.complex64, device=device)
        img1 = torch.randn(nb, nx, ny1, dtype=torch.complex64, device=device)
        order, v = 8, round(1 - 1 / 1.4, 6)
        w0, i0 = _merge_weight_table(ny0, nyout, order, v)
        w1, i1 = _merge_weight_table(ny1, nyout, order, v)
        w0, i0, w1, i1 = (t.to(device) for t in (w0, i0, w1, i1))
        return (img0, img1, w0, i0, w1, i1, nb, nx, ny0, ny1, nyout,
                w0.shape[1], w1.shape[1], 0.5, 0.25,
                300.0, -30.2, 50.0, 300.4, -29.8, 50.5, 300.2, -30.0, 50.25,
                80.05)

    def test_merge2_matches_torch_reference(self):
        args = self._merge2_args("cpu")
        (img0, img1, w0, i0, w1, i1, nb, nx, ny0, ny1, nyout, order0, order1,
         dx, dy, ox0, oy0, z0, ox1, oy1, z1, oxp, oyp, zp, ref_phase) = args
        out = torch.ops.torchbp.cfbp_merge2.default(*args)

        x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
        y = torch.arange(nyout, dtype=torch.float64)[None, :] * dy

        def dist(ox, oy, z):
            return torch.sqrt((ox + x).float() ** 2 + (oy + y).float() ** 2 + z**2)

        dp = dist(oxp, oyp, zp)
        ref = 0
        for img, w, idx, (ox, oy, z) in [
            (img0, w0, i0, (ox0, oy0, z0)),
            (img1, w1, i1, (ox1, oy1, z1)),
        ]:
            order = w.shape[1]
            gidx = idx[:, None].long() + torch.arange(order)[None, :]
            g = img[..., gidx.reshape(-1)].reshape(nb, nx, nyout, order)
            interp = (g * w.to(img.dtype)).sum(-1)
            ph = torch.pi * ref_phase * (dist(ox, oy, z) - dp)
            ref = ref + interp * torch.polar(torch.ones_like(ph), ph)
        # The re-reference carrier pi * ref_phase * (d - dp) with d ~ 300 m
        # in float32 amplifies a single ulp of either distance to ~8e-3 of
        # relative amplitude, and the kernel's summation order is compiler
        # dependent (FMA contraction), so bit-agreement with the torch
        # reference cannot be expected. Real indexing/weight bugs decorrelate
        # the phase entirely and show up as rel ~ 1.
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 0.05)

    def _merge2_opcheck(self, device):
        opcheck(
            torch.ops.torchbp.cfbp_merge2,
            self._merge2_args(device),
            test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
        )

    def test_merge2_opcheck_cpu(self):
        self._merge2_opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_merge2_opcheck_cuda(self):
        self._merge2_opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_merge2_cuda_matches_cpu(self):
        # CPU (MT19937) and CUDA (Philox) randn give different values for the
        # same seed, so the inputs must be built once and copied, not
        # regenerated per device. Tolerance: see
        # test_merge2_matches_torch_reference; CPU additionally hoists
        # x^2 + z^2 out of the pixel loop while CUDA sums per pixel, and the
        # CUDA build uses --use_fast_math sqrt, so a few ulp of independent
        # distance rounding is expected.
        args = self._merge2_args("cpu")
        gargs = tuple(a.cuda() if torch.is_tensor(a) else a for a in args)
        out_cuda = torch.ops.torchbp.cfbp_merge2.default(*gargs)
        out_cpu = torch.ops.torchbp.cfbp_merge2.default(*args)
        rel = ((out_cuda.cpu() - out_cpu).abs().max() / out_cpu.abs().max()).item()
        self.assertLess(rel, 0.05)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_matches_direct(self):
        out_cuda = self._compare("cuda", stages=2)
        out_cpu = self._compare("cpu", stages=2)
        rel = ((out_cuda.cpu() - out_cpu).abs().max() / out_cpu.abs().max()).item()
        self.assertLess(rel, 0.05)


class TestCFBPAdaptive(TestCase):
    """Range-adaptive CFBP on a long track with near zero ground range.

    Plain cfbp cannot represent the subaperture images on the subsampled
    grid at near range in this geometry; cfbp_adaptive must still match
    direct backprojection.
    """

    fc = 6e9
    r_res = 0.375
    sweep_samples = 512
    nsweeps = 1024
    grid = {"x": (2.0, 52.0), "y": (-25.0, 25.0), "nx": 128, "ny": 128}

    def _scene(self):
        torch.manual_seed(7)
        c0 = 299792458.0
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = 0.25 * c0 / self.fc * (
            torch.arange(self.nsweeps) - self.nsweeps / 2
        )
        pos[:, 2] = 10.0
        tx = torch.tensor([3.0, 5.0, 8.0, 14.0, 25.0, 40.0, 50.0])
        ty = torch.tensor([0.0, 15.0, -10.0, 20.0, -20.0, 5.0, 0.0])
        data = torch.zeros(self.nsweeps, self.sweep_samples, dtype=torch.complex64)
        m_idx = torch.arange(self.nsweeps)
        for x, y in zip(tx, ty):
            t = torch.tensor([x, y, 0.0])
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = d / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += w[valid] * phase[valid]
        return data, pos

    def test_matches_direct_near_range(self):
        data, pos = self._scene()
        ref = torchbp.ops.backprojection_cart_2d(
            data, self.grid, self.fc, self.r_res, pos
        )
        out = torchbp.ops.cfbp_adaptive(
            data, self.grid, self.fc, self.r_res, pos, stages=4,
            data_oversample=1.2
        )
        self.assertEqual(out.shape, ref.shape)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 0.02, f"cfbp_adaptive relative error {rel:.2e}")
        # Plain cfbp must be much worse in this geometry, otherwise the
        # scene does not actually stress the near range.
        out_plain = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=4
        )
        rel_plain = ((out_plain - ref).abs().max() / ref.abs().max()).item()
        self.assertGreater(rel_plain, 10 * rel)

    def test_blocks(self):
        _, pos = self._scene()
        blocks = torchbp.ops.cfbp_adaptive_blocks(
            self.grid, pos, self.fc, self.r_res, stages=4, data_oversample=1.2
        )
        # Blocks cover all rows contiguously and density decreases with range.
        self.assertEqual(blocks[0][0], 0)
        self.assertEqual(blocks[-1][1], self.grid["nx"])
        for b, b_next in zip(blocks, blocks[1:]):
            self.assertEqual(b[1], b_next[0])
        ks = [b[2] for b in blocks]
        self.assertEqual(ks, sorted(ks, reverse=True))
        self.assertGreater(ks[0], ks[-1])


