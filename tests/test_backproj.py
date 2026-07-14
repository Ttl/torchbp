#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp
from torch import Tensor
from random import uniform


class TestBackprojectionPolar(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        # Make sure that scene is in view
        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        def make_nondiff_tensor(size, dtype=torch.float32):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        nbatch = 2
        sweeps = 2
        sweep_samples = 64
        grid = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 4, "ntheta": 4}
        args = {
            "data": make_tensor((nbatch, sweeps, sweep_samples), dtype=torch.complex64),
            "grid": grid,
            "fc": 6e9,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, sweeps, 3), dtype=torch.float32),
            "d0": 0.2,
            "dealias": False,
            "att": None,
            "g": None,
            "g_extent": None,
            "data_fmod": uniform(0, 2 * torch.pi),
            "alias_fmod": 0,
        }
        return [args]

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        samples = self.sample_inputs("cuda")
        for sample in samples:
            res_gpu = torchbp.ops.backprojection_polar_2d(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.backprojection_polar_2d(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu, atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu_grad(self):
        samples = self.sample_inputs("cuda", requires_grad=True)
        for sample in samples:
            sample_cpu = {
                k: sample[k].detach().cpu()
                if isinstance(sample[k], torch.Tensor)
                else sample[k]
                for k in sample.keys()
            }
            for k in sample.keys():
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad:
                    sample_cpu[k].requires_grad = True

            res_gpu = torchbp.ops.backprojection_polar_2d(**sample)
            loss_gpu = torch.mean(torch.abs(res_gpu))
            loss_gpu.backward()
            grads_gpu = [
                sample[k].cpu()
                for k in sample.keys()
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad
            ]

            res_cpu = torchbp.ops.backprojection_polar_2d(**sample_cpu)
            loss_cpu = torch.mean(torch.abs(res_cpu))
            loss_cpu.backward()
            grads_cpu = [
                sample_cpu[k]
                for k in sample_cpu.keys()
                if isinstance(sample_cpu[k], torch.Tensor)
                and sample_cpu[k].requires_grad
            ]
            torch.testing.assert_close(grads_cpu, grads_gpu, atol=1e-3, rtol=1e-2)

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.backprojection_polar_2d,
                list(args.values()),
                eps=5e-4,  # This test is very sensitive to eps
                rtol=0.2,  # Also to rtol
                atol=0.05,
            )

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_backprojection_polar_2d_args(**args)
            opcheck(
                torch.ops.torchbp.backprojection_polar_2d,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestBackprojectionPolarKnabCpu(TestCase):
    """Tests for the CPU table-based backprojection_polar_2d_knab."""

    fc = 1e9
    c0 = 299792458.0

    def _random_inputs(self, nsweeps=8, sweep_samples=64, nr=16, ntheta=8):
        torch.manual_seed(7)
        grid = {"r": (5, 15), "theta": (-0.5, 0.5), "nr": nr, "ntheta": ntheta}
        data = torch.randn(nsweeps, sweep_samples, dtype=torch.complex64)
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-1, 1, nsweeps)
        pos[:, 2] = 2.0
        return data, grid, pos

    def _reference_knab(self, data, grid, fc, r_res, pos, d0, order,
                        oversample, data_fmod=0.0):
        """Brute-force knab backprojection with exact double weights.

        Same geometry, pixel validity gating and per-tap edge masking as the
        kernel; the kernel's nearest-table-row weights differ from the exact
        weights by the fraction quantization only.
        """
        nsweeps, ns = data.shape
        nr, ntheta = grid["nr"], grid["ntheta"]
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        dr = (r1 - r0) / nr
        dt = (t1 - t0) / ntheta
        a = order / 2.0
        v = 1.0 - 1.0 / oversample
        ref_phase = 4.0 * fc / self.c0

        def knab_w(x):
            if abs(x) >= a or x == 0.0:
                return float(x == 0.0)
            w = np.sin(np.pi * x) / (np.pi * x)
            if v > 0:
                w *= np.cosh(np.pi * v * a * np.sqrt(1.0 - (x / a) ** 2))
                w /= np.cosh(np.pi * v * a)
            return w

        d_np = data.numpy()
        pos_np = pos.numpy().astype(np.float64)
        img = np.zeros((nr, ntheta), np.complex128)
        for i in range(nr):
            r = r0 + dr * i
            for j in range(ntheta):
                th = t0 + dt * j
                x = r * np.sqrt(max(0.0, 1.0 - th * th))
                y = r * th
                acc = 0.0
                for m in range(nsweeps):
                    px, py, pz = pos_np[m]
                    d = np.sqrt(r * r + px * px + py * py + pz * pz
                                - 2.0 * (x * px + y * py))
                    sx = (d + d0) / r_res
                    id0 = int(sx)
                    if not (sx >= 0.0 and id0 + 1 < ns):
                        continue
                    frac = sx - id0
                    base = id0 - (order // 2 - 1)
                    s = 0.0
                    for k in range(order):
                        t = base + k
                        if t < 0 or t >= ns:
                            continue
                        s += knab_w(frac + (order // 2 - 1) - k) * d_np[m, t]
                    acc += s * np.exp(1j * (np.pi * ref_phase * d
                                            - data_fmod * sx))
                img[i, j] = acc
        return torch.from_numpy(img).to(torch.complex64)

    def test_matches_reference_cpu(self):
        data, grid, pos = self._random_inputs()
        # d0 and r_res chosen so sx spans past both data edges: covers
        # invalid pixels and partial interpolation windows at the edges.
        r_res = 8.0 / data.shape[1]
        d0 = -6.0
        for order, oversample in [(2, 1.0), (4, 1.25), (6, 1.5), (8, 2.0)]:
            res = torchbp.ops.backprojection_polar_2d_knab(
                data, grid, self.fc, r_res, pos, d0=d0, order=order,
                oversample=oversample, data_fmod=0.7)[0]
            ref = self._reference_knab(
                data, grid, self.fc, r_res, pos, d0, order, oversample,
                data_fmod=0.7)
            rel = torch.norm(res - ref) / torch.norm(ref)
            self.assertLess(rel.item(), 5e-3,
                            f"order={order} oversample={oversample}")

    def test_odd_order_rejected_cpu(self):
        data, grid, pos = self._random_inputs()
        with self.assertRaises(RuntimeError):
            torchbp.ops.backprojection_polar_2d_knab(
                data, grid, self.fc, 0.15, pos, order=5)

    def test_constant_g_matches_no_g_cpu(self):
        # A constant antenna pattern with normalization is a no-op.
        data, grid, pos = self._random_inputs()
        g = torch.ones(8, 16)
        g_extent = [-torch.pi / 2, -torch.pi, torch.pi / 2, torch.pi]
        att = torch.zeros(data.shape[0], 3)
        ref = torchbp.ops.backprojection_polar_2d_knab(
            data, grid, self.fc, 0.15, pos, order=6, oversample=1.5)
        res = torchbp.ops.backprojection_polar_2d_knab(
            data, grid, self.fc, 0.15, pos, order=6, oversample=1.5,
            att=att, g=g, g_extent=g_extent)
        torch.testing.assert_close(res, ref, atol=1e-5, rtol=1e-5)

    def test_zero_dem_matches_no_dem_cpu(self):
        data, grid, pos = self._random_inputs()
        dem = torch.zeros(grid["nr"], grid["ntheta"])
        ref = torchbp.ops.backprojection_polar_2d_knab(
            data, grid, self.fc, 0.15, pos, order=6, oversample=1.5)
        res = torchbp.ops.backprojection_polar_2d_knab(
            data, grid, self.fc, 0.15, pos, order=6, oversample=1.5, dem=dem)
        torch.testing.assert_close(res, ref)

    def test_batched_cpu(self):
        # Batched input matches per-batch processing.
        data, grid, pos = self._random_inputs()
        data2 = torch.stack([data, data.flip(0)])
        pos2 = torch.stack([pos, pos.flip(0)])
        res = torchbp.ops.backprojection_polar_2d_knab(
            data2, grid, self.fc, 0.15, pos2, order=6, oversample=1.5)
        for b in range(2):
            ref = torchbp.ops.backprojection_polar_2d_knab(
                data2[b], grid, self.fc, 0.15, pos2[b], order=6,
                oversample=1.5)
            torch.testing.assert_close(res[b], ref[0])


class TestBackprojectionPolarDem(TestCase):
    """Tests for the optional polar-grid DEM input of backprojection_polar_2d."""

    def _random_inputs(self, device, nr=32, ntheta=16):
        torch.manual_seed(42)
        nbatch = 1
        nsweeps = 8
        sweep_samples = 512
        grid = {"r": (30, 60), "theta": (-0.5, 0.5), "nr": nr, "ntheta": ntheta}
        data = torch.randn(nbatch, nsweeps, sweep_samples, device=device,
                           dtype=torch.complex64)
        pos = torch.zeros(nbatch, nsweeps, 3, device=device)
        pos[..., 1] = torch.linspace(-1, 1, nsweeps, device=device)
        pos[..., 2] = 25.0
        return data, grid, pos

    def _test_zero_dem_matches_no_dem(self, device):
        data, grid, pos = self._random_inputs(device)
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        ref = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        torch.testing.assert_close(res, ref)

    def test_zero_dem_matches_no_dem_cpu(self):
        self._test_zero_dem_matches_no_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_dem_matches_no_dem_cuda(self):
        self._test_zero_dem_matches_no_dem("cuda")

    def _test_constant_dem_equals_shifted_pos(self, device):
        # Pixels at a constant height h see exactly the same distances as
        # z=0 pixels with the platform lowered by h.
        data, grid, pos = self._random_inputs(device)
        h = 7.0
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        pos_shift = pos.clone()
        pos_shift[..., 2] -= h
        ref = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos_shift)
        torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

    def test_constant_dem_equals_shifted_pos_cpu(self):
        self._test_constant_dem_equals_shifted_pos("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_constant_dem_equals_shifted_pos_cuda(self):
        self._test_constant_dem_equals_shifted_pos("cuda")

    def _make_ramp_dem(self, grid, dem_nr, dem_ntheta, device):
        # Bilinear in (r, theta), so bilinear interpolation between DEM
        # samples reproduces it exactly on aligned grids.
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        r = r0 + (r1 - r0) / dem_nr * torch.arange(dem_nr, device=device)
        t = t0 + (t1 - t0) / dem_ntheta * torch.arange(dem_ntheta, device=device)
        return 0.2 * r[:, None] + 3.0 * t[None, :] + 2.0

    def _test_downsampled_dem(self, device):
        data, grid, pos = self._random_inputs(device, nr=32, ntheta=16)
        # Constant DEM: downsampling changes nothing anywhere.
        h = 5.0
        dem_fine = torch.full((32, 16), h, device=device)
        dem_coarse = torch.full((8, 4), h, device=device)
        res_f = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_fine)
        res_c = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_coarse)
        torch.testing.assert_close(res_c, res_f)

        # Ramp DEM: exact away from the edge-clamped last coarse cell.
        # fr = idr * dem_nr / nr is within the coarse grid for
        # idr <= (dem_nr - 1) * nr / dem_nr.
        dem_fine = self._make_ramp_dem(grid, 32, 16, device)
        dem_coarse = self._make_ramp_dem(grid, 8, 4, device)
        res_f = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_fine)
        res_c = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_coarse)
        torch.testing.assert_close(res_c[:, :29, :13], res_f[:, :29, :13],
                                   atol=1e-2, rtol=1e-2)

    def test_downsampled_dem_cpu(self):
        self._test_downsampled_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_downsampled_dem_cuda(self):
        self._test_downsampled_dem("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu_dem(self):
        data, grid, pos = self._random_inputs("cuda")
        torch.manual_seed(123)
        dem = 5.0 + 2.0 * torch.randn(grid["nr"] // 4, grid["ntheta"] // 4,
                                      device="cuda")
        res_gpu = torchbp.ops.backprojection_polar_2d(
            data, grid, 6e9, 0.15, pos, dem=dem).cpu()
        res_cpu = torchbp.ops.backprojection_polar_2d(
            data.cpu(), grid, 6e9, 0.15, pos.cpu(), dem=dem.cpu())
        torch.testing.assert_close(res_cpu, res_gpu, atol=1e-3, rtol=1e-2)

    def _test_dealias_constant_dem_equals_shifted_pos(self, device):
        # With a DEM the dealias carrier is DEM-referenced:
        # sqrt(r^2 + (z0 - h)^2) with a constant DEM h equals the flat r-only
        # carrier with the platform lowered by h.
        data, grid, pos = self._random_inputs(device)
        h = 7.0
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        res = torchbp.ops.backprojection_polar_2d(
            data[0], grid, 6e9, 0.15, pos[0], dealias=True, dem=dem)
        pos_shift = pos[0].clone()
        pos_shift[:, 2] -= h
        ref = torchbp.ops.backprojection_polar_2d(
            data[0], grid, 6e9, 0.15, pos_shift, dealias=True)
        torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

    def test_dealias_constant_dem_cpu(self):
        self._test_dealias_constant_dem_equals_shifted_pos("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dealias_constant_dem_cuda(self):
        self._test_dealias_constant_dem_equals_shifted_pos("cuda")

    def _test_dem_grad_raises(self, device):
        data, grid, pos = self._random_inputs(device)
        pos.requires_grad = True
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        with self.assertRaises(ValueError):
            torch.mean(torch.abs(res)).backward()

    def test_dem_grad_raises_cpu(self):
        self._test_dem_grad_raises("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dem_grad_raises_cuda(self):
        self._test_dem_grad_raises("cuda")

    def _opcheck_dem(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_args

        data, grid, pos = self._random_inputs(device)
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        cpp_args = _prepare_backprojection_polar_2d_args(
            data, grid, 6e9, 0.15, pos, dem=dem)
        opcheck(
            torch.ops.torchbp.backprojection_polar_2d,
            cpp_args,
            test_utils=["test_schema", "test_faketensor"]
        )

    def test_opcheck_dem_cpu(self):
        self._opcheck_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_dem_cuda(self):
        self._opcheck_dem("cuda")

    def _test_dem_point_target(self, device):
        # A target above the ground plane focuses at the wrong ground range
        # with the flat-earth assumption and at the true ground range with a
        # DEM at the target height.
        c0 = 299792458.0
        fc = 6e9
        bw = 200e6
        tsweep = 100e-6
        fs = 10e6
        nsamples = int(fs * tsweep)
        oversample = 2
        r_res = c0 / (2 * bw * oversample)
        nsweeps = 128
        alt = 30.0
        h = 12.0
        target_r = 80.0

        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-10, 10, nsweeps, device=device)
        pos[:, 2] = alt
        target_pos = torch.tensor([[target_r, 0.0, h]], device=device)
        target_rcs = torch.ones((1, 1), device=device)

        raw = torchbp.util.generate_fmcw_data(
            target_pos, target_rcs, pos, fc, bw, tsweep, fs, rvp=False)
        w = torch.hamming_window(nsamples, periodic=False, device=device)
        data = torch.fft.ifft(raw * w[None, :], dim=-1, n=nsamples * oversample)

        grid = {"r": (60.0, 100.0), "theta": (-0.3, 0.3), "nr": 320, "ntheta": 128}
        dr = (grid["r"][1] - grid["r"][0]) / grid["nr"]

        img_flat = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos)[0]
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        img_dem = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos, dem=dem)[0]

        idr_dem = torch.argmax(torch.abs(img_dem)).item() // grid["ntheta"]
        idr_flat = torch.argmax(torch.abs(img_flat)).item() // grid["ntheta"]

        r_dem = grid["r"][0] + dr * idr_dem
        r_flat = grid["r"][0] + dr * idr_flat
        # Flat-earth image focuses where sqrt(r^2 + alt^2) equals the true
        # slant range sqrt(target_r^2 + (alt - h)^2).
        d_true = np.sqrt(target_r**2 + (alt - h) ** 2)
        r_flat_expected = np.sqrt(d_true**2 - alt**2)

        self.assertLess(abs(r_dem - target_r), 2 * dr)
        self.assertLess(abs(r_flat - r_flat_expected), 2 * dr)

    def test_dem_point_target_cpu(self):
        self._test_dem_point_target("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dem_point_target_cuda(self):
        self._test_dem_point_target("cuda")


class TestBackprojectionPolarAntennaPattern(TestCase):
    """Test that antenna pattern weighting is correctly normalized."""

    def _test_uniform_pattern_normalization(self, device):
        """Test that uniform antenna pattern (g=1) is correctly normalized.

        With uniform pattern g=1, when all sweeps contribute, we should get the
        same result as without antenna pattern normalization.
        """
        nbatch = 1
        sweep_samples = 64
        nsweeps = 10
        fc = 1e9
        grid = {"r": (5, 10), "theta": (0.-1, 0.1), "nr": 4, "ntheta": 4}

        pos = torch.zeros([nbatch, nsweeps, 3], dtype=torch.float32, device=device)
        pos[:,:,1] = torch.linspace(-nsweeps/2, nsweeps/2, nsweeps) * 0.25 * 3e8 / fc

        # Random data
        data = torch.randn(nbatch, nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        # Create uniform antenna pattern (two-way gain = 1)
        g = torch.ones(10, 10, device=device, dtype=torch.float32)
        g_extent = [-torch.pi/2, -torch.pi, torch.pi/2, torch.pi]
        att = torch.zeros(nbatch, nsweeps, 3, device=device, dtype=torch.float32)

        result_no_pattern = torchbp.ops.backprojection_polar_2d(
            data=data,
            grid=grid,
            fc=6e9,
            r_res=0.15,
            pos=pos
        )

        result_pattern = torchbp.ops.backprojection_polar_2d(
            data=data,
            grid=grid,
            fc=6e9,
            r_res=0.15,
            pos=pos,
            att=att,
            g=g,
            g_extent=g_extent
        )

        torch.testing.assert_close(result_pattern, result_no_pattern, atol=1e-5, rtol=1e-4)

    def test_uniform_pattern_normalization_cpu(self):
        self._test_uniform_pattern_normalization("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_uniform_pattern_normalization_cuda(self):
        self._test_uniform_pattern_normalization("cuda")

    @staticmethod
    def _make_nonsymmetric_antenna(device):
        """Inputs with a strongly non-symmetric antenna pattern.

        The pattern is narrow in azimuth and wide in elevation on a non-square
        (nel != naz) grid, so reading it transposed (swapping the el/az axes or
        their sizes) changes the result. The yaw is swept across the aperture so
        every sweep samples a different part of the azimuth pattern.
        """
        torch.manual_seed(0)
        nsweeps = 24
        sweep_samples = 64
        fc = 6e9
        r_res = 0.15
        grid = {"r": (4, 8), "theta": (-0.3, 0.3), "nr": 16, "ntheta": 24}

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
        att[:, 2] = torch.linspace(-0.3, 0.3, nsweeps, device=device)              # yaw sweep
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nonsymmetric_pattern_cpu_gpu(self):
        """CPU and CUDA backprojection must agree with a non-symmetric pattern,
        for both the normalized and unnormalized (FFBP) paths."""
        a = self._make_nonsymmetric_antenna("cuda")

        def to_cpu(v):
            return v.cpu() if isinstance(v, torch.Tensor) else v

        a_cpu = {k: to_cpu(v) for k, v in a.items()}
        for normalize in (True, False):
            out_gpu = torchbp.ops.backprojection_polar_2d(
                a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
                att=a["att"], g=a["g"], g_extent=a["g_extent"], normalize=normalize,
            )
            out_cpu = torchbp.ops.backprojection_polar_2d(
                a_cpu["data"], a_cpu["grid"], a_cpu["fc"], a_cpu["r_res"], a_cpu["pos"],
                att=a_cpu["att"], g=a_cpu["g"], g_extent=a_cpu["g_extent"],
                normalize=normalize,
            )
            torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-3)


class TestBackprojectionPolarLanczos(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        # Make sure that scene is in view
        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        def make_nondiff_tensor(size, dtype=torch.float32):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        nbatch = 2
        sweeps = 2
        sweep_samples = 64
        grid = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 4, "ntheta": 4}
        args = {
            "data": make_tensor((nbatch, sweeps, sweep_samples), dtype=torch.complex64),
            "grid": grid,
            "fc": 6e9,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, sweeps, 3), dtype=torch.float32),
            "d0": 0.2,
            "dealias": False,
            "order": 4,
            "att": None,
            "g": None,
            "g_extent": None,
            "data_fmod": uniform(0, 2 * torch.pi),
            "alias_fmod": 0,
        }
        return [args]

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_basic_execution(self):
        """Test that lanczos backprojection executes without errors."""
        samples = self.sample_inputs("cuda")
        for sample in samples:
            result = torchbp.ops.backprojection_polar_2d_lanczos(**sample)
            # Check output shape
            nbatch = sample["data"].shape[0]
            nr = sample["grid"]["nr"]
            ntheta = sample["grid"]["ntheta"]
            self.assertEqual(result.shape, (nbatch, nr, ntheta))
            # Check that result is not NaN or Inf
            self.assertFalse(torch.isnan(result).any())
            self.assertFalse(torch.isinf(result).any())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_comparison_with_linear(self):
        """Test that lanczos produces similar results to linear interpolation."""
        samples = self.sample_inputs("cuda")
        for sample in samples:
            # Remove lanczos-specific parameters for linear version
            sample_linear = {k: v for k, v in sample.items() if k != "order"}

            res_lanczos = torchbp.ops.backprojection_polar_2d_lanczos(**sample)
            res_linear = torchbp.ops.backprojection_polar_2d(**sample_linear)

            # Results should be reasonably similar (lanczos is higher order)
            # but not identical due to different interpolation methods
            self.assertEqual(res_lanczos.shape, res_linear.shape)

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_lanczos_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_backprojection_polar_2d_lanczos_args(**args)
            # Only test schema - no gradient or faketensor support
            opcheck(
                torch.ops.torchbp.backprojection_polar_2d_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestBackprojectionPolarKnab(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        # Make sure that scene is in view
        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        def make_nondiff_tensor(size, dtype=torch.float32):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        nbatch = 2
        sweeps = 2
        sweep_samples = 64
        grid = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 4, "ntheta": 4}
        args = {
            "data": make_tensor((nbatch, sweeps, sweep_samples), dtype=torch.complex64),
            "grid": grid,
            "fc": 6e9,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, sweeps, 3), dtype=torch.float32),
            "d0": 0.2,
            "dealias": False,
            "order": 4,
            "oversample": 1.5,
            "att": None,
            "g": None,
            "g_extent": None,
            "data_fmod": uniform(0, 2 * torch.pi),
            "alias_fmod": 0,
        }
        return [args]

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_basic_execution(self):
        """Test that knab backprojection executes without errors."""
        samples = self.sample_inputs("cuda")
        for sample in samples:
            result = torchbp.ops.backprojection_polar_2d_knab(**sample)
            # Check output shape
            nbatch = sample["data"].shape[0]
            nr = sample["grid"]["nr"]
            ntheta = sample["grid"]["ntheta"]
            self.assertEqual(result.shape, (nbatch, nr, ntheta))
            # Check that result is not NaN or Inf
            self.assertFalse(torch.isnan(result).any())
            self.assertFalse(torch.isinf(result).any())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_comparison_with_linear(self):
        """Test that knab produces similar results to linear interpolation."""
        samples = self.sample_inputs("cuda")
        for sample in samples:
            # Remove knab-specific parameters for linear version
            sample_linear = {k: v for k, v in sample.items() if k not in ("order", "oversample")}

            res_knab = torchbp.ops.backprojection_polar_2d_knab(**sample)
            res_linear = torchbp.ops.backprojection_polar_2d(**sample_linear)

            # Results should be reasonably similar (knab is higher order)
            # but not identical due to different interpolation methods
            self.assertEqual(res_knab.shape, res_linear.shape)

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_knab_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_backprojection_polar_2d_knab_args(**args)
            # Only test schema - no gradient or faketensor support
            opcheck(
                torch.ops.torchbp.backprojection_polar_2d_knab,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        # CUDA evaluates the kernel transcendentally per tap, CPU through the
        # precomputed table; the nearest-row quantization is ~1e-3 relative.
        samples = self.sample_inputs("cuda")
        for sample in samples:
            res_gpu = torchbp.ops.backprojection_polar_2d_knab(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.backprojection_polar_2d_knab(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu, atol=1e-3, rtol=1e-2)


class TestBackprojectionCart(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        # Make sure that scene is in view
        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        def make_nondiff_tensor(size, dtype=torch.float32):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        nbatch = 2
        sweeps = 2
        sweep_samples = 128
        grid = {"x": (2, 10), "y": (-5, 5), "nx": 4, "ny": 4}
        args = {
            "data": make_tensor((nbatch, sweeps, sweep_samples), dtype=torch.complex64),
            "grid": grid,
            "fc": 6e9,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, sweeps, 3), dtype=torch.float32),
            "beamwidth": 3.14,
            "d0": 0.2,
            "data_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.backprojection_cart_2d,
                list(args.values()),
                eps=5e-4,  # This test is very sensitive to eps
                rtol=0.2,  # Also to rtol
                atol=0.05,
            )

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        from torchbp.ops.backproj import _prepare_backprojection_cart_2d_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_backprojection_cart_2d_args(**args)
            opcheck(
                torch.ops.torchbp.backprojection_cart_2d,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestGPGABackprojection2D(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        ntargets = 3
        nsweeps = 5
        sweep_samples = 64

        args = {
            "target_pos": make_tensor((ntargets, 3), dtype=torch.float32),
            "data": make_tensor((nsweeps, sweep_samples), dtype=torch.complex64),
            "pos": make_pos_tensor((nsweeps, 3), dtype=torch.float32),
            "fc": 6e9,
            "r_res": 0.15,
            "d0": 0.2,
            "data_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            target_pos = args["target_pos"]
            data = args["data"]
            pos = args["pos"]
            nsweeps = data.shape[0]
            sweep_samples = data.shape[1]
            ntargets = target_pos.shape[0]

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.gpga_backprojection_2d,
                (target_pos, data, pos, sweep_samples, nsweeps, args["fc"],
                 args["r_res"], ntargets, args["d0"], args["data_fmod"]),
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def test_cpu_reference(self):
        """Compare the CPU op against a direct Python reference implementation."""
        c0 = 299792458.0
        ntargets, nsweeps, sweep_samples = 3, 5, 64
        fc, r_res, d0, data_fmod = 6e9, 0.15, 0.2, 0.7
        torch.manual_seed(1)
        # Place targets close to the platform so ranges land inside the sweep.
        data = torch.randn(nsweeps, sweep_samples, dtype=torch.complex64)
        pos = torch.randn(nsweeps, 3)
        target_pos = pos.mean(dim=0) + 0.3 * torch.randn(ntargets, 3)

        out = torchbp.ops.gpga_backprojection_2d_core(
            target_pos, data, pos, fc, r_res, d0=d0, data_fmod=data_fmod)

        ref_phase = 4.0 * fc / c0
        delta_r = 1.0 / r_res
        ref = torch.zeros(ntargets, nsweeps, dtype=torch.complex64)
        for t in range(ntargets):
            for s in range(nsweeps):
                d = torch.linalg.norm(target_pos[t] - pos[s]).item()
                sx = delta_r * (d + d0)
                i0 = int(sx)
                i1 = i0 + 1
                if i0 < 0 or i1 >= sweep_samples:
                    continue
                fr = sx - i0
                sval = (1 - fr) * data[s, i0] + fr * data[s, i1]
                angle = torch.pi * ref_phase * d - data_fmod * sx
                ref[t, s] = sval * torch.exp(1j * torch.tensor(angle))
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for args in self.sample_inputs("cpu"):
            kw = dict(fc=args["fc"], r_res=args["r_res"], d0=args["d0"],
                      data_fmod=args["data_fmod"])
            out_cpu = torchbp.ops.gpga_backprojection_2d_core(
                args["target_pos"], args["data"], args["pos"], **kw)
            out_gpu = torchbp.ops.gpga_backprojection_2d_core(
                args["target_pos"].cuda(), args["data"].cuda(),
                args["pos"].cuda(), **kw).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)


class TestBlocksvdAlpha(TestCase):
    def sample_inputs(self, device):
        torch.manual_seed(2)
        nr, ntheta = 12, 10
        nsweeps, sweep_samples = 7, 200
        args = {
            "img": torch.randn(nr, ntheta, device=device, dtype=torch.complex64),
            "data": torch.randn(
                nsweeps, sweep_samples, device=device, dtype=torch.complex64
            ),
            "pos": 0.1 * torch.randn(nsweeps, 3, device=device),
            # (ri0, ri1, ti0, ti1, sweep_lo, sweep_hi): full grid, partial
            # rectangles with partial sweep windows, empty sweep window,
            # degenerate pixel rectangle
            "blocks": torch.tensor(
                [
                    [0, nr, 0, ntheta, 0, nsweeps],
                    [0, 6, 0, 5, 2, 6],
                    [6, nr, 5, ntheta, 0, nsweeps],
                    [0, 6, 5, ntheta, 0, 0],
                    [3, 3, 2, 8, 0, nsweeps],
                ],
                device=device,
            ),
            "fc": 6e9,
            "r_res": 0.15,
            "r0": 10.0,
            "dr": 0.5,
            "theta0": -0.4,
            "dtheta": 0.08,
            "d0": 0.2,
            "data_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

    def _reference(self, args):
        """Compose the same result from gpga_backprojection_2d_core + GEMV."""
        img, data, pos = args["img"], args["data"], args["pos"]
        blocks = args["blocks"]
        nsweeps = data.shape[0]
        ref = torch.zeros(
            blocks.shape[0], nsweeps, dtype=torch.complex64, device=img.device
        )
        for b in range(blocks.shape[0]):
            ri0, ri1, ti0, ti1, lo, hi = [int(v) for v in blocks[b]]
            if ri1 <= ri0 or ti1 <= ti0 or hi <= lo:
                continue
            r = args["r0"] + args["dr"] * torch.arange(
                ri0, ri1, device=img.device, dtype=torch.float32
            )
            t = args["theta0"] + args["dtheta"] * torch.arange(
                ti0, ti1, device=img.device, dtype=torch.float32
            )
            R, T = torch.meshgrid(r, t, indexing="ij")
            x = (R * torch.sqrt(torch.clamp(1.0 - T**2, min=0.0))).reshape(-1)
            y = (R * T).reshape(-1)
            target_pos = torch.stack([x, y, torch.zeros_like(x)], dim=1)
            B = torchbp.ops.gpga_backprojection_2d_core(
                target_pos, data[lo:hi], pos[lo:hi], args["fc"],
                args["r_res"], d0=args["d0"], data_fmod=args["data_fmod"],
            )
            patch = img[ri0:ri1, ti0:ti1].reshape(-1)
            ref[b, lo:hi] = torch.conj(patch) @ B
        return ref

    def test_cpu_reference(self):
        for args in self.sample_inputs("cpu"):
            out = torchbp.ops.blocksvd_alpha(
                args["img"], args["data"], args["pos"], args["blocks"],
                args["fc"], args["r_res"], args["r0"], args["dr"],
                args["theta0"], args["dtheta"], d0=args["d0"],
                data_fmod=args["data_fmod"],
            )
            ref = self._reference(args)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def _opcheck(self, device):
        for args in self.sample_inputs(device):
            data = args["data"]
            blocks = args["blocks"].to(torch.int32)
            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.blocksvd_alpha,
                (args["img"], data, args["pos"], blocks, data.shape[1],
                 data.shape[0], blocks.shape[0], args["img"].shape[1],
                 args["fc"], args["r_res"], args["r0"], args["dr"],
                 args["theta0"], args["dtheta"], args["d0"],
                 args["data_fmod"]),
                test_utils=["test_schema"],
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for args in self.sample_inputs("cpu"):
            kw = dict(
                fc=args["fc"], r_res=args["r_res"], r0=args["r0"],
                dr=args["dr"], theta0=args["theta0"], dtheta=args["dtheta"],
                d0=args["d0"], data_fmod=args["data_fmod"],
            )
            out_cpu = torchbp.ops.blocksvd_alpha(
                args["img"], args["data"], args["pos"], args["blocks"], **kw)
            out_gpu = torchbp.ops.blocksvd_alpha(
                args["img"].cuda(), args["data"].cuda(), args["pos"].cuda(),
                args["blocks"].cuda(), **kw).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)


class TestBackprojectionPolar2DTxPower(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )
            return x

        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        nbatch = 2
        nsweeps = 4
        grid = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 8, "ntheta": 8}
        g_extent = [-0.5, -1.0, 0.5, 1.0]  # [g_el0, g_az0, g_el1, g_az1]
        args = {
            "wa": make_tensor((nbatch, nsweeps)),
            "g": make_tensor((16, 16)),
            "g_extent": g_extent,
            "grid": grid,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, nsweeps, 3)),
            "att": make_tensor((nbatch, nsweeps, 3)),
            "normalization": "sigma",
        }
        return [args]

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_tx_power_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_backprojection_polar_2d_tx_power_args(**args)
            opcheck(
                torch.ops.torchbp.backprojection_polar_2d_tx_power,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestBackprojectionCart2DTxPower(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size):
            return torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )

        def make_pos_tensor(size):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )
            # Push the platform below and behind the scene so the ground pixels
            # fall inside a reasonable look/azimuth cone.
            x = x.clone()
            with torch.no_grad():
                x[..., 2] = 100.0
            return x

        nbatch = 2
        nsweeps = 4
        grid = {"x": (30.0, 200.0), "y": (-100.0, 100.0), "nx": 8, "ny": 8}
        g_extent = [-1.4, -1.2, 1.4, 1.2]  # [g_el0, g_az0, g_el1, g_az1]
        args = {
            "wa": make_tensor((nbatch, nsweeps)).abs(),
            "g": make_tensor((16, 16)).abs(),
            "g_extent": g_extent,
            "grid": grid,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, nsweeps, 3)),
            "att": make_tensor((nbatch, nsweeps, 3)),
            "normalization": "sigma",
        }
        return [args]

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_cart_2d_tx_power_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_backprojection_cart_2d_tx_power_args(**args)
            opcheck(
                torch.ops.torchbp.backprojection_cart_2d_tx_power,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestProjectionCart2D(TestCase):
    """Tests for projection_cart_2d (direct scatter kernel) and
    projection_cart_2d_nufft (NUFFT-based O(N log M) kernel)."""

    # SAR parameters matching realistic interferometry use
    FC   = 6.1e9
    BW   = 500e6
    TSWEEP = 200e-6
    FS   = 30e6
    GAMMA = BW / TSWEEP       # 2.5e12

    def _make_inputs(self, device, *, nx=8, ny=16, nsweeps=4, nbatch=1,
                     x_range=(30.0, 200.0), y_range=(-100.0, 100.0),
                     altitude=100.0, seed=42):
        torch.manual_seed(seed)
        sweep_samples = int(self.FS * self.TSWEEP)
        grid = {"x": x_range, "y": y_range, "nx": nx, "ny": ny}

        if nbatch == 1:
            img = torch.randn(nx, ny, dtype=torch.complex64, device=device)
            pos = torch.zeros(nsweeps, 3, dtype=torch.float32, device=device)
            pos[:, 1] = torch.linspace(-5.0, 5.0, nsweeps)
            pos[:, 2] = altitude
        else:
            img = torch.randn(nbatch, nx, ny, dtype=torch.complex64, device=device)
            pos = torch.zeros(nbatch, nsweeps, 3, dtype=torch.float32, device=device)
            pos[:, :, 1] = torch.linspace(-5.0, 5.0, nsweeps)
            pos[:, :, 2] = altitude

        return dict(img=img, pos=pos, grid=grid,
                    fc=self.FC, fs=self.FS, gamma=self.GAMMA,
                    sweep_samples=sweep_samples, d0=0.0,
                    use_rvp=False, normalization="gamma")

    def sample_inputs(self, device, *, requires_grad=False):
        """Minimal inputs for opcheck (small sweep_samples avoids timeout)."""
        torch.manual_seed(0)
        nx, ny, nsweeps, sweep_samples = 4, 4, 2, 8
        grid = {"x": (-2, 2), "y": (-2, 2), "nx": nx, "ny": ny}
        pos = torch.zeros(nsweeps, 3, dtype=torch.float32, device=device)
        pos[:, 2] = 5.0
        return [{
            "img": torch.randn(nx, ny, dtype=torch.complex64, device=device),
            "pos": pos,
            "grid": grid,
            "fc": 6e9,
            "fs": 2e6,
            "gamma": 1e12,
            "sweep_samples": sweep_samples,
            "d0": 0.2,
            "normalization": "sigma",
        }]

    # ------------------------------------------------------------------
    # opcheck for both ops
    # ------------------------------------------------------------------

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_projection_cart_2d_args

        samples = self.sample_inputs(device)
        for args in samples:
            cpp_args = _prepare_projection_cart_2d_args(**args)
            opcheck(
                torch.ops.torchbp.projection_cart_2d,
                cpp_args,
                test_utils=["test_schema", "test_faketensor"],
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def _opcheck_nufft(self, device):
        from torchbp.ops.backproj import _prepare_projection_cart_2d_nufft_args

        samples = self.sample_inputs(device)
        for args in samples:
            args_nufft = {k: v for k, v in args.items() if k != "vel"}
            cpp_args = _prepare_projection_cart_2d_nufft_args(**args_nufft)
            opcheck(
                torch.ops.torchbp.projection_cart_2d_nufft,
                cpp_args,
                test_utils=["test_schema", "test_faketensor"],
            )

    def test_opcheck_nufft_cpu(self):
        self._opcheck_nufft("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_nufft_cuda(self):
        self._opcheck_nufft("cuda")

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_output_shape(self):
        """Both ops produce [nbatch, nsweeps, sweep_samples] output."""
        inputs = self._make_inputs("cuda", nx=8, ny=16, nsweeps=4, nbatch=1)
        M = inputs["sweep_samples"]
        nsweeps = inputs["pos"].shape[0]

        out_direct = torchbp.ops.projection_cart_2d(
            **inputs, vel=torch.zeros_like(inputs["pos"])
        )
        out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)

        self.assertEqual(out_direct.shape, (1, nsweeps, M))
        self.assertEqual(out_nufft.shape,  (1, nsweeps, M))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_output_shape_batched(self):
        """Batched inputs produce [nbatch, nsweeps, sweep_samples] output."""
        nbatch = 3
        inputs = self._make_inputs("cuda", nx=8, ny=16, nsweeps=4, nbatch=nbatch)
        M = inputs["sweep_samples"]
        nsweeps = inputs["pos"].shape[1]

        out_direct = torchbp.ops.projection_cart_2d(
            **inputs, vel=torch.zeros_like(inputs["pos"])
        )
        out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)

        self.assertEqual(out_direct.shape, (nbatch, nsweeps, M))
        self.assertEqual(out_nufft.shape,  (nbatch, nsweeps, M))

    # ------------------------------------------------------------------
    # NUFFT vs direct kernel agreement
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nufft_matches_direct(self):
        """NUFFT output must agree with direct kernel to within 5e-3 relative."""
        inputs = self._make_inputs("cuda")
        out_direct = torchbp.ops.projection_cart_2d(
            **inputs, vel=torch.zeros_like(inputs["pos"])
        )
        out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)

        ref_scale = out_direct.abs().max()
        rel_err = (out_direct - out_nufft).abs().max() / (ref_scale + 1e-30)
        self.assertLess(rel_err.item(), 5e-3,
                        f"NUFFT relative error {rel_err.item():.2e} exceeds 5e-3")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nufft_matches_direct_batched(self):
        """NUFFT / direct agreement holds for nbatch > 1."""
        inputs = self._make_inputs("cuda", nbatch=2)
        out_direct = torchbp.ops.projection_cart_2d(
            **inputs, vel=torch.zeros_like(inputs["pos"])
        )
        out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)

        ref_scale = out_direct.abs().max()
        rel_err = (out_direct - out_nufft).abs().max() / (ref_scale + 1e-30)
        self.assertLess(rel_err.item(), 5e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nufft_matches_direct_rvp(self):
        """NUFFT / direct agreement holds with use_rvp=True."""
        inputs = self._make_inputs("cuda")
        inputs["use_rvp"] = True
        out_direct = torchbp.ops.projection_cart_2d(
            **inputs, vel=torch.zeros_like(inputs["pos"])
        )
        out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)

        ref_scale = out_direct.abs().max()
        rel_err = (out_direct - out_nufft).abs().max() / (ref_scale + 1e-30)
        self.assertLess(rel_err.item(), 5e-3)

    def test_nufft_matches_direct_cpu(self):
        """CPU NUFFT agrees with the CPU direct kernel across branches.

        Runs without CUDA, so it validates the CPU NUFFT op directly. Covers
        use_rvp on/off, both normalizations, and the antenna-pattern path.
        """
        nsweeps = 4
        base = self._make_inputs("cpu", nx=16, ny=24, nsweeps=nsweeps)
        att = torch.zeros(nsweeps, 3, dtype=torch.float32)
        g = torch.ones(16, 16, dtype=torch.float32)
        g_extent = [-torch.pi / 2, -torch.pi, torch.pi / 2, torch.pi]

        configs = [
            dict(use_rvp=False, normalization="sigma"),
            dict(use_rvp=True,  normalization="sigma"),
            dict(use_rvp=False, normalization="gamma"),
            dict(use_rvp=True,  normalization="gamma"),
            dict(use_rvp=False, normalization="gamma", g=g, att=att, g_extent=g_extent),
        ]
        for cfg in configs:
            inputs = {**base, **cfg}
            out_direct = torchbp.ops.projection_cart_2d(**inputs, vel=None)
            out_nufft = torchbp.ops.projection_cart_2d_nufft(**inputs)
            ref_scale = out_direct.abs().max()
            rel_err = (out_direct - out_nufft).abs().max() / (ref_scale + 1e-30)
            self.assertLess(rel_err.item(), 5e-3,
                f"NUFFT/direct CPU mismatch {rel_err.item():.2e} for {cfg}")

    # ------------------------------------------------------------------
    # Direct formula check (single pixel)
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_single_pixel_direct_formula(self):
        """Projection of a single bright pixel matches the analytical sum.

        For a scene with one non-zero pixel at position (x_p, y_p, 0) the
        projected signal at sweep i, sample j is:

            data[i, j] = (1/d^2) * exp(-2πi * (fc + γ*j/fs) * τ)

        where τ = 2d/c (sigma normalization, no RVP).
        The reference must use the same c = 299792458 as the CUDA kernel.
        """
        import math
        device = "cuda"
        c = 299792458.0  # must match kC0 in util.h

        x_p, y_p = 100.0, 0.0
        grid = {"x": (x_p, x_p + 1.0), "y": (y_p, y_p + 1.0), "nx": 1, "ny": 1}
        amp = torch.tensor([[1.0 + 0.0j]], dtype=torch.complex64, device=device)

        nsweeps = 3
        pos = torch.zeros(nsweeps, 3, dtype=torch.float32, device=device)
        pos[:, 0] = x_p
        pos[:, 1] = torch.linspace(-10.0, 10.0, nsweeps)
        pos[:, 2] = 80.0

        sweep_samples = 512
        out = torchbp.ops.projection_cart_2d(
            img=amp, pos=pos, grid=grid,
            fc=self.FC, fs=self.FS, gamma=self.GAMMA,
            sweep_samples=sweep_samples, d0=0.0,
            use_rvp=False, normalization="sigma",
            vel=torch.zeros_like(pos),
        )[0]  # [nsweeps, sweep_samples]

        j = torch.arange(sweep_samples, dtype=torch.float64, device=device)
        ref = torch.zeros(nsweeps, sweep_samples, dtype=torch.complex128, device=device)
        for i in range(nsweeps):
            dx = x_p - pos[i, 0].item()
            dy = y_p - pos[i, 1].item()
            dz = 0.0  - pos[i, 2].item()
            d  = math.sqrt(dx*dx + dy*dy + dz*dz)
            tau = 2.0 * d / c
            w   = 1.0 / (d * d)
            # Kernel: exp(iπ·τ(-2γj/fs - 2fc)) = exp(-2πi(fc + γj/fs)τ)
            phase = self.FC * tau + self.GAMMA * tau / self.FS * j
            ref[i] = w * torch.exp(-2j * torch.pi * phase)

        out_d = out.to(torch.complex128)
        rel = (out_d - ref).abs().max() / (ref.abs().max() + 1e-30)
        # float32 arithmetic in the kernel introduces ~1e-3 relative error
        self.assertLess(rel.item(), 3e-3,
                        f"Single-pixel formula error {rel.item():.2e} exceeds 3e-3")

    # ------------------------------------------------------------------
    # DEM support
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dem_changes_output(self):
        """Passing a non-zero DEM changes the projected output."""
        inputs = self._make_inputs("cuda", nx=8, ny=16)
        nx, ny = inputs["img"].shape[-2], inputs["img"].shape[-1]

        out_flat = torchbp.ops.projection_cart_2d_nufft(**inputs)

        inputs_dem = {**inputs,
                      "dem": torch.full((nx, ny), 10.0,
                                        dtype=torch.float32, device="cuda")}
        out_dem = torchbp.ops.projection_cart_2d_nufft(**inputs_dem)

        self.assertFalse(torch.allclose(out_flat, out_dem),
                         "DEM had no effect on projection output")

    # ------------------------------------------------------------------
    # Normalization modes
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_normalizations_differ(self):
        """sigma and gamma normalization produce different results."""
        inputs_sigma = self._make_inputs("cuda")
        inputs_gamma = {**inputs_sigma, "normalization": "sigma"}

        out_sigma = torchbp.ops.projection_cart_2d_nufft(**inputs_sigma)
        out_gamma = torchbp.ops.projection_cart_2d_nufft(**inputs_gamma)

        self.assertFalse(torch.allclose(out_sigma, out_gamma))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_matches_cuda(self):
        """CPU projection_cart_2d output matches the CUDA kernel.

        Both kernels do float32 math and feed an identically-computed
        float32 phase into sin/cos, so the only real difference is the CPU
        fast-polynomial sincospi vs CUDA's sincospif. Covers the vel /
        use_rvp / normalization / antenna-pattern branches the two
        implementations share. Tolerance matches the other parity tests.
        """
        base = self._make_inputs("cpu", nx=8, ny=16, nsweeps=4)
        img, pos = base["img"], base["pos"]
        common = dict(grid=base["grid"], fc=base["fc"], fs=base["fs"],
                      gamma=base["gamma"], sweep_samples=base["sweep_samples"],
                      d0=base["d0"])

        # Nonzero along-track velocity exercises the HAS_VEL path.
        vel = torch.zeros_like(pos)
        vel[..., 0] = 20.0

        # Uniform unit gain over the full sphere keeps every pixel in-bounds,
        # so the antenna branch (asin/atan2 geometry, bounds check, interp2d)
        # runs without the result depending on interpolation noise.
        nsweeps = pos.shape[0]
        att = torch.zeros(nsweeps, 3, dtype=torch.float32)
        g = torch.ones(16, 16, dtype=torch.float32)
        g_extent = [-torch.pi / 2, -torch.pi, torch.pi / 2, torch.pi]

        configs = [
            dict(name="sigma",         use_rvp=False, normalization="sigma", vel=None),
            dict(name="sigma+rvp",     use_rvp=True,  normalization="sigma", vel=None),
            dict(name="gamma",         use_rvp=False, normalization="gamma", vel=None),
            dict(name="vel",           use_rvp=False, normalization="sigma", vel=vel),
            dict(name="vel+rvp+gamma", use_rvp=True,  normalization="gamma", vel=vel),
            dict(name="antenna",       use_rvp=False, normalization="gamma", vel=None,
                 g=g, att=att, g_extent=g_extent),
        ]

        for cfg in configs:
            vel_c = cfg.get("vel")
            extra_cpu, extra_cuda = {}, {}
            if "g" in cfg:
                extra_cpu = dict(g=cfg["g"], att=cfg["att"], g_extent=cfg["g_extent"])
                extra_cuda = dict(g=cfg["g"].cuda(), att=cfg["att"].cuda(),
                                  g_extent=cfg["g_extent"])

            out_cpu = torchbp.ops.projection_cart_2d(
                img=img, pos=pos, **common,
                use_rvp=cfg["use_rvp"], normalization=cfg["normalization"],
                vel=vel_c, **extra_cpu)

            out_cuda = torchbp.ops.projection_cart_2d(
                img=img.cuda(), pos=pos.cuda(), **common,
                use_rvp=cfg["use_rvp"], normalization=cfg["normalization"],
                vel=(vel_c.cuda() if vel_c is not None else None),
                **extra_cuda).cpu()

            ref_scale = out_cuda.abs().max()
            rel = (out_cpu - out_cuda).abs().max() / (ref_scale + 1e-30)
            self.assertLess(rel.item(), 5e-3,
                f"CPU/CUDA mismatch {rel.item():.2e} for config '{cfg['name']}'")


class TestGPGABackprojection2DLanczos(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=False, dtype=dtype
            )
            return x

        def make_pos_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=False, dtype=dtype
            )
            x = x - torch.max(x[:, 0]) - 2
            return x

        nsweeps = 4
        sweep_samples = 64
        ntargets = 3
        args = {
            "target_pos": make_tensor((ntargets, 3), dtype=torch.float32),
            "data": make_tensor((nsweeps, sweep_samples), dtype=torch.complex64),
            "pos": make_pos_tensor((nsweeps, 3), dtype=torch.float32),
            "fc": 6e9,
            "r_res": 0.15,
            "d0": 0.2,
            "order": 6,
            "data_fmod": 0.0,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            # Prepare C++ operator arguments
            target_pos = args["target_pos"]
            data = args["data"]
            pos = args["pos"]
            fc = args["fc"]
            r_res = args["r_res"]
            d0 = args["d0"]
            order = args["order"]
            data_fmod = args["data_fmod"]

            nsweeps = data.shape[0]
            sweep_samples = data.shape[1]
            ntargets = target_pos.shape[0]

            cpp_args = (target_pos, data, pos, sweep_samples, nsweeps,
                       fc, r_res, ntargets, d0, order, data_fmod)

            opcheck(
                torch.ops.torchbp.gpga_backprojection_2d_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


