#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import torchbp
from torch import Tensor
import torch.nn.functional as F
from conftest import requires_cuda


class TestBpPolarRangeDealias(TestCase):
    """bp_polar_range_dealias/alias custom op: flat and DEM-referenced carriers."""

    grid = {"r": (30.0, 60.0), "theta": (-0.5, 0.5), "nr": 32, "ntheta": 16}
    fc = 6e9

    def _img(self, device, batch=False):
        torch.manual_seed(7)
        shape = (32, 16) if not batch else (2, 32, 16)
        return torch.randn(shape, dtype=torch.complex64, device=device)

    def _dealias_ref(self, img, origin, fc, grid, alias_fmod=0):
        # The previous pure-PyTorch implementation, as reference.
        r0, r1 = grid["r"]
        theta0, theta1 = grid["theta"]
        nr, ntheta = grid["nr"], grid["ntheta"]
        dr = (r1 - r0) / nr
        dtheta = (theta1 - theta0) / ntheta
        er = torch.arange(nr, device=img.device)
        r = r0 + dr * er
        theta = theta0 + dtheta * torch.arange(ntheta, device=img.device)
        x = r[:, None] * torch.sqrt(1 - torch.square(theta))[None, :]
        y = r[:, None] * theta[None, :]
        d = torch.sqrt((x - origin[0])**2 + (y - origin[1])**2 + origin[2]**2)
        c0 = 299792458
        phase = torch.exp(-1j * 4 * torch.pi * fc * d / c0
                          + 1j * alias_fmod * er[:, None])
        if img.dim() == 3:
            phase = phase.unsqueeze(0)
        return phase * img

    def _test_flat_matches_reference(self, device):
        img = self._img(device)
        origin = torch.tensor([1.5, -2.0, 25.0], device=device)
        res = torchbp.util.bp_polar_range_dealias(
            img, origin, self.fc, self.grid, alias_fmod=0.3)
        ref = self._dealias_ref(img, origin, self.fc, self.grid, alias_fmod=0.3)
        # Both evaluate a ~25e3 rad carrier in float32 with different but
        # equivalent distance expressions; a few mrad of independent
        # rounding is expected.
        torch.testing.assert_close(res, ref, atol=2e-2, rtol=2e-2)
        # Batched input
        imgb = self._img(device, batch=True)
        res = torchbp.util.bp_polar_range_dealias(
            imgb, origin, self.fc, self.grid)
        ref = self._dealias_ref(imgb, origin, self.fc, self.grid)
        torch.testing.assert_close(res, ref, atol=2e-2, rtol=2e-2)

    def test_flat_matches_reference_cpu(self):
        self._test_flat_matches_reference("cpu")

    @requires_cuda
    def test_flat_matches_reference_cuda(self):
        self._test_flat_matches_reference("cuda")

    def _test_zero_dem_matches_no_dem(self, device):
        img = self._img(device)
        origin = torch.tensor([0.0, 0.0, 25.0], device=device)
        dem = torch.zeros(8, 4, device=device)
        res = torchbp.util.bp_polar_range_dealias(
            img, origin, self.fc, self.grid, dem=dem)
        ref = torchbp.util.bp_polar_range_dealias(
            img, origin, self.fc, self.grid)
        torch.testing.assert_close(res, ref)

    def test_zero_dem_matches_no_dem_cpu(self):
        self._test_zero_dem_matches_no_dem("cpu")

    @requires_cuda
    def test_zero_dem_matches_no_dem_cuda(self):
        self._test_zero_dem_matches_no_dem("cuda")

    def _test_alias_dealias_roundtrip(self, device):
        img = self._img(device)
        origin = torch.tensor([0.0, 0.0, 25.0], device=device)
        torch.manual_seed(8)
        dem = 5.0 + 2.0 * torch.randn(8, 4, device=device)
        res = torchbp.util.bp_polar_range_dealias(
            img, origin, self.fc, self.grid, alias_fmod=0.3, dem=dem)
        back = torchbp.util.bp_polar_range_alias(
            res, origin, self.fc, self.grid, alias_fmod=0.3, dem=dem)
        torch.testing.assert_close(back, img, atol=1e-5, rtol=1e-5)

    def test_alias_dealias_roundtrip_cpu(self):
        self._test_alias_dealias_roundtrip("cpu")

    @requires_cuda
    def test_alias_dealias_roundtrip_cuda(self):
        self._test_alias_dealias_roundtrip("cuda")

    def _test_matches_bp_dealias_dem(self, device):
        # bp(dem, dealias=True) must equal bp_polar_range_dealias with the
        # same dem applied on the non-dealiased image, with origin
        # [0, 0, mean platform z] (the carrier reference the kernel uses).
        torch.manual_seed(42)
        nsweeps = 8
        data = torch.randn(nsweeps, 512, device=device, dtype=torch.complex64)
        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-1, 1, nsweeps, device=device)
        pos[:, 2] = 25.0
        torch.manual_seed(9)
        dem = 5.0 + 2.0 * torch.randn(8, 4, device=device)

        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, 0.15, pos, dealias=True,
            alias_fmod=0.3, dem=dem)[0]
        img = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, 0.15, pos, dem=dem)[0]
        origin = torch.tensor([0.0, 0.0, 25.0], device=device)
        res = torchbp.util.bp_polar_range_dealias(
            img, origin, self.fc, self.grid, alias_fmod=0.3, dem=dem)
        torch.testing.assert_close(res, ref, atol=1e-3, rtol=1e-3)

    def test_matches_bp_dealias_dem_cpu(self):
        self._test_matches_bp_dealias_dem("cpu")

    @requires_cuda
    def test_matches_bp_dealias_dem_cuda(self):
        self._test_matches_bp_dealias_dem("cuda")

    def test_opcheck(self):
        img = self._img("cpu")
        dem = torch.zeros(8, 4)
        opcheck(
            torch.ops.torchbp.polar_range_dealias,
            (img, dem, 1, 32, 16, self.fc, 30.0, 30.0/32, -0.5, 1.0/16,
             0.0, 0.0, 25.0, 0.0),
            test_utils=["test_schema", "test_faketensor"],
        )


class TestDemToPolar(TestCase):
    def test_planar_dem(self):
        # Bilinear sampling of a planar surface is exact in the interior.
        cart_grid = {"x": (0.0, 100.0), "y": (-50.0, 50.0), "nx": 100, "ny": 100}
        polar_grid = {"r": (20.0, 80.0), "theta": (-0.5, 0.5), "nr": 64, "ntheta": 32}

        nx, ny = cart_grid["nx"], cart_grid["ny"]
        dx = (cart_grid["x"][1] - cart_grid["x"][0]) / nx
        dy = (cart_grid["y"][1] - cart_grid["y"][0]) / ny
        xc = cart_grid["x"][0] + dx * torch.arange(nx)
        yc = cart_grid["y"][0] + dy * torch.arange(ny)
        dem_cart = 0.1 * xc[:, None] + 0.05 * yc[None, :] + 3.0

        dem_polar = torchbp.util.dem_to_polar(dem_cart, cart_grid, polar_grid)

        r0, r1 = polar_grid["r"]
        t0, t1 = polar_grid["theta"]
        nr, ntheta = polar_grid["nr"], polar_grid["ntheta"]
        r = r0 + (r1 - r0) / nr * torch.arange(nr)
        sin_t = t0 + (t1 - t0) / ntheta * torch.arange(ntheta)
        cos_t = torch.sqrt(torch.clamp(1 - sin_t**2, min=0))
        x = r[:, None] * cos_t[None, :]
        y = r[:, None] * sin_t[None, :]
        expected = 0.1 * x + 0.05 * y + 3.0

        self.assertEqual(dem_polar.shape, (nr, ntheta))
        torch.testing.assert_close(dem_polar, expected, atol=1e-3, rtol=1e-5)

    def test_origin_rotation(self):
        # With a rotation of pi/2 the grid x axis points along +y in the DEM
        # frame; heights are returned relative to the origin z.
        cart_grid = {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "nx": 200, "ny": 200}
        polar_grid = {"r": (20.0, 50.0), "theta": (-0.3, 0.3), "nr": 16, "ntheta": 8}

        nx, ny = cart_grid["nx"], cart_grid["ny"]
        dx = (cart_grid["x"][1] - cart_grid["x"][0]) / nx
        dy = (cart_grid["y"][1] - cart_grid["y"][0]) / ny
        xc = cart_grid["x"][0] + dx * torch.arange(nx)
        yc = cart_grid["y"][0] + dy * torch.arange(ny)
        dem_cart = 0.1 * yc[None, :].expand(nx, ny).clone()

        origin = torch.tensor([5.0, -10.0, 2.0])
        rotation = np.pi / 2
        dem_polar = torchbp.util.dem_to_polar(
            dem_cart, cart_grid, polar_grid, origin=origin, rotation=rotation)

        r0, r1 = polar_grid["r"]
        t0, t1 = polar_grid["theta"]
        nr, ntheta = polar_grid["nr"], polar_grid["ntheta"]
        r = r0 + (r1 - r0) / nr * torch.arange(nr)
        sin_t = t0 + (t1 - t0) / ntheta * torch.arange(ntheta)
        cos_t = torch.sqrt(torch.clamp(1 - sin_t**2, min=0))
        # Rotated by pi/2: x_dem = origin_x - r*sin, y_dem = origin_y + r*cos
        y_dem = origin[1] + r[:, None] * cos_t[None, :]
        expected = 0.1 * y_dem - origin[2]

        torch.testing.assert_close(dem_polar, expected, atol=1e-3, rtol=1e-5)


class TestGenerateFMCWAntennaOrientation(TestCase):
    """generate_fmcw_data must apply the azimuth pattern in azimuth (not elevation).

    This is a device-independent check: the simulator samples ``g`` with
    ``F.grid_sample`` whose last-axis order is (x=width=azimuth, y=height=
    elevation). Getting it wrong applies an azimuth-narrow beam in elevation, so
    the per-sweep gain stops depending on azimuth. A CPU-vs-GPU comparison cannot
    catch this (both devices would be transposed identically).
    """

    def test_azimuth_pattern_applied_in_azimuth(self):
        device = "cpu"
        fc, bw, tsweep, fs = 6e9, 200e6, 100e-6, 4e6
        nsweeps = 41

        # Target broadside; platform at origin, on the ground (look angle 0), so
        # the elevation angle into the pattern is ~0 for every sweep and only the
        # azimuth changes (through the swept yaw).
        target_pos = torch.tensor([[100.0, 0.0, 0.0]], device=device)
        target_rcs = torch.ones((1, 1), device=device)
        pos = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)

        # Pattern that varies ONLY in azimuth (constant across elevation rows).
        nel, naz = 8, 128
        az_axis = torch.linspace(-1.0, 1.0, naz)
        profile = torch.exp(-(az_axis / 0.2) ** 2)          # azimuth Gaussian
        g = profile[None, :].repeat(nel, 1).to(torch.float32)
        g_extent = [-0.5, -1.0, 0.5, 1.0]                   # [el0, az0, el1, az1]

        att = torch.zeros((nsweeps, 3), dtype=torch.float32, device=device)
        yaw = torch.linspace(-0.7, 0.7, nsweeps)
        att[:, 2] = yaw                                     # az_deg = -yaw

        kw = dict(g_extent=g_extent, att=att)
        data_g = torchbp.util.generate_fmcw_data(
            target_pos, target_rcs, pos, fc, bw, tsweep, fs, g=g, **kw)
        data_1 = torchbp.util.generate_fmcw_data(
            target_pos, target_rcs, pos, fc, bw, tsweep, fs,
            g=torch.ones_like(g), **kw)

        applied = data_g.abs().mean(dim=-1) / data_1.abs().mean(dim=-1)

        # Reference: az_deg = atan2(0, 100) - yaw = -yaw, sampled on the profile.
        az_deg = -yaw
        ref = torch.from_numpy(
            np.interp(az_deg.numpy(), az_axis.numpy(), profile.numpy())
        ).to(torch.float32)

        # Applied gain must follow the azimuth profile (and therefore vary a lot
        # across the sweep). The transposed bug makes it ~constant in azimuth.
        self.assertGreater(applied.max() / applied.min(), 5.0)
        torch.testing.assert_close(applied, ref, atol=2e-2, rtol=5e-2)


class TestWienerNormalize(TestCase):
    def _smooth_illumination(self, nb0, nb1):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, nb0), torch.linspace(0, 1, nb1), indexing="ij"
        )
        return (0.3 + 0.7 * torch.sin(xx * 3.14159) * torch.cos(0.4 * yy)).clamp_min(
            0.05
        ).float()

    def test_matching_shape_matches_formula(self):
        """With equal shapes wiener_normalize is the plain elementwise estimate."""
        from torchbp.util import wiener_normalize

        txp = self._smooth_illumination(48, 32)
        sar = torch.randn(48, 32, dtype=torch.complex64) * txp
        eps = 0.05
        out = wiener_normalize(sar, txp, eps=eps)
        ref = sar * txp / (txp * txp + eps**2)
        torch.testing.assert_close(out, ref)

    def test_interp_approximates_full_res_wiener(self):
        """A coarse tx_power is interpolated up; the result tracks the exact
        full-resolution Wiener estimate to ~1% over smooth illumination."""
        from torchbp.util import wiener_normalize

        txp = self._smooth_illumination(17, 13)
        v = F.interpolate(
            txp[None, None], size=(128, 96), mode="bilinear", align_corners=True
        )[0, 0]
        sar = torch.randn(128, 96, dtype=torch.complex64) * v + 0.01 * torch.randn(
            128, 96, dtype=torch.complex64
        )
        eps = 0.05
        out = wiener_normalize(sar, txp, eps=eps)
        ref = sar * v / (v * v + eps**2)
        self.assertEqual(out.shape, sar.shape)
        rel = (out - ref).abs().mean() / ref.abs().mean()
        self.assertLess(float(rel), 0.05)

    def test_interp_batched_matches_per_channel(self):
        from torchbp.util import wiener_normalize

        txp = self._smooth_illumination(9, 7)
        sar = torch.randn(64, 48, dtype=torch.complex64)
        eps = 0.05
        out2d = wiener_normalize(sar, txp, eps=eps)
        out3d = wiener_normalize(sar[None], txp[None], eps=eps)
        torch.testing.assert_close(out3d[0], out2d)

    def test_interp_auto_eps_is_finite(self):
        from torchbp.util import wiener_normalize

        txp = self._smooth_illumination(9, 7)
        txp[:, 0] = float("nan")  # un-illuminated no-data edge
        sar = torch.randn(64, 48, dtype=torch.complex64)
        out = wiener_normalize(sar, txp)  # eps auto-estimated
        self.assertEqual(out.shape, sar.shape)
        self.assertTrue(torch.isfinite(out).all())


class TestLowpassFilterWindow(TestCase):
    """Regression tests for FFT lowpass filtering."""

    def test_string_window_matches_precalculated(self):
        """String-window path must match the precalculated-Tensor path.

        Regression: `fft_lowpass_filter_window` used to try to unpack
        `w, pad_size = fft_lowpass_filter_precalculate_window(...)`, but
        that helper only returns the window Tensor, so any call passing a
        string window (e.g. the "boxcar" default of `gpga`)
        raised `ValueError: too many values to unpack`.
        """
        from torchbp.util import (
            fft_lowpass_filter_window,
            fft_lowpass_filter_precalculate_window,
        )

        torch.manual_seed(0)
        data = torch.randn(4, 64, dtype=torch.complex64)
        for window in ("boxcar", "hamming", "hann"):
            for width in (11, 20, 33):
                out_str = fft_lowpass_filter_window(
                    data, window=window, window_width=width
                )
                w = fft_lowpass_filter_precalculate_window(
                    data.shape[-1], width, data.device, window, fast_len=True
                )
                out_tensor = fft_lowpass_filter_window(
                    data, window=w, window_width=width
                )
                self.assertTrue(torch.isfinite(out_str).all())
                self.assertEqual(out_str.shape, data.shape)
                torch.testing.assert_close(out_str, out_tensor)


