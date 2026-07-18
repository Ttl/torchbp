#!/usr/bin/env python
"""DEM support in the polar tx_power kernels (direct + ffbp).

The DEM height fixes the range and the antenna elevation lookup
(h = pos_z - z, same correction as the DEM-aware compute_illumination), and
the sigma/gamma normalizations use the local terrain slope:

  sigma: projected-area factor (sin(inc) - u*cos(inc)) / N with up-range
         slope u and normal magnitude N = sqrt(1 + |grad z|^2); the azimuth
         slope enters only through N (not the naive sin of local incidence).
  gamma: additionally divided by the cosine of the local incidence angle
         (u*sin(inc) + cos(inc)) / N (terrain-flattened gamma).

Both clamp at the flat-formula nadir floor with the DEM-adjusted reference
height, so a constant DEM is exactly equivalent to lowering the platform.
"""
import unittest

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase

import torchbp
from torchbp.util import polar_dem_slopes
from conftest import requires_cuda

NORMALIZATIONS = ["beta", "sigma", "gamma", "point"]


def _terrain(r, t):
    return 4.0 + 2.5 * torch.sin(2 * torch.pi * (r - 40.0) / 25.0) + 2.0 * t


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


def _grid_axes(grid, device):
    r0, r1 = grid["r"]
    t0, t1 = grid["theta"]
    nr, ntheta = grid["nr"], grid["ntheta"]
    rr = r0 + (r1 - r0) / nr * torch.arange(nr, device=device)
    tt = t0 + (t1 - t0) / ntheta * torch.arange(ntheta, device=device)
    return rr, tt


class TestTxPowerDemDirect(TestCase):
    grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}

    def _tx_power(self, device="cpu", dem=None, normalization="sigma",
                  azimuth_resolution=True, pos_shift=0.0, alt=20.0):
        g, g_extent = _antenna(device)
        wa, pos, att = _straight_track(device, alt=alt)
        if pos_shift != 0.0:
            pos = pos.clone()
            pos[:, 2] += pos_shift
        return torchbp.ops.backprojection_polar_2d_tx_power(
            wa, g, g_extent, self.grid, 0.15, pos, att,
            normalization=normalization,
            azimuth_resolution=azimuth_resolution, dem=dem)[0]

    def test_zero_dem_matches_no_dem(self):
        device = "cpu"
        dem = torch.zeros(self.grid["nr"], self.grid["ntheta"], device=device)
        for norm in NORMALIZATIONS:
            for az_res in [True, False]:
                ref = self._tx_power(device, dem=None, normalization=norm,
                                     azimuth_resolution=az_res)
                out = self._tx_power(device, dem=dem, normalization=norm,
                                     azimuth_resolution=az_res)
                torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6,
                                           equal_nan=True)

    def test_constant_dem_equals_shifted_pos(self):
        """A constant DEM at height h must equal flat earth with the platform
        lowered by h, for every normalization: zero slopes reduce the terrain
        factors to the flat formulas with h = pos_z - h_dem, and the nadir
        floor uses the DEM-adjusted reference height."""
        device = "cpu"
        h = 6.0
        dem = torch.full((self.grid["nr"], self.grid["ntheta"]), h,
                         device=device)
        for norm in NORMALIZATIONS:
            ref = self._tx_power(device, dem=None, normalization=norm,
                                 pos_shift=-h)
            out = self._tx_power(device, dem=dem, normalization=norm)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6,
                                       equal_nan=True)

    def test_tilted_plane_analytic(self):
        """On a tilted plane z = a*x + b the per-pixel ratio of beta to
        sigma/gamma power isolates the terrain factor; compare against the
        hand-computed projected-area and terrain-flattened-gamma factors
        from the exact single-sweep geometry."""
        device = "cpu"
        a, b = 0.15, 3.0
        nr, ntheta = self.grid["nr"], self.grid["ntheta"]
        rr, tt = _grid_axes(self.grid, device)
        x = rr[:, None] * torch.sqrt(torch.clamp(1 - tt[None, :]**2, min=0.0))
        y = rr[:, None] * tt[None, :]
        dem = (a * x + b).float()

        g, g_extent = _antenna(device, az_width=2.0, el_width=2.0)
        alt = 25.0
        wa = torch.ones(1, device=device)
        pos = torch.tensor([[0.0, 0.0, alt]], device=device)
        att = torch.zeros(1, 3, device=device)
        d_c = float(np.sqrt(60.0**2 + alt**2))
        att[0, 0] = -float(np.arcsin(alt / d_c))

        outs = {}
        for norm in NORMALIZATIONS[:3]:
            outs[norm] = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, self.grid, 0.15, pos, att,
                normalization=norm, azimuth_resolution=False, dem=dem)[0]

        # Exact geometry: single sweep at the origin.
        h = alt - dem
        d = torch.sqrt(x**2 + y**2 + h**2)
        Rg = torch.sqrt(x**2 + y**2)
        s = x * a  # px*dzdx + py*dzdy with dzdy = 0
        N = float(np.sqrt(1 + a**2))
        sinl_sigma = (Rg**2 - s * h) / (Rg * d * N)
        sinl_gamma = (Rg**2 - s * h) / (Rg * (s + h))

        # Interior pixels where the clamp is inactive and gain is meaningful.
        dr = (self.grid["r"][1] - self.grid["r"][0]) / nr
        floor = torch.sqrt(2.0 * dr / torch.clamp(alt - dem, min=1e-3))
        beta2 = outs["beta"] ** 2
        mask = (beta2 > 1e-3 * float(beta2.max())) \
            & (sinl_sigma > 1.5 * floor) & (sinl_gamma > 1.5 * floor * d / h)
        mask[:2] = False
        mask[-2:] = False
        mask[:, :2] = False
        mask[:, -2:] = False
        self.assertGreater(int(mask.sum()), 1000)

        ratio_sigma = beta2[mask] / (outs["sigma"][mask] ** 2)
        ratio_gamma = beta2[mask] / (outs["gamma"][mask] ** 2)
        # Finite differences of the DEM are second order; slopes on the polar
        # grid are not exact for a plane, so allow a small tolerance.
        err_s = (ratio_sigma / sinl_sigma[mask] - 1).abs()
        err_g = (ratio_gamma / sinl_gamma[mask] - 1).abs()
        self.assertLess(float(err_s.max()), 5e-3)
        self.assertLess(float(err_g.max()), 5e-3)

    def test_coarse_dem(self):
        device = "cpu"
        nr, ntheta = self.grid["nr"], self.grid["ntheta"]
        # Constant coarse DEM must match constant fine DEM exactly.
        h = 5.0
        fine = torch.full((nr, ntheta), h, device=device)
        coarse = torch.full((nr // 8, ntheta // 8), h, device=device)
        out_f = self._tx_power(device, dem=fine)
        out_c = self._tx_power(device, dem=coarse)
        torch.testing.assert_close(out_c, out_f, rtol=1e-5, atol=1e-7,
                                   equal_nan=True)
        # Smooth terrain: coarse within tolerance of fine away from edges.
        rr, tt = _grid_axes(self.grid, device)
        dem = _terrain(rr[:, None], tt[None, :]).float()
        rr_c = rr[::2]
        tt_c = tt[::2]
        dem_c = _terrain(rr_c[:, None], tt_c[None, :]).float()
        out_f = self._tx_power(device, dem=dem)
        out_c = self._tx_power(device, dem=dem_c)
        m = torch.isfinite(out_f) & torch.isfinite(out_c) \
            & (out_f > 1e-3 * float(out_f[torch.isfinite(out_f)].max()))
        m[:4] = False
        m[-4:] = False
        m[:, 4:] &= m[:, :-4].clone()
        err = (out_c[m] / out_f[m] - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), 0.02)
        # A DEM finer than the image grid (the sar_process_torch.py pattern:
        # full-resolution DEM on a reduced tx_power grid) must also agree.
        r0, r1 = self.grid["r"]
        t0, t1 = self.grid["theta"]
        rr_f = r0 + (r1 - r0) / (4 * nr) * torch.arange(4 * nr, device=device)
        tt_f = t0 + (t1 - t0) / (4 * ntheta) * torch.arange(
            4 * ntheta, device=device)
        dem_f = _terrain(rr_f[:, None], tt_f[None, :]).float()
        out_fine = self._tx_power(device, dem=dem_f)
        err = (out_fine[m] / out_f[m] - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), 0.02)

    def test_slant_dem_raises(self):
        device = "cpu"
        g, g_extent = _antenna(device)
        wa, pos, att = _straight_track(device, alt=0.0)
        dem = torch.zeros(self.grid["nr"], self.grid["ntheta"], device=device)
        with self.assertRaises(NotImplementedError):
            torchbp.ops.backprojection_polar_2d_tx_power_slant(
                wa, g, g_extent, self.grid, 0.15, pos, att, altitude=20.0,
                normalization="sigma", dem=dem)

    @requires_cuda
    def test_cpu_cuda_agree(self):
        device = "cuda"
        rr, tt = _grid_axes(self.grid, "cpu")
        dem = _terrain(rr[:, None], tt[None, :]).float()
        for norm in NORMALIZATIONS:
            out_cpu = self._tx_power("cpu", dem=dem, normalization=norm)
            out_gpu = self._tx_power(device, dem=dem.to(device),
                                     normalization=norm).cpu()
            torch.testing.assert_close(out_gpu, out_cpu, rtol=1e-3, atol=1e-5,
                                       equal_nan=True)


class TestPolarDemSlopes(TestCase):
    def _plane_check(self, theta_psi):
        device = "cpu"
        a, c, b = 0.2, -0.1, 3.0
        if theta_psi:
            grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 64,
                    "ntheta": 128}
            t_is_psi = True
        else:
            grid = {"r": (40, 80), "theta": (-0.5, 0.5), "nr": 64,
                    "ntheta": 128}
            t_is_psi = False
        rr, tt = _grid_axes(grid, device)
        if t_is_psi:
            sin_p = torch.sin(tt)
            cos_p = torch.cos(tt)
        else:
            sin_p = tt
            cos_p = torch.sqrt(1 - tt**2)
        x = rr[:, None] * cos_p[None, :]
        y = rr[:, None] * sin_p[None, :]
        dem = (a * x + c * y + b).float()
        dem3 = polar_dem_slopes(dem, grid, theta_psi=theta_psi)
        self.assertEqual(tuple(dem3.shape), (3, 64, 128))
        torch.testing.assert_close(dem3[0], dem)
        interior = dem3[:, 2:-2, 2:-2]
        self.assertLess(float((interior[1] - a).abs().max()), 1e-3)
        self.assertLess(float((interior[2] - c).abs().max()), 1e-3)

    def test_plane_sin_theta_grid(self):
        self._plane_check(theta_psi=False)

    def test_plane_psi_grid(self):
        self._plane_check(theta_psi=True)

    def test_constant_dem_zero_slopes(self):
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 32, "ntheta": 64}
        dem = torch.full((32, 64), 7.0)
        dem3 = polar_dem_slopes(dem, grid)
        self.assertEqual(float(dem3[1].abs().max()), 0.0)
        self.assertEqual(float(dem3[2].abs().max()), 0.0)


class TestTxPowerDemFfbp(TestCase):
    grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}

    def _run(self, device="cpu", dem=None, normalization="sigma",
             pos_shift=0.0, direct=False, **kw):
        g, g_extent = _antenna(device)
        wa, pos, att = _straight_track(device)
        if pos_shift != 0.0:
            pos = pos.clone()
            pos[:, 2] += pos_shift
        if direct:
            return torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, self.grid, 0.15, pos, att,
                normalization=normalization, dem=dem)[0]
        kw.setdefault("stages", 3)
        kw.setdefault("downsample_r", 1.0)
        kw.setdefault("downsample_theta", 1.0)
        kw.setdefault("min_nsweeps", 16)
        return torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, self.grid, 0.15, pos, att,
            normalization=normalization, dem=dem, **kw)

    def _assert_matches(self, out, ref, q95=0.03, emax=0.15, margin=4,
                        rel_thresh=1e-3):
        self.assertEqual(out.shape, ref.shape)
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.99)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        thresh = rel_thresh * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 100)
        err = (out_c[mask] / ref_c[mask] - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), q95)
        self.assertLess(float(err.max()), emax)

    def test_zero_dem_matches_no_dem(self):
        device = "cpu"
        dem = torch.zeros(self.grid["nr"], self.grid["ntheta"], device=device)
        for norm in ["beta", "sigma", "gamma"]:
            ref = self._run(device, dem=None, normalization=norm)
            out = self._run(device, dem=dem, normalization=norm)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6,
                                       equal_nan=True)

    def test_constant_dem_equals_shifted_pos(self):
        device = "cpu"
        h = 6.0
        dem = torch.full((self.grid["nr"], self.grid["ntheta"]), h,
                         device=device)
        for norm in ["beta", "sigma", "gamma"]:
            ref = self._run(device, dem=None, normalization=norm,
                            pos_shift=-h)
            out = self._run(device, dem=dem, normalization=norm)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6,
                                       equal_nan=True)

    def test_terrain_matches_direct(self):
        """Gamma illumination plunges toward zero in the grazing bands of the
        terrain (cos of local incidence -> 0); inside those near-null bands
        the interpolated ffbp maps have a large relative error against near
        zero references, like the documented illumination-edge caveat. Assert
        agreement above -20 dB of the maximum for gamma."""
        device = "cpu"
        rr, tt = _grid_axes(self.grid, device)
        dem = _terrain(rr[:, None], tt[None, :]).float()
        for norm in ["beta", "sigma", "gamma"]:
            rel_thresh = 0.1 if norm == "gamma" else 1e-3
            ref = self._run(device, dem=dem, normalization=norm, direct=True)
            out = self._run(device, dem=dem, normalization=norm)
            self._assert_matches(out, ref, q95=0.04, emax=0.3,
                                 rel_thresh=rel_thresh)

    def test_coarse_dem_matches_direct(self):
        """A downsampled DEM through ffbp against the direct kernel with the
        same downsampled DEM."""
        device = "cpu"
        rr, tt = _grid_axes(self.grid, device)
        dem = _terrain(rr[::4, None], tt[None, ::4]).float()
        ref = self._run(device, dem=dem, normalization="sigma", direct=True)
        out = self._run(device, dem=dem, normalization="sigma")
        self._assert_matches(out, ref, q95=0.04, emax=0.2)

    def test_altitude_dem_raises(self):
        device = "cpu"
        dem = torch.zeros(self.grid["nr"], self.grid["ntheta"], device=device)
        with self.assertRaises(NotImplementedError):
            self._run(device, dem=dem, altitude=20.0)

    @requires_cuda
    def test_cpu_cuda_agree(self):
        rr, tt = _grid_axes(self.grid, "cpu")
        dem = _terrain(rr[:, None], tt[None, :]).float()
        out_cpu = self._run("cpu", dem=dem)
        out_gpu = self._run("cuda", dem=dem.to("cuda")).cpu()
        torch.testing.assert_close(out_gpu, out_cpu, rtol=1e-3, atol=1e-5,
                                   equal_nan=True)


if __name__ == "__main__":
    unittest.main()
