#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
import torchbp
from conftest import requires_cuda


class TestAFBP(TestCase):
    """afbp must match direct polar backprojection including pixel phase."""

    fc = 6e9
    r_res = 0.5
    nsamples = 512
    nsweeps = 256
    grid = {"r": (100.0, 200.0), "theta": (-0.2, 0.2), "nr": 128, "ntheta": 128}

    def _scene(self, device="cpu", z0=0.0):
        c0 = 299792458.0
        lam = c0 / self.fc
        pos = torch.zeros((self.nsweeps, 3), device=device)
        pos[:, 1] = lam / 4 * (torch.arange(self.nsweeps, device=device) - self.nsweeps / 2)
        pos[:, 2] = z0
        # Point targets, including ones near the theta extents where the
        # fusion guard band matters.
        targets = [(150.0, 0.0), (120.0, 0.15), (180.0, -0.12), (105.0, -0.18)]
        data = torch.zeros((self.nsweeps, self.nsamples), dtype=torch.complex64, device=device)
        i = torch.arange(self.nsamples, dtype=torch.float64, device=device)
        for r, th in targets:
            gr = np.sqrt(max(r * r - z0 * z0, 1.0))
            tx = gr * np.sqrt(1 - th * th)
            ty = gr * th
            d = torch.sqrt((pos[:, 0].double() - tx) ** 2
                           + (pos[:, 1].double() - ty) ** 2
                           + (pos[:, 2].double()) ** 2)
            # Range compressed envelope with 2x oversampled bandwidth.
            env = torch.special.sinc((i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        return data, pos

    def _compare(self, device, z0=0.0, nsub=8, dealias=False, data_fmod=0.0,
                 alias_fmod=0.0, tol=2e-2):
        data, pos = self._scene(device, z0)
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, dealias=dealias,
            data_fmod=data_fmod, alias_fmod=alias_fmod)[0]
        out = torchbp.ops.afbp(
            data, self.grid, self.fc, self.r_res, pos, nsub=nsub,
            dealias=dealias, data_fmod=data_fmod, alias_fmod=alias_fmod)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, tol)

    def test_matches_direct_cpu(self):
        for nsub in (2, 4, 8):
            self._compare("cpu", nsub=nsub)

    def test_matches_direct_dealias_cpu(self):
        self._compare("cpu", dealias=True)

    def test_matches_direct_altitude_cpu(self):
        self._compare("cpu", z0=40.0)
        self._compare("cpu", z0=40.0, dealias=True)

    def test_matches_direct_fmod_cpu(self):
        self._compare("cpu", data_fmod=0.3)
        self._compare("cpu", dealias=True, alias_fmod=-0.5)
        self._compare("cpu", dealias=True, data_fmod=0.3, alias_fmod=-0.3)

    @requires_cuda
    def test_matches_direct_cuda(self):
        self._compare("cuda")
        self._compare("cuda", z0=40.0, dealias=True)

    def test_uneven_split(self):
        # nsweeps, ntheta not divisible by nsub
        data, pos = self._scene()
        data = data[:249]
        pos = pos[:249]
        grid = dict(self.grid, ntheta=122)
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos, dealias=True)[0]
        out = torchbp.ops.afbp(data, grid, self.fc, self.r_res, pos, nsub=7,
                               dealias=True)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_reversed_track(self):
        data, pos = self._scene()
        data = data.flip(0)
        pos = pos.flip(0)
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, dealias=True)[0]
        out = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                               nsub=8, dealias=True)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_data_interp_method_knab(self):
        # Subaperture backprojections with knab range interpolation; the
        # fusion is interpolation-free, so the output must match the direct
        # knab backprojection like the linear case matches linear.
        data, pos = self._scene()
        method = ("knab", 6, 2.0)
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos,
            interp_method=method)[0]
        out = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                               nsub=8, data_interp_method=method)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_data_interp_method_grad_raises(self):
        data, pos = self._scene()
        data.requires_grad_(True)
        with self.assertRaises(ValueError):
            torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                             nsub=8, data_interp_method=("knab", 6, 2.0))

    def test_nsub1_fallback_exact(self):
        data, pos = self._scene()
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos)[0]
        out = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos, nsub=1)
        self.assertEqual((out - ref).abs().max().item(), 0.0)

    def test_antenna_pattern_unnormalized(self):
        data, pos = self._scene(z0=30.0)
        att = torch.zeros((self.nsweeps, 3))
        el = torch.linspace(-1.0, 1.0, 16)
        az = torch.linspace(-1.2, 1.2, 64)
        g = torch.exp(-el[:, None] ** 2 / 0.8) * torch.exp(-az[None, :] ** 2 / 0.5)
        g_extent = [-1.0, -1.2, 1.0, 1.2]
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, dealias=True,
            att=att, g=g, g_extent=g_extent, normalize=False)[0]
        out = torchbp.ops.afbp(
            data, self.grid, self.fc, self.r_res, pos, nsub=4, dealias=True,
            att=att, g=g, g_extent=g_extent, normalize=False)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 2e-2)
        # normalize=True matches the same accumulation normalized with the
        # illumination moment maps (the antenna-weighted ffbp convention).
        from torchbp.ops.ffbp import (
            compute_subaperture_illumination, _weighted_normalize)
        w1, w2 = compute_subaperture_illumination(
            pos, att, g, g_extent, self.grid, decimation=1)
        ref_n = _weighted_normalize(ref.clone(), w1, w2)
        out_n = torchbp.ops.afbp(
            data, self.grid, self.fc, self.r_res, pos, nsub=4, dealias=True,
            att=att, g=g, g_extent=g_extent, weight_map_downsample=1)
        rel = ((out_n - ref_n).abs().max() / ref_n.abs().max()).item()
        self.assertLess(rel, 2e-2)

    @staticmethod
    def _terrain(r, th):
        """Analytic terrain height, used for both the DEM tensor and the
        target heights so targets lie on the DEM surface."""
        return 15.0 + 10.0 * torch.sin(
            2 * torch.pi * (r - 100.0) / 70.0) * torch.cos(3.0 * th)

    def _dem(self, dem_nr=64, dem_ntheta=48):
        # Cell start convention, matching the kernels' index-ratio mapping.
        r0, r1 = self.grid["r"]
        t0, t1 = self.grid["theta"]
        r = r0 + (r1 - r0) / dem_nr * torch.arange(dem_nr, dtype=torch.float64)
        t = t0 + (t1 - t0) / dem_ntheta * torch.arange(dem_ntheta, dtype=torch.float64)
        return self._terrain(r[:, None], t[None, :]).float()

    def _dem_scene(self, z0=60.0):
        c0 = 299792458.0
        lam = c0 / self.fc
        pos = torch.zeros((self.nsweeps, 3))
        pos[:, 1] = lam / 4 * (torch.arange(self.nsweeps) - self.nsweeps / 2)
        pos[:, 2] = z0
        targets = [(150.0, 0.0), (120.0, 0.15), (180.0, -0.12), (105.0, -0.18)]
        data = torch.zeros((self.nsweeps, self.nsamples), dtype=torch.complex64)
        i = torch.arange(self.nsamples, dtype=torch.float64)
        for gr, th in targets:
            tx = gr * np.sqrt(1 - th * th)
            ty = gr * th
            tz = float(self._terrain(torch.tensor(gr, dtype=torch.float64),
                                     torch.tensor(th, dtype=torch.float64)))
            d = torch.sqrt((pos[:, 0].double() - tx) ** 2
                           + (pos[:, 1].double() - ty) ** 2
                           + (pos[:, 2].double() - tz) ** 2)
            env = torch.special.sinc((i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        return data, pos, self._dem()

    def test_matches_direct_dem_cpu(self):
        for dealias in (False, True):
            data, pos, dem = self._dem_scene()
            ref = torchbp.ops.backprojection_polar_2d(
                data, self.grid, self.fc, self.r_res, pos, dealias=dealias,
                dem=dem)[0]
            out = torchbp.ops.afbp(
                data, self.grid, self.fc, self.r_res, pos, nsub=8,
                dealias=dealias, dem=dem)
            rel = ((out - ref).abs().max() / ref.abs().max()).item()
            self.assertLess(rel, 2e-2)

    def test_batched_fusion_parity_dem(self):
        data, pos, dem = self._dem_scene()
        out1 = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                                nsub=8, dealias=True, dem=dem)
        out2 = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                                nsub=8, dealias=True, dem=dem,
                                _batched_fusion=True)
        rel = ((out1 - out2).abs().max() / out1.abs().max()).item()
        self.assertLess(rel, 1e-4)

    def test_gradient_dem_raises(self):
        # The backprojection kernels have no DEM gradient: the forward
        # runs and backward raises, like backprojection_polar_2d.
        data, pos, dem = self._dem_scene()
        d1 = data.clone().requires_grad_(True)
        img1 = torchbp.ops.afbp(d1, self.grid, self.fc, self.r_res, pos,
                                nsub=4, dem=dem)
        with self.assertRaises(ValueError):
            img1.abs().pow(2).sum().backward()

    def test_antenna_pattern_dem(self):
        data, pos, dem = self._dem_scene()
        att = torch.zeros((self.nsweeps, 3))
        el = torch.linspace(-1.0, 1.0, 16)
        az = torch.linspace(-1.2, 1.2, 64)
        g = torch.exp(-el[:, None] ** 2 / 0.8) * torch.exp(-az[None, :] ** 2 / 0.5)
        g_extent = [-1.0, -1.2, 1.0, 1.2]
        ref = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos, dealias=True,
            att=att, g=g, g_extent=g_extent, normalize=False, dem=dem)[0]
        out = torchbp.ops.afbp(
            data, self.grid, self.fc, self.r_res, pos, nsub=4, dealias=True,
            att=att, g=g, g_extent=g_extent, normalize=False, dem=dem)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 2e-2)
        from torchbp.ops.ffbp import (
            compute_subaperture_illumination, _weighted_normalize)
        w1, w2 = compute_subaperture_illumination(
            pos, att, g, g_extent, self.grid, decimation=1, dem=dem)
        ref_n = _weighted_normalize(ref.clone(), w1, w2)
        out_n = torchbp.ops.afbp(
            data, self.grid, self.fc, self.r_res, pos, nsub=4, dealias=True,
            att=att, g=g, g_extent=g_extent, weight_map_downsample=1,
            dem=dem)
        rel = ((out_n - ref_n).abs().max() / ref_n.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_gradient_matches_direct(self):
        data, pos = self._scene()
        d1 = data.clone().requires_grad_(True)
        img1 = torchbp.ops.afbp(d1, self.grid, self.fc, self.r_res, pos, nsub=4)
        img1.abs().pow(2).sum().backward()
        d2 = data.clone().requires_grad_(True)
        img2 = torchbp.ops.backprojection_polar_2d(
            d2, self.grid, self.fc, self.r_res, pos)[0]
        img2.abs().pow(2).sum().backward()
        rel = ((d1.grad - d2.grad).norm() / d2.grad.norm()).item()
        self.assertLess(rel, 5e-2)

    def test_full_theta_extent(self):
        # theta extent up to +-1: the internal guard band would extend past
        # the polar domain where the kernel returns NaN. Regression test:
        # those columns must be zeroed, not poison the fusion FFTs.
        data, pos = self._scene(z0=40.0)
        # theta oversampled 1.5x over the full-aperture bandwidth, as in
        # normal use; a critically sampled grid leaves the fusion no
        # spectral margin.
        grid = {"r": (1.0, 200.0), "theta": (-1.0, 1.0), "nr": 128, "ntheta": 384}
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos, dealias=True)[0]
        out = torchbp.ops.afbp(data, grid, self.fc, self.r_res, pos, nsub=8,
                               dealias=True)
        self.assertEqual(int(torch.isnan(out).sum()), 0)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 5e-2)

    def test_batched_fusion_parity(self):
        # The batched (GPU) fusion path must match the per-block loop path.
        data, pos = self._scene(z0=40.0)
        out1 = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                                nsub=8, dealias=True)
        out2 = torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                                nsub=8, dealias=True, _batched_fusion=True)
        rel = ((out1 - out2).abs().max() / out1.abs().max()).item()
        self.assertLess(rel, 1e-4)

    def test_antenna_cutoff_warns(self):
        # Gain table ending inside the grid theta extent with
        # non-negligible edge gain: the fusion cannot represent the hard
        # cutoff step, so afbp warns. Tapered to zero at the table edge
        # there is no step and no warning.
        import warnings as _warnings
        data, pos = self._scene()
        att = torch.zeros((self.nsweeps, 3))
        el = torch.linspace(-1.0, 1.0, 8)
        az = torch.linspace(-0.1, 0.1, 16)
        g = torch.exp(-el[:, None] ** 2) * torch.exp(-az[None, :] ** 2)
        g_extent = [-1.0, -0.1, 1.0, 0.1]
        with self.assertWarnsRegex(UserWarning, "gain table ends inside"):
            torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                             nsub=4, att=att, g=g, g_extent=g_extent,
                             normalize=False)
        g2, g2_extent = torchbp.util.taper_antenna_pattern(g, g_extent, 0.05)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            torchbp.ops.afbp(data, self.grid, self.fc, self.r_res, pos,
                             nsub=4, att=att, g=g2, g_extent=g2_extent,
                             normalize=False)

    def test_patch_alias_warns(self):
        # Theta step too coarse for the subaperture spectrum patch.
        data, pos = self._scene()
        grid = dict(self.grid, ntheta=16)
        with self.assertWarns(UserWarning):
            torchbp.ops.afbp(data, grid, self.fc, self.r_res, pos, nsub=8)

    def test_ffbp_afbp_base(self):
        # ffbp with afbp base must match plain ffbp.
        data, pos = self._scene(z0=30.0)
        img1 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0)
        img2 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                afbp_nsub=4)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_ffbp_afbp_base_antenna(self):
        # Antenna-pattern-weighted ffbp with afbp base must match the plain
        # weighted ffbp: afbp feeds the same unnormalized accumulation into
        # the weight-map Wiener normalization.
        data, pos = self._scene(z0=30.0)
        att = torch.zeros((self.nsweeps, 3))
        att[:, 2] = 0.05 * torch.sin(
            2 * torch.pi * torch.arange(self.nsweeps) / self.nsweeps)
        el = torch.linspace(-1.0, 1.0, 16)
        az = torch.linspace(-1.2, 1.2, 64)
        g = torch.exp(-el[:, None] ** 2 / 0.8) * torch.exp(-az[None, :] ** 2 / 0.35)
        g_extent = [-1.0, -1.2, 1.0, 1.2]
        img1 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                att=att, g=g, g_extent=g_extent)
        img2 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                att=att, g=g, g_extent=g_extent, afbp_nsub=4)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 2e-2)

    def test_ffbp_afbp_nsub_noop_warns(self):
        # Full-extent grid with lambda/4 pulse spacing: the scene theta
        # extent exceeds the grating replica clearance, the afbp base
        # level can never split at any leaf, and ffbp warns up front that
        # afbp_nsub is a no-op.
        data, pos = self._scene(z0=40.0)
        grid = {"r": (1.0, 200.0), "theta": (-1.0, 1.0),
                "nr": 128, "ntheta": 384}
        with self.assertWarnsRegex(UserWarning, "has no effect"):
            torchbp.ops.ffbp(data, grid, self.fc, self.r_res, pos,
                             stages=2, dealias=True, afbp_nsub=4)

    def test_ffbp_afbp_base_dem(self):
        # ffbp with afbp base and a DEM must match plain ffbp with the DEM.
        # The tolerance is looser than the flat-ground parity: the terrain
        # theta variation within a range row is outside the afbp fusion
        # model and costs ~2 % of peak amplitude at the steepest-slope
        # target on this scene (phase stays exact, see the afbp dem note).
        data, pos, dem = self._dem_scene()
        img1 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                dem=dem)
        img2 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                dem=dem, afbp_nsub=4)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 3e-2)

    def test_ffbp_afbp_base_subaperture_gain_cutoff(self):
        # A gain table that ends inside the scene theta extent: per-pulse
        # gain in the afbp base puts the hard cutoff step of the table
        # edge into the subaperture images, which the wavenumber fusion
        # cannot represent (illumination dims and smears near the edge).
        # antenna_leaf_gain="subaperture" forms the afbp base without the
        # pattern and applies the gain in the image domain; it must match
        # the direct base level.
        c0 = 299792458.0
        lam = c0 / self.fc
        nsweeps = 256
        pos = torch.zeros((nsweeps, 3))
        pos[:, 1] = lam / 8 * (torch.arange(nsweeps) - nsweeps / 2)
        pos[:, 2] = 40.0
        grid = {"r": (100.0, 200.0), "theta": (-0.8, 0.8),
                "nr": 128, "ntheta": 256}
        gen = torch.Generator().manual_seed(0)
        ntg = 24
        gr = 100.0 + 100.0 * torch.rand(ntg, generator=gen)
        th = -0.78 + 1.56 * torch.rand(ntg, generator=gen)
        # Gentle terrain: this test isolates the gain cutoff handling, so
        # keep the (documented) terrain theta-slope amplitude effect of
        # the fusion well below it.
        terrain = lambda r, t: 0.3 * self._terrain(r, t)
        tz = terrain(gr.double(), th.double())
        tx = gr.double() * torch.sqrt(1 - th.double() ** 2)
        ty = gr.double() * th.double()
        nsamples = 512
        data = torch.zeros((nsweeps, nsamples), dtype=torch.complex64)
        i = torch.arange(nsamples, dtype=torch.float64)
        for k in range(ntg):
            d = torch.sqrt((pos[:, 0].double() - tx[k]) ** 2
                           + (pos[:, 1].double() - ty[k]) ** 2
                           + (pos[:, 2].double() - tz[k]) ** 2)
            env = torch.special.sinc(
                (i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        rr = r0 + (r1 - r0) / 64 * torch.arange(64, dtype=torch.float64)
        ttg = t0 + (t1 - t0) / 64 * torch.arange(64, dtype=torch.float64)
        dem = terrain(rr[:, None], ttg[None, :]).float()
        att = torch.zeros((nsweeps, 3))
        el = torch.linspace(-1.2, 1.2, 16)
        # Azimuth extent 0.6 rad: hard cutoff at theta = sin(0.6) = 0.565,
        # inside the +-0.8 scene extent.
        az = torch.linspace(-0.6, 0.6, 64)
        g = torch.exp(-el[:, None] ** 2 / 1.0) * torch.exp(-az[None, :] ** 2 / 0.8)
        g_extent = [-1.2, -0.6, 1.2, 0.6]
        kw = dict(stages=2, dealias=True, grid_oversample=1.5,
                  att=att, g=g, g_extent=g_extent, dem=dem,
                  weight_map_downsample=4,
                  antenna_leaf_gain="subaperture")
        img1 = torchbp.ops.ffbp(data, grid, self.fc, self.r_res, pos, **kw)
        img2 = torchbp.ops.ffbp(data, grid, self.fc, self.r_res, pos,
                                afbp_nsub=4, **kw)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 3e-2)

    def test_ffbp_afbp_base_antenna_dem(self):
        # Antenna-weighted ffbp with afbp base and a DEM must match the
        # plain weighted ffbp with the same DEM. Tolerance as in
        # test_ffbp_afbp_base_dem (terrain-slope amplitude error of the
        # afbp fusion).
        data, pos, dem = self._dem_scene()
        att = torch.zeros((self.nsweeps, 3))
        att[:, 2] = 0.05 * torch.sin(
            2 * torch.pi * torch.arange(self.nsweeps) / self.nsweeps)
        el = torch.linspace(-1.0, 1.0, 16)
        az = torch.linspace(-1.2, 1.2, 64)
        g = torch.exp(-el[:, None] ** 2 / 0.8) * torch.exp(-az[None, :] ** 2 / 0.35)
        g_extent = [-1.0, -1.2, 1.0, 1.2]
        img1 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                att=att, g=g, g_extent=g_extent, dem=dem)
        img2 = torchbp.ops.ffbp(data, self.grid, self.fc, self.r_res, pos,
                                stages=2, dealias=True, grid_oversample=2.0,
                                att=att, g=g, g_extent=g_extent, dem=dem,
                                afbp_nsub=4)
        rel = ((img1 - img2).abs().max() / img1.abs().max()).item()
        self.assertLess(rel, 3e-2)


