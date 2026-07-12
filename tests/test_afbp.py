#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
import unittest
import torchbp


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

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
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


