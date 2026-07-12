#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
import unittest
import torchbp


class TestInsarRmeBlocksvd(TestCase):
    """End-to-end InSAR RME on synthetic point-scatterer data."""

    fc = 6e9
    r_res = 0.3
    grid_polar = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 64,
                  "ntheta": 48}
    nsweeps = 64
    sweep_samples = 512

    def _make_data(self, targets, amps, pos):
        """Point responses consistent with the backprojection phase model."""
        c0 = 299792458.0
        data = torch.zeros(
            pos.shape[0], self.sweep_samples, dtype=torch.complex64
        )
        m_idx = torch.arange(pos.shape[0])
        for t, a in zip(targets, amps):
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = d / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += a * w[valid] * phase[valid]
        return data

    def _scene(self):
        torch.manual_seed(3)
        ntargets = 40
        r = 85.0 + 30.0 * torch.rand(ntargets)
        t = -0.2 + 0.4 * torch.rand(ntargets)
        targets = torch.stack(
            [r * torch.sqrt(1 - t**2), r * t, torch.zeros_like(r)], dim=1
        )
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = torch.linspace(-2.0, 2.0, self.nsweeps)
        pos[:, 2] = 30.0
        return targets, amps, pos

    def test_recovers_x_error(self):
        targets, amps, pos = self._scene()
        data_m = self._make_data(targets, amps, pos)
        img_m = torchbp.ops.backprojection_polar_2d(
            data_m, self.grid_polar, self.fc, self.r_res, pos
        )[0]

        # Slave measured at pos + [dx, 0, 0] but backprojected at pos;
        # blocksvd should recover the zero-mean dx profile. A few cycles
        # per aperture: a slower error is close to the unobservable
        # linear trend.
        dx = 2e-3 * torch.sin(
            2 * torch.pi * 3 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_err = pos.clone()
        pos_err[:, 0] += dx
        data_s = self._make_data(targets, amps, pos_err)

        pos_new, phi = torchbp.autofocus.insar_rme_blocksvd(
            data_s, pos, img_m, self.fc, self.r_res, self.grid_polar,
            n_az_blocks=8, n_r_blocks=4,
        )
        d_corr = pos_new[:, 0] - pos[:, 0]
        resid = dx - d_corr
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.4 * dx.pow(2).mean().sqrt().item(),
        )

    def test_variants_run(self):
        targets, amps, pos = self._scene()
        data_m = self._make_data(targets, amps, pos)
        img_m = torchbp.ops.backprojection_polar_2d(
            data_m, self.grid_polar, self.fc, self.r_res, pos
        )[0]
        coh = torch.rand(
            self.grid_polar["nr"], self.grid_polar["ntheta"]
        ) * 0.5 + 0.5
        for row_weight in ("coherence", "power", "uniform"):
            for aperture_mask in (True, False):
                pos_new, phi = torchbp.autofocus.insar_rme_blocksvd(
                    data_m, pos, img_m, self.fc, self.r_res, self.grid_polar,
                    n_az_blocks=4, n_r_blocks=2, row_weight=row_weight,
                    aperture_mask=aperture_mask, spatial_coherence=coh,
                    phi_lowpass=9,
                )
                self.assertTrue(torch.isfinite(phi).all())
                # Master vs its own data: no motion error (sidelobes of
                # the point-target scene leave a small residual)
                self.assertLess(phi.abs().max().item(), 0.3)

    def test_strata_runs(self):
        targets, amps, pos = self._scene()
        data_m = self._make_data(targets, amps, pos)
        img_m = torchbp.ops.backprojection_polar_2d(
            data_m, self.grid_polar, self.fc, self.r_res, pos
        )[0]
        dx = 2e-3 * torch.sin(
            2 * torch.pi * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_err = pos.clone()
        pos_err[:, 0] += dx
        data_s = self._make_data(targets, amps, pos_err)
        pos_new, delta = torchbp.autofocus.insar_rme_blocksvd_strata(
            data_s, pos, img_m, self.fc, self.r_res, self.grid_polar,
            n_strata=3, n_az_blocks_per_strata=8, estimate_z=True,
            phi_lowpass=9,
        )
        self.assertTrue(torch.isfinite(delta).all())
        resid = dx - delta[:, 0]
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.6 * dx.pow(2).mean().sqrt().item(),
        )


class TestPga(TestCase):
    """Plain phase gradient autofocus on a synthetic point-target image."""

    @staticmethod
    def _sharpness(img):
        p = img.abs() ** 2
        return ((p**2).sum() / (p.sum() ** 2)).item()

    def test_recovers_phase_error(self):
        torch.manual_seed(3)
        nr, ntheta = 96, 256
        # Sparse bright point targets over weak clutter
        img = 0.01 * torch.randn(nr, ntheta, dtype=torch.complex64)
        r_idx = torch.randint(0, nr, (16,))
        t_idx = torch.randint(0, ntheta, (16,))
        img[r_idx, t_idx] += (1.0 + torch.rand(16)) * torch.exp(
            2j * torch.pi * torch.rand(16)
        )

        # Corrupt with a smooth azimuth phase error (the model pga inverts:
        # img_focused = ifft(fft(img) * exp(-1j*phi)))
        k = torch.arange(ntheta)
        phi_true = 2.0 * torch.sin(2 * torch.pi * 2 * k / ntheta) + torch.cos(
            2 * torch.pi * 5 * k / ntheta
        )
        img_bad = torch.fft.ifft(
            torch.fft.fft(img, axis=-1) * torch.exp(1j * phi_true)[None, :], axis=-1
        )

        img_focus, phi = torchbp.autofocus.pga(img_bad.clone())

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(phi).all())
        self.assertGreater(
            self._sharpness(img_focus), 2.0 * self._sharpness(img_bad)
        )
        # Recovered phase must match the injected error up to the
        # unobservable linear trend and constant offset.
        from torchbp.util import detrend, unwrap
        resid = detrend(unwrap(phi - phi_true))
        resid = resid - resid.mean()
        # Passing runs give ~0.2x; a broken azimuth shift gives ~1x.
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.4 * phi_true.std().item(),
        )


class TestGpgaBpPolar(TestCase):
    """End-to-end GPGA polar autofocus on synthetic point-scatterer data."""

    fc = 6e9
    r_res = 0.3
    grid_polar = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 64,
                  "ntheta": 64}
    nsweeps = 128
    sweep_samples = 512

    def _make_data(self, targets, amps, pos):
        """Point responses consistent with the backprojection phase model."""
        c0 = 299792458.0
        data = torch.zeros(
            pos.shape[0], self.sweep_samples, dtype=torch.complex64
        )
        m_idx = torch.arange(pos.shape[0])
        for t, a in zip(targets, amps):
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = d / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += a * w[valid] * phase[valid]
        return data

    def _scene(self):
        torch.manual_seed(5)
        ntargets = 12
        r = 90.0 + 20.0 * torch.rand(ntargets)
        t = -0.15 + 0.3 * torch.rand(ntargets)
        targets = torch.stack(
            [r * torch.sqrt(1 - t**2), r * t, torch.zeros_like(r)], dim=1
        )
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, self.nsweeps)
        pos[:, 2] = 30.0
        return targets, amps, pos

    @staticmethod
    def _sharpness(img):
        # Inverse participation ratio of the intensity: higher for a
        # well-focused (peaky) image, lower for a blurred one.
        p = img.abs() ** 2
        return (p**2).sum() / (p.sum() ** 2)

    def test_focuses_range_motion_error(self):
        targets, amps, pos = self._scene()
        # True platform has a smooth zero-mean range (X) motion error;
        # the data are formed with it but backprojected at the nominal
        # positions, defocusing the image. GPGA must recover it.
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid_polar, self.fc, self.r_res, pos
        )[0]
        img_focus, phi = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            max_iters=8, target_threshold_db=15,
        )

        # Autofocus must run to completion (the lowpass path is exercised
        # once window_width drops below the number of sweeps) and produce
        # a finite, sharper image.
        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(phi).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

        # The recovered range correction should track the injected error.
        from torchbp.util import detrend
        c0 = 299792458.0
        d = phi * c0 / (4 * torch.pi * self.fc)
        # Sign/linear-trend are unobservable, so compare detrended and
        # take the better-matching sign.
        resid = min(
            detrend(dx - d).pow(2).mean().sqrt().item(),
            detrend(dx + d).pow(2).mean().sqrt().item(),
        )
        self.assertLess(resid, 0.4 * dx.pow(2).mean().sqrt().item())

    def test_ffbp_image_formation(self):
        # algorithm="ffbp" swaps the image formation for fast factorized
        # backprojection but should drive the same autofocus solution.
        targets, amps, pos = self._scene()
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)

        common = dict(max_iters=8, target_threshold_db=15)
        img_bp, phi_bp = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar, **common
        )
        img_ff, phi_ff = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            algorithm="ffbp", image_opts={"stages": 4}, **common
        )

        self.assertTrue(torch.isfinite(img_ff).all())
        self.assertTrue(torch.isfinite(phi_ff).all())
        self.assertEqual(img_ff.shape, img_bp.shape)
        # The recovered phase error must agree with the exact-backprojection
        # path (both estimate the same platform motion error).
        corr = torch.corrcoef(torch.stack([phi_bp, phi_ff]))[0, 1]
        self.assertGreater(corr.item(), 0.9)


class TestGpgaBpPolarTde(TestGpgaBpPolar):
    """End-to-end GPGA TDE (3D position) autofocus on synthetic data.

    Inherits the scene/data helpers from TestGpgaBpPolar; the parent's test
    methods are disabled by overriding them.
    """

    # Don't re-run the parent's tests in this class.
    def test_focuses_range_motion_error(self):
        pass

    def test_ffbp_image_formation(self):
        pass

    def test_focuses_and_recovers_position(self):
        targets, amps, pos = self._scene()
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid_polar, self.fc, self.r_res, pos
        )[0]
        img_focus, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            azimuth_divisions=2, range_divisions=2, estimate_z=False,
            max_iters=8, target_threshold_db=15,
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(pos_new).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

        # The solved X correction should track the injected error
        # (linear trend is unobservable).
        from torchbp.util import detrend
        d = pos_new[:, 0] - pos[:, 0]
        resid = detrend(dx - d).pow(2).mean().sqrt().item()
        self.assertLess(resid, 0.5 * dx.pow(2).mean().sqrt().item())

    def test_dead_blocks_and_ffbp_initial_image(self):
        # Regression: blocks with no targets (grid extends past the data's
        # max range) used to crash on an empty reduction, and the initial
        # image ignored use_ffbp. Also exercises the weighted block-center
        # computation.
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        grid_dead = dict(self.grid_polar)
        grid_dead["r"] = (80.0, 400.0)
        grid_dead["nr"] = 128

        img, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fc, self.r_res, grid_dead,
            azimuth_divisions=2, range_divisions=4, estimate_z=False,
            max_iters=2, algorithm="ffbp", image_opts={"stages": 3},
        )
        self.assertTrue(torch.isfinite(img).all())
        self.assertTrue(torch.isfinite(pos_new).all())


class TestGpgaCartesian(TestGpgaBpPolar):
    """End-to-end GPGA on a Cartesian grid (BP and CFBP image formation).

    Reuses the polar scene/data helpers but images on a CartesianGrid,
    exercising the grid-agnostic pixel->world mapping and Cartesian image
    formers. The parent's polar-only tests are disabled.
    """

    grid_cart = {"x": (85.0, 115.0), "y": (-20.0, 20.0), "nx": 96, "ny": 128}

    # Disable the inherited polar tests.
    def test_focuses_range_motion_error(self):
        pass

    def test_ffbp_image_formation(self):
        pass

    def _scene_with_error(self):
        targets, amps, pos = self._scene()
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)
        return data, pos

    def test_cart_bp_focuses_range_motion_error(self):
        data, pos = self._scene_with_error()
        img_blur = torchbp.ops.backprojection_cart_2d(
            data, self.grid_cart, self.fc, self.r_res, pos
        )[0]
        img_focus, phi = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_cart,
            algorithm="bp", max_iters=8, target_threshold_db=15,
        )
        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(phi).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

    def test_cfbp_image_formation(self):
        # algorithm="cfbp" swaps the Cartesian image formation for Cartesian
        # factorized backprojection but should drive the same autofocus
        # solution as direct Cartesian backprojection.
        data, pos = self._scene_with_error()
        common = dict(max_iters=8, target_threshold_db=15)
        img_bp, phi_bp = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_cart,
            algorithm="bp", **common
        )
        img_cf, phi_cf = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_cart,
            algorithm="cfbp", image_opts={"stages": 4}, **common
        )
        self.assertTrue(torch.isfinite(img_cf).all())
        self.assertTrue(torch.isfinite(phi_cf).all())
        self.assertEqual(img_cf.shape, img_bp.shape)
        corr = torch.corrcoef(torch.stack([phi_bp, phi_cf]))[0, 1]
        self.assertGreater(corr.item(), 0.9)

    def test_tde_cart_focuses(self):
        data, pos = self._scene_with_error()
        img_blur = torchbp.ops.backprojection_cart_2d(
            data, self.grid_cart, self.fc, self.r_res, pos
        )[0]
        img_focus, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fc, self.r_res, self.grid_cart,
            azimuth_divisions=2, range_divisions=2, estimate_z=False,
            algorithm="bp", max_iters=8, target_threshold_db=15,
        )
        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(pos_new).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

    def test_antenna_args_rejected_for_cartesian(self):
        data, pos = self._scene_with_error()
        with self.assertRaises(ValueError):
            torchbp.autofocus.gpga(
                None, data, pos, self.fc, self.r_res, self.grid_cart,
                algorithm="cfbp", g=torch.ones(4, 4), max_iters=1,
            )


class TestGpgaDem(TestGpgaBpPolar):
    """GPGA with a DEM: scatterers on a sloped plane instead of z=0.

    Reuses the polar scene/data helpers; the parent's z=0 tests are
    disabled. The DEM is coarser than the image grid to also exercise the
    bilinear target-height interpolation.
    """

    dem_nr = 32
    dem_ntheta = 32

    # Disable the inherited z=0 tests.
    def test_focuses_range_motion_error(self):
        pass

    def test_ffbp_image_formation(self):
        pass

    @staticmethod
    def _plane_z(x, y):
        return 4.0 + 0.08 * (x - 80.0) + 0.05 * y

    def _dem(self):
        # DEM sample [i, j] corresponds to image pixel (i * nr / dem_nr,
        # j * ntheta / dem_ntheta), i.e. r/theta at fraction i/dem_nr of
        # the grid extent (the backprojection kernel convention).
        r0, r1 = self.grid_polar["r"]
        t0, t1 = self.grid_polar["theta"]
        r = r0 + (r1 - r0) * torch.arange(self.dem_nr) / self.dem_nr
        t = t0 + (t1 - t0) * torch.arange(self.dem_ntheta) / self.dem_ntheta
        x = r[:, None] * torch.sqrt(1 - t[None, :] ** 2)
        y = r[:, None] * t[None, :]
        return self._plane_z(x, y).to(torch.float32)

    def _dem_scene_with_error(self):
        targets, amps, pos = self._scene()
        targets[:, 2] = self._plane_z(targets[:, 0], targets[:, 1])
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)
        return data, pos, dx

    def test_zero_dem_matches_no_dem(self):
        # A zero DEM must reproduce the z=0 solution.
        targets, amps, pos = self._scene()
        dx = 4e-3 * torch.sin(
            2 * torch.pi * 2 * torch.arange(self.nsweeps) / self.nsweeps
        )
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)

        common = dict(max_iters=4, target_threshold_db=15)
        _, phi = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar, **common
        )
        _, phi_dem = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            dem=torch.zeros(self.dem_nr, self.dem_ntheta), **common
        )
        self.assertLess((phi - phi_dem).abs().max().item(), 1e-2)

    def test_gpga_dem_focuses(self):
        data, pos, dx = self._dem_scene_with_error()
        dem = self._dem()

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid_polar, self.fc, self.r_res, pos, dem=dem
        )[0]
        img_focus, phi = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            max_iters=8, target_threshold_db=15, dem=dem,
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(phi).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

        from torchbp.util import detrend
        c0 = 299792458.0
        d = phi * c0 / (4 * torch.pi * self.fc)
        resid = min(
            detrend(dx - d).pow(2).mean().sqrt().item(),
            detrend(dx + d).pow(2).mean().sqrt().item(),
        )
        self.assertLess(resid, 0.4 * dx.pow(2).mean().sqrt().item())

    def test_gpga_ffbp_dem(self):
        # algorithm="ffbp" with a DEM should drive the same autofocus
        # solution as exact backprojection with the DEM.
        data, pos, dx = self._dem_scene_with_error()
        dem = self._dem()

        common = dict(max_iters=8, target_threshold_db=15, dem=dem)
        _, phi_bp = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar, **common
        )
        _, phi_ff = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            algorithm="ffbp", image_opts={"stages": 4}, **common
        )
        self.assertTrue(torch.isfinite(phi_ff).all())
        corr = torch.corrcoef(torch.stack([phi_bp, phi_ff]))[0, 1]
        self.assertGreater(corr.item(), 0.9)

    def test_tde_dem_focuses_and_recovers(self):
        data, pos, dx = self._dem_scene_with_error()
        dem = self._dem()

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid_polar, self.fc, self.r_res, pos, dem=dem
        )[0]
        img_focus, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            azimuth_divisions=2, range_divisions=2, estimate_z=False,
            max_iters=8, target_threshold_db=15, dem=dem,
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(pos_new).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )

        from torchbp.util import detrend
        d = pos_new[:, 0] - pos[:, 0]
        resid = detrend(dx - d).pow(2).mean().sqrt().item()
        self.assertLess(resid, 0.5 * dx.pow(2).mean().sqrt().item())

    def test_dem_rejected_for_unsupported_algorithms(self):
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        dem = torch.zeros(self.dem_nr, self.dem_ntheta)
        with self.assertRaises(ValueError):
            torchbp.autofocus.gpga(
                None, data, pos, self.fc, self.r_res, self.grid_polar,
                algorithm="afbp", image_opts={"nsub": 4}, dem=dem,
                max_iters=1,
            )
        grid_cart = {"x": (85.0, 115.0), "y": (-20.0, 20.0),
                     "nx": 64, "ny": 64}
        with self.assertRaises(ValueError):
            torchbp.autofocus.gpga(
                None, data, pos, self.fc, self.r_res, grid_cart,
                algorithm="bp", dem=dem, max_iters=1,
            )
        with self.assertRaises(ValueError):
            torchbp.autofocus.gpga_tde(
                None, data, pos, self.fc, self.r_res, grid_cart,
                azimuth_divisions=2, range_divisions=2,
                algorithm="cfbp", dem=dem, max_iters=1,
            )


