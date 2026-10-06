#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
import torchbp

from conftest import requires_cuda, C0, fmcw_scene


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


class TestInsarRmeMultisquint(TestCase):
    """Multisquint InSAR RME: X and along-track (Y) recovery on synthetic
    point-scatterer data.

    ntheta is chosen so the theta-spectrum Nyquist covers the look bands
    (f = (2/wl)*cos(el)*y up to y = 2 m); the theta extent gives the
    aspect-angle diversity the Y estimate needs.
    """

    fc = 6e9
    r_res = 0.3
    grid_polar = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 64,
                  "ntheta": 128}
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
        torch.manual_seed(7)
        ntargets = 60
        r = 85.0 + 30.0 * torch.rand(ntargets)
        t = -0.22 + 0.44 * torch.rand(ntargets)
        targets = torch.stack(
            [r * torch.sqrt(1 - t**2), r * t, torch.zeros_like(r)], dim=1
        )
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = torch.linspace(-2.0, 2.0, self.nsweeps)
        pos[:, 2] = 30.0
        return targets, amps, pos

    def _master(self, targets, amps, pos):
        data_m = self._make_data(targets, amps, pos)
        return torchbp.ops.backprojection_polar_2d(
            data_m, self.grid_polar, self.fc, self.r_res, pos
        )[0]

    def test_recovers_xy_error(self):
        from torchbp.util import detrend
        targets, amps, pos = self._scene()
        img_m = self._master(targets, amps, pos)

        # Slave measured with smooth zero-mean X and Y errors (integer
        # cycles per aperture; constant/linear parts are unobservable)
        # but backprojected at the nominal positions.
        n = torch.arange(self.nsweeps)
        dx = 2e-3 * torch.sin(2 * torch.pi * 3 * n / self.nsweeps)
        dy = 8e-3 * torch.sin(2 * torch.pi * 2 * n / self.nsweeps + 0.7)
        pos_err = pos.clone()
        pos_err[:, 0] += dx
        pos_err[:, 1] += dy
        data_s = self._make_data(targets, amps, pos_err)
        img_s = torchbp.ops.backprojection_polar_2d(
            data_s, self.grid_polar, self.fc, self.r_res, pos
        )[0]

        pos_new, delta = torchbp.autofocus.insar_rme_multisquint(
            img_m, img_s, pos, self.fc, self.grid_polar,
            n_looks=16, estimate_y=True,
        )
        self.assertTrue(torch.isfinite(delta).all())
        # X must stay accurate with the Y error present and vice versa
        # (cross-leakage shows up as a residual the size of the other
        # axis's error). Y is the noisier observable (sin(aspect)
        # projection); allow it a looser bound.
        for ax, err, lim in ((0, dx, 0.5), (1, dy, 0.7)):
            resid = detrend(err - delta[:, ax])
            self.assertLess(
                resid.pow(2).mean().sqrt().item(),
                lim * err.pow(2).mean().sqrt().item(),
                msg=f"axis {ax}",
            )

    def test_x_only_default_leaves_y_zero(self):
        from torchbp.util import detrend
        targets, amps, pos = self._scene()
        img_m = self._master(targets, amps, pos)

        n = torch.arange(self.nsweeps)
        dx = 2e-3 * torch.sin(2 * torch.pi * 3 * n / self.nsweeps)
        pos_err = pos.clone()
        pos_err[:, 0] += dx
        data_s = self._make_data(targets, amps, pos_err)
        img_s = torchbp.ops.backprojection_polar_2d(
            data_s, self.grid_polar, self.fc, self.r_res, pos
        )[0]

        pos_new, delta = torchbp.autofocus.insar_rme_multisquint(
            img_m, img_s, pos, self.fc, self.grid_polar, n_looks=16,
        )
        self.assertEqual(delta[:, 1].abs().max().item(), 0.0)
        resid = detrend(dx - delta[:, 0])
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.5 * dx.pow(2).mean().sqrt().item(),
        )

    def test_y_only_recovery(self):
        # A pure Y error must not leak into X.
        from torchbp.util import detrend
        targets, amps, pos = self._scene()
        img_m = self._master(targets, amps, pos)

        n = torch.arange(self.nsweeps)
        dy = 8e-3 * torch.sin(2 * torch.pi * 2 * n / self.nsweeps + 0.7)
        pos_err = pos.clone()
        pos_err[:, 1] += dy
        data_s = self._make_data(targets, amps, pos_err)
        img_s = torchbp.ops.backprojection_polar_2d(
            data_s, self.grid_polar, self.fc, self.r_res, pos
        )[0]

        pos_new, delta = torchbp.autofocus.insar_rme_multisquint(
            img_m, img_s, pos, self.fc, self.grid_polar,
            n_looks=16, estimate_y=True,
        )
        resid = detrend(dy - delta[:, 1])
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.7 * dy.pow(2).mean().sqrt().item(),
        )
        self.assertLess(
            delta[:, 0].pow(2).mean().sqrt().item(),
            0.5 * dy.pow(2).mean().sqrt().item(),
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


class TestPgaWindowEstimate(TestCase):
    """Blur-width estimation from the peak-centered noncoherent sum."""

    def test_tracks_blur_width(self):
        torch.manual_seed(3)
        nr, N = 256, 4096
        img = 0.01 * torch.randn(nr, N, dtype=torch.complex64)
        r_idx = torch.randint(0, nr, (500,))
        t_idx = torch.randint(0, N, (500,))
        img[r_idx, t_idx] += (1.0 + torch.rand(500)) * torch.exp(
            2j * torch.pi * torch.rand(500)
        )
        k = torch.arange(N).float()
        # Quadratic phase error: kernel extent ~ +-2*beta/pi bins.
        phi = 20.0 * (2 * k / N - 1) ** 2

        def corrupt(p):
            return torch.fft.ifft(
                torch.fft.fft(img, axis=-1) * torch.exp(1j * p)[None, :],
                axis=-1,
            )

        w_focused = torchbp.autofocus.pga_window_estimate(img)
        w_blur = torchbp.autofocus.pga_window_estimate(corrupt(phi))
        w_blur3 = torchbp.autofocus.pga_window_estimate(corrupt(3 * phi))
        # A focused image collapses to the mainlobe width.
        self.assertLess(w_focused, 32)
        # The window must cover the kernel (~26 and ~76 bins) with
        # margin, without ballooning to the image size, and must grow
        # with the blur.
        self.assertGreater(w_blur, 26)
        self.assertLess(w_blur, 300)
        self.assertGreater(w_blur3, 76)
        self.assertLess(w_blur3, 900)
        self.assertGreater(w_blur3, w_blur)


class TestPgaXz(TestCase):
    """Image-domain range/elevation (x, z) PGA on a polar image."""

    fc = 6e9
    h = 60.0
    r_res = 0.3
    grid = {"r": (60.0, 140.0), "theta": (-0.25, 0.25), "nr": 128,
            "ntheta": 256}
    nsweeps = 128
    sweep_samples = 512

    @staticmethod
    def _sharpness(img):
        p = img.abs() ** 2
        return ((p**2).sum() / (p.sum() ** 2)).item()

    def _geometry(self):
        """Per-row cos/sin elevation angle, platform above scene at z=0."""
        r0, r1 = self.grid["r"]
        nr = self.grid["nr"]
        r = r0 + (r1 - r0) / nr * torch.arange(nr)
        slant = torch.sqrt(r**2 + self.h**2)
        return r / slant, -self.h / slant

    def test_recovers_xz_error(self):
        # Corrupt a sparse point-target image through the physical model
        # pga_xz inverts: phase (4*pi*fc/c) * (cos_el*dx + sin_el*dz) at
        # spectrum bin f, with the per-row cos(el) scaling of the
        # spectral axis. Both profiles must be recovered on the common
        # axis, verifying the per-bin x/z separation and the axis
        # rescaling (dz is at a frequency where ignoring the scaling
        # would misalign the blocks by about a cycle).
        torch.manual_seed(3)
        nr, ntheta = self.grid["nr"], self.grid["ntheta"]
        img = 0.01 * torch.randn(nr, ntheta, dtype=torch.complex64)
        r_idx = torch.randint(0, nr, (64,))
        t_idx = torch.randint(0, ntheta, (64,))
        img[r_idx, t_idx] += (1.0 + torch.rand(64)) * torch.exp(
            2j * torch.pi * torch.rand(64)
        )

        cos_el, sin_el = self._geometry()
        f = (torch.arange(ntheta) - ntheta // 2).to(torch.float32)

        def dx_fun(q):
            return 4e-3 * torch.sin(2 * torch.pi * 2 * q / ntheta)

        def dz_fun(q):
            return 8e-3 * torch.sin(2 * torch.pi * 4 * q / ntheta + 1.0)

        k_wave = 4 * torch.pi * self.fc / 299792458.0
        # Row's bin f sees the common-axis profile at f / gamma(r).
        q = f[None, :] / (cos_el / cos_el.max())[:, None]
        phi_true = k_wave * (
            cos_el[:, None] * dx_fun(q) + sin_el[:, None] * dz_fun(q)
        )
        img_bad = torch.fft.ifft(
            torch.fft.ifftshift(
                torch.fft.fftshift(torch.fft.fft(img, axis=-1), dim=-1)
                * torch.exp(1j * phi_true),
                dim=-1,
            ),
            axis=-1,
        )

        img_focus, d = torchbp.autofocus.pga_xz(
            img_bad.clone(), self.grid, self.fc, self.h, range_divisions=4
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(d).all())
        self.assertGreater(
            self._sharpness(img_focus), 2.0 * self._sharpness(img_bad)
        )
        # Recovered profiles must match the injected ones up to the
        # unobservable linear trend, without x/z cross-leakage.
        from torchbp.util import detrend
        for comp, true in ((0, dx_fun(f)), (1, dz_fun(f))):
            resid = detrend(true - d[comp])
            self.assertLess(
                resid.pow(2).mean().sqrt().item(),
                0.4 * true.pow(2).mean().sqrt().item(),
            )

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

    def test_focuses_backprojected_motion_error(self):
        # End-to-end: data simulated with true x and z platform motion
        # errors, backprojected at the nominal positions. This validates
        # the geometry conventions against the real kernel (a sign error
        # in the correction would make the image worse, not better).
        torch.manual_seed(5)
        ntargets = 24
        r = 70.0 + 60.0 * torch.rand(ntargets)
        t = -0.2 + 0.4 * torch.rand(ntargets)
        targets = torch.stack(
            [r * torch.sqrt(1 - t**2), r * t, torch.zeros_like(r)], dim=1
        )
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, self.nsweeps)
        pos[:, 2] = self.h

        n = torch.arange(self.nsweeps)
        dx = 4e-3 * torch.sin(2 * torch.pi * 2 * n / self.nsweeps)
        dz = 15e-3 * torch.sin(2 * torch.pi * 3 * n / self.nsweeps + 1.0)
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        pos_true[:, 2] += dz
        data = self._make_data(targets, amps, pos_true)

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos
        )[0]
        img_focus, d = torchbp.autofocus.pga_xz(
            img_blur.clone(), self.grid, self.fc, self.h,
            range_divisions=4,
        )

        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(d).all())
        self.assertGreater(
            self._sharpness(img_focus), 1.3 * self._sharpness(img_blur)
        )
        # The range-variant z correction must add focus over plain
        # (space-invariant) pga on the same image.
        img_pga, _ = torchbp.autofocus.pga(img_blur.clone())
        self.assertGreater(
            self._sharpness(img_focus), 1.2 * self._sharpness(img_pga)
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

    def _tde_scene_with_x_error(self):
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
        return data, pos, dx, img_blur

    def test_coarse_window_schedule_recovers_position(self):
        # Two-stage schedule: coarse iterations with the sin(aspect) mode
        # projected out of the block phase gradient and an error-driven
        # window jump, then regular iterations. Must still focus the scene
        # and track the injected error; the coarse stage must actually run
        # (initial window above coarse_window) and the run must converge
        # in fewer iterations than the plain schedule needs to walk the
        # window down.
        from torchbp.util import detrend
        data, pos, dx, img_blur = self._tde_scene_with_x_error()
        img_focus, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            azimuth_divisions=2, range_divisions=2, estimate_z=False,
            max_iters=8, target_threshold_db=15,
            coarse_window=self.nsweeps // 8,
        )
        self.assertTrue(torch.isfinite(img_focus).all())
        self.assertTrue(torch.isfinite(pos_new).all())
        self.assertGreater(
            self._sharpness(img_focus).item(),
            1.3 * self._sharpness(img_blur).item(),
        )
        d = pos_new[:, 0] - pos[:, 0]
        resid = detrend(dx - d).pow(2).mean().sqrt().item()
        self.assertLess(resid, 0.5 * dx.pow(2).mean().sqrt().item())
        with self.assertRaises(ValueError):
            torchbp.autofocus.gpga_tde(
                None, data, pos, self.fc, self.r_res, self.grid_polar,
                azimuth_divisions=2, range_divisions=2, estimate_z=False,
                max_iters=1, coarse_window=0,
            )

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

    def test_afbp_matches_bp_with_dem(self):
        # algorithm="afbp" with a DEM must reproduce the direct-bp phase
        # estimate (the afbp image matches bp including pixel phase).
        data, pos, dx = self._dem_scene_with_error()
        dem = self._dem()
        common = dict(max_iters=2, target_threshold_db=15, dem=dem)
        _, phi_bp = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar, **common
        )
        _, phi_afbp = torchbp.autofocus.gpga(
            None, data, pos, self.fc, self.r_res, self.grid_polar,
            algorithm="afbp", image_opts={"nsub": 4}, **common
        )
        self.assertTrue(torch.isfinite(phi_afbp).all())
        self.assertLess((phi_bp - phi_afbp).abs().max().item(), 1e-2)

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




class TestBatchedEigh(TestCase):
    """_batched_eigh matches torch.linalg.eigh with a bounded workspace.

    cuSOLVER's batched Jacobi solver takes a workspace proportional to the
    batch size and thousands of times the input size, so the per-sweep 3x3
    solve in gpga_tde asked for 10 GB on a 40k-sweep collection.
    """

    def _sym(self, batch, n):
        a = torch.randn(batch, 4 * n, n)
        return (a.transpose(-1, -2) @ a).contiguous()

    def test_matches_unchunked(self):
        for n in (2, 3):
            a = self._sym(700, n)
            ref_e, ref_v = torch.linalg.eigh(a)
            # Force many chunks: workspace budget of a few matrices.
            e, v = torchbp.autofocus._batched_eigh(
                a, max_workspace=8192 * n * n * a.element_size() * 3
            )
            self.assertEqual(e, ref_e)
            self.assertEqual(v, ref_v)

    def test_reconstructs_input(self):
        a = self._sym(700, 3)
        e, v = torchbp.autofocus._batched_eigh(
            a, max_workspace=8192 * 9 * a.element_size() * 3
        )
        rec = v @ torch.diag_embed(e) @ v.transpose(-1, -2)
        self.assertEqual(rec, a, atol=1e-4, rtol=1e-4)

    def test_passthrough_shapes(self):
        # Unbatched input, and a batch shape with more than one leading dim.
        a2 = self._sym(1, 3)[0]
        e, v = torchbp.autofocus._batched_eigh(a2)
        self.assertEqual(e, torch.linalg.eigh(a2)[0])
        a4 = self._sym(60, 3).reshape(5, 12, 3, 3)
        e, v = torchbp.autofocus._batched_eigh(
            a4, max_workspace=8192 * 9 * a4.element_size() * 3
        )
        self.assertEqual(e.shape, torch.Size([5, 12, 3]))
        self.assertEqual(v.shape, torch.Size([5, 12, 3, 3]))
        self.assertEqual(e, torch.linalg.eigh(a4)[0])


class TestAntennaWeightMemory(TestCase):
    """Antenna weight chunking and the weight cache budget.

    The per-block caches hold one float per (target, sweep), unbounded they
    reached several GB on a block-divided long collection causing OOM.
    """

    g_extent = [-0.6, -0.5, 0.3, 0.5]
    nsweeps = 200
    ntargets = 40

    def setUp(self):
        super().setUp()
        torch.manual_seed(1)
        self.g = torch.rand(16, 32)
        n = self.nsweeps
        self.pos = torch.stack([
            torch.zeros(n), torch.linspace(0, 40, n), torch.full((n,), 30.0)
        ], dim=-1)
        self.att = torch.zeros(n, 3)
        self.att[:, 0] = -0.17

    def _tpos(self, keys):
        k = torch.as_tensor(list(keys), dtype=torch.float32)
        return torch.stack(
            [60 + 2 * k, -20 + 1.5 * k, torch.zeros_like(k)], dim=-1
        ).to(self.pos.device)

    def _direct(self, tpos):
        return torchbp.autofocus._antenna_weights(
            tpos, self.pos, self.att, self.g, self.g_extent
        )

    def test_chunked_antenna_weights_match(self):
        tpos = self._tpos(range(self.ntargets))
        saved = torchbp.autofocus._ANT_WEIGHT_CHUNK_BYTES
        try:
            torchbp.autofocus._ANT_WEIGHT_CHUNK_BYTES = 1 << 40
            ref = self._direct(tpos)
            # Budget for ~4 targets per chunk, so the loop runs many times.
            torchbp.autofocus._ANT_WEIGHT_CHUNK_BYTES = (
                20 * self.nsweeps * tpos.element_size() * 4
            )
            chunked = self._direct(tpos)
        finally:
            torchbp.autofocus._ANT_WEIGHT_CHUNK_BYTES = saved
        self.assertEqual(chunked, ref)

    def test_cache_rows_follow_keys(self):
        cache = torchbp.autofocus._AntennaWeightCache(
            self.att, self.g, self.g_extent
        )
        keys = list(range(self.ntargets))
        tpos = self._tpos(keys)
        w1, _, _ = cache.get(keys, tpos, self.pos)
        self.assertEqual(w1, self._direct(tpos))

        # Reordered subset of cached keys: rows must follow the given order.
        sub = [30, 3, 21, 7]
        w2, _, _ = cache.get(sub, self._tpos(sub), self.pos)
        self.assertEqual(w2, w1[[keys.index(k) for k in sub]])

        # Interleaved hits and misses land in the right rows.
        mixed = [21, 900, 3, 901]
        w3, _, _ = cache.get(mixed, self._tpos(mixed), self.pos)
        direct = self._direct(self._tpos(mixed))
        self.assertEqual(w3[0], w1[21])
        self.assertEqual(w3[2], w1[3])
        self.assertEqual(w3[1], direct[1])
        self.assertEqual(w3[3], direct[3])

        # A key dropped by the previous calls is recomputed, not resurrected.
        w4, _, _ = cache.get([30], self._tpos([30]), self.pos)
        self.assertEqual(w4, self._direct(self._tpos([30])))

    def test_cache_derived_products(self):
        cache = torchbp.autofocus._AntennaWeightCache(
            self.att, self.g, self.g_extent
        )
        keys = list(range(self.ntargets))
        for _ in range(2):
            w, wn, wpair = cache.get(keys, self._tpos(keys), self.pos)
            self.assertEqual(
                wn, w / torch.clamp(w.amax(1, keepdim=True), min=1e-12)
            )
            self.assertEqual(
                wpair, wn * torch.nn.functional.pad(wn[..., :-1], (1, 0))
            )

    def test_budget_zero_never_retains(self):
        budget = torchbp.autofocus._WeightCacheBudget(0)
        cache = torchbp.autofocus._AntennaWeightCache(
            self.att, self.g, self.g_extent, budget
        )
        keys = list(range(self.ntargets))
        tpos = self._tpos(keys)
        for _ in range(3):
            w, _, _ = cache.get(keys, tpos, self.pos)
            # Nothing retained, so every call recomputes exactly.
            self.assertEqual(w, self._direct(tpos))
        self.assertEqual(budget.used, 0)

    def test_budget_shared_across_caches(self):
        keys = list(range(self.ntargets))
        tpos = self._tpos(keys)
        row_bytes = self.ntargets * self.nsweeps * tpos.element_size()
        budget = torchbp.autofocus._WeightCacheBudget(row_bytes * 2)
        caches = [
            torchbp.autofocus._AntennaWeightCache(
                self.att, self.g, self.g_extent, budget
            )
            for _ in range(5)
        ]
        for _ in range(2):
            for cache in caches:
                w, _, _ = cache.get(keys, tpos, self.pos)
                self.assertEqual(w, self._direct(tpos))
            self.assertLessEqual(budget.used, budget.limit)
        # Only the caches that fit under the shared budget retain rows.
        self.assertEqual([c._held > 0 for c in caches],
                         [True, True, False, False, False])

    def test_empty_keys(self):
        cache = torchbp.autofocus._AntennaWeightCache(
            self.att, self.g, self.g_extent
        )
        w, wn, wpair = cache.get([], self._tpos([]), self.pos)
        for t in (w, wn, wpair):
            self.assertEqual(t.shape, torch.Size([0, self.nsweeps]))


class TestPhaseToPos(TestCase):
    """phase_to_pos recovers an injected x position error from pga phase."""

    fc = 6e9
    r_res = 0.3
    grid = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 64,
            "ntheta": 256}
    nsweeps = 256
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

    @staticmethod
    def _detrend_on(x, u):
        A = torch.stack([torch.ones_like(u), u], dim=1)
        sol = torch.linalg.lstsq(A, x[:, None]).solution
        return x - (A @ sol)[:, 0]

    def test_recovers_x_error_nonlinear_track(self):
        torch.manual_seed(5)
        ntargets = 12
        r = 90.0 + 20.0 * torch.rand(ntargets)
        t = -0.15 + 0.3 * torch.rand(ntargets)
        targets = torch.stack(
            [r * torch.sqrt(1 - t**2), r * t, torch.zeros_like(r)], dim=1
        )
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        # Slightly non-linear nominal track: non-uniform along-track
        # spacing plus a known x bow and z wiggle.
        f = torch.arange(self.nsweeps) / self.nsweeps
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, self.nsweeps) * (
            1 + 0.05 * torch.sin(2 * torch.pi * f)
        )
        pos[:, 0] = 0.05 * torch.sin(torch.pi * f)
        pos[:, 2] = 30.0 + 0.02 * torch.sin(2 * torch.pi * f)

        dx = 4e-3 * torch.sin(2 * torch.pi * 2 * f + 0.5)
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        data = self._make_data(targets, amps, pos_true)

        img_blur = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fc, self.r_res, pos
        )[0]
        img_s = torchbp.util.shift_spectrum(img_blur.clone())
        _, phi = torchbp.autofocus.pga(img_s)
        dx_est = torchbp.autofocus.phase_to_pos(phi, self.grid, self.fc, pos)

        # The mean and linear trend are unobservable; compare detrended
        # in the along-track coordinate.
        u = pos[:, 1]
        resid = self._detrend_on(dx - dx_est, u)
        self.assertLess(
            resid.pow(2).mean().sqrt().item(),
            0.3 * self._detrend_on(dx, u).pow(2).mean().sqrt().item(),
        )

        # The non-shifted path must give the same result.
        _, phi2 = torchbp.autofocus.pga(img_blur.clone())
        dx_est2 = torchbp.autofocus.phase_to_pos(
            phi2, self.grid, self.fc, pos, shifted=False
        )
        self.assertLess((dx_est - dx_est2).abs().max().item(), 1e-5)


class _FmcwAutofocusScene:
    """Sweep-start referenced FMCW data of a few point targets on a slightly
    non-linear track with a sinusoidal x position error. The spectral center
    ``fcenter = fstart + bw / 2`` differs from the phase reference ``fstart``
    by a factor of 1.5 (1 GHz sweep starting at 1 GHz), so a phase to
    distance conversion with the wrong frequency fails clearly.
    """

    fstart = 1.0e9
    bw = 1.0e9
    tsweep = 100e-6
    fs = 20e6
    nsweeps = 128
    grid = {"r": (90.0, 110.0), "theta": (-0.2, 0.2), "nr": 200,
            "ntheta": 128}
    dx_amp = 8e-3

    @property
    def fcenter(self):
        return self.fstart + self.bw / 2

    def _scene(self):
        n = self.nsweeps
        t = torch.arange(n) / n
        pos = torch.zeros(n, 3)
        pos[:, 1] = torch.linspace(-n / 2, n / 2, n) * (
            0.25 * C0 / (self.fstart + self.bw)) * (
            1 + 0.05 * torch.sin(2 * torch.pi * t))
        pos[:, 0] = 0.05 * torch.sin(torch.pi * t)
        dx = self.dx_amp * torch.sin(2 * torch.pi * 2 * t + 0.5)
        pos_true = pos.clone()
        pos_true[:, 0] += dx
        # Five well separated targets; more targets with overlapping range
        # profiles degrade the gpga estimate for reasons unrelated to fc.
        targets = torch.tensor(
            [[100.0, 0.0, 0.0], [105.0, 10.0, 0.0], [97.0, -5.0, 0.0],
             [102.0, -10.0, 0.0], [95.0, 5.0, 0.0]])
        wa = torch.hamming_window(n, periodic=False)
        data, data_fmod, r_res = fmcw_scene(
            targets, pos_true, self.fstart, self.bw, self.tsweep, self.fs,
            wa=wa)
        return data, data_fmod, r_res, pos, dx

    @staticmethod
    def _detrend_on(x, u):
        A = torch.stack([torch.ones_like(u), u], dim=1)
        sol = torch.linalg.lstsq(A, x[:, None]).solution
        return x - (A @ sol)[:, 0]

    def _residual(self, dx_est, dx, u):
        """Detrended rms residual relative to the detrended true error."""
        r = self._detrend_on(dx - dx_est, u).pow(2).mean().sqrt()
        return (r / self._detrend_on(dx, u).pow(2).mean().sqrt()).item()


class TestPhaseToPosFcenter(_FmcwAutofocusScene, TestCase):
    """phase_to_pos must be given the spectral center frequency, not the
    phase reference the image was formed with."""

    fstart = 2.0e9  # fcenter / fstart = 1.25

    def test_fcenter_recovers_fstart_does_not(self):
        torch.manual_seed(1)
        data, data_fmod, r_res, pos, dx = self._scene()
        img = torchbp.ops.backprojection_polar_2d(
            data, self.grid, self.fstart, r_res, pos, dealias=True,
            data_fmod=data_fmod)[0]
        img = torchbp.util.shift_spectrum(img)
        _, phi = torchbp.autofocus.pga(img, estimator="wls")
        u = pos[:, 1]
        dx_c = torchbp.autofocus.phase_to_pos(phi, self.grid, self.fcenter, pos)
        dx_s = torchbp.autofocus.phase_to_pos(phi, self.grid, self.fstart, pos)
        res_c = self._residual(dx_c, dx, u)
        res_s = self._residual(dx_s, dx, u)
        self.assertLess(res_c, 0.4, f"fcenter residual {res_c:.2f}")
        self.assertGreater(res_s, 0.5, f"fstart residual {res_s:.2f}")


class TestGpgaFcenter(_FmcwAutofocusScene, TestCase):
    """gpga and gpga_tde convert phase to distance with ``fcenter``. With it
    a pure x error is solved in one iteration and the returned phase of
    gpga is in ``fcenter`` units; with the default ``fcenter = fc`` and
    sweep-start data the single-step correction is scaled by
    ``fcenter / fc``."""

    def test_gpga_single_step(self):
        torch.manual_seed(2)
        data, data_fmod, r_res, pos, dx = self._scene()
        u = pos[:, 1]
        common = dict(max_iters=1, target_threshold_db=15, data_fmod=data_fmod)
        _, phi = torchbp.autofocus.gpga(
            None, data, pos, self.fstart, r_res, self.grid,
            fcenter=self.fcenter, **common)
        d = torchbp.util.phase_to_distance(phi, self.fcenter)
        res = self._residual(d, dx, u)
        self.assertLess(res, 0.3, f"gpga(fcenter) residual {res:.2f}")

        # Default fcenter = fc: phi is in fc units and the applied step is
        # 1.5x too large.
        _, phi_fc = torchbp.autofocus.gpga(
            None, data, pos, self.fstart, r_res, self.grid, **common)
        d_fc = torchbp.util.phase_to_distance(phi_fc, self.fstart)
        res_fc = self._residual(d_fc, dx, u)
        self.assertGreater(res_fc, 0.35, f"gpga(fc) residual {res_fc:.2f}")
        # Same phase estimate, only the conversion differs.
        d_fc_c = torchbp.util.phase_to_distance(phi_fc, self.fcenter)
        self.assertLess(self._residual(d_fc_c, dx, u), 0.3)

    def test_gpga_tde_single_step(self):
        torch.manual_seed(3)
        data, data_fmod, r_res, pos, dx = self._scene()
        u = pos[:, 1]
        common = dict(azimuth_divisions=2, range_divisions=2, max_iters=1,
                      estimate_z=False, target_threshold_db=15,
                      data_fmod=data_fmod)
        _, pos_new = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fstart, r_res, self.grid,
            fcenter=self.fcenter, **common)
        res = self._residual(pos_new[:, 0] - pos[:, 0], dx, u)
        self.assertLess(res, 0.3, f"gpga_tde(fcenter) residual {res:.2f}")
        _, pos_fc = torchbp.autofocus.gpga_tde(
            None, data, pos, self.fstart, r_res, self.grid, **common)
        res_fc = self._residual(pos_fc[:, 0] - pos[:, 0], dx, u)
        self.assertGreater(res_fc, 0.35, f"gpga_tde(fc) residual {res_fc:.2f}")


# ----------------------------------------------------------------------
# CUDA re-runs.
#
# The tests above build every tensor with the default-device factory
# functions, so pointing the default device at the GPU re-runs the whole
# file against the CUDA kernels without duplicating any scene setup.
# ----------------------------------------------------------------------


class _OnCuda:
    """Run the parent class's tests with the default device on the GPU.

    The random draws are additionally forced through the CPU generator, so
    each CUDA test sees the *same* scene as its CPU counterpart: a failure
    here is a kernel difference and not a different random scene.
    """

    _rng_fns = ("randn", "rand", "randint", "randperm")

    def setUp(self):
        self._orig_rng = {n: getattr(torch, n) for n in self._rng_fns}
        for name, fn in self._orig_rng.items():
            def wrapper(*args, _fn=fn, **kwargs):
                device = kwargs.pop("device", None)
                return _fn(*args, device="cpu", **kwargs).to(
                    torch.get_default_device() if device is None else device)
            setattr(torch, name, wrapper)
        torch.set_default_device("cuda")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch.set_default_device("cpu")
        for name, fn in self._orig_rng.items():
            setattr(torch, name, fn)


@requires_cuda
class TestInsarRmeBlocksvdCuda(_OnCuda, TestInsarRmeBlocksvd):
    pass


@requires_cuda
class TestInsarRmeMultisquintCuda(_OnCuda, TestInsarRmeMultisquint):
    pass


@requires_cuda
class TestBatchedEighCuda(_OnCuda, TestBatchedEigh):
    """The chunking only engages on CUDA. This is the path that matters."""

    def test_workspace_is_bounded(self):
        # The unchunked cuSOLVER call takes ~270 kB per 3x3 float32 matrix,
        # so this batch alone would ask for over 5 GB of workspace.
        a = self._sym(20000, 3)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        e, v = torchbp.autofocus._batched_eigh(a)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - base
        # Outputs are 60000*(3+9)*4 bytes
        self.assertLess(extra, 4 * torchbp.autofocus._EIGH_WORKSPACE_BYTES)


@requires_cuda
class TestAntennaWeightMemoryCuda(_OnCuda, TestAntennaWeightMemory):

    def test_antenna_weights_transient_bounded(self):
        # The bilinear lookup holds ~20 [ntargets, nsweeps] temporaries at
        # once; unchunked this is where a large block's weights blew up.
        n, ntgt = 20000, 600
        pos = torch.stack([
            torch.zeros(n), torch.linspace(0, 400, n), torch.full((n,), 110.0)
        ], dim=-1)
        att = torch.zeros(n, 3)
        att[:, 0] = -0.17
        k = torch.linspace(0, 1, ntgt)
        tpos = torch.stack(
            [200 + 800 * k, -100 + 600 * k, torch.zeros_like(k)], dim=-1
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        w = torchbp.autofocus._antenna_weights(
            tpos, pos, att, self.g, self.g_extent
        )
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - base
        self.assertEqual(w.shape, torch.Size([ntgt, n]))
        self.assertLess(
            extra, 2 * torchbp.autofocus._ANT_WEIGHT_CHUNK_BYTES
        )


@requires_cuda
class TestPgaCuda(_OnCuda, TestPga):
    pass


@requires_cuda
class TestPgaWindowEstimateCuda(_OnCuda, TestPgaWindowEstimate):
    pass


@requires_cuda
class TestPgaXzCuda(_OnCuda, TestPgaXz):
    pass


@requires_cuda
class TestGpgaBpPolarCuda(_OnCuda, TestGpgaBpPolar):
    pass


@requires_cuda
class TestGpgaBpPolarTdeCuda(_OnCuda, TestGpgaBpPolarTde):
    pass


@requires_cuda
class TestGpgaCartesianCuda(_OnCuda, TestGpgaCartesian):
    pass


@requires_cuda
class TestGpgaDemCuda(_OnCuda, TestGpgaDem):
    pass


@requires_cuda
class TestPhaseToPosCuda(_OnCuda, TestPhaseToPos):
    pass


@requires_cuda
class TestPhaseToPosFcenterCuda(_OnCuda, TestPhaseToPosFcenter):
    pass


@requires_cuda
class TestGpgaFcenterCuda(_OnCuda, TestGpgaFcenter):
    pass
