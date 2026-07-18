#!/usr/bin/env python
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase
import torchbp
from conftest import requires_cuda


class TestAinsworth(TestCase):
    """Ainsworth polarimetric calibration on synthetic fully-polarimetric data
    generated with the NUFFT projector.

    A patch of distributed clutter is passed through a known
    polarimetric distortion M_true, each channel is forward projected with
    projection_cart_2d_nufft and backprojected, and ainsworth must
    recover M_true.
    """

    # SAR parameters
    FC = 6e9
    BW = 200e6
    TSWEEP = 100e-6
    FS = 20e6

    def _distortion(self, device):
        """Known distortion matrix in [HH, HV, VH, VV] order (zero common-mode
        crosstalk, non-trivial channel imbalance k and alpha)."""
        import torchbp.polarimetry as pol
        u, v = 0.08 + 0.03j, -0.05 + 0.06j
        return pol.distortion_matrix(
            alpha=1.15 * np.exp(1j * 0.20),    # cross-pol channel imbalance
            k=0.85 * np.exp(-1j * 0.35),       # co-pol HH/VV imbalance (needs corner)
            u=u, v=v, w=-v, z=-u,              # zero common-mode -> fully recoverable
            pol_order=["HH", "HV", "VH", "VV"],
        ).to(device)

    def _make_scene(self, device, nx, ny):
        """Distributed clutter (reflection-symmetric) + one corner reflector,
        in [HH, HV, VH, VV] order."""
        def cn(scale):
              z = scale * (torch.randn(nx, ny) + 1j*torch.randn(nx, ny)) / np.sqrt(2)
              return z.to(device)

        shh, svv = cn(1.0), cn(1.0)        # co-pol, independent
        shv = cn(0.4)                      # cross-pol, independent of co-pol
        svh = shv.clone()                  # reciprocity
        # Trihedral corner reflector: HH=VV bright, HV=VH=0.
        cx, cy = nx // 2, ny // 2
        shh[cx, cy] = svv[cx, cy] = 200.0
        shv[cx, cy] = svh[cx, cy] = 0.0
        return torch.stack([shh, shv, svh, svv])

    def _run(self, device):
        import torchbp.polarimetry as pol
        torch.manual_seed(0)
        nsamples = int(self.FS * self.TSWEEP)
        nsweeps = 256
        wl = 3e8 / self.FC
        altitude = 100.0
        element_spacing = 0.25 * wl

        grid_proj = {"x": (60.0, 260.0), "y": (-90.0, 90.0), "nx": 256, "ny": 256}
        r0, r1, theta_limit = 90.0, 260.0, 0.35
        res = 3e8 / (2 * self.BW)
        nr = int(1.3 * (r1 - r0) / res)
        ntheta = int(1 + nsweeps * 1.3 * (element_spacing / wl) * theta_limit / 0.25)
        grid_polar = {"r": (r0, r1), "theta": (-theta_limit, theta_limit),
                      "nr": nr, "ntheta": ntheta}

        pos = torch.zeros(nsweeps, 3, dtype=torch.float32, device=device)
        pos[:, 1] = torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps) * element_spacing
        pos[:, 2] = altitude

        order = ["HH", "HV", "VH", "VV"]
        M_true = self._distortion(device)
        S = self._make_scene(device, grid_proj["nx"], grid_proj["ny"]).reshape(4, -1)
        O = (M_true @ S).reshape(4, grid_proj["nx"], grid_proj["ny"])

        # NUFFT projection + range compression + backprojection per channel.
        oversample = 2
        data_fmod = -torch.pi * (1 - (oversample - 1) / oversample)
        n = nsamples * oversample
        r_res = 3e8 / (2 * self.BW * oversample)
        win = torch.tensor(np.hamming(nsamples)[None, :], dtype=torch.float32, device=device)
        fmod_f = torch.exp(1j * data_fmod * torch.arange(n, device=device))[None, :]

        def make_image(scene):
            data = torchbp.ops.projection_cart_2d_nufft(
                scene, pos, grid_proj, self.FC, self.FS, self.BW / self.TSWEEP,
                nsamples, use_rvp=False, normalization="gamma")[0]
            data = torch.fft.ifft(data * win, dim=-1, n=n) * fmod_f
            return torchbp.ops.backprojection_polar_2d(
                data, grid_polar, self.FC, r_res, pos, dealias=False,
                data_fmod=data_fmod)[0]

        sar = torch.stack([make_image(O[c]) for c in range(4)])

        # Locate the corner reflector (brightest pixel) and measure its HH/VV
        # ratio to resolve k; mask it out of the crosstalk statistics.
        mag = sar[0].abs()
        pk = torch.argmax(mag)
        pr, pt = int(pk // mag.shape[1]), int(pk % mag.shape[1])
        self.assertGreater((mag[pr, pt] / mag.median()).item(), 10.0,
                           "corner reflector not dominant in focused image")
        corner_hh_vv = (sar[0, pr, pt] / sar[3, pr, pt]).item()
        weight = torch.ones(nr, ntheta, device=device)
        ww = 6
        weight[max(0, pr - ww):pr + ww + 1, max(0, pt - ww):pt + ww + 1] = 0.0

        Minv = pol.ainsworth(sar, weight=weight, pol_order=order,
                             corner_hh_vv=corner_hh_vv)

        # Calibration must invert M_true up to an overall complex scalar.
        R = Minv @ M_true
        R = R / R[0, 0]
        eye = torch.eye(4, dtype=R.dtype, device=device)
        offdiag = (R * (1 - eye)).abs().max().item()
        diagerr = (torch.diagonal(R) - 1).abs().max().item()
        # Crosstalk and cross-pol imbalance are recovered to numerical precision;
        # the small diagonal error is the corner-resolved HH/VV imbalance k.
        self.assertLess(offdiag, 0.01, f"residual crosstalk {offdiag:.3f} too large")
        self.assertLess(diagerr, 0.025, f"residual imbalance {diagerr:.3f} too large")

    def test_ainsworth_cpu(self):
        self._run("cpu")

    @requires_cuda
    def test_ainsworth_cuda(self):
        self._run("cuda")


class TestPolAntennaRotation(TestCase):
    """pol_antenna_rotation must equal the scattering-matrix rotation
    S' = R(theta) @ S @ R(theta).T per pixel."""

    def test_matches_matrix_rotation(self):
        import math
        from torchbp.polarimetry import pol_antenna_rotation
        torch.manual_seed(0)
        order = ["HH", "HV", "VH", "VV"]
        theta = 0.3
        npx = 8
        # Non-reciprocal random scattering matrices expose asymmetric errors.
        ch = [torch.randn(npx, dtype=torch.complex64) for _ in range(4)]
        img = torch.stack(ch).reshape(4, npx, 1)
        out = pol_antenna_rotation(img, theta, pol_order=order).reshape(4, npx)

        c, s = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
        hh, hv, vh, vv = ch
        for i in range(npx):
            S = torch.tensor([[hh[i], hv[i]], [vh[i], vv[i]]], dtype=torch.complex64)
            Sp = R @ S @ R.T
            ref = torch.stack([Sp[0, 0], Sp[0, 1], Sp[1, 0], Sp[1, 1]])
            self.assertEqual(out[:, i], ref, rtol=1e-5, atol=1e-5)

    def test_trace_preserved_and_round_trip(self):
        from torchbp.polarimetry import pol_antenna_rotation
        torch.manual_seed(1)
        order = ["HH", "HV", "VH", "VV"]
        img = torch.randn(4, 6, 1, dtype=torch.complex64)
        rot = pol_antenna_rotation(img, 0.4, pol_order=order)
        # A rotation preserves the scattering-matrix trace HH + VV.
        self.assertEqual(rot[0] + rot[3], img[0] + img[3], rtol=1e-5, atol=1e-5)
        # Rotating by +theta then -theta returns the original image.
        back = pol_antenna_rotation(rot, -0.4, pol_order=order)
        self.assertEqual(back, img, rtol=1e-5, atol=1e-5)


class TestDistortionMatrix(TestCase):
    """distortion_matrix builds the forward model that ainsworth inverts."""

    def test_defaults_identity(self):
        from torchbp.polarimetry import distortion_matrix
        self.assertEqual(distortion_matrix(), torch.eye(4, dtype=torch.complex64),
                         rtol=1e-6, atol=1e-6)

    def test_inverted_by_ainsworth(self):
        from torchbp.polarimetry import distortion_matrix, ainsworth
        order = ["HH", "HV", "VH", "VV"]
        u, v = 0.08 + 0.03j, -0.05 + 0.06j
        M = distortion_matrix(alpha=1.15 * np.exp(1j * 0.20), k=0.85 * np.exp(-1j * 0.35),
                              u=u, v=v, w=-v, z=-u, pol_order=order)   # zero common-mode
        torch.manual_seed(0)
        N = 300
        g = lambda: (torch.randn(N * N) + 1j * torch.randn(N * N)) / np.sqrt(2)
        shv = 0.4 * g()
        S = torch.stack([g(), shv, shv.clone(), g()])
        O = (M @ S).reshape(4, N, N)
        Minv = ainsworth(O, pol_order=order, corner_hh_vv=(M[0, 0] / M[3, 3]).item())
        R = Minv @ M
        R = R / R[0, 0]
        off = (R * (1 - torch.eye(4, dtype=R.dtype))).abs().max().item()
        self.assertLess(off, 0.01)


class TestOrientationAngle(TestCase):
    """orientation_angle / orientation_angle_image must recover the rotation
    angle applied to a reflection-symmetric scene by pol_antenna_rotation."""

    order = ["HH", "HV", "VH", "VV"]

    def _scene(self, n, seed=0):
        import math
        torch.manual_seed(seed)
        g = lambda: (torch.randn(n, n) + 1j * torch.randn(n, n)) / math.sqrt(2)
        shv = 0.4 * g()
        # co-pol independent, cross-pol independent of co-pol, S_vh = S_hv.
        return torch.stack([g(), shv, shv.clone(), g()])

    def test_global_recovers_rotation(self):
        import math
        from torchbp.polarimetry import pol_antenna_rotation, orientation_angle
        S = self._scene(256)
        # An unrotated reflection-symmetric scene has ~zero orientation.
        self.assertLess(
            abs(math.degrees(orientation_angle(S, pol_order=self.order).item())), 1.0
        )
        for deg_true in [12.0, -20.0, 35.0]:
            Sr = pol_antenna_rotation(S, math.radians(deg_true), pol_order=self.order)
            est = math.degrees(orientation_angle(Sr, pol_order=self.order).item())
            self.assertLess(abs(est - deg_true), 1.5)

    def test_image_map(self):
        import math
        from torchbp.polarimetry import pol_antenna_rotation, orientation_angle_image
        S = pol_antenna_rotation(self._scene(96), math.radians(15.0), pol_order=self.order)
        m = orientation_angle_image(S, window=(11, 11), pol_order=self.order)
        self.assertEqual(m.shape, (96, 96))
        # The averaged map should be centred on the applied 15 degrees.
        self.assertLess(abs(math.degrees(m.mean().item()) - 15.0), 3.0)


