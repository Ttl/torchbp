#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp


class TestResample2DLanczos(TestCase):
    """Test generic 2D Lanczos resampling."""

    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=requires_grad, dtype=dtype)

        args = {
            "img": make_tensor((2, 16, 16), dtype=dtype),
            "shift_r": torch.zeros(16, 16, device=device, dtype=torch.float32),
            "shift_az": torch.zeros(16, 16, device=device, dtype=torch.float32),
            "order": 6,
        }
        return [args]

    def _test_zero_shift_identity(self, device, dtype):
        """Zero shift should return the input unchanged (within kernel boundary effects)."""
        Nr, Naz = 32, 32
        img = torch.randn(1, Nr, Naz, device=device, dtype=dtype)
        shift_r = torch.zeros(Nr, Naz, device=device, dtype=torch.float32)
        shift_az = torch.zeros(Nr, Naz, device=device, dtype=torch.float32)
        out = torchbp.ops.resample_2d_lanczos(img, shift_r, shift_az, order=6)
        # Interior pixels should match (boundary pixels affected by kernel support)
        margin = 4
        err = torch.max(torch.abs(out[0, margin:-margin, margin:-margin] - img[0, margin:-margin, margin:-margin]))
        self.assertLess(err.item(), 1e-5)

    def _test_smooth_function(self, device, dtype):
        """Interpolate a smooth analytical function and verify error is small.

        Uses f(r, az) = exp(j * (0.3*r + 0.2*az)) for complex,
        or f(r, az) = cos(0.3*r) * cos(0.2*az) for real.
        These are smooth, band-limited functions that Lanczos should interpolate well.
        """
        Nr, Naz = 64, 64
        r = torch.arange(Nr, device=device, dtype=torch.float32)
        az = torch.arange(Naz, device=device, dtype=torch.float32)

        # Shift: constant sub-pixel shift
        shift_val_r = 0.37
        shift_val_az = -0.23
        shift_r = torch.full((Nr, Naz), shift_val_r, device=device, dtype=torch.float32)
        shift_az = torch.full((Nr, Naz), shift_val_az, device=device, dtype=torch.float32)

        if dtype == torch.complex64:
            # Complex exponential: exact values at shifted coordinates
            freq_r, freq_az = 0.1, 0.07
            img = torch.exp(1j * (freq_r * r[:, None] + freq_az * az[None, :]))
            img = img.unsqueeze(0).to(dtype=dtype, device=device)
            expected = torch.exp(1j * (freq_r * (r[:, None] + shift_val_r) + freq_az * (az[None, :] + shift_val_az)))
            expected = expected.to(dtype=dtype, device=device)
        else:
            freq_r, freq_az = 0.1, 0.07
            img = torch.cos(freq_r * r[:, None]) * torch.cos(freq_az * az[None, :])
            img = img.unsqueeze(0).to(dtype=dtype, device=device)
            expected = (torch.cos(freq_r * (r[:, None] + shift_val_r))
                        * torch.cos(freq_az * (az[None, :] + shift_val_az)))
            expected = expected.to(dtype=dtype, device=device)

        out = torchbp.ops.resample_2d_lanczos(img, shift_r, shift_az, order=6)

        # Check interior (avoid boundary effects from kernel support)
        margin = 6
        err = torch.abs(out[0, margin:-margin, margin:-margin] - expected[margin:-margin, margin:-margin])
        max_err = err.max().item()
        rms_err = torch.sqrt(torch.mean(err ** 2)).item()
        # Unnormalized Lanczos-6 has ~0.01 max error from kernel weight sum != 1
        self.assertLess(max_err, 0.02, f"Max error {max_err:.6f} too large for smooth function")
        self.assertLess(rms_err, 0.01, f"RMS error {rms_err:.6f} too large for smooth function")

    def _test_varying_shift(self, device, dtype):
        """Test with a spatially varying shift field."""
        Nr, Naz = 64, 64
        r = torch.arange(Nr, device=device, dtype=torch.float32)
        az = torch.arange(Naz, device=device, dtype=torch.float32)

        freq_r, freq_az = 0.1, 0.07
        if dtype == torch.complex64:
            img = torch.exp(1j * (freq_r * r[:, None] + freq_az * az[None, :]))
            img = img.unsqueeze(0).to(dtype=dtype, device=device)
        else:
            img = torch.cos(freq_r * r[:, None]) * torch.cos(freq_az * az[None, :])
            img = img.unsqueeze(0).to(dtype=dtype, device=device)

        # Varying shift: 0 to 2 pixels across the image
        shift_r = 2.0 * r[:, None].expand(Nr, Naz) / Nr
        shift_az = torch.zeros(Nr, Naz, device=device, dtype=torch.float32)
        shift_r = shift_r.to(device=device)

        # Compute expected by evaluating the analytical function at shifted coords
        r_shifted = r[:, None] + shift_r
        if dtype == torch.complex64:
            expected = torch.exp(1j * (freq_r * r_shifted + freq_az * az[None, :]))
            expected = expected.to(dtype=dtype, device=device)
        else:
            expected = torch.cos(freq_r * r_shifted) * torch.cos(freq_az * az[None, :])
            expected = expected.to(dtype=dtype, device=device)

        out = torchbp.ops.resample_2d_lanczos(img, shift_r, shift_az, order=6)

        margin = 6
        err = torch.abs(out[0, margin:-margin, margin:-margin] - expected[margin:-margin, margin:-margin])
        max_err = err.max().item()
        self.assertLess(max_err, 0.02, f"Max error {max_err:.6f} too large for varying shift")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_shift_complex(self):
        self._test_zero_shift_identity("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_shift_float(self):
        self._test_zero_shift_identity("cuda", torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_complex(self):
        self._test_smooth_function("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_float(self):
        self._test_smooth_function("cuda", torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_varying_shift_complex(self):
        self._test_varying_shift("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_varying_shift_float(self):
        self._test_varying_shift("cuda", torch.float32)

    def test_zero_shift_complex_cpu(self):
        self._test_zero_shift_identity("cpu", torch.complex64)

    def test_zero_shift_float_cpu(self):
        self._test_zero_shift_identity("cpu", torch.float32)

    def test_smooth_complex_cpu(self):
        self._test_smooth_function("cpu", torch.complex64)

    def test_smooth_float_cpu(self):
        self._test_smooth_function("cpu", torch.float32)

    def test_varying_shift_complex_cpu(self):
        self._test_varying_shift("cpu", torch.complex64)

    def test_varying_shift_float_cpu(self):
        self._test_varying_shift("cpu", torch.float32)

    def _test_cpu_cuda(self, dtype):
        Nr, Naz = 32, 32
        img = torch.randn(2, Nr, Naz, dtype=dtype)
        shift_r = torch.full((Nr, Naz), 0.37, dtype=torch.float32)
        shift_az = torch.full((Nr, Naz), -0.23, dtype=torch.float32)
        out_cpu = torchbp.ops.resample_2d_lanczos(img, shift_r, shift_az, order=6)
        out_gpu = torchbp.ops.resample_2d_lanczos(
            img.cuda(), shift_r.cuda(), shift_az.cuda(), order=6).cpu()
        torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_complex(self):
        self._test_cpu_cuda(torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_float(self):
        self._test_cpu_cuda(torch.float32)

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = (
                args["img"],
                args["shift_r"],
                args["shift_az"],
                args["img"].shape[0],  # nbatch
                args["img"].shape[1],  # Nr
                args["img"].shape[2],  # Naz
                args["order"],
            )
            opcheck(
                torch.ops.torchbp.resample_2d_lanczos,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestResample2DKnab(TestCase):
    """Test generic 2D Knab resampling."""

    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=requires_grad, dtype=dtype)

        args = {
            "img": make_tensor((2, 16, 16), dtype=dtype),
            "shift_r": torch.zeros(16, 16, device=device, dtype=torch.float32),
            "shift_az": torch.zeros(16, 16, device=device, dtype=torch.float32),
            "order": 6,
            "oversample": 1.5,
        }
        return [args]

    def _test_zero_shift_identity(self, device, dtype):
        Nr, Naz = 32, 32
        img = torch.randn(1, Nr, Naz, device=device, dtype=dtype)
        shift_r = torch.zeros(Nr, Naz, device=device, dtype=torch.float32)
        shift_az = torch.zeros(Nr, Naz, device=device, dtype=torch.float32)
        out = torchbp.ops.resample_2d_knab(img, shift_r, shift_az, order=6, oversample=1.5)
        margin = 4
        err = torch.max(torch.abs(out[0, margin:-margin, margin:-margin] - img[0, margin:-margin, margin:-margin]))
        self.assertLess(err.item(), 1e-5)

    def _test_smooth_function(self, device, dtype):
        Nr, Naz = 64, 64
        r = torch.arange(Nr, device=device, dtype=torch.float32)
        az = torch.arange(Naz, device=device, dtype=torch.float32)

        shift_val_r = 0.37
        shift_val_az = -0.23
        shift_r = torch.full((Nr, Naz), shift_val_r, device=device, dtype=torch.float32)
        shift_az = torch.full((Nr, Naz), shift_val_az, device=device, dtype=torch.float32)

        freq_r, freq_az = 0.1, 0.07
        if dtype == torch.complex64:
            img = torch.exp(1j * (freq_r * r[:, None] + freq_az * az[None, :])).unsqueeze(0).to(dtype=dtype)
            expected = torch.exp(1j * (freq_r * (r[:, None] + shift_val_r) + freq_az * (az[None, :] + shift_val_az))).to(dtype=dtype)
        else:
            img = (torch.cos(freq_r * r[:, None]) * torch.cos(freq_az * az[None, :])).unsqueeze(0).to(dtype=dtype)
            expected = (torch.cos(freq_r * (r[:, None] + shift_val_r)) * torch.cos(freq_az * (az[None, :] + shift_val_az))).to(dtype=dtype)

        out = torchbp.ops.resample_2d_knab(img, shift_r, shift_az, order=6, oversample=1.5)

        margin = 6
        err = torch.abs(out[0, margin:-margin, margin:-margin] - expected[margin:-margin, margin:-margin])
        max_err = err.max().item()
        rms_err = torch.sqrt(torch.mean(err ** 2)).item()
        self.assertLess(max_err, 0.02, f"Max error {max_err:.6f} too large for smooth function")
        self.assertLess(rms_err, 0.01, f"RMS error {rms_err:.6f} too large for smooth function")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_shift_complex(self):
        self._test_zero_shift_identity("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_shift_float(self):
        self._test_zero_shift_identity("cuda", torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_complex(self):
        self._test_smooth_function("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_float(self):
        self._test_smooth_function("cuda", torch.float32)

    def test_zero_shift_complex_cpu(self):
        self._test_zero_shift_identity("cpu", torch.complex64)

    def test_zero_shift_float_cpu(self):
        self._test_zero_shift_identity("cpu", torch.float32)

    def test_smooth_complex_cpu(self):
        self._test_smooth_function("cpu", torch.complex64)

    def test_smooth_float_cpu(self):
        self._test_smooth_function("cpu", torch.float32)

    def _test_cpu_cuda(self, dtype):
        Nr, Naz = 32, 32
        img = torch.randn(2, Nr, Naz, dtype=dtype)
        shift_r = torch.full((Nr, Naz), 0.37, dtype=torch.float32)
        shift_az = torch.full((Nr, Naz), -0.23, dtype=torch.float32)
        out_cpu = torchbp.ops.resample_2d_knab(
            img, shift_r, shift_az, order=6, oversample=1.5)
        out_gpu = torchbp.ops.resample_2d_knab(
            img.cuda(), shift_r.cuda(), shift_az.cuda(), order=6, oversample=1.5).cpu()
        torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_complex(self):
        self._test_cpu_cuda(torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_float(self):
        self._test_cpu_cuda(torch.float32)

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = (
                args["img"],
                args["shift_r"],
                args["shift_az"],
                args["img"].shape[0],
                args["img"].shape[1],
                args["img"].shape[2],
                args["order"],
                args["oversample"],
            )
            opcheck(
                torch.ops.torchbp.resample_2d_knab,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestResample1DLanczos(TestCase):
    """Test generic 1D signal rate change with Lanczos interpolation."""

    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=requires_grad, dtype=dtype)

        args = {
            "img": make_tensor((2, 32), dtype=dtype),
            "N": 32,
            "M": 64,
            "order": 6,
        }
        return [args]

    def _test_identity(self, device, dtype):
        """num == N must reproduce the input exactly (within kernel boundary effects)."""
        N = 64
        x = torch.randn(3, N, device=device, dtype=dtype)
        out = torchbp.ops.resample_1d_lanczos(x, N, order=6)
        self.assertEqual(out.shape, x.shape)
        margin = 4
        err = torch.max(torch.abs(out[:, margin:-margin] - x[:, margin:-margin]))
        self.assertLess(err.item(), 1e-5)

    def _test_smooth_upsample(self, device, dtype):
        """Upsample a band-limited signal and compare against the analytical values."""
        N = 128
        M = 256
        t = torch.arange(N, device=device, dtype=torch.float32)
        freq = 0.1
        if dtype == torch.complex64:
            x = torch.exp(1j * freq * t).to(dtype=dtype)
        else:
            x = torch.cos(freq * t).to(dtype=dtype)

        out = torchbp.ops.resample_1d_lanczos(x, M, order=8)
        self.assertEqual(out.shape[0], M)

        # Output sample k maps to input position k * N / M.
        tk = torch.arange(M, device=device, dtype=torch.float32) * (N / M)
        if dtype == torch.complex64:
            expected = torch.exp(1j * freq * tk).to(dtype=dtype)
        else:
            expected = torch.cos(freq * tk).to(dtype=dtype)

        margin = 8
        err = torch.abs(out[margin:-margin] - expected[margin:-margin])
        max_err = err.max().item()
        self.assertLess(max_err, 0.02, f"Max error {max_err:.6f} too large for upsampling")

    def _test_axis(self, device, dtype):
        """Resampling along an arbitrary axis matches moving that axis to the end."""
        x = torch.randn(4, 40, 5, device=device, dtype=dtype)
        for axis in [0, 1, 2, -1, -2]:
            num = 2 * x.shape[axis]
            out = torchbp.ops.resample_1d_lanczos(x, num, axis=axis, order=6)
            exp_shape = list(x.shape)
            exp_shape[axis] = num
            self.assertEqual(list(out.shape), exp_shape)
            ref = torchbp.ops.resample_1d_lanczos(
                x.movedim(axis, -1), num, axis=-1, order=6).movedim(-1, axis)
            torch.testing.assert_close(out, ref)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_identity_complex(self):
        self._test_identity("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_identity_float(self):
        self._test_identity("cuda", torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_complex(self):
        self._test_smooth_upsample("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_float(self):
        self._test_smooth_upsample("cuda", torch.float32)

    def test_identity_complex_cpu(self):
        self._test_identity("cpu", torch.complex64)

    def test_identity_float_cpu(self):
        self._test_identity("cpu", torch.float32)

    def test_smooth_complex_cpu(self):
        self._test_smooth_upsample("cpu", torch.complex64)

    def test_smooth_float_cpu(self):
        self._test_smooth_upsample("cpu", torch.float32)

    def test_axis_complex_cpu(self):
        self._test_axis("cpu", torch.complex64)

    def test_axis_float_cpu(self):
        self._test_axis("cpu", torch.float32)

    def _test_cpu_cuda(self, dtype):
        x = torch.randn(3, 33, 5, dtype=dtype)
        for num in [33, 66, 116, 16]:
            out_cpu = torchbp.ops.resample_1d_lanczos(x, num, axis=1, order=6)
            out_gpu = torchbp.ops.resample_1d_lanczos(
                x.cuda(), num, axis=1, order=6).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_complex(self):
        self._test_cpu_cuda(torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_float(self):
        self._test_cpu_cuda(torch.float32)

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = (
                args["img"],
                args["img"].shape[0],  # nbatch
                args["N"],
                args["M"],
                args["order"],
            )
            opcheck(
                torch.ops.torchbp.resample_1d_lanczos,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestResample1DKnab(TestCase):
    """Test generic 1D signal rate change with Knab interpolation."""

    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=requires_grad, dtype=dtype)

        args = {
            "img": make_tensor((2, 32), dtype=dtype),
            "N": 32,
            "M": 64,
            "order": 6,
            "oversample": 1.5,
        }
        return [args]

    def _test_identity(self, device, dtype):
        N = 64
        x = torch.randn(3, N, device=device, dtype=dtype)
        out = torchbp.ops.resample_1d_knab(x, N, order=6, oversample=1.5)
        self.assertEqual(out.shape, x.shape)
        margin = 4
        err = torch.max(torch.abs(out[:, margin:-margin] - x[:, margin:-margin]))
        self.assertLess(err.item(), 1e-5)

    def _test_smooth_upsample(self, device, dtype):
        N = 128
        M = 256
        t = torch.arange(N, device=device, dtype=torch.float32)
        freq = 0.1
        if dtype == torch.complex64:
            x = torch.exp(1j * freq * t).to(dtype=dtype)
        else:
            x = torch.cos(freq * t).to(dtype=dtype)

        out = torchbp.ops.resample_1d_knab(x, M, order=8, oversample=2.0)
        self.assertEqual(out.shape[0], M)

        tk = torch.arange(M, device=device, dtype=torch.float32) * (N / M)
        if dtype == torch.complex64:
            expected = torch.exp(1j * freq * tk).to(dtype=dtype)
        else:
            expected = torch.cos(freq * tk).to(dtype=dtype)

        margin = 8
        err = torch.abs(out[margin:-margin] - expected[margin:-margin])
        max_err = err.max().item()
        self.assertLess(max_err, 0.02, f"Max error {max_err:.6f} too large for upsampling")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_identity_complex(self):
        self._test_identity("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_identity_float(self):
        self._test_identity("cuda", torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_complex(self):
        self._test_smooth_upsample("cuda", torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_smooth_float(self):
        self._test_smooth_upsample("cuda", torch.float32)

    def test_identity_complex_cpu(self):
        self._test_identity("cpu", torch.complex64)

    def test_identity_float_cpu(self):
        self._test_identity("cpu", torch.float32)

    def test_smooth_complex_cpu(self):
        self._test_smooth_upsample("cpu", torch.complex64)

    def test_smooth_float_cpu(self):
        self._test_smooth_upsample("cpu", torch.float32)

    def _test_cpu_cuda(self, dtype):
        x = torch.randn(3, 33, 5, dtype=dtype)
        for num in [33, 66, 116, 16]:
            out_cpu = torchbp.ops.resample_1d_knab(x, num, axis=1, order=6, oversample=1.5)
            out_gpu = torchbp.ops.resample_1d_knab(
                x.cuda(), num, axis=1, order=6, oversample=1.5).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_complex(self):
        self._test_cpu_cuda(torch.complex64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_float(self):
        self._test_cpu_cuda(torch.float32)

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = (
                args["img"],
                args["img"].shape[0],
                args["N"],
                args["M"],
                args["order"],
                args["oversample"],
            )
            opcheck(
                torch.ops.torchbp.resample_1d_knab,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


