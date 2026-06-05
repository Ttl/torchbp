#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from random import uniform


class TestCoherence2D(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "img0": make_tensor((2, 3, 3), dtype=dtype),
            "img1": make_tensor((2, 3, 3), dtype=dtype),
            "Navg": (2, 3),
        }
        return [args]

    def _test_gradients(self, device, dtype=torch.complex64):
        samples = self.sample_inputs(device, requires_grad=True, dtype=dtype)
        eps = 1e-4
        rtol = 0.05
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.coherence_2d,
                list(args.values()),
                eps=eps,
                rtol=rtol,
            )

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def test_cpu_self_coherence(self):
        """Coherence of an image with itself is 1 everywhere."""
        img = torch.randn(2, 16, 16, dtype=torch.complex64)
        c = torchbp.ops.coherence_2d(img, img, (3, 3))
        torch.testing.assert_close(c, torch.ones_like(c))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for args in self.sample_inputs("cpu"):
            img0, img1 = args["img0"], args["img1"]
            out_cpu = torchbp.ops.coherence_2d(img0, img1, args["Navg"])
            out_gpu = torchbp.ops.coherence_2d(
                img0.cuda(), img1.cuda(), args["Navg"]).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_grad(self):
        """Backward must agree between CPU and CUDA."""
        for args in self.sample_inputs("cpu", requires_grad=False):
            i0 = args["img0"]
            i1 = args["img1"]
            grads = {}
            for dev in ("cpu", "cuda"):
                a = i0.detach().to(dev).requires_grad_(True)
                b = i1.detach().to(dev).requires_grad_(True)
                out = torchbp.ops.coherence_2d(a, b, args["Navg"])
                out.abs().sum().backward()
                grads[dev] = (a.grad.cpu(), b.grad.cpu())
            torch.testing.assert_close(grads["cpu"][0], grads["cuda"][0], rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(grads["cpu"][1], grads["cuda"][1], rtol=1e-3, atol=1e-4)

    def _opcheck(self, device):
        from torchbp.ops.coherence import _prepare_coherence_2d_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_coherence_2d_args(**args)
            opcheck(
                torch.ops.torchbp.coherence_2d,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestEntropy(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        def make_nondiff_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        args = {"img": make_tensor((3, 3), dtype=dtype)}
        return [args]

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_ref(self):
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

            res_gpu = torchbp.ops.entropy(sample["img"])
            res_gpu.backward()
            grads_gpu = [
                sample[k].cpu()
                for k in sample.keys()
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad
            ]

            res_cpu = torchbp.util.entropy(sample_cpu["img"])
            res_cpu.backward()
            grads_cpu = [
                sample_cpu[k]
                for k in sample_cpu.keys()
                if isinstance(sample_cpu[k], torch.Tensor)
                and sample_cpu[k].requires_grad
            ]
            torch.testing.assert_close(grads_cpu, grads_gpu)
            torch.testing.assert_close(res_gpu.cpu(), res_cpu)

    def _opcheck(self, device):
        # Test the underlying C++ operators
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            img = args["img"]
            nbatch = 1 if img.dim() == 2 else img.shape[0]

            # Test abs_sum operator
            opcheck(
                torch.ops.torchbp.abs_sum,
                (img, nbatch),
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

            # Test entropy operator (need norm from abs_sum)
            norm = torch.ops.torchbp.abs_sum.default(img, nbatch)
            opcheck(
                torch.ops.torchbp.entropy,
                (img, norm, nbatch),
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestPolarInterpLinear(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.float32):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        def make_nondiff_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        nbatch = 2
        grid_polar = {"r": (10, 20), "theta": (-1, 1), "nr": 2, "ntheta": 2}
        grid_polar_new = {"r": (12, 18), "theta": (-0.8, 0.8), "nr": 3, "ntheta": 3}
        dorigin = 0.1 * make_tensor((nbatch, 3), dtype=dtype)
        args = {
            "img": make_tensor(
                (nbatch, grid_polar["nr"], grid_polar["ntheta"]), dtype=complex_dtype
            ),
            "dorigin": dorigin,
            "grid_polar": grid_polar,
            "fc": 6e9,
            "rotation": 0.3,
            "grid_polar_new": grid_polar_new,
            "z0": 2,
            "alias_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

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

            res_gpu = torchbp.ops.polar_interp_linear(**sample)
            loss_gpu = torch.mean(torch.abs(res_gpu))
            loss_gpu.backward()
            grads_gpu = [
                sample[k].cpu()
                for k in sample.keys()
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad
            ]

            res_cpu = torchbp.ops.polar_interp_linear(**sample_cpu)
            loss_cpu = torch.mean(torch.abs(res_cpu))
            loss_cpu.backward()
            grads_cpu = [
                sample_cpu[k]
                for k in sample_cpu.keys()
                if isinstance(sample_cpu[k], torch.Tensor)
                and sample_cpu[k].requires_grad
            ]
            torch.testing.assert_close(grads_cpu, grads_gpu, atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        samples = self.sample_inputs("cuda")
        for sample in samples:
            res_gpu = torchbp.ops.polar_interp_linear(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.polar_interp_linear(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu, rtol=5e-4, atol=5e-4)

    def _test_gradients(self, device, dtype=torch.float32):
        samples = self.sample_inputs(device, requires_grad=True, dtype=dtype)
        eps = 1e-3 if dtype == torch.float32 else 1e-4
        rtol = 0.15 if dtype == torch.float32 else 0.05
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.polar_interp_linear,
                list(args.values()),
                eps=eps,  # This test is very sensitive to eps
                rtol=rtol,  # Also to rtol
            )

    def test_gradients_cpu(self):
        self._test_gradients("cpu")
        self._test_gradients("cpu", dtype=torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        from torchbp.ops.polar_interp import _prepare_polar_interp_linear_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_polar_interp_linear_args(**args)
            opcheck(
                torch.ops.torchbp.polar_interp_linear,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestPolarToCartLinear(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.float32):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        def make_nondiff_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, requires_grad=False, dtype=dtype)

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        nbatch = 2
        grid_polar = {"r": (10, 20), "theta": (-1, 1), "nr": 2, "ntheta": 2}
        grid_cart = {"x": (12, 18), "y": (-5, 5), "nx": 3, "ny": 3}
        origin = 0.1 * make_tensor((nbatch, 3), dtype=dtype)
        origin[:, 2] += 4  # Offset height
        args = {
            "img": make_tensor(
                (nbatch, grid_polar["nr"], grid_polar["ntheta"]), dtype=complex_dtype
            ),
            "origin": origin,
            "grid_polar": grid_polar,
            "grid_cart": grid_cart,
            "fc": 6e9,
            "rotation": 0.1,
            "alias_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip("polar_to_cart_linear_grad has no CPU implementation")
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

            res_gpu = torchbp.ops.polar_to_cart_linear(**sample)
            loss_gpu = torch.mean(torch.abs(res_gpu))
            loss_gpu.backward()
            grads_gpu = [
                sample[k].cpu()
                for k in sample.keys()
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad
            ]

            res_cpu = torchbp.ops.polar_to_cart_linear(**sample_cpu)
            loss_cpu = torch.mean(torch.abs(res_cpu))
            loss_cpu.backward()
            grads_cpu = [
                sample_cpu[k]
                for k in sample_cpu.keys()
                if isinstance(sample_cpu[k], torch.Tensor)
                and sample_cpu[k].requires_grad
            ]
            torch.testing.assert_close(grads_cpu, grads_gpu, atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        samples = self.sample_inputs("cuda")
        for sample in samples:
            res_gpu = torchbp.ops.polar_to_cart_linear(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.polar_to_cart_linear(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu, rtol=5e-4, atol=5e-4)

    def _test_gradients(self, device, dtype=torch.float32):
        samples = self.sample_inputs(device, requires_grad=True, dtype=dtype)
        eps = 5e-4 if dtype == torch.float32 else 1e-4
        rtol = 0.15 if dtype == torch.float32 else 0.05
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.polar_to_cart_linear,
                list(args.values()),
                eps=eps,  # This test is very sensitive to eps
                rtol=rtol,  # Also to rtol
            )

    @unittest.skip("polar_to_cart_linear_grad has no CPU implementation")
    def test_gradients_cpu(self):
        self._test_gradients("cpu")
        self._test_gradients("cpu", dtype=torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        from torchbp.ops.polar_interp import _prepare_polar_to_cart_linear_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_polar_to_cart_linear_args(**args)
            opcheck(
                torch.ops.torchbp.polar_to_cart_linear,
                cpp_args,
                test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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


class TestCfar2D(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.float32):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x.abs()

        args = {
            "img": make_tensor((2, 10, 10), dtype=dtype),
            "Navg": (3, 3),
            "Nguard": (1, 1),
            "threshold": 2.0,
            "peaks_only": False,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            img = args["img"]
            nbatch = 1 if img.dim() == 2 else img.shape[0]
            N0 = img.shape[-2]
            N1 = img.shape[-1]

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.cfar_2d,
                (img, nbatch, N0, N1, args["Navg"][0], args["Navg"][1],
                 args["Nguard"][0], args["Nguard"][1], args["threshold"], args["peaks_only"]),
                test_utils=["test_schema"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestDivMul2DInterpLinear(TestCase):
    def sample_inputs_div(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "img1": make_tensor((2, 5, 5), dtype=dtype),
            "img2": make_tensor((2, 3, 3), dtype=dtype),
        }
        return [args]

    def sample_inputs_mul(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "img1": make_tensor((2, 5, 5), dtype=dtype),
            "img2": make_tensor((2, 3, 3), dtype=dtype),
        }
        return [args]

    def _opcheck_div(self, device):
        samples = self.sample_inputs_div(device, requires_grad=False)
        for args in samples:
            img1 = args["img1"]
            img2 = args["img2"]
            nbatch = 1 if img1.dim() == 2 else img1.shape[0]
            na0, na1 = img1.shape[-2], img1.shape[-1]
            nb0, nb1 = img2.shape[-2], img2.shape[-1]

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.div_2d_interp_linear,
                (img1, img2, nbatch, na0, na1, nb0, nb1),
                test_utils=["test_schema"]
            )

    def _opcheck_mul(self, device):
        samples = self.sample_inputs_mul(device, requires_grad=False)
        for args in samples:
            img1 = args["img1"]
            img2 = args["img2"]
            nbatch = 1 if img1.dim() == 2 else img1.shape[0]
            na0, na1 = img1.shape[-2], img1.shape[-1]
            nb0, nb1 = img2.shape[-2], img2.shape[-1]

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.mul_2d_interp_linear,
                (img1, img2, nbatch, na0, na1, nb0, nb1),
                test_utils=["test_schema"]
            )

    def test_opcheck_div_cpu(self):
        self._opcheck_div("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_div_cuda(self):
        self._opcheck_div("cuda")

    def test_opcheck_mul_cpu(self):
        self._opcheck_mul("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_mul_cuda(self):
        self._opcheck_mul("cuda")

    def test_cpu_matching_shape(self):
        """With matching shapes the op reduces to plain elementwise mul/div.

        The interpolation maps output index i to b[i*(N-1)/(N-1)] = b[i], so
        interior pixels match exactly. The last row/col differ because the CUDA
        kernel clamps the source index to N-2 (replicated on the CPU side), so
        compare only the interior.
        """
        for b_dtype in (torch.complex64, torch.float32):
            a = torch.randn(2, 8, 8, dtype=torch.complex64)
            b = torch.randn(2, 8, 8, dtype=b_dtype)
            mul = torchbp.ops.mul_2d_interp_linear(a, b)
            torch.testing.assert_close(mul[:, :-1, :-1], (a * b)[:, :-1, :-1])
            div = torchbp.ops.div_2d_interp_linear(a, b)
            torch.testing.assert_close(div[:, :-1, :-1], (a / b)[:, :-1, :-1])

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_div(self):
        for args in self.sample_inputs_div("cpu"):
            img1, img2 = args["img1"], args["img2"]
            out_cpu = torchbp.ops.div_2d_interp_linear(img1, img2)
            out_gpu = torchbp.ops.div_2d_interp_linear(img1.cuda(), img2.cuda()).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_mul(self):
        for args in self.sample_inputs_mul("cpu"):
            img1, img2 = args["img1"], args["img2"]
            out_cpu = torchbp.ops.mul_2d_interp_linear(img1, img2)
            out_gpu = torchbp.ops.mul_2d_interp_linear(img1.cuda(), img2.cuda()).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_mul_mixed_dtype(self):
        """Complex image times a real weight map (the polarimetry calibration path)."""
        a = torch.randn(2, 5, 5, dtype=torch.complex64)
        b = torch.randn(2, 3, 3, dtype=torch.float32)
        out_cpu = torchbp.ops.mul_2d_interp_linear(a, b)
        out_gpu = torchbp.ops.mul_2d_interp_linear(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)


class TestSubpixelCorrelation(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.float32):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "im_m": make_tensor((2, 10, 10), dtype=dtype),
            "im_s": make_tensor((2, 10, 10), dtype=dtype),
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False, dtype=torch.complex64)
        for args in samples:
            im_m = args["im_m"]
            im_s = args["im_s"]
            nbatch = 1 if im_m.dim() == 2 else im_m.shape[0]
            Nx = im_m.shape[-2]
            Ny = im_m.shape[-1]
            mean_m = torch.mean(im_m, dim=(-2, -1))
            mean_s = torch.mean(im_s, dim=(-2, -1))

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.subpixel_correlation,
                (im_m, im_s, mean_m, mean_s, nbatch, Nx, Ny),
                test_utils=["test_schema"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestLeeFilter(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "img": make_tensor((2, 10, 10), dtype=dtype),
            "wx": 3,
            "wy": 3,
            "cu": 0.5,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            img = args["img"]
            nbatch = 1 if img.dim() == 2 else img.shape[0]
            Nx = img.shape[-2]
            Ny = img.shape[-1]
            wx = args["wx"] // 2
            wy = args["wy"] // 2

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.lee_filter,
                (img, nbatch, Nx, Ny, wx, wy, args["cu"]),
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def test_cpu_constant_image(self):
        """A constant image has zero local variance, so the filter returns it
        unchanged (the variation weight is zero)."""
        for dtype in (torch.float32, torch.complex64):
            img = torch.full((1, 16, 16), 3.0, dtype=dtype)
            out = torchbp.ops.lee_filter(img, 3, 3, 0.5)
            torch.testing.assert_close(out, torch.full((1, 16, 16), 3.0, dtype=torch.float32))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for dtype in (torch.float32, torch.complex64):
            for args in self.sample_inputs("cpu", dtype=dtype):
                img = args["img"]
                kw = dict(wx=args["wx"], wy=args["wy"], cu=args["cu"])
                out_cpu = torchbp.ops.lee_filter(img, **kw)
                out_gpu = torchbp.ops.lee_filter(img.cuda(), **kw).cpu()
                torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)


class TestPowerCoherence2D(TestCase):
    def sample_inputs(self, device, *, requires_grad=False, dtype=torch.complex64):
        def make_tensor(size, dtype=dtype):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=dtype
            )
            return x

        args = {
            "img0": make_tensor((2, 10, 10), dtype=dtype),
            "img1": make_tensor((2, 10, 10), dtype=dtype),
            "Navg": (3, 3),
            "corr_output": True,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            img0 = args["img0"]
            img1 = args["img1"]
            nbatch = 1 if img0.dim() == 2 else img0.shape[0]
            N0 = img0.shape[-2]
            N1 = img0.shape[-1]

            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.power_coherence_2d,
                (img0, img1, nbatch, N0, N1, args["Navg"][0], args["Navg"][1], args["corr_output"]),
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def test_cpu_self_coherence(self):
        """Coherence of an image with itself is 1 everywhere."""
        img = torch.randn(2, 16, 16, dtype=torch.complex64)
        pc = torchbp.ops.power_coherence_2d(img, img, (3, 3), corr_output=True)
        torch.testing.assert_close(pc, torch.ones_like(pc))
        pc_raw = torchbp.ops.power_coherence_2d(img, img, (3, 3), corr_output=False)
        torch.testing.assert_close(pc_raw, torch.ones_like(pc_raw))

    def test_cpu_uncorrelated_bounded(self):
        """Coherence stays within [0, 1] for uncorrelated inputs."""
        img0 = torch.randn(1, 32, 32, dtype=torch.complex64)
        img1 = torch.randn(1, 32, 32, dtype=torch.complex64)
        pc = torchbp.ops.power_coherence_2d(img0, img1, (4, 4))
        self.assertGreaterEqual(pc.min().item(), 0.0)
        self.assertLessEqual(pc.max().item(), 1.0)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for args in self.sample_inputs("cpu"):
            img0, img1 = args["img0"], args["img1"]
            out_cpu = torchbp.ops.power_coherence_2d(
                img0, img1, args["Navg"], args["corr_output"])
            out_gpu = torchbp.ops.power_coherence_2d(
                img0.cuda(), img1.cuda(), args["Navg"], args["corr_output"]).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)


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


class TestFFBPMerge2Poly(TestCase):
    """Test polynomial approximation version against reference Knab implementation."""

    def sample_inputs(self, device, *, dtype=torch.float32):
        def make_tensor(size, dtype=dtype):
            return torch.randn(size, device=device, dtype=dtype)

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        # Create two polar grid images with slightly different grids
        grid_polar0 = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 50, "ntheta": 40}
        grid_polar1 = {"r": (105, 195), "theta": (-0.75, 0.75), "nr": 48, "ntheta": 38}
        grid_polar_new = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 60, "ntheta": 50}

        img0 = make_tensor((grid_polar0["nr"], grid_polar0["ntheta"]), dtype=complex_dtype)
        img1 = make_tensor((grid_polar1["nr"], grid_polar1["ntheta"]), dtype=complex_dtype)
        dorigin0 = 0.1 * make_tensor((3,), dtype=dtype)
        dorigin1 = 0.1 * make_tensor((3,), dtype=dtype)

        args = {
            "img0": img0,
            "img1": img1,
            "dorigin0": dorigin0,
            "dorigin1": dorigin1,
            "grid_polars": [grid_polar0, grid_polar1],
            "fc": 10e9,
            "grid_polar_new": grid_polar_new,
            "z0": 1.0,
            "order": 6,
            "oversample": 1.5,
            "alias": False,
            "alias_fmod": 0.0,
            "output_alias": True,
        }
        return [args]

    def _poly_vs_knab_reference(self, device):
        """Compare polynomial approximation against reference Knab implementation."""
        samples = self.sample_inputs(device)

        for args in samples:
            # Run reference Knab implementation
            result_knab = torchbp.ops.ffbp_merge2_knab(**args)

            # Run polynomial approximation version
            result_poly = torchbp.ops.ffbp_merge2_poly(**args)

            # They should be reasonably close (polynomial is an approximation)
            # The polynomial trades accuracy for speed
            # Allow for moderate differences due to approximation
            torch.testing.assert_close(
                result_poly,
                result_knab,
                rtol=0.05,
                atol=1e-3
            )

    def test_poly_vs_knab_reference_cpu(self):
        self._poly_vs_knab_reference("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_poly_vs_knab_reference(self):
        self._poly_vs_knab_reference("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_poly_with_precomputed_coefs(self):
        """Test that precomputed polynomial coefficients give same result."""
        samples = self.sample_inputs("cuda")

        for args in samples:
            # Compute polynomial coefficients once
            from torchbp.ops.polar_interp import compute_knab_poly_coefs_full
            poly_coefs = compute_knab_poly_coefs_full(args["order"], args["oversample"])

            # Run with automatic coefficient computation
            result_auto = torchbp.ops.ffbp_merge2_poly(**args)

            # Run with precomputed coefficients
            result_precomp = torchbp.ops.ffbp_merge2_poly(**args, poly_coefs=poly_coefs)

            # Should be identical
            torch.testing.assert_close(result_precomp, result_auto)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_different_orders(self):
        """Test polynomial approximation with different interpolation orders."""
        samples = self.sample_inputs("cuda")

        for args in samples:
            for order in [4, 6, 8]:
                args_order = args.copy()
                args_order["order"] = order

                # Should not raise an error
                result = torchbp.ops.ffbp_merge2_poly(**args_order)

                # Check output shape
                expected_shape = (
                    args["grid_polar_new"]["nr"],
                    args["grid_polar_new"]["ntheta"]
                )
                self.assertEqual(result.shape, expected_shape)


class TestComputeIllumination(TestCase):
    """Test antenna pattern illumination map computation (compute_illumination op)."""

    def _make_inputs(self, device):
        torch.manual_seed(0)
        nsweeps = 8
        pos = torch.randn(nsweeps, 3, device=device, dtype=torch.float32)
        att = torch.zeros(nsweeps, 3, device=device, dtype=torch.float32)
        g = torch.ones(16, 16, device=device, dtype=torch.float32)
        g_extent = [-0.5, -1.0, 0.5, 1.0]
        grid = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 50, "ntheta": 40}
        return pos, att, g, g_extent, grid

    def _test_basic(self, device):
        from torchbp.ops import compute_subaperture_illumination

        pos, att, g, g_extent, grid = self._make_inputs(device)
        w1_map, w2_map = compute_subaperture_illumination(
            pos, att, g, g_extent, grid, decimation=1
        )
        # Output shape matches the (undecimated) polar grid
        self.assertEqual(tuple(w1_map.shape), (grid["nr"], grid["ntheta"]))
        self.assertEqual(tuple(w2_map.shape), (grid["nr"], grid["ntheta"]))
        for m in (w1_map, w2_map):
            self.assertFalse(torch.isnan(m).any())
            self.assertFalse(torch.isinf(m).any())
        # With a uniform unit gain pattern the sum of squared gains never exceeds
        # the sum of gains.
        self.assertTrue(torch.all(w2_map <= w1_map + 1e-4))

    def test_basic_cpu(self):
        self._test_basic("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_basic_cuda(self):
        self._test_basic("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        from torchbp.ops import compute_subaperture_illumination

        pos, att, g, g_extent, grid = self._make_inputs("cuda")
        w1_gpu, w2_gpu = compute_subaperture_illumination(
            pos, att, g, g_extent, grid, decimation=1
        )
        w1_cpu, w2_cpu = compute_subaperture_illumination(
            pos.cpu(), att.cpu(), g.cpu(), g_extent, grid, decimation=1
        )
        torch.testing.assert_close(w1_cpu, w1_gpu.cpu(), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(w2_cpu, w2_gpu.cpu(), atol=1e-4, rtol=1e-3)


class TestFFBPMerge2PolyWeighted(TestCase):
    """Test antenna-pattern-weighted polynomial FFBP merge (ffbp_merge2_poly_weighted op)."""

    def _make_inputs(self, device):
        from torchbp.ops import compute_subaperture_illumination

        torch.manual_seed(0)
        nsweeps = 8
        g = torch.ones(16, 16, device=device, dtype=torch.float32)
        g_extent = [-0.5, -1.0, 0.5, 1.0]
        grid0 = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 50, "ntheta": 40}
        grid1 = {"r": (105, 195), "theta": (-0.75, 0.75), "nr": 48, "ntheta": 38}
        grid_new = {"r": (100, 200), "theta": (-0.8, 0.8), "nr": 60, "ntheta": 50}

        def illum(grid):
            pos = torch.randn(nsweeps, 3, device=device, dtype=torch.float32)
            att = torch.zeros(nsweeps, 3, device=device, dtype=torch.float32)
            return compute_subaperture_illumination(
                pos, att, g, g_extent, grid, decimation=1
            )

        w1_0, w2_0 = illum(grid0)
        w1_1, w2_1 = illum(grid1)
        img0 = torch.randn(grid0["nr"], grid0["ntheta"], device=device, dtype=torch.complex64)
        img1 = torch.randn(grid1["nr"], grid1["ntheta"], device=device, dtype=torch.complex64)
        dorigin0 = 0.1 * torch.randn(3, device=device, dtype=torch.float32)
        dorigin1 = 0.1 * torch.randn(3, device=device, dtype=torch.float32)
        return dict(
            img0=img0, img1=img1, dorigin0=dorigin0, dorigin1=dorigin1,
            grid_polars=[grid0, grid1], fc=10e9, grid_polar_new=grid_new,
            z0=1.0, order=6, oversample=1.5,
            w1_map0=w1_0, w2_map0=w2_0, weight_grid0=grid0,
            w1_map1=w1_1, w2_map1=w2_1, weight_grid1=grid1,
            output_weight_map=False,
        ), grid_new

    def _test_basic(self, device):
        from torchbp.ops import ffbp_merge2_poly_weighted

        args, grid_new = self._make_inputs(device)
        merged, w1_out, w2_out, weight_grid = ffbp_merge2_poly_weighted(**args)
        self.assertEqual(tuple(merged.shape), (grid_new["nr"], grid_new["ntheta"]))
        self.assertFalse(torch.isnan(merged).any())
        self.assertFalse(torch.isinf(merged).any())
        # No weight map requested
        self.assertIsNone(w1_out)
        self.assertIsNone(w2_out)

    def test_basic_cpu(self):
        self._test_basic("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_basic_cuda(self):
        self._test_basic("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu(self):
        from torchbp.ops import ffbp_merge2_poly_weighted

        args_gpu, _ = self._make_inputs("cuda")
        merged_gpu, *_ = ffbp_merge2_poly_weighted(**args_gpu)

        def to_cpu(v):
            return v.cpu() if isinstance(v, torch.Tensor) else v

        args_cpu = {k: to_cpu(v) for k, v in args_gpu.items()}
        merged_cpu, *_ = ffbp_merge2_poly_weighted(**args_cpu)
        # The polynomial-Knab kernel is an approximation, and CPU (plain float
        # Horner) vs CUDA (__fmaf_rn + --use_fast_math) diverge at the ULP level
        # per term. Compounded over the kernel taps and amplified by the
        # weighted-merge normalization, a handful of low-magnitude pixels reach
        # ~0.02 absolute difference. Use an atol matching that approximation
        # envelope (cf. the atol=1e-3, rtol=0.05 poly-vs-knab tolerance above).
        torch.testing.assert_close(merged_cpu, merged_gpu.cpu(), atol=3e-2, rtol=5e-2)


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


class TestPolarInterpLanczos(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=False, dtype=dtype
            )
            return x

        grid = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 8, "ntheta": 8}
        grid_new = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 16, "ntheta": 16}
        args = {
            "img": make_tensor((8, 8), dtype=torch.complex64),
            "dorigin": make_tensor((3,), dtype=torch.float32) * 0.1,
            "grid_polar": grid,
            "fc": 6e9,
            "rotation": 0.0,
            "grid_polar_new": grid_new,
            "z0": 0.0,
            "order": 6,
            "alias_fmod": 0.0,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            # Prepare C++ operator arguments
            img = args["img"]
            dorigin = args["dorigin"]
            fc = args["fc"]
            rotation = args["rotation"]
            z0 = args["z0"]
            order = args["order"]
            alias_fmod = args["alias_fmod"]

            grid = args["grid_polar"]
            r1_0, r1_1 = grid["r"]
            theta1_0, theta1_1 = grid["theta"]
            ntheta1 = grid["ntheta"]
            nr1 = grid["nr"]
            dtheta1 = (theta1_1 - theta1_0) / ntheta1
            dr1 = (r1_1 - r1_0) / nr1

            grid_new = args["grid_polar_new"]
            r3_0, r3_1 = grid_new["r"]
            theta3_0, theta3_1 = grid_new["theta"]
            ntheta3 = grid_new["ntheta"]
            nr3 = grid_new["nr"]
            dtheta3 = (theta3_1 - theta3_0) / ntheta3
            dr3 = (r3_1 - r3_0) / nr3

            nbatch = 1

            cpp_args = (img, dorigin, nbatch, rotation, fc,
                       r1_0, dr1, theta1_0, dtheta1, nr1, ntheta1,
                       r3_0, dr3, theta3_0, dtheta3, nr3, ntheta3,
                       z0, order, alias_fmod)

            opcheck(
                torch.ops.torchbp.polar_interp_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestPolarToCartLanczos(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=False, dtype=dtype
            )
            return x

        grid_polar = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 8, "ntheta": 8}
        grid_cart = {"x": (-5, 5), "y": (-5, 5), "nx": 16, "ny": 16}
        args = {
            "img": make_tensor((8, 8), dtype=torch.complex64),
            "origin": make_tensor((3,), dtype=torch.float32) * 0.1,
            "grid_polar": grid_polar,
            "grid_cart": grid_cart,
            "fc": 6e9,
            "rotation": 0.0,
            "alias_fmod": 0.0,
            "order": 6,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            # Prepare C++ operator arguments
            img = args["img"]
            origin = args["origin"]
            fc = args["fc"]
            rotation = args["rotation"]
            alias_fmod = args["alias_fmod"]
            order = args["order"]

            grid_polar = args["grid_polar"]
            r0, r1 = grid_polar["r"]
            theta0, theta1 = grid_polar["theta"]
            ntheta = grid_polar["ntheta"]
            nr = grid_polar["nr"]
            dtheta = (theta1 - theta0) / ntheta
            dr = (r1 - r0) / nr

            grid_cart = args["grid_cart"]
            x0, x1 = grid_cart["x"]
            y0, y1 = grid_cart["y"]
            nx = grid_cart["nx"]
            ny = grid_cart["ny"]
            dx = (x1 - x0) / nx
            dy = (y1 - y0) / ny

            nbatch = 1

            cpp_args = (img, origin, nbatch, rotation, fc,
                       r0, dr, theta0, dtheta, nr, ntheta,
                       x0, y0, dx, dy, nx, ny, alias_fmod, order)

            opcheck(
                torch.ops.torchbp.polar_to_cart_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestFFBPMerge2Lanczos(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size, dtype=torch.float32):
            x = torch.randn(
                size, device=device, requires_grad=False, dtype=dtype
            )
            return x

        grid0 = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 8, "ntheta": 8}
        grid1 = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 8, "ntheta": 8}
        grid_new = {"r": (1, 10), "theta": (-0.9, 0.9), "nr": 16, "ntheta": 16}
        args = {
            "img0": make_tensor((8, 8), dtype=torch.complex64),
            "img1": make_tensor((8, 8), dtype=torch.complex64),
            "dorigin0": make_tensor((3,), dtype=torch.float32) * 0.1,
            "dorigin1": make_tensor((3,), dtype=torch.float32) * 0.1,
            "grid_polars": [grid0, grid1],
            "fc": 6e9,
            "grid_polar_new": grid_new,
            "z0": 0.0,
            "order": 6,
            "alias": False,
            "alias_fmod": 0.0,
            "output_alias": True,
        }
        return [args]

    def _opcheck(self, device):
        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            # Prepare C++ operator arguments
            img0 = args["img0"]
            img1 = args["img1"]
            dorigin0 = args["dorigin0"]
            dorigin1 = args["dorigin1"]
            fc = args["fc"]
            z0 = args["z0"]
            order = args["order"]
            alias = args["alias"]
            alias_fmod = args["alias_fmod"]
            output_alias = args["output_alias"]

            nimages = 2
            r0 = torch.zeros(nimages, dtype=torch.float32, device=device)
            dr0 = torch.zeros(nimages, dtype=torch.float32, device=device)
            theta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
            dtheta0 = torch.zeros(nimages, dtype=torch.float32, device=device)
            Nr0 = torch.zeros(nimages, dtype=torch.int32, device=device)
            Ntheta0 = torch.zeros(nimages, dtype=torch.int32, device=device)

            for i, grid in enumerate(args["grid_polars"]):
                r1_0, r1_1 = grid["r"]
                theta1_0, theta1_1 = grid["theta"]
                ntheta1 = grid["ntheta"]
                nr1 = grid["nr"]
                dtheta1 = (theta1_1 - theta1_0) / ntheta1
                dr1 = (r1_1 - r1_0) / nr1
                r0[i] = r1_0
                dr0[i] = dr1
                theta0[i] = theta1_0
                dtheta0[i] = dtheta1
                Nr0[i] = nr1
                Ntheta0[i] = ntheta1

            grid_new = args["grid_polar_new"]
            r3_0, r3_1 = grid_new["r"]
            theta3_0, theta3_1 = grid_new["theta"]
            ntheta3 = grid_new["ntheta"]
            nr3 = grid_new["nr"]
            dtheta3 = (theta3_1 - theta3_0) / ntheta3
            dr3 = (r3_1 - r3_0) / nr3

            dorigin = torch.stack((dorigin0, dorigin1), dim=0)

            alias_mode = 0
            if alias:
                if not output_alias:
                    alias_mode = 1
                else:
                    alias_mode = 2
            elif not output_alias:
                alias_mode = 3

            cpp_args = (img0, img1, dorigin, fc,
                       r0, dr0, theta0, dtheta0, Nr0, Ntheta0,
                       r3_0, dr3, theta3_0, dtheta3, nr3, ntheta3,
                       z0, order, alias_mode, alias_fmod)

            opcheck(
                torch.ops.torchbp.ffbp_merge2_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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


if __name__ == "__main__":
    unittest.main()
