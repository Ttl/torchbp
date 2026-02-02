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

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
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
    @unittest.skip
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

    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip
    def test_cpu_and_gpu(self):
        samples = self.sample_inputs("cuda")
        for sample in samples:
            res_gpu = torchbp.ops.polar_to_cart_linear(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.polar_to_cart_linear(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu)

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

    @unittest.skip
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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip
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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
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

    # def test_gradients_cpu(self):
    #    self._test_gradients("cpu")

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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
    def test_opcheck_div_cpu(self):
        self._opcheck_div("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_div_cuda(self):
        self._opcheck_div("cuda")

    @unittest.skip("CPU implementation not available")
    def test_opcheck_mul_cpu(self):
        self._opcheck_mul("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_mul_cuda(self):
        self._opcheck_mul("cuda")


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

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_poly_vs_knab_reference(self):
        """Compare polynomial approximation against reference Knab implementation."""
        samples = self.sample_inputs("cuda")

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

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestProjectionCart2D(TestCase):
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

        nbatch = 1
        nsweeps = 2
        sweep_samples = 8
        grid = {"x": (-2, 2), "y": (-2, 2), "nx": 4, "ny": 4}
        args = {
            "img": make_tensor((nbatch, 4, 4), dtype=torch.complex64),
            "pos": make_pos_tensor((nbatch, nsweeps, 3), dtype=torch.float32),
            "grid": grid,
            "fc": 6e9,
            "fs": 2e6,
            "gamma": 1e12,
            "sweep_samples": sweep_samples,
            "d0": 0.2,
            "normalization": "sigma",
        }
        return [args]

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_projection_cart_2d_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_projection_cart_2d_args(**args)
            opcheck(
                torch.ops.torchbp.projection_cart_2d,
                cpp_args,
                test_utils=["test_schema"]
            )

    @unittest.skip("CPU implementation not available")
    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
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

    @unittest.skip("CPU implementation not available")
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


if __name__ == "__main__":
    unittest.main()
