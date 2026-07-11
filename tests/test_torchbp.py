#!/usr/bin/env python
import torch
import numpy as np
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

    def test_gradients_cpu(self):
        # float64 not tested: CPU forward supports only complex64/float32
        self._test_gradients("cpu")

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


class TestCartToPolarLinear(TestCase):
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
        grid_cart = {"x": (12, 18), "y": (-5, 5), "nx": 3, "ny": 3}
        grid_polar = {"r": (10, 20), "theta": (-1, 1), "nr": 2, "ntheta": 2}
        origin = 0.1 * make_tensor((nbatch, 3), dtype=dtype)
        origin[:, 2] += 4  # Offset height
        args = {
            "img": make_tensor(
                (nbatch, grid_cart["nx"], grid_cart["ny"]), dtype=complex_dtype
            ),
            "origin": origin,
            "grid_cart": grid_cart,
            "grid_polar": grid_polar,
            "fc": 6e9,
            "rotation": 0.1,
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

            res_gpu = torchbp.ops.cart_to_polar_linear(**sample)
            loss_gpu = torch.mean(torch.abs(res_gpu))
            loss_gpu.backward()
            grads_gpu = [
                sample[k].cpu()
                for k in sample.keys()
                if isinstance(sample[k], torch.Tensor) and sample[k].requires_grad
            ]

            res_cpu = torchbp.ops.cart_to_polar_linear(**sample_cpu)
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
            res_gpu = torchbp.ops.cart_to_polar_linear(**sample).cpu()
            sample_cpu = {
                k: sample[k].cpu() if isinstance(sample[k], torch.Tensor) else sample[k]
                for k in sample.keys()
            }
            res_cpu = torchbp.ops.cart_to_polar_linear(**sample_cpu)
            torch.testing.assert_close(res_cpu, res_gpu, rtol=5e-4, atol=5e-4)

    def test_inverse_of_polar_to_cart(self):
        # polar -> cart -> polar round trip on a smooth image recovers the
        # input away from the grid edges.
        device = "cpu"
        grid_polar = {"r": (50.0, 100.0), "theta": (-0.5, 0.5), "nr": 128, "ntheta": 128}
        grid_cart = {"x": (60.0, 90.0), "y": (-25.0, 25.0), "nx": 512, "ny": 512}
        fc = 6e9

        # Bandlimited smooth complex image on the polar grid
        img = torch.randn(16, 16, dtype=torch.complex64, device=device)
        img = torch.fft.ifft2(F.pad(torch.fft.fft2(img), (0, 112, 0, 112)))
        img = img / img.abs().max()

        origin = torch.tensor([0.0, 0.0, 30.0], device=device)
        cart = torchbp.ops.polar_to_cart(img, origin, grid_polar, grid_cart, fc)
        back = torchbp.ops.cart_to_polar(cart[0], origin, grid_cart, grid_polar, fc)

        r = torch.linspace(*grid_polar["r"], grid_polar["nr"] + 1)[:-1]
        t = torch.linspace(*grid_polar["theta"], grid_polar["ntheta"] + 1)[:-1]
        rg, tg = torch.meshgrid(r, t, indexing="ij")
        x = rg * torch.sqrt(1 - tg**2)
        y = rg * tg
        mask = (x > 62) & (x < 88) & (y > -23) & (y < 23)
        err = (back[0] - img).abs()[mask].max().item()
        self.assertLess(err, 0.05)

    def _test_gradients(self, device, dtype=torch.float32):
        samples = self.sample_inputs(device, requires_grad=True, dtype=dtype)
        eps = 5e-4 if dtype == torch.float32 else 1e-4
        rtol = 0.15 if dtype == torch.float32 else 0.05
        for args in samples:
            torch.autograd.gradcheck(
                torchbp.ops.cart_to_polar_linear,
                list(args.values()),
                eps=eps,  # This test is very sensitive to eps
                rtol=rtol,  # Also to rtol
            )

    def test_gradients_cpu(self):
        # float64 not tested: CPU forward supports only complex64/float32
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        from torchbp.ops.polar_interp import _prepare_cart_to_polar_linear_args

        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            cpp_args = _prepare_cart_to_polar_linear_args(**args)
            opcheck(
                torch.ops.torchbp.cart_to_polar_linear,
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


class TestBackprojectionPolarDem(TestCase):
    """Tests for the optional polar-grid DEM input of backprojection_polar_2d."""

    def _random_inputs(self, device, nr=32, ntheta=16):
        torch.manual_seed(42)
        nbatch = 1
        nsweeps = 8
        sweep_samples = 512
        grid = {"r": (30, 60), "theta": (-0.5, 0.5), "nr": nr, "ntheta": ntheta}
        data = torch.randn(nbatch, nsweeps, sweep_samples, device=device,
                           dtype=torch.complex64)
        pos = torch.zeros(nbatch, nsweeps, 3, device=device)
        pos[..., 1] = torch.linspace(-1, 1, nsweeps, device=device)
        pos[..., 2] = 25.0
        return data, grid, pos

    def _test_zero_dem_matches_no_dem(self, device):
        data, grid, pos = self._random_inputs(device)
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        ref = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        torch.testing.assert_close(res, ref)

    def test_zero_dem_matches_no_dem_cpu(self):
        self._test_zero_dem_matches_no_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_zero_dem_matches_no_dem_cuda(self):
        self._test_zero_dem_matches_no_dem("cuda")

    def _test_constant_dem_equals_shifted_pos(self, device):
        # Pixels at a constant height h see exactly the same distances as
        # z=0 pixels with the platform lowered by h.
        data, grid, pos = self._random_inputs(device)
        h = 7.0
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        pos_shift = pos.clone()
        pos_shift[..., 2] -= h
        ref = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos_shift)
        torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

    def test_constant_dem_equals_shifted_pos_cpu(self):
        self._test_constant_dem_equals_shifted_pos("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_constant_dem_equals_shifted_pos_cuda(self):
        self._test_constant_dem_equals_shifted_pos("cuda")

    def _make_ramp_dem(self, grid, dem_nr, dem_ntheta, device):
        # Bilinear in (r, theta), so bilinear interpolation between DEM
        # samples reproduces it exactly on aligned grids.
        r0, r1 = grid["r"]
        t0, t1 = grid["theta"]
        r = r0 + (r1 - r0) / dem_nr * torch.arange(dem_nr, device=device)
        t = t0 + (t1 - t0) / dem_ntheta * torch.arange(dem_ntheta, device=device)
        return 0.2 * r[:, None] + 3.0 * t[None, :] + 2.0

    def _test_downsampled_dem(self, device):
        data, grid, pos = self._random_inputs(device, nr=32, ntheta=16)
        # Constant DEM: downsampling changes nothing anywhere.
        h = 5.0
        dem_fine = torch.full((32, 16), h, device=device)
        dem_coarse = torch.full((8, 4), h, device=device)
        res_f = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_fine)
        res_c = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_coarse)
        torch.testing.assert_close(res_c, res_f)

        # Ramp DEM: exact away from the edge-clamped last coarse cell.
        # fr = idr * dem_nr / nr is within the coarse grid for
        # idr <= (dem_nr - 1) * nr / dem_nr.
        dem_fine = self._make_ramp_dem(grid, 32, 16, device)
        dem_coarse = self._make_ramp_dem(grid, 8, 4, device)
        res_f = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_fine)
        res_c = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem_coarse)
        torch.testing.assert_close(res_c[:, :29, :13], res_f[:, :29, :13],
                                   atol=1e-2, rtol=1e-2)

    def test_downsampled_dem_cpu(self):
        self._test_downsampled_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_downsampled_dem_cuda(self):
        self._test_downsampled_dem("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_and_gpu_dem(self):
        data, grid, pos = self._random_inputs("cuda")
        torch.manual_seed(123)
        dem = 5.0 + 2.0 * torch.randn(grid["nr"] // 4, grid["ntheta"] // 4,
                                      device="cuda")
        res_gpu = torchbp.ops.backprojection_polar_2d(
            data, grid, 6e9, 0.15, pos, dem=dem).cpu()
        res_cpu = torchbp.ops.backprojection_polar_2d(
            data.cpu(), grid, 6e9, 0.15, pos.cpu(), dem=dem.cpu())
        torch.testing.assert_close(res_cpu, res_gpu, atol=1e-3, rtol=1e-2)

    def _test_dem_grad_raises(self, device):
        data, grid, pos = self._random_inputs(device)
        pos.requires_grad = True
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        res = torchbp.ops.backprojection_polar_2d(data, grid, 6e9, 0.15, pos, dem=dem)
        with self.assertRaises(ValueError):
            torch.mean(torch.abs(res)).backward()

    def test_dem_grad_raises_cpu(self):
        self._test_dem_grad_raises("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dem_grad_raises_cuda(self):
        self._test_dem_grad_raises("cuda")

    def _opcheck_dem(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_polar_2d_args

        data, grid, pos = self._random_inputs(device)
        dem = torch.zeros(grid["nr"], grid["ntheta"], device=device)
        cpp_args = _prepare_backprojection_polar_2d_args(
            data, grid, 6e9, 0.15, pos, dem=dem)
        opcheck(
            torch.ops.torchbp.backprojection_polar_2d,
            cpp_args,
            test_utils=["test_schema", "test_faketensor"]
        )

    def test_opcheck_dem_cpu(self):
        self._opcheck_dem("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_dem_cuda(self):
        self._opcheck_dem("cuda")

    def _test_dem_point_target(self, device):
        # A target above the ground plane focuses at the wrong ground range
        # with the flat-earth assumption and at the true ground range with a
        # DEM at the target height.
        c0 = 299792458.0
        fc = 6e9
        bw = 200e6
        tsweep = 100e-6
        fs = 10e6
        nsamples = int(fs * tsweep)
        oversample = 2
        r_res = c0 / (2 * bw * oversample)
        nsweeps = 128
        alt = 30.0
        h = 12.0
        target_r = 80.0

        pos = torch.zeros(nsweeps, 3, device=device)
        pos[:, 1] = torch.linspace(-10, 10, nsweeps, device=device)
        pos[:, 2] = alt
        target_pos = torch.tensor([[target_r, 0.0, h]], device=device)
        target_rcs = torch.ones((1, 1), device=device)

        raw = torchbp.util.generate_fmcw_data(
            target_pos, target_rcs, pos, fc, bw, tsweep, fs, rvp=False)
        w = torch.hamming_window(nsamples, periodic=False, device=device)
        data = torch.fft.ifft(raw * w[None, :], dim=-1, n=nsamples * oversample)

        grid = {"r": (60.0, 100.0), "theta": (-0.3, 0.3), "nr": 320, "ntheta": 128}
        dr = (grid["r"][1] - grid["r"][0]) / grid["nr"]

        img_flat = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos)[0]
        dem = torch.full((grid["nr"], grid["ntheta"]), h, device=device)
        img_dem = torchbp.ops.backprojection_polar_2d(
            data, grid, fc, r_res, pos, dem=dem)[0]

        idr_dem = torch.argmax(torch.abs(img_dem)).item() // grid["ntheta"]
        idr_flat = torch.argmax(torch.abs(img_flat)).item() // grid["ntheta"]

        r_dem = grid["r"][0] + dr * idr_dem
        r_flat = grid["r"][0] + dr * idr_flat
        # Flat-earth image focuses where sqrt(r^2 + alt^2) equals the true
        # slant range sqrt(target_r^2 + (alt - h)^2).
        d_true = np.sqrt(target_r**2 + (alt - h) ** 2)
        r_flat_expected = np.sqrt(d_true**2 - alt**2)

        self.assertLess(abs(r_dem - target_r), 2 * dr)
        self.assertLess(abs(r_flat - r_flat_expected), 2 * dr)

    def test_dem_point_target_cpu(self):
        self._test_dem_point_target("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_dem_point_target_cuda(self):
        self._test_dem_point_target("cuda")


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

    @staticmethod
    def _make_nonsymmetric_antenna(device):
        """Inputs with a strongly non-symmetric antenna pattern.

        The pattern is narrow in azimuth and wide in elevation on a non-square
        (nel != naz) grid, so reading it transposed (swapping the el/az axes or
        their sizes) changes the result. The yaw is swept across the aperture so
        every sweep samples a different part of the azimuth pattern.
        """
        torch.manual_seed(0)
        nsweeps = 24
        sweep_samples = 64
        fc = 6e9
        r_res = 0.15
        grid = {"r": (4, 8), "theta": (-0.3, 0.3), "nr": 16, "ntheta": 24}

        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = (
            torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps, device=device)
            * 0.25 * 3e8 / fc
        )
        pos[:, 2] = 2.0   # low altitude so the scene is within sweep range

        data = torch.randn(nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        nel, naz = 16, 24
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-2.0, 2.0, naz, device=device)
        gain = torch.exp(-(el[:, None] / 0.8) ** 2) * torch.exp(-(az[None, :] / 0.25) ** 2)
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]

        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        att[:, 0] = -float(np.arcsin(2.0 / 6.0))   # roll: point elevation beam at scene
        att[:, 2] = torch.linspace(-0.3, 0.3, nsweeps, device=device)              # yaw sweep
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nonsymmetric_pattern_cpu_gpu(self):
        """CPU and CUDA backprojection must agree with a non-symmetric pattern,
        for both the normalized and unnormalized (FFBP) paths."""
        a = self._make_nonsymmetric_antenna("cuda")

        def to_cpu(v):
            return v.cpu() if isinstance(v, torch.Tensor) else v

        a_cpu = {k: to_cpu(v) for k, v in a.items()}
        for normalize in (True, False):
            out_gpu = torchbp.ops.backprojection_polar_2d(
                a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
                att=a["att"], g=a["g"], g_extent=a["g_extent"], normalize=normalize,
            )
            out_cpu = torchbp.ops.backprojection_polar_2d(
                a_cpu["data"], a_cpu["grid"], a_cpu["fc"], a_cpu["r_res"], a_cpu["pos"],
                att=a_cpu["att"], g=a_cpu["g"], g_extent=a_cpu["g_extent"],
                normalize=normalize,
            )
            torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-3)


class TestFFBPAntennaPattern(TestCase):
    """End-to-end antenna-pattern-weighted FFBP must match between CPU and CUDA."""

    @staticmethod
    def _make_inputs(device):
        torch.manual_seed(0)
        nsweeps = 64
        sweep_samples = 128
        fc = 6e9
        r_res = 0.15
        grid = {"r": (4, 8), "theta": (-0.3, 0.3), "nr": 64, "ntheta": 128}

        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = (
            torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps, device=device)
            * 0.25 * 3e8 / fc
        )
        pos[:, 2] = 2.0   # low altitude so the scene is within sweep range

        data = torch.randn(nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        nel, naz = 16, 24
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-2.0, 2.0, naz, device=device)
        gain = torch.exp(-(el[:, None] / 0.8) ** 2) * torch.exp(-(az[None, :] / 0.25) ** 2)
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]

        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        att[:, 0] = -float(np.arcsin(2.0 / 6.0))   # roll: point elevation beam at scene
        att[:, 2] = torch.linspace(-0.3, 0.3, nsweeps, device=device)
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    @staticmethod
    def _make_narrow_beam_inputs(device):
        """Narrow azimuth beam scanned across a wide swath. Many pixels are barely
        illuminated, so the W1/W2 normalization divides by near-zero illumination
        and (without regularization)."""
        torch.manual_seed(0)
        nsweeps = 128
        sweep_samples = 128
        fc = 6e9
        r_res = 0.15
        grid = {"r": (5, 12), "theta": (-0.5, 0.5), "nr": 96, "ntheta": 192}

        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = (
            torch.linspace(-nsweeps / 2, nsweeps / 2, nsweeps, device=device)
            * 0.25 * 3e8 / fc
        )
        pos[:, 2] = 3.0

        data = torch.randn(nsweeps, sweep_samples, device=device, dtype=torch.complex64)

        nel, naz = 16, 64
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-1.5, 1.5, naz, device=device)
        # Very narrow azimuth beam
        gain = torch.exp(-(el[:, None] / 0.6) ** 2) * torch.exp(-(az[None, :] / 0.04) ** 2)
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]

        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        att[:, 0] = -float(np.arcsin(3.0 / 8.0))
        att[:, 2] = torch.linspace(-0.45, 0.45, nsweeps, device=device) # scan
        return dict(data=data, grid=grid, fc=fc, r_res=r_res, pos=pos,
                    att=att, g=g, g_extent=g_extent)

    def test_weighted_ffbp_no_blowup_downsample1(self):
        """Weighted FFBP at downsample=1 must stay bounded at swath
        edges instead of producing a large spike from dividing by near-zero
        illumination."""
        a = self._make_narrow_beam_inputs("cpu")
        kw = dict(stages=3, divisions=2, dealias=True,
                  att=a["att"], g=a["g"], g_extent=a["g_extent"])
        img = torchbp.ops.ffbp(a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
                               weight_map_downsample=1, **kw)
        self.assertTrue(torch.isfinite(img).all())
        # Output must be of the same order as the unweighted backprojection.
        ref = torchbp.ops.backprojection_polar_2d(
            a["data"], a["grid"], a["fc"], a["r_res"], a["pos"], dealias=True)
        self.assertLess(float(img.abs().max()), 50.0 * float(ref.abs().max()))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_weighted_ffbp_cpu_gpu(self):
        a = self._make_inputs("cuda")

        def to_cpu(v):
            return v.cpu() if isinstance(v, torch.Tensor) else v

        a_cpu = {k: to_cpu(v) for k, v in a.items()}
        kw = dict(stages=2, divisions=2, dealias=True)
        out_gpu = torchbp.ops.ffbp(
            a["data"], a["grid"], a["fc"], a["r_res"], a["pos"],
            att=a["att"], g=a["g"], g_extent=a["g_extent"], **kw,
        )
        out_cpu = torchbp.ops.ffbp(
            a_cpu["data"], a_cpu["grid"], a_cpu["fc"], a_cpu["r_res"], a_cpu["pos"],
            att=a_cpu["att"], g=a_cpu["g"], g_extent=a_cpu["g_extent"], **kw,
        )
        # Poly-Knab merge differs at the ULP level between CPU and CUDA and is
        # amplified by the weighted normalization (cf. TestFFBPMerge2PolyWeighted).
        torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=3e-2, rtol=5e-2)


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


class TestBlocksvdAlpha(TestCase):
    def sample_inputs(self, device):
        torch.manual_seed(2)
        nr, ntheta = 12, 10
        nsweeps, sweep_samples = 7, 200
        args = {
            "img": torch.randn(nr, ntheta, device=device, dtype=torch.complex64),
            "data": torch.randn(
                nsweeps, sweep_samples, device=device, dtype=torch.complex64
            ),
            "pos": 0.1 * torch.randn(nsweeps, 3, device=device),
            # (ri0, ri1, ti0, ti1, sweep_lo, sweep_hi): full grid, partial
            # rectangles with partial sweep windows, empty sweep window,
            # degenerate pixel rectangle
            "blocks": torch.tensor(
                [
                    [0, nr, 0, ntheta, 0, nsweeps],
                    [0, 6, 0, 5, 2, 6],
                    [6, nr, 5, ntheta, 0, nsweeps],
                    [0, 6, 5, ntheta, 0, 0],
                    [3, 3, 2, 8, 0, nsweeps],
                ],
                device=device,
            ),
            "fc": 6e9,
            "r_res": 0.15,
            "r0": 10.0,
            "dr": 0.5,
            "theta0": -0.4,
            "dtheta": 0.08,
            "d0": 0.2,
            "data_fmod": uniform(0, 2 * torch.pi),
        }
        return [args]

    def _reference(self, args):
        """Compose the same result from gpga_backprojection_2d_core + GEMV."""
        img, data, pos = args["img"], args["data"], args["pos"]
        blocks = args["blocks"]
        nsweeps = data.shape[0]
        ref = torch.zeros(
            blocks.shape[0], nsweeps, dtype=torch.complex64, device=img.device
        )
        for b in range(blocks.shape[0]):
            ri0, ri1, ti0, ti1, lo, hi = [int(v) for v in blocks[b]]
            if ri1 <= ri0 or ti1 <= ti0 or hi <= lo:
                continue
            r = args["r0"] + args["dr"] * torch.arange(
                ri0, ri1, device=img.device, dtype=torch.float32
            )
            t = args["theta0"] + args["dtheta"] * torch.arange(
                ti0, ti1, device=img.device, dtype=torch.float32
            )
            R, T = torch.meshgrid(r, t, indexing="ij")
            x = (R * torch.sqrt(torch.clamp(1.0 - T**2, min=0.0))).reshape(-1)
            y = (R * T).reshape(-1)
            target_pos = torch.stack([x, y, torch.zeros_like(x)], dim=1)
            B = torchbp.ops.gpga_backprojection_2d_core(
                target_pos, data[lo:hi], pos[lo:hi], args["fc"],
                args["r_res"], d0=args["d0"], data_fmod=args["data_fmod"],
            )
            patch = img[ri0:ri1, ti0:ti1].reshape(-1)
            ref[b, lo:hi] = torch.conj(patch) @ B
        return ref

    def test_cpu_reference(self):
        for args in self.sample_inputs("cpu"):
            out = torchbp.ops.blocksvd_alpha(
                args["img"], args["data"], args["pos"], args["blocks"],
                args["fc"], args["r_res"], args["r0"], args["dr"],
                args["theta0"], args["dtheta"], d0=args["d0"],
                data_fmod=args["data_fmod"],
            )
            ref = self._reference(args)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def _opcheck(self, device):
        for args in self.sample_inputs(device):
            data = args["data"]
            blocks = args["blocks"].to(torch.int32)
            # Only test schema - no gradient support for this op
            opcheck(
                torch.ops.torchbp.blocksvd_alpha,
                (args["img"], data, args["pos"], blocks, data.shape[1],
                 data.shape[0], blocks.shape[0], args["img"].shape[1],
                 args["fc"], args["r_res"], args["r0"], args["dr"],
                 args["theta0"], args["dtheta"], args["d0"],
                 args["data_fmod"]),
                test_utils=["test_schema"],
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda(self):
        for args in self.sample_inputs("cpu"):
            kw = dict(
                fc=args["fc"], r_res=args["r_res"], r0=args["r0"],
                dr=args["dr"], theta0=args["theta0"], dtheta=args["dtheta"],
                d0=args["d0"], data_fmod=args["data_fmod"],
            )
            out_cpu = torchbp.ops.blocksvd_alpha(
                args["img"], args["data"], args["pos"], args["blocks"], **kw)
            out_gpu = torchbp.ops.blocksvd_alpha(
                args["img"].cuda(), args["data"].cuda(), args["pos"].cuda(),
                args["blocks"].cuda(), **kw).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)


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


class TestFFBPRemainderSweeps(TestCase):
    """ffbp must use every sweep when divisions does not divide nsweeps.

    Regression test: the subaperture split used to truncate to
    nsweeps // divisions * divisions sweeps at every recursion level,
    silently dropping the remainder.
    """

    def _last_sweep_energy(self, nsweeps, stages, divisions=2):
        grid = {"r": (80.0, 120.0), "theta": (-0.25, 0.25), "nr": 32, "ntheta": 32}
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, nsweeps)
        pos[:, 2] = 30.0
        # Only the last sweep carries signal. If it is dropped the image is
        # exactly zero.
        data = torch.zeros(nsweeps, 64, dtype=torch.complex64)
        data[-1, 50] = 1.0
        img = torchbp.ops.ffbp(
            data, grid, 6e9, 2.0, pos, stages=stages, divisions=divisions
        )
        return img.abs().sum().item()

    def test_odd_nsweeps(self):
        for stages in (1, 2, 3):
            self.assertGreater(self._last_sweep_energy(513, stages), 0.0)

    def test_divisions3(self):
        self.assertGreater(self._last_sweep_energy(1000, 2, divisions=3), 0.0)


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


class TestBackprojectionCart2DTxPower(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(size):
            return torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )

        def make_pos_tensor(size):
            x = torch.randn(
                size, device=device, requires_grad=requires_grad, dtype=torch.float32
            )
            # Push the platform below and behind the scene so the ground pixels
            # fall inside a reasonable look/azimuth cone.
            x = x.clone()
            with torch.no_grad():
                x[..., 2] = 100.0
            return x

        nbatch = 2
        nsweeps = 4
        grid = {"x": (30.0, 200.0), "y": (-100.0, 100.0), "nx": 8, "ny": 8}
        g_extent = [-1.4, -1.2, 1.4, 1.2]  # [g_el0, g_az0, g_el1, g_az1]
        args = {
            "wa": make_tensor((nbatch, nsweeps)).abs(),
            "g": make_tensor((16, 16)).abs(),
            "g_extent": g_extent,
            "grid": grid,
            "r_res": 0.15,
            "pos": make_pos_tensor((nbatch, nsweeps, 3)),
            "att": make_tensor((nbatch, nsweeps, 3)),
            "normalization": "sigma",
        }
        return [args]

    def _opcheck(self, device):
        from torchbp.ops.backproj import _prepare_backprojection_cart_2d_tx_power_args

        samples = self.sample_inputs(device, requires_grad=False)
        for args in samples:
            cpp_args = _prepare_backprojection_cart_2d_tx_power_args(**args)
            opcheck(
                torch.ops.torchbp.backprojection_cart_2d_tx_power,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestFFBPTxPower(TestCase):
    """backprojection_polar_2d_tx_power_ffbp must match the direct backprojection_polar_2d_tx_power."""

    @staticmethod
    def _antenna(device, az_width=0.25, el_width=0.8):
        nel, naz = 16, 24
        el = torch.linspace(-1.2, 1.2, nel, device=device)
        az = torch.linspace(-2.0, 2.0, naz, device=device)
        gain = torch.exp(-(el[:, None] / el_width) ** 2) * torch.exp(
            -(az[None, :] / az_width) ** 2
        )
        g = gain.to(torch.float32)
        g_extent = [el[0].item(), az[0].item(), el[-1].item(), az[-1].item()]
        return g, g_extent

    @staticmethod
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

    def _assert_matches(self, out, ref, q95=0.03, emax=0.1, margin=4, min_ratio=None):
        self.assertEqual(out.shape, ref.shape)
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
        # Infinite (unilluminated) pixels should agree except for isolated
        # interpolation edge cases.
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.99)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        self.assertGreater(int(finite.sum()), 0)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 100)
        ratio = out_c[mask] / ref_c[mask]
        err = (ratio - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), q95)
        self.assertLess(float(err.max()), emax)
        if min_ratio is not None:
            self.assertGreater(float(ratio.min()), min_ratio)

    def _run_both(self, wa, g, g_extent, grid, pos, att, device="cpu", altitude=0.0,
                  **kw):
        kw.setdefault("stages", 3)
        kw.setdefault("downsample_r", 1.0)
        kw.setdefault("downsample_theta", 1.0)
        kw.setdefault("min_nsweeps", 16)
        if altitude > 0.0:
            ref = torchbp.ops.backprojection_polar_2d_tx_power_slant(
                wa, g, g_extent, grid, 0.15, pos, att, altitude=altitude,
                normalization=kw.get("normalization"),
                azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        else:
            ref = torchbp.ops.backprojection_polar_2d_tx_power(
                wa, g, g_extent, grid, 0.15, pos, att,
                normalization=kw.get("normalization"),
                azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        out = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, grid, 0.15, pos, att, altitude=altitude, **kw)
        return out, ref

    def test_straight_track(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma")
        self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_long_track_narrow_theta(self):
        """Subapertures at the track ends see the scene at local theta far
        outside the narrow output theta range. A shared theta grid would lose
        their contributions entirely."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.8)
        wa, pos, att = self._straight_track(
            device, nsweeps=256, span=300.0, alt=20.0, r_center=120.0)
        grid = {"r": (100, 140), "theta": (-0.05, 0.05), "nr": 64, "ntheta": 64}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", stages=4)
        # min_ratio catches a systematic deficit from dropped subapertures.
        self._assert_matches(out, ref, q95=0.03, emax=0.1, min_ratio=0.9)

    def test_curved_track(self):
        device = "cpu"
        torch.manual_seed(0)
        nsweeps = 256
        g, g_extent = self._antenna(device, az_width=0.5)
        R = 150.0
        alpha = torch.linspace(-0.5, 0.5, nsweeps, device=device)
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 0] = R - R * torch.cos(alpha)
        pos[:, 1] = R * torch.sin(alpha)
        pos[:, 2] = 20.0
        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        # Yaw tracks the scene center so every pulse illuminates the scene.
        att[:, 2] = torch.atan2(-pos[:, 1], 100.0 - pos[:, 0])
        att[:, 0] = -float(np.arcsin(20.0 / np.sqrt(100.0**2 + 20.0**2)))
        wa = (torch.hann_window(nsweeps, device=device) + 0.1).to(torch.float32)
        grid = {"r": (80, 120), "theta": (-0.3, 0.3), "nr": 64, "ntheta": 128}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", stages=4)
        self._assert_matches(out, ref, q95=0.04, emax=0.15)

    def test_thin_beam_deep_stages(self):
        """A beam that is thin compared to the output grid theta step used to
        poison the psi moments with NaN (0/0 in the Welford update when the
        pulse weight underflows to zero) and undersample the illumination
        rolloff in the subaperture maps. Both showed up as heavy azimuth
        errors that grew with the number of stages."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.05)
        wa, pos, att = self._straight_track(
            device, nsweeps=512, span=100.0, alt=40.0, r_center=150.0)
        grid = {"r": (80, 220), "theta": (-0.6, 0.6), "nr": 128, "ntheta": 256}
        out, ref = self._run_both(
            wa, g, g_extent, grid, pos, att, normalization="sigma",
            stages=100, downsample_r=4.0, downsample_theta=4.0,
            min_nsweeps=32)
        self.assertFalse(bool(torch.isnan(out).any()))
        out_c = out[4:-4, 4:-4]
        ref_c = ref[4:-4, 4:-4]
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.98)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 1000)
        err = (out_c[mask] / ref_c[mask] - 1).abs()
        self.assertLess(float(err.median()), 0.01)
        # Isolated illumination edge pixels can still be off, quantile only.
        self.assertLess(float(torch.quantile(err, 0.95)), 0.05)

    def test_zero_gain_no_nan(self):
        """Antenna patterns with exact zeros inside the table (measured or
        rotated patterns) and windows with zero endpoints must not produce
        NaN moments in either the direct kernel or the accumulator maps."""
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum

        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.1)
        g = torch.where(g > 1e-3 * g.max(), g, torch.zeros_like(g))
        wa, pos, att = self._straight_track(device, nsweeps=128)
        wa = torch.hann_window(128).to(torch.float32)  # exact zero endpoints
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 64, "ntheta": 128}
        acc = _backprojection_polar_2d_tx_power_accum(
            wa, g, g_extent, grid, pos, att, "sigma", (80 - 40) / 64, 20.0)
        self.assertTrue(bool(torch.isfinite(acc).all()))
        ref = torchbp.ops.backprojection_polar_2d_tx_power(
            wa, g, g_extent, grid, 0.15, pos, att, normalization="sigma")[0]
        self.assertFalse(bool(torch.isnan(ref).any()))
        # Illuminated pixels (W > 0 needs at least two pulses for a nonzero
        # aperture) must be finite in the direct output too.
        self.assertGreater(int((torch.isfinite(ref) & (acc[1] > 0)).sum()), 1000)

    def test_slant(self):
        device = "cpu"
        alt = 30.0
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, alt=0.0, r_center=60.0)
        att[:, 0] = -float(np.arcsin(alt / 60.0))
        grid = {"r": (25, 90), "theta": (-0.5, 0.5), "nr": 96, "ntheta": 128}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", altitude=alt)
        # Shadow zone below nadir must be exactly zero in both.
        r_vec = 25 + (90 - 25) / 96 * torch.arange(96)
        t_vec = -0.5 + 1.0 / 128 * torch.arange(128)
        rg2 = r_vec[:, None] ** 2 * (1 - t_vec[None, :] ** 2) - alt**2
        shadow = rg2 < 0
        self.assertGreater(int(shadow.sum()), 0)
        self.assertTrue((ref[shadow] == 0).all())
        self.assertTrue((out[shadow] == 0).all())
        # The slant range to ground range mapping is singular at the shadow
        # boundary and the accumulator fields cannot be interpolated
        # accurately there. Exclude the steep fold zone (ground range less
        # than half the slant range) in addition to the shadow zone itself.
        steep = rg2 < (r_vec[:, None] / 2) ** 2
        out = torch.where(steep, torch.inf, out)
        ref = torch.where(steep, torch.inf, ref)
        self._assert_matches(out, ref, q95=0.04, emax=0.15)

    def test_normalization_variants(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, nsweeps=64)
        grid = {"r": (40, 80), "theta": (-0.4, 0.4), "nr": 48, "ntheta": 64}
        for normalization in ["beta", "sigma", "gamma", "point"]:
            out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                      normalization=normalization, stages=2)
            self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_azimuth_resolution_false(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", azimuth_resolution=False)
        self.assertTrue(torch.isfinite(out).all())
        self._assert_matches(out, ref, q95=0.02, emax=0.05)

    def test_downsampled_subaperture_maps(self):
        """Coarse subaperture maps must stay reasonably accurate."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma",
                                  downsample_r=4.0, downsample_theta=4.0)
        self._assert_matches(out, ref, q95=0.05, emax=0.2)

    def _merge_exactness(self, device):
        """Merging two half track accumulator maps on the same grid must equal
        the full track accumulator map (Chan's formula is exact)."""
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum
        from torchbp.ops import ffbp_tx_power_merge2

        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device, nsweeps=64)
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 48, "ntheta": 64}
        kw = dict(normalization="sigma", dr_ref=(80 - 40) / 48, h_ref=20.0)
        full = _backprojection_polar_2d_tx_power_accum(
            wa, g, g_extent, grid, pos, att, **kw)
        acc0 = _backprojection_polar_2d_tx_power_accum(
            wa[:32], g, g_extent, grid, pos[:32], att[:32], **kw)
        acc1 = _backprojection_polar_2d_tx_power_accum(
            wa[32:], g, g_extent, grid, pos[32:], att[32:], **kw)
        zero = torch.zeros(3, device=device)
        merged = ffbp_tx_power_merge2(
            acc0, acc1, zero, zero.clone(), [grid, grid], grid)
        # The first and last rows and columns are on the edge of the bilinear
        # interpolation support of the merge kernel where one ulp of rounding
        # in the coordinate computation flips the bounds check.
        torch.testing.assert_close(
            merged[:, 1:-1, 1:-1], full[:, 1:-1, 1:-1], atol=1e-5, rtol=1e-2)

    def test_merge_exactness_cpu(self):
        self._merge_exactness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_merge_exactness_cuda(self):
        self._merge_exactness("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_accum_cpu_cuda(self):
        from torchbp.ops.backproj import _backprojection_polar_2d_tx_power_accum

        for altitude in (0.0, 30.0):
            g, g_extent = self._antenna("cuda", az_width=0.4)
            wa, pos, att = self._straight_track(
                "cuda", nsweeps=64, alt=0.0 if altitude > 0 else 20.0)
            att[:, 0] = -float(np.arcsin(max(altitude, 20.0) / 65.0))
            grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 48, "ntheta": 64}
            kw = dict(normalization="sigma", dr_ref=(80 - 40) / 48,
                      h_ref=altitude if altitude > 0 else 20.0, altitude=altitude)
            out_gpu = _backprojection_polar_2d_tx_power_accum(
                wa, g, g_extent, grid, pos, att, **kw)
            out_cpu = _backprojection_polar_2d_tx_power_accum(
                wa.cpu(), g.cpu(), g_extent, grid, pos.cpu(), att.cpu(), **kw)
            torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_ffbp_tx_power_cpu_cuda(self):
        g, g_extent = self._antenna("cuda", az_width=0.4)
        wa, pos, att = self._straight_track("cuda")
        grid = {"r": (40, 80), "theta": (-0.6, 0.6), "nr": 96, "ntheta": 192}
        kw = dict(stages=3, downsample_r=1.0, downsample_theta=1.0,
                  min_nsweeps=16, normalization="sigma")
        out_gpu = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        out_cpu = torchbp.ops.backprojection_polar_2d_tx_power_ffbp(
            wa.cpu(), g.cpu(), g_extent, grid, 0.15, pos.cpu(), att.cpu(), **kw)
        finite = torch.isfinite(out_cpu) & torch.isfinite(out_gpu.cpu())
        self.assertGreater(float(finite.float().mean()), 0.95)
        torch.testing.assert_close(
            out_cpu[finite], out_gpu.cpu()[finite], atol=1e-3, rtol=1e-2)

    def _opcheck(self, device):
        wa = torch.rand(4, device=device) + 0.5
        pos = torch.randn(4, 3, device=device)
        pos[:, 0] -= 5.0
        att = torch.zeros(4, 3, device=device)
        g = torch.rand(8, 8, device=device)
        opcheck(
            torch.ops.torchbp.backprojection_polar_2d_tx_power_accum,
            (wa, pos, att, g, -1.0, -0.5, 2.0 / 8, 1.0 / 8, 8, 8, 4,
             1.0, 0.5, -0.9, 0.2, 8, 8, 1, 0.5, 10.0, 0.0, 0),
            test_utils=["test_schema"],
        )
        acc0 = torch.rand(4, 8, 8, device=device)
        acc1 = torch.rand(4, 8, 8, device=device)
        dorigin = torch.zeros(2, 3, device=device)
        r0 = torch.ones(2, device=device)
        dr0 = torch.full((2,), 0.5, device=device)
        theta0 = torch.full((2,), -0.9, device=device)
        dtheta0 = torch.full((2,), 0.2, device=device)
        nr0 = torch.full((2,), 8, dtype=torch.int32, device=device)
        ntheta0 = torch.full((2,), 8, dtype=torch.int32, device=device)
        opcheck(
            torch.ops.torchbp.ffbp_tx_power_merge2,
            (acc0, acc1, dorigin, r0, dr0, theta0, dtheta0, nr0, ntheta0,
             1.0, 0.5, -0.9, 0.2, 8, 8, 0.0, 0, 0),
            test_utils=["test_schema"],
        )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    def test_matches_polar_at_coincident_ground_points(self):
        """The Cartesian tx_power kernel maps a pixel's ground position
        directly from the grid; the polar kernel maps (r, theta) to the same
        ground point. Evaluated at coincident ground positions with the
        azimuth-resolution normalization off, they must agree."""
        import math
        from torchbp.ops.backproj import (
            backprojection_cart_2d_tx_power,
            backprojection_polar_2d_tx_power,
        )

        torch.manual_seed(1)
        nsweeps = 32
        wa = torch.rand(nsweeps)
        g = torch.rand(16, 16)
        g_extent = [-1.4, -1.2, 1.4, 1.2]
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-20, 20, nsweeps)
        pos[:, 2] = 100.0
        att = torch.zeros(nsweeps, 3)

        r0, r1, nr = 150.0, 260.0, 40
        t0, t1, nt = -0.3, 0.3, 40
        pgrid = {"r": (r0, r1), "theta": (t0, t1), "nr": nr, "ntheta": nt}
        dr = (r1 - r0) / nr
        dt = (t1 - t0) / nt
        ir, it = 20, 25
        r = r0 + ir * dr
        t = t0 + it * dt
        x = r * math.sqrt(1 - t * t)
        y = r * t
        cgrid = {"x": (x, x + 1.0), "y": (y, y + 1.0), "nx": 1, "ny": 1}

        for norm in ["beta", "sigma", "gamma", "point"]:
            pol = backprojection_polar_2d_tx_power(
                wa, g, g_extent, pgrid, 0.15, pos, att,
                normalization=norm, azimuth_resolution=False)[0]
            cart = backprojection_cart_2d_tx_power(
                wa, g, g_extent, cgrid, 0.15, pos, att,
                normalization=norm, azimuth_resolution=False)[0, 0, 0]
            torch.testing.assert_close(
                cart, pol[ir, it], atol=1e-6, rtol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_match(self):
        from torchbp.ops.backproj import backprojection_cart_2d_tx_power

        args = self.sample_inputs("cpu")[0]
        out_cpu = backprojection_cart_2d_tx_power(**args)
        gargs = {k: (v.cuda() if torch.is_tensor(v) else v)
                 for k, v in args.items()}
        out_cuda = backprojection_cart_2d_tx_power(**gargs).cpu()
        fin = torch.isfinite(out_cpu) & torch.isfinite(out_cuda)
        torch.testing.assert_close(out_cpu[fin], out_cuda[fin],
                                   atol=1e-5, rtol=1e-4)
        self.assertTrue(
            (torch.isinf(out_cpu) == torch.isinf(out_cuda)).all())


class TestCartTxPowerCFBP(TestCase):
    """backprojection_cart_2d_tx_power_cfbp must match the direct
    backprojection_cart_2d_tx_power."""

    @staticmethod
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

    @staticmethod
    def _straight_track(device, nsweeps=128, span=32.0, alt=20.0, r_center=60.0):
        pos = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        pos[:, 1] = torch.linspace(-span / 2, span / 2, nsweeps, device=device)
        pos[:, 2] = alt
        att = torch.zeros([nsweeps, 3], dtype=torch.float32, device=device)
        d = float(np.sqrt(r_center**2 + alt**2))
        att[:, 0] = -float(np.arcsin(alt / d))
        wa = (torch.hann_window(nsweeps, device=device) + 0.1).to(torch.float32)
        return wa, pos, att

    def _assert_matches(self, out, ref, q95=0.03, emax=0.1, margin=4):
        self.assertEqual(out.shape, ref.shape)
        # No illuminated pixel of the direct result may be dropped (turned to
        # inf) by the factorization, including the image border. Guards against
        # the merge losing edge rows/columns for want of interpolation margin.
        missing = torch.isfinite(ref) & ~torch.isfinite(out)
        self.assertEqual(int(missing.sum()), 0)
        out_c = out[margin:-margin, margin:-margin]
        ref_c = ref[margin:-margin, margin:-margin]
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.99)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        self.assertGreater(int(finite.sum()), 0)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 100)
        ratio = out_c[mask] / ref_c[mask]
        err = (ratio - 1).abs()
        self.assertLess(float(torch.quantile(err, 0.95)), q95)
        self.assertLess(float(err.max()), emax)

    def _run_both(self, wa, g, g_extent, grid, pos, att, **kw):
        kw.setdefault("stages", 3)
        kw.setdefault("downsample_x", 1.0)
        kw.setdefault("downsample_y", 1.0)
        kw.setdefault("min_nsweeps", 16)
        ref = torchbp.ops.backprojection_cart_2d_tx_power(
            wa, g, g_extent, grid, 0.15, pos, att,
            normalization=kw.get("normalization"),
            azimuth_resolution=kw.get("azimuth_resolution", True))[0]
        out = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        return out, ref

    def test_straight_track(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        for norm in [None, "sigma", "gamma", "point"]:
            out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                      normalization=norm)
            self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_no_azimuth_resolution(self):
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(device)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        out, ref = self._run_both(wa, g, g_extent, grid, pos, att,
                                  normalization="sigma", azimuth_resolution=False)
        self._assert_matches(out, ref, q95=0.03, emax=0.1)

    def test_downsampled_deep_stages(self):
        """Coarse subaperture maps (downsample 4) merged over many stages must
        stay accurate and free of NaN."""
        device = "cpu"
        g, g_extent = self._antenna(device, az_width=0.4)
        wa, pos, att = self._straight_track(
            device, nsweeps=512, span=60.0, alt=20.0, r_center=60.0)
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 256, "ny": 512}
        out, ref = self._run_both(
            wa, g, g_extent, grid, pos, att, normalization="sigma",
            stages=100, downsample_x=4.0, downsample_y=4.0, min_nsweeps=32)
        self.assertFalse(bool(torch.isnan(out).any()))
        # No illuminated direct pixel dropped, even at the image border.
        self.assertEqual(int((torch.isfinite(ref) & ~torch.isfinite(out)).sum()), 0)
        out_c = out[4:-4, 4:-4]
        ref_c = ref[4:-4, 4:-4]
        inf_agree = (torch.isinf(out_c) == torch.isinf(ref_c)).float().mean()
        self.assertGreater(float(inf_agree), 0.98)
        finite = torch.isfinite(ref_c) & torch.isfinite(out_c)
        thresh = 1e-3 * float(ref_c[torch.isfinite(ref_c)].max())
        mask = finite & (ref_c > thresh)
        self.assertGreater(int(mask.sum()), 1000)
        err = (out_c[mask] / ref_c[mask] - 1).abs()
        self.assertLess(float(err.median()), 0.01)
        # Isolated illumination edge pixels can still be off, quantile only.
        self.assertLess(float(torch.quantile(err, 0.95)), 0.05)

    def _opcheck(self, device):
        wa = torch.rand(4, device=device) + 0.5
        pos = torch.randn(4, 3, device=device)
        pos[:, 2] = 20.0
        att = torch.zeros(4, 3, device=device)
        g = torch.rand(8, 8, device=device)
        opcheck(
            torch.ops.torchbp.backprojection_cart_2d_tx_power_accum,
            (wa, pos, att, g, -1.0, -0.5, 2.0 / 8, 1.0 / 8, 8, 8, 4,
             45.0, 0.25, -25.0, 0.2, 8, 8, 1, 0.25, 20.0),
            test_utils=["test_schema"],
        )
        acc0 = torch.rand(4, 8, 8, device=device)
        acc1 = torch.rand(4, 8, 8, device=device)
        opcheck(
            torch.ops.torchbp.cart_tx_power_merge2,
            (acc0, acc1,
             45.0, 0.25, -25.0, 0.2, 8, 8,
             45.0, 0.25, -25.0, 0.2, 8, 8,
             45.0, 0.25, -25.0, 0.2, 8, 8),
            test_utils=["test_schema"],
        )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_match(self):
        g, g_extent = self._antenna("cuda", az_width=0.4)
        wa, pos, att = self._straight_track("cuda")
        grid = {"x": (45, 75), "y": (-25, 25), "nx": 128, "ny": 256}
        kw = dict(stages=3, downsample_x=1.0, downsample_y=1.0,
                  min_nsweeps=16, normalization="sigma")
        out_gpu = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa, g, g_extent, grid, 0.15, pos, att, **kw)
        out_cpu = torchbp.ops.backprojection_cart_2d_tx_power_cfbp(
            wa.cpu(), g.cpu(), g_extent, grid, 0.15, pos.cpu(), att.cpu(), **kw)
        finite = torch.isfinite(out_cpu) & torch.isfinite(out_gpu.cpu())
        self.assertGreater(float(finite.float().mean()), 0.95)
        torch.testing.assert_close(
            out_cpu[finite], out_gpu.cpu()[finite], atol=1e-3, rtol=1e-2)


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

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
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


class TestCFBP(TestCase):
    """Cartesian factorized backprojection against direct backprojection."""

    fc = 6e9
    r_res = 0.3
    grid = {"x": (60.0, 160.0), "y": (-25.0, 25.0), "nx": 256, "ny": 512}
    sweep_samples = 1024

    def _make_data(self, targets, amps, pos, d0=0.0, data_fmod=0.0):
        """Point responses consistent with the backprojection phase model."""
        c0 = 299792458.0
        data = torch.zeros(
            pos.shape[0], self.sweep_samples, dtype=torch.complex64
        )
        m_idx = torch.arange(pos.shape[0])
        for t, a in zip(targets, amps):
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = (d + d0) / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += a * w[valid] * phase[valid]
        if data_fmod != 0.0:
            mod = torch.exp(1j * data_fmod * torch.arange(self.sweep_samples))
            data = data * mod[None, :]
        return data

    def _scene(self, nsweeps=512):
        torch.manual_seed(5)
        ntargets = 12
        x = 70.0 + 80.0 * torch.rand(ntargets)
        y = -20.0 + 40.0 * torch.rand(ntargets)
        targets = torch.stack([x, y, torch.zeros_like(x)], dim=1)
        amps = (1.0 + torch.rand(ntargets)).to(torch.complex64)
        pos = torch.zeros(nsweeps, 3)
        pos[:, 1] = torch.linspace(-3.0, 3.0, nsweeps)
        pos[:, 2] = 20.0
        return targets, amps, pos

    def _compare(self, device, stages, nsweeps=512, grid=None, d0=0.0,
                 data_fmod=0.0, divisions=2, tol=0.05,
                 interp_method=("knab", 8, 1.4)):
        if grid is None:
            grid = self.grid
        targets, amps, pos = self._scene(nsweeps)
        data = self._make_data(targets, amps, pos, d0=d0, data_fmod=data_fmod)
        data = data.to(device)
        pos = pos.to(device)
        ref = torchbp.ops.backprojection_cart_2d(
            data, grid, self.fc, self.r_res, pos, d0=d0, data_fmod=data_fmod
        )
        out = torchbp.ops.cfbp(
            data, grid, self.fc, self.r_res, pos, stages=stages,
            divisions=divisions, d0=d0, data_fmod=data_fmod,
            interp_method=interp_method
        )
        self.assertEqual(out.shape, ref.shape)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, tol, f"cfbp relative error {rel:.2e} exceeds {tol}")
        return out

    def test_stages1_matches_direct(self):
        self._compare("cpu", stages=1, tol=0.03)

    def test_stages2_matches_direct(self):
        self._compare("cpu", stages=2)

    def test_stages3_matches_direct(self):
        # Enough sweeps that all three stages actually recurse.
        self._compare("cpu", stages=3, nsweeps=1040)

    def test_odd_nsweeps(self):
        # divisions does not divide nsweeps: no sweep may be dropped.
        self._compare("cpu", stages=2, nsweeps=511)

    def test_odd_ny(self):
        self._compare("cpu", stages=2, grid=dict(self.grid, ny=255))
        self._compare("cpu", stages=2, grid=dict(self.grid, ny=250))

    def test_divisions3(self):
        self._compare("cpu", stages=2, nsweeps=1040, divisions=3)

    def test_d0(self):
        self._compare("cpu", stages=2, d0=-0.5)

    def test_data_fmod(self):
        self._compare("cpu", stages=2, data_fmod=-torch.pi / 2)

    def test_grid_object(self):
        from torchbp.grid import CartesianGrid
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        grid_obj = CartesianGrid(
            x_range=self.grid["x"], y_range=self.grid["y"],
            nx=self.grid["nx"], ny=self.grid["ny"],
        )
        out_dict = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2
        )
        out_obj = torchbp.ops.cfbp(
            data, grid_obj, self.fc, self.r_res, pos, stages=2
        )
        torch.testing.assert_close(out_dict, out_obj)

    def test_data_gradient(self):
        targets, amps, pos = self._scene(256)
        data = self._make_data(targets, amps, pos).requires_grad_(True)
        out = torchbp.ops.cfbp(data, self.grid, self.fc, self.r_res, pos, stages=2)
        out.abs().mean().backward()
        self.assertTrue(torch.isfinite(data.grad).all())
        self.assertGreater(data.grad.abs().sum().item(), 0)

    def test_interp_method_fft(self):
        self._compare("cpu", stages=2, interp_method="fft")

    def test_knab_matches_fft(self):
        # The merge kernel should differ from the exact FFT merge only by
        # interpolation error.
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        out_fft = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2,
            interp_method="fft"
        )
        out_knab = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=2,
            interp_method=("knab", 8, 1.4)
        )
        rel = ((out_knab - out_fft).abs().max() / out_fft.abs().max()).item()
        self.assertLess(rel, 1e-2)

    def test_guard_y0(self):
        # Without a guard band the y edges differ (FFT wrap-around vs window
        # truncation); the interior should still match direct backprojection.
        grid = self.grid
        targets, amps, pos = self._scene()
        data = self._make_data(targets, amps, pos)
        ref = torchbp.ops.backprojection_cart_2d(
            data, grid, self.fc, self.r_res, pos
        )
        out = torchbp.ops.cfbp(
            data, grid, self.fc, self.r_res, pos, stages=2, guard_y=0.0
        )
        m = int(0.05 * grid["ny"])
        rel = ((out - ref)[..., m:-m].abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 0.05)

    def _merge2_args(self, device):
        from torchbp.ops.cfbp import _merge_weight_table

        torch.manual_seed(2)
        nb, nx, ny0, ny1, nyout = 2, 16, 100, 130, 260
        img0 = torch.randn(nb, nx, ny0, dtype=torch.complex64, device=device)
        img1 = torch.randn(nb, nx, ny1, dtype=torch.complex64, device=device)
        order, v = 8, round(1 - 1 / 1.4, 6)
        w0, i0 = _merge_weight_table(ny0, nyout, order, v)
        w1, i1 = _merge_weight_table(ny1, nyout, order, v)
        w0, i0, w1, i1 = (t.to(device) for t in (w0, i0, w1, i1))
        return (img0, img1, w0, i0, w1, i1, nb, nx, ny0, ny1, nyout,
                w0.shape[1], w1.shape[1], 0.5, 0.25,
                300.0, -30.2, 50.0, 300.4, -29.8, 50.5, 300.2, -30.0, 50.25,
                80.05)

    def test_merge2_matches_torch_reference(self):
        args = self._merge2_args("cpu")
        (img0, img1, w0, i0, w1, i1, nb, nx, ny0, ny1, nyout, order0, order1,
         dx, dy, ox0, oy0, z0, ox1, oy1, z1, oxp, oyp, zp, ref_phase) = args
        out = torch.ops.torchbp.cfbp_merge2.default(*args)

        x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
        y = torch.arange(nyout, dtype=torch.float64)[None, :] * dy

        def dist(ox, oy, z):
            return torch.sqrt((ox + x).float() ** 2 + (oy + y).float() ** 2 + z**2)

        dp = dist(oxp, oyp, zp)
        ref = 0
        for img, w, idx, (ox, oy, z) in [
            (img0, w0, i0, (ox0, oy0, z0)),
            (img1, w1, i1, (ox1, oy1, z1)),
        ]:
            order = w.shape[1]
            gidx = idx[:, None].long() + torch.arange(order)[None, :]
            g = img[..., gidx.reshape(-1)].reshape(nb, nx, nyout, order)
            interp = (g * w.to(img.dtype)).sum(-1)
            ph = torch.pi * ref_phase * (dist(ox, oy, z) - dp)
            ref = ref + interp * torch.polar(torch.ones_like(ph), ph)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 1e-4)

    def _merge2_opcheck(self, device):
        opcheck(
            torch.ops.torchbp.cfbp_merge2,
            self._merge2_args(device),
            test_utils=["test_schema", "test_autograd_registration", "test_faketensor"]
        )

    def test_merge2_opcheck_cpu(self):
        self._merge2_opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_merge2_opcheck_cuda(self):
        self._merge2_opcheck("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_merge2_cuda_matches_cpu(self):
        out_cuda = torch.ops.torchbp.cfbp_merge2.default(*self._merge2_args("cuda"))
        out_cpu = torch.ops.torchbp.cfbp_merge2.default(*self._merge2_args("cpu"))
        rel = ((out_cuda.cpu() - out_cpu).abs().max() / out_cpu.abs().max()).item()
        self.assertLess(rel, 1e-4)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_matches_direct(self):
        out_cuda = self._compare("cuda", stages=2)
        out_cpu = self._compare("cpu", stages=2)
        rel = ((out_cuda.cpu() - out_cpu).abs().max() / out_cpu.abs().max()).item()
        self.assertLess(rel, 0.05)


class TestCFBPAdaptive(TestCase):
    """Range-adaptive CFBP on a long track with near zero ground range.

    Plain cfbp cannot represent the subaperture images on the subsampled
    grid at near range in this geometry; cfbp_adaptive must still match
    direct backprojection.
    """

    fc = 6e9
    r_res = 0.375
    sweep_samples = 512
    nsweeps = 1024
    grid = {"x": (2.0, 52.0), "y": (-25.0, 25.0), "nx": 128, "ny": 128}

    def _scene(self):
        torch.manual_seed(7)
        c0 = 299792458.0
        pos = torch.zeros(self.nsweeps, 3)
        pos[:, 1] = 0.25 * c0 / self.fc * (
            torch.arange(self.nsweeps) - self.nsweeps / 2
        )
        pos[:, 2] = 10.0
        tx = torch.tensor([3.0, 5.0, 8.0, 14.0, 25.0, 40.0, 50.0])
        ty = torch.tensor([0.0, 15.0, -10.0, 20.0, -20.0, 5.0, 0.0])
        data = torch.zeros(self.nsweeps, self.sweep_samples, dtype=torch.complex64)
        m_idx = torch.arange(self.nsweeps)
        for x, y in zip(tx, ty):
            t = torch.tensor([x, y, 0.0])
            d = torch.linalg.norm(t[None, :] - pos, dim=1)
            sx = d / self.r_res
            phase = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)
            for k in range(-2, 3):
                idx = torch.floor(sx).long() + k
                w = torch.clamp(1.5 - (idx.float() - sx).abs(), 0, 1)
                valid = (idx >= 0) & (idx < self.sweep_samples)
                data[m_idx[valid], idx[valid]] += w[valid] * phase[valid]
        return data, pos

    def test_matches_direct_near_range(self):
        data, pos = self._scene()
        ref = torchbp.ops.backprojection_cart_2d(
            data, self.grid, self.fc, self.r_res, pos
        )
        out = torchbp.ops.cfbp_adaptive(
            data, self.grid, self.fc, self.r_res, pos, stages=4,
            data_oversample=1.2
        )
        self.assertEqual(out.shape, ref.shape)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        self.assertLess(rel, 0.02, f"cfbp_adaptive relative error {rel:.2e}")
        # Plain cfbp must be much worse in this geometry, otherwise the
        # scene does not actually stress the near range.
        out_plain = torchbp.ops.cfbp(
            data, self.grid, self.fc, self.r_res, pos, stages=4
        )
        rel_plain = ((out_plain - ref).abs().max() / ref.abs().max()).item()
        self.assertGreater(rel_plain, 10 * rel)

    def test_blocks(self):
        _, pos = self._scene()
        blocks = torchbp.ops.cfbp_adaptive_blocks(
            self.grid, pos, self.fc, self.r_res, stages=4, data_oversample=1.2
        )
        # Blocks cover all rows contiguously and density decreases with range.
        self.assertEqual(blocks[0][0], 0)
        self.assertEqual(blocks[-1][1], self.grid["nx"])
        for b, b_next in zip(blocks, blocks[1:]):
            self.assertEqual(b[1], b_next[0])
        ks = [b[2] for b in blocks]
        self.assertEqual(ks, sorted(ks, reverse=True))
        self.assertGreater(ks[0], ks[-1])


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


class TestFFBPLongBaseline(TestCase):
    """On a long baseline the subapertures away from the merge origin see the
    far/near scene at a shifted range (largest toward |theta| = 1). Their
    grids must extend in range so the merge lookups stay inside them,
    otherwise edge targets at high |theta| lose the contribution of the
    offset subapertures and dim. See the range guard band in _ffbp_impl."""

    fc = 6e9
    r_res = 0.5
    nsamples = 600
    nsweeps = 1024

    def _scene(self, grid, targets, device="cpu"):
        c0 = 299792458.0
        lam = c0 / self.fc
        pos = torch.zeros((self.nsweeps, 3), device=device)
        # Aperture ~25 m: offset of the top-level subapertures from the merge
        # origin is a several-meter fraction of the ~100 m range, enough to
        # push rp off the grid by many range bins at |theta| ~ 0.8.
        pos[:, 1] = (lam / 4) * (torch.arange(self.nsweeps, device=device)
                                 - self.nsweeps / 2)
        data = torch.zeros((self.nsweeps, self.nsamples),
                           dtype=torch.complex64, device=device)
        i = torch.arange(self.nsamples, dtype=torch.float64, device=device)
        for r, th in targets:
            tx = r * np.sqrt(1 - th * th)
            ty = r * th
            d = torch.sqrt((pos[:, 0].double() - tx) ** 2
                           + (pos[:, 1].double() - ty) ** 2
                           + pos[:, 2].double() ** 2)
            env = torch.special.sinc(
                (i[None, :] * self.r_res - d[:, None]) / (2 * self.r_res))
            ph = torch.exp(-1j * 4 * torch.pi * self.fc / c0 * d)[:, None]
            data += (env * ph).to(torch.complex64)
        return data, pos

    def _peaks(self, ref, out, grid, targets):
        (r0, r1), (t0, t1) = grid["r"], grid["theta"]
        dr = (r1 - r0) / grid["nr"]
        dth = (t1 - t0) / grid["ntheta"]
        ratios = []
        for r, th in targets:
            ri = int(round((r - r0) / dr))
            ti = int(round((th - t0) / dth))
            a = ref[max(0, ri - 3):ri + 4, max(0, ti - 3):ti + 4].abs().max()
            b = out[max(0, ri - 3):ri + 4, max(0, ti - 3):ti + 4].abs().max()
            ratios.append((b / a).item())
        return ratios

    def test_edge_targets_recovered_cpu(self):
        # Targets pinned to the range edges at high |theta|, where an offset
        # subaperture's rp leaves the [r0, r1] grid without the range guard.
        c0 = 299792458.0
        lam = c0 / self.fc
        aperture = (lam / 4) * (self.nsweeps - 1)
        r0f, r1f, th0, th1 = 100.0, 180.0, -0.85, 0.85
        # Sample azimuth finely enough for the aperture (else the scene
        # aliases regardless of the merge geometry).
        ntheta = int(np.ceil((th1 - th0) / (lam / (2 * aperture))))
        grid = {"r": (r0f, r1f), "theta": (th0, th1), "nr": 200,
                "ntheta": ntheta}
        targets = [(140.0, 0.0), (178.0, 0.82), (178.0, -0.82),
                   (102.0, 0.82), (102.0, -0.82)]
        data, pos = self._scene(grid, targets)
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos)[0]
        out = torchbp.ops.ffbp(
            data, grid, self.fc, self.r_res, pos, stages=6, grid_oversample=2.0)
        ratios = self._peaks(ref, out, grid, targets)
        # Broadside (no range shift) is a control; the edge targets are the
        # ones the range guard rescues. Without the guard they sit near 0.58.
        for (r, th), ratio in zip(targets, ratios):
            self.assertGreater(ratio, 0.9,
                               f"target r={r} th={th} dimmed to {ratio:.3f}")

    def test_odd_divisions_cpu(self):
        # The pairwise merge chain must land on the node centroid and merge
        # adjacent subapertures for ANY divisions, not just powers of two.
        # Odd/non-power-of-two divisions previously drifted the output frame
        # (0.5/0.5 origin averaging) and merged non-adjacent subapertures
        # (gapped aperture -> azimuth aliasing), collapsing targets (e.g.
        # divisions=3 broadside to 0.13, divisions=5 to 0.31). The result must
        # not depend on divisions: every count should match divisions=2.
        r0f, r1f, th0, th1 = 100.0, 180.0, -0.3, 0.3
        grid = {"r": (r0f, r1f), "theta": (th0, th1), "nr": 200, "ntheta": 256}
        # Broadside (th=0) has no range shift, so it isolates the frame-drift
        # and merge-order bugs from the range-guard geometry.
        targets = [(140.0, 0.0), (120.0, 0.2), (170.0, -0.22), (110.0, 0.25)]
        data, pos = self._scene(grid, targets)
        ref = torchbp.ops.backprojection_polar_2d(
            data, grid, self.fc, self.r_res, pos)[0]
        base = self._peaks(ref, torchbp.ops.ffbp(
            data, grid, self.fc, self.r_res, pos, stages=5, divisions=2,
            grid_oversample=2.0), grid, targets)
        for divisions in (3, 4, 5, 7):
            out = torchbp.ops.ffbp(
                data, grid, self.fc, self.r_res, pos, stages=5,
                divisions=divisions, grid_oversample=2.0)
            ratios = self._peaks(ref, out, grid, targets)
            for (r, th), ratio, base_ratio in zip(targets, ratios, base):
                self.assertAlmostEqual(
                    ratio, base_ratio, delta=0.05,
                    msg=f"divisions={divisions} target r={r} th={th}: peak "
                        f"{ratio:.3f} vs divisions=2 {base_ratio:.3f}")


if __name__ == "__main__":
    unittest.main()
