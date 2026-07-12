#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp


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


