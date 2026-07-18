#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp
from torch import Tensor
from conftest import requires_cuda


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

    @requires_cuda
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

    @requires_cuda
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


