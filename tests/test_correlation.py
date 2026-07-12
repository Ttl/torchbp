#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp


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


