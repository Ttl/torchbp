#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp


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


