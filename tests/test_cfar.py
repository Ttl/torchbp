#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import torchbp
from conftest import requires_cuda


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

    @requires_cuda
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


