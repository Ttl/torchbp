#!/usr/bin/env python
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import torchbp
from torch import Tensor
import torch.nn.functional as F
from random import uniform
from conftest import requires_cuda


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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
    def test_opcheck_div_cuda(self):
        self._opcheck_div("cuda")

    def test_opcheck_mul_cpu(self):
        self._opcheck_mul("cpu")

    @requires_cuda
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

    @requires_cuda
    def test_cpu_cuda_div(self):
        for args in self.sample_inputs_div("cpu"):
            img1, img2 = args["img1"], args["img2"]
            out_cpu = torchbp.ops.div_2d_interp_linear(img1, img2)
            out_gpu = torchbp.ops.div_2d_interp_linear(img1.cuda(), img2.cuda()).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)

    @requires_cuda
    def test_cpu_cuda_mul(self):
        for args in self.sample_inputs_mul("cpu"):
            img1, img2 = args["img1"], args["img2"]
            out_cpu = torchbp.ops.mul_2d_interp_linear(img1, img2)
            out_gpu = torchbp.ops.mul_2d_interp_linear(img1.cuda(), img2.cuda()).cpu()
            torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)

    @requires_cuda
    def test_cpu_cuda_mul_mixed_dtype(self):
        """Complex image times a real weight map (the polarimetry calibration path)."""
        a = torch.randn(2, 5, 5, dtype=torch.complex64)
        b = torch.randn(2, 3, 3, dtype=torch.float32)
        out_cpu = torchbp.ops.mul_2d_interp_linear(a, b)
        out_gpu = torchbp.ops.mul_2d_interp_linear(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-4, atol=1e-5)


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

    @requires_cuda
    def test_poly_vs_knab_reference(self):
        self._poly_vs_knab_reference("cuda")

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
    def test_basic_cuda(self):
        self._test_basic("cuda")

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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
                       z0, order, alias_mode, alias_fmod, None)

            opcheck(
                torch.ops.torchbp.ffbp_merge2_lanczos,
                cpp_args,
                test_utils=["test_schema"]
            )

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @requires_cuda
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


