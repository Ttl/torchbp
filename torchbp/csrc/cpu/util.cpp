#include "util.h"

// CPU element-wise interp-combine ops. Mirrors cuda/util.cu.
namespace torchbp {

// Element-wise a (*|/) interp2d(b). Output grid [Na0, Na1] is mapped onto the
// input grid [Nb0, Nb1] with bilinear interpolation, matching the CUDA kernel
// in cuda/util.cu bit-for-bit.
template<typename T, typename T2>
static void interp2d_combine_kernel_cpu(const T* a, const T2* b, T* out,
        int Na0, int Na1, int Nb0, int Nb1, bool is_div, int idx, int idbatch) {
    const int id0 = idx / Na1;
    const int id1 = idx % Na1;

    if (id0 >= Na0 || id1 >= Na1) return;

    // Map from output grid [0, Na0-1] x [0, Na1-1] to input grid [0, Nb0-1] x [0, Nb1-1]
    float b0_float = (float)id0 * (Nb0 - 1) / (Na0 - 1);
    float b1_float = (float)id1 * (Nb1 - 1) / (Na1 - 1);

    int b0_int = (int)floorf(b0_float);
    int b1_int = (int)floorf(b1_float);
    float b0_frac = b0_float - b0_int;
    float b1_frac = b1_float - b1_int;

    // Clamp to valid range
    b0_int = std::min(b0_int, Nb0 - 2);
    b1_int = std::min(b1_int, Nb1 - 2);
    b0_int = std::max(b0_int, 0);
    b1_int = std::max(b1_int, 0);

    const T2 v = interp2d<T2>(&b[idbatch * Nb1 * Nb0], Nb0, Nb1, b0_int, b0_frac, b1_int, b1_frac);
    const int oidx = idbatch * Na1 * Na0 + id0 * Na1 + id1;
    if (is_div) {
        out[oidx] = a[oidx] / static_cast<T>(v);
    } else {
        out[oidx] = a[oidx] * static_cast<T>(v);
    }
}

static at::Tensor interp2d_combine_cpu(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1,
          bool is_div) {
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor out = torch::zeros({nbatch, Na0, Na1}, a_contig.options());

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    const bool a_complex = a.scalar_type() == at::ScalarType::ComplexFloat;
    const bool b_complex = b.scalar_type() == at::ScalarType::ComplexFloat;

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int idx = 0; idx < Na0 * Na1; idx++) {
            if (a_complex && b_complex) {
                interp2d_combine_kernel_cpu<complex64_t, complex64_t>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        (const complex64_t*)b_contig.data_ptr<c10::complex<float>>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (a_complex && !b_complex) {
                interp2d_combine_kernel_cpu<complex64_t, float>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        b_contig.data_ptr<float>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (!a_complex && !b_complex) {
                interp2d_combine_kernel_cpu<float, float>(
                        a_contig.data_ptr<float>(),
                        b_contig.data_ptr<float>(),
                        out.data_ptr<float>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else {
                AT_ERROR("Unsupported dtype combination for interp2d combine");
            }
        }
    }
    return out;
}

at::Tensor div_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/true);
}

at::Tensor mul_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/false);
}

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("div_2d_interp_linear", &div_2d_interp_linear_cpu);
  m.impl("mul_2d_interp_linear", &mul_2d_interp_linear_cpu);
}

}
