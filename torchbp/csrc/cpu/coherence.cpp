#include "util.h"

// CPU coherence ops. Mirrors cuda/coherence.cu.
namespace torchbp {

// "Quick and dirty" power coherence over a moving window. Mirrors
// power_coherence_2d_kernel in cuda/coherence.cu.
static void power_coherence_2d_kernel_cpu(
          const complex64_t* img0, const complex64_t* img1, float* out,
          int N0, int N1, int w0, int w1, bool corr_output, int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    float corr = 0.0f;
    float p0 = 0.0f;
    float p1 = 0.0f;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            float v0_abs2 = v0.real() * v0.real() + v0.imag() * v0.imag();
            float v1_abs2 = v1.real() * v1.real() + v1.imag() * v1.imag();
            p0 += v0_abs2 * v0_abs2;
            p1 += v1_abs2 * v1_abs2;
            corr += v0_abs2 * v1_abs2;
        }
    }
    float denom = sqrtf(p0 * p1);
    float v = denom > 0.0f ? corr / denom : 0.0f;
    if (corr_output) {
        v = v > 0.5f ? sqrtf(2.0f * v - 1.0f) : 0.0f;
    }
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}

at::Tensor power_coherence_2d_cpu(
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1,
          int64_t corr_output) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
    at::Tensor out = torch::empty({nbatch, N0, N1}, options);

    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();
    float* out_ptr = out.data_ptr<float>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            power_coherence_2d_kernel_cpu(
                    img0_ptr, img1_ptr, out_ptr,
                    N0, N1, w0, w1, (bool)corr_output, id0, idbatch);
        }
    }
    return out;
}
// Interferometric coherence over a moving window. Mirrors coherence_2d_kernel
// in cuda/coherence.cu.
static void coherence_2d_kernel_cpu(
          const complex64_t* img0, const complex64_t* img1, float* out,
          int N0, int N1, int w0, int w1, int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * std::conj(v1);
            Navg += 1;
        }
    }
    corr /= static_cast<float>(Navg);
    p0 /= static_cast<float>(Navg);
    p1 /= static_cast<float>(Navg);
    float denom = sqrtf(p0 * p1);
    float v = denom > 0.0f ? std::abs(corr) / denom : 0.0f;
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}

at::Tensor coherence_2d_cpu(
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
    at::Tensor out = torch::empty({nbatch, N0, N1}, options);

    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();
    float* out_ptr = out.data_ptr<float>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            coherence_2d_kernel_cpu(
                    img0_ptr, img1_ptr, out_ptr, N0, N1, w0, w1, id0, idbatch);
        }
    }
    return out;
}

// Gradient of coherence_2d. Scatters the per-output-pixel gradient back to
// every input pixel in the window. Mirrors coherence_2d_grad_kernel in
// cuda/coherence.cu (CUDA atomicAdd -> "#pragma omp atomic" here).
static void coherence_2d_grad_kernel_cpu(
          const float* grad, const complex64_t* img0, const complex64_t* img1,
          int N0, int N1, int w0, int w1,
          complex64_t* img0_grad, complex64_t* img1_grad,
          int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * std::conj(v1);
            Navg += 1;
        }
    }
    corr /= static_cast<float>(Navg);
    p0 /= static_cast<float>(Navg);
    p1 /= static_cast<float>(Navg);
    float p0p1 = p0 * p1;
    if (p0p1 <= 0.0f) {
        return;
    }
    float dout2 = (corr.real() * corr.real() + corr.imag() * corr.imag()) / p0p1;
    float dout = sqrtf(dout2);

    float dout_ddout2 = dout > 0.0f
        ? grad[idbatch * N0 * N1 + idy * N1 + idx] / dout
        : 0.0f;
    complex64_t dout_dc = dout_ddout2 * corr / p0p1 / static_cast<float>(Navg);
    float dout_dnorm1 = dout_ddout2 * (-dout2 / p0) / static_cast<float>(Navg);
    float dout_dnorm2 = dout_ddout2 * (-dout2 / p1) / static_cast<float>(Navg);

    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];

            if (img0_grad != nullptr) {
                complex64_t gx1 = dout_dc * v1 + dout_dnorm1 * v0;
                float* g0 = reinterpret_cast<float*>(&img0_grad[idbatch * N0 * N1 + y * N1 + x]);
                #pragma omp atomic
                g0[0] += gx1.real();
                #pragma omp atomic
                g0[1] += gx1.imag();
            }
            if (img1_grad != nullptr) {
                complex64_t gx2 = std::conj(dout_dc) * v0 + dout_dnorm2 * v1;
                float* g1 = reinterpret_cast<float*>(&img1_grad[idbatch * N0 * N1 + y * N1 + x]);
                #pragma omp atomic
                g1[0] += gx2.real();
                #pragma omp atomic
                g1[1] += gx2.imag();
            }
        }
    }
}

std::vector<at::Tensor> coherence_2d_grad_cpu(
          const at::Tensor &grad,
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1) {
    TORCH_CHECK(grad.dtype() == at::kFloat);
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor grad_contig = grad.contiguous();

    const float* grad_ptr = grad_contig.data_ptr<float>();
    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();

    at::Tensor img0_grad;
    at::Tensor img1_grad;
    complex64_t* img0_grad_ptr = nullptr;
    complex64_t* img1_grad_ptr = nullptr;

    if (img0.requires_grad()) {
        img0_grad = torch::zeros_like(img0);
        img0_grad_ptr = (complex64_t*)img0_grad.data_ptr<c10::complex<float>>();
    } else {
        img0_grad = torch::Tensor();
    }
    if (img1.requires_grad()) {
        img1_grad = torch::zeros_like(img1);
        img1_grad_ptr = (complex64_t*)img1_grad.data_ptr<c10::complex<float>>();
    } else {
        img1_grad = torch::Tensor();
    }

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            coherence_2d_grad_kernel_cpu(
                    grad_ptr, img0_ptr, img1_ptr, N0, N1, w0, w1,
                    img0_grad_ptr, img1_grad_ptr, id0, idbatch);
        }
    }

    std::vector<at::Tensor> ret;
    ret.push_back(img0_grad);
    ret.push_back(img1_grad);
    return ret;
}

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("coherence_2d", &coherence_2d_cpu);
  m.impl("coherence_2d_grad", &coherence_2d_grad_cpu);
  m.impl("power_coherence_2d", &power_coherence_2d_cpu);
}

}
