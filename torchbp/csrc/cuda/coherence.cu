#include "util.h"

namespace torchbp {

__global__ void coherence_2d_kernel(
          const complex64_t* img0,
          const complex64_t* img1,
          float *out,
          int N0,
          int N1,
          int w0,
          int w1) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = id0 % N1;
    const int idy = id0 / N1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id0 >= N0 * N1) {
        return;
    }

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = max(-w0, -idx);
    int end_i = min(w0, N1 - 1 - idx);
    int start_j = max(-w1, -idy);
    int end_j = min(w1, N0 - 1 - idy);
    for (int y=idy + start_j; y <= idy + end_j; y++) {
        for (int x=idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * cuda::std::conj(v1);
            Navg += 1;
        }
    }
    float v;
    corr /= Navg;
    p0 /= Navg;
    p1 /= Navg;
    v = abs(corr) / sqrtf(p0 * p1);
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}


__global__ void coherence_2d_grad_kernel(
          const float* grad,
          const complex64_t* img0,
          const complex64_t* img1,
          int N0,
          int N1,
          int w0,
          int w1,
          complex64_t* img0_grad,
          complex64_t* img1_grad) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = id0 % N1;
    const int idy = id0 / N1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id0 >= N0 * N1) {
        return;
    }

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = max(-w0, -idx);
    int end_i = min(w0, N1 - 1 - idx);
    int start_j = max(-w1, -idy);
    int end_j = min(w1, N0 - 1 - idy);
    for (int y=idy + start_j; y <= idy + end_j; y++) {
        for (int x=idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * cuda::std::conj(v1);
            Navg += 1;
        }
    }
    corr /= Navg;
    p0 /= Navg;
    p1 /= Navg;
    float dout2 = (corr.real()*corr.real() + corr.imag()*corr.imag()) / (p0 * p1);
    float dout = sqrtf(dout2);

    float dout_ddout2 = grad[idbatch * N0 * N1 + idy * N1 + idx] * 1.0f/dout;
    complex64_t dout_dc = dout_ddout2 * corr/(p0 * p1) / static_cast<float>(Navg);
    float dout_dnorm1 = dout_ddout2 * (-dout2 / p0) / static_cast<float>(Navg);
    float dout_dnorm2 = dout_ddout2 * (-dout2 / p1) / static_cast<float>(Navg);

    for (int y=idy + start_j; y <= idy + end_j; y++) {
        for (int x=idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];

            if (img0_grad != nullptr) {
                complex64_t dout_c_dx1 = dout_dc * v1;
                complex64_t dout_norm1_dx1 = dout_dnorm1 * v0;
                complex64_t gx1 = dout_c_dx1 + dout_norm1_dx1;
                float2 *g0 = (float2*)&img0_grad[idbatch * N0 * N1 + y * N1 + x];
                atomicAdd(&g0->x, gx1.real());
                atomicAdd(&g0->y, gx1.imag());
            }
            if (img1_grad != nullptr) {
                complex64_t dout_c_dx2 = cuda::std::conj(dout_dc) * v0;
                complex64_t dout_norm2_dx2 = dout_dnorm2 * v1;
                complex64_t gx2 = dout_c_dx2 + dout_norm2_dx2;
                float2 *g1 = (float2*)&img1_grad[idbatch * N0 * N1 + y * N1 + x];
                atomicAdd(&g1->x, gx2.real());
                atomicAdd(&g1->y, gx2.imag());
            }
        }
    }
}


at::Tensor coherence_2d_cuda(
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1) {
	TORCH_CHECK(img0.dtype() == at::kComplexFloat);
	TORCH_CHECK(img1.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
	at::Tensor img0_contig = img0.contiguous();
	at::Tensor img1_contig = img1.contiguous();
	c10::complex<float>* img0_ptr = img0_contig.data_ptr<c10::complex<float>>();
	c10::complex<float>* img1_ptr = img1_contig.data_ptr<c10::complex<float>>();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
	at::Tensor out = torch::empty({nbatch, N0, N1}, options);

	float* out_ptr = out.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (N0 * N1 + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    coherence_2d_kernel<<<block_count, thread_per_block>>>(
          (complex64_t*)img0_ptr,
          (complex64_t*)img1_ptr,
          out_ptr,
          N0,
          N1,
          w0,
          w1);

	return out;
}


std::vector<at::Tensor> coherence_2d_grad_cuda(
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
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
	at::Tensor img0_contig = img0.contiguous();
	at::Tensor img1_contig = img1.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	float* grad_ptr = grad_contig.data_ptr<float>();
	c10::complex<float>* img0_ptr = img0_contig.data_ptr<c10::complex<float>>();
	c10::complex<float>* img1_ptr = img1_contig.data_ptr<c10::complex<float>>();

	at::Tensor img0_grad;
	at::Tensor img1_grad;

	c10::complex<float>* img0_grad_ptr = nullptr;
	c10::complex<float>* img1_grad_ptr = nullptr;

    if (img0.requires_grad()) {
        img0_grad = torch::zeros_like(img0);
        img0_grad_ptr = img0_grad.data_ptr<c10::complex<float>>();
    } else {
        img0_grad = torch::Tensor();
    }

    if (img1.requires_grad()) {
        img1_grad = torch::zeros_like(img0);
        img1_grad_ptr = img1_grad.data_ptr<c10::complex<float>>();
    } else {
        img1_grad = torch::Tensor();
    }

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (N0 * N1 + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    coherence_2d_grad_kernel<<<block_count, thread_per_block>>>(
          grad_ptr,
          (complex64_t*)img0_ptr,
          (complex64_t*)img1_ptr,
          N0,
          N1,
          w0,
          w1,
          (complex64_t*)img0_grad_ptr,
          (complex64_t*)img1_grad_ptr);

    std::vector<at::Tensor> ret;
    ret.push_back(img0_grad);
    ret.push_back(img1_grad);
	return ret;
}


__global__ void power_coherence_2d_kernel(
          const complex64_t* img0,
          const complex64_t* img1,
          float *out,
          int N0,
          int N1,
          int w0,
          int w1,
          bool corr_output) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = id0 % N1;
    const int idy = id0 / N1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id0 >= N0 * N1) {
        return;
    }

    float corr = 0.0f;
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = max(-w0, -idx);
    int end_i = min(w0, N1 - 1 - idx);
    int start_j = max(-w1, -idy);
    int end_j = min(w1, N0 - 1 - idy);
    for (int y=idy + start_j; y <= idy + end_j; y++) {
        for (int x=idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            float v0_abs2 = v0.real() * v0.real() + v0.imag() * v0.imag();
            float v1_abs2 = v1.real() * v1.real() + v1.imag() * v1.imag();
            p0 += v0_abs2 * v0_abs2;
            p1 += v1_abs2 * v1_abs2;
            corr += v0_abs2 * v1_abs2;
            Navg += 1;
        }
    }
    float v;
    corr /= Navg;
    p0 /= Navg;
    p1 /= Navg;
    v = corr / sqrtf(p0 * p1);
    if (corr_output) {
        v = v > 0.5f ? sqrtf(2*v - 1) : 0.0f;
    }
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}


at::Tensor power_coherence_2d_cuda(
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
	TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
	at::Tensor img0_contig = img0.contiguous();
	at::Tensor img1_contig = img1.contiguous();
	c10::complex<float>* img0_ptr = img0_contig.data_ptr<c10::complex<float>>();
	c10::complex<float>* img1_ptr = img1_contig.data_ptr<c10::complex<float>>();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
	at::Tensor out = torch::empty({nbatch, N0, N1}, options);

	float* out_ptr = out.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (N0 * N1 + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    power_coherence_2d_kernel<<<block_count, thread_per_block>>>(
          (complex64_t*)img0_ptr,
          (complex64_t*)img1_ptr,
          out_ptr,
          N0,
          N1,
          w0,
          w1,
          corr_output);

	return out;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("coherence_2d", &coherence_2d_cuda);
  m.impl("coherence_2d_grad", &coherence_2d_grad_cuda);
  m.impl("power_coherence_2d", &power_coherence_2d_cuda);
}

}
