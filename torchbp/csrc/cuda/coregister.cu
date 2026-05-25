#include "util.h"

namespace torchbp {

__global__ void subpixel_correlation_kernel(
          const complex64_t* im_m,
          const complex64_t* im_s,
          const complex64_t* mean_m,
          const complex64_t* mean_s,
          float* out_a,
          complex64_t* out_b,
          float* out_c,
          int N0,
          int N1) {
    const int idt = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idt % (N1-1);
    const int idx = idt / (N1-1);
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= N0-1 || idy >= N1-1) {
        return;
    }

    complex64_t mm = mean_m[idbatch];
    complex64_t ms = mean_s[idbatch];

    complex64_t s00 = im_s[idbatch * N0 * N1 + idx * N1 + idy] - ms;
    complex64_t s01 = im_s[idbatch * N0 * N1 + idx * N1 + idy + 1] - ms;
    complex64_t s10 = im_s[idbatch * N0 * N1 + (idx + 1) * N1 + idy] - ms;
    complex64_t s11 = im_s[idbatch * N0 * N1 + (idx + 1) * N1 + idy + 1] - ms;

    complex64_t vm = im_m[idbatch * N0 * N1 + idx * N1 + idy] - mm;
    complex64_t a0 = s00;
    complex64_t a1 = s00 - s01;
    complex64_t a2 = s00 - s10;
    complex64_t a3 = s00 - s10 - s01 + s11;

    complex64_t b[4];
    b[0] = vm * cuda::std::conj(a0);
    b[1] = vm * cuda::std::conj(a1);
    b[2] = vm * cuda::std::conj(a2);
    b[3] = vm * cuda::std::conj(a3);

    float r[9];
    r[0] = cuda::std::real(a0 * cuda::std::conj(a0));
    r[1] = cuda::std::real(a0 * cuda::std::conj(a1) + cuda::std::conj(a0) * a1);
    r[2] = cuda::std::real(a0 * cuda::std::conj(a2) + cuda::std::conj(a0) * a2);
    r[3] = cuda::std::real(a1 * cuda::std::conj(a1));
    r[4] = cuda::std::real(a2 * cuda::std::conj(a2));
    r[5] = cuda::std::real(a0 * cuda::std::conj(a3) + cuda::std::conj(a0) * a3 +
                       a1 * cuda::std::conj(a2) + cuda::std::conj(a1) * a2);
    r[6] = cuda::std::real(a1 * cuda::std::conj(a3) + cuda::std::conj(a1) * a3);
    r[7] = cuda::std::real(a2 * cuda::std::conj(a3) + cuda::std::conj(a2) * a3);
    r[8] = cuda::std::real(a3 * cuda::std::conj(a3));

    float c = vm.real() * vm.real() + vm.imag() * vm.imag();

    float2 *out_b2 = (float2*)&out_b[idbatch * 4];
    float *out_a2 = &out_a[idbatch * 9];
    for (int i=0; i < 4; i++) {
        atomicAdd(&out_b2[i].x, b[i].real());
        atomicAdd(&out_b2[i].y, b[i].imag());
    }

    for (int i=0; i < 9; i++) {
        atomicAdd(&out_a2[i], r[i]);
    }

    atomicAdd(&out_c[idbatch], c);
}


std::vector<at::Tensor> subpixel_correlation_cuda(
          const at::Tensor &im_m,
          const at::Tensor &im_s,
          const at::Tensor &mean_m,
          const at::Tensor &mean_s,
          int64_t nbatch,
          int64_t N0,
          int64_t N1) {
	TORCH_CHECK(im_m.dtype() == at::kComplexFloat);
	TORCH_CHECK(im_s.dtype() == at::kComplexFloat);
	TORCH_CHECK(mean_m.dtype() == at::kComplexFloat);
	TORCH_CHECK(mean_s.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(im_m.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(im_s.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(mean_m.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(mean_s.device().type() == at::DeviceType::CUDA);

	at::Tensor im_m_contig = im_m.contiguous();
	at::Tensor im_s_contig = im_s.contiguous();
	at::Tensor mean_m_contig = mean_m.contiguous();
	at::Tensor mean_s_contig = mean_s.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(im_m.device());
    auto options_r =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(im_m.device());
	at::Tensor out_a = torch::zeros({nbatch, 9}, options_r);
	at::Tensor out_b = torch::zeros({nbatch, 4}, options);
	at::Tensor out_c = torch::zeros({nbatch}, options_r);

    c10::complex<float>* im_m_ptr = im_m_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* im_s_ptr = im_s_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* mean_m_ptr = mean_m_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* mean_s_ptr = mean_s_contig.data_ptr<c10::complex<float>>();
    float* out_a_ptr = out_a.data_ptr<float>();
    c10::complex<float>* out_b_ptr = out_b.data_ptr<c10::complex<float>>();
    float* out_c_ptr = out_c.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = (N0 - 1) * (N1 - 1);
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

   subpixel_correlation_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  (complex64_t*)im_m_ptr,
                  (complex64_t*)im_s_ptr,
                  (complex64_t*)mean_m_ptr,
                  (complex64_t*)mean_s_ptr,
                  out_a_ptr,
                  (complex64_t*)out_b_ptr,
                  out_c_ptr,
                  N0, N1);

    std::vector<at::Tensor> ret;
    ret.push_back(out_a);
    ret.push_back(out_b);
    ret.push_back(out_c);
	return ret;
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("subpixel_correlation", &subpixel_correlation_cuda);
}

}
