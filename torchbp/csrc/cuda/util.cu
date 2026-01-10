#include "util.h"

namespace torchbp {

template<typename T, typename T2>
__global__ void div_2d_interp_linear_kernel(
          const T* a,
          const T2* b,
          T* out,
          const int Na0,
          const int Na1,
          const int Nb0,
          const int Nb1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int id0 = idx / Na1;
    const int id1 = idx % Na1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (id0 >= Na0 || id1 >= Na1) return;

    // Calculate interpolation indices
    // Map from output grid [0, Na0-1] x [0, Na1-1] to input grid [0, Nb0-1] x [0, Nb1-1]
    float b0_float = (float)id0 * (Nb0 - 1) / (Na0 - 1);
    float b1_float = (float)id1 * (Nb1 - 1) / (Na1 - 1);

    int b0_int = (int)floorf(b0_float);
    int b1_int = (int)floorf(b1_float);
    float b0_frac = b0_float - b0_int;
    float b1_frac = b1_float - b1_int;

    // Clamp to valid range
    b0_int = min(b0_int, Nb0 - 2);
    b1_int = min(b1_int, Nb1 - 2);
    b0_int = max(b0_int, 0);
    b1_int = max(b1_int, 0);

    const T2 v = interp2d<T2>(&b[idbatch * Nb1 * Nb0], Nb0, Nb1, b0_int, b0_frac, b1_int, b1_frac);
    out[idbatch * Na1 * Na0 + id0 * Na1 + id1] = a[idbatch * Na1 * Na0 + id0 * Na1 + id1] / (T)v;
}

at::Tensor div_2d_interp_linear_cuda(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1) {
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(a.dtype())
        .layout(torch::kStrided)
        .device(a.device());
	at::Tensor out = torch::zeros({nbatch, Na0, Na1}, options);

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Na0 * Na1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Float && b.scalar_type() == at::ScalarType::Float) {
        div_2d_interp_linear_kernel<float, float><<<block_count, thread_per_block, 0, stream>>>(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            Na0, Na1, Nb0, Nb1);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::ComplexFloat) {
        div_2d_interp_linear_kernel<complex64_t, complex64_t><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const complex64_t*>(b_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::Float) {
        div_2d_interp_linear_kernel<complex64_t, float><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            b_contig.data_ptr<float>(),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1);
    } else {
        AT_ERROR("Unsupported dtype combination for div_2d_interp_linear");
    }

	return out;
}

template<typename T, typename T2>
__global__ void mul_2d_interp_linear_kernel(
          const T* a,
          const T2* b,
          T* out,
          const int Na0,
          const int Na1,
          const int Nb0,
          const int Nb1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int id0 = idx / Na1;
    const int id1 = idx % Na1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (id0 >= Na0 || id1 >= Na1) return;

    // Calculate interpolation indices
    // Map from output grid [0, Na0-1] x [0, Na1-1] to input grid [0, Nb0-1] x [0, Nb1-1]
    float b0_float = (float)id0 * (Nb0 - 1) / (Na0 - 1);
    float b1_float = (float)id1 * (Nb1 - 1) / (Na1 - 1);

    int b0_int = (int)floorf(b0_float);
    int b1_int = (int)floorf(b1_float);
    float b0_frac = b0_float - b0_int;
    float b1_frac = b1_float - b1_int;

    // Clamp to valid range
    b0_int = min(b0_int, Nb0 - 2);
    b1_int = min(b1_int, Nb1 - 2);
    b0_int = max(b0_int, 0);
    b1_int = max(b1_int, 0);

    const T2 v = interp2d<T2>(&b[idbatch * Nb1 * Nb0], Nb0, Nb1, b0_int, b0_frac, b1_int, b1_frac);
    out[idbatch * Na1 * Na0 + id0 * Na1 + id1] = a[idbatch * Na1 * Na0 + id0 * Na1 + id1] * (T)v;
}

at::Tensor mul_2d_interp_linear_cuda(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1) {
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(a.dtype())
        .layout(torch::kStrided)
        .device(a.device());
	at::Tensor out = torch::zeros({nbatch, Na0, Na1}, options);

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Na0 * Na1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Float && b.scalar_type() == at::ScalarType::Float) {
        mul_2d_interp_linear_kernel<float, float><<<block_count, thread_per_block, 0, stream>>>(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            Na0, Na1, Nb0, Nb1);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::ComplexFloat) {
        mul_2d_interp_linear_kernel<complex64_t, complex64_t><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const complex64_t*>(b_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::Float) {
        mul_2d_interp_linear_kernel<complex64_t, float><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            b_contig.data_ptr<float>(),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1);
    } else {
        AT_ERROR("Unsupported dtype combination for mul_2d_interp_linear");
    }

	return out;
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("div_2d_interp_linear", &div_2d_interp_linear_cuda);
  m.impl("mul_2d_interp_linear", &mul_2d_interp_linear_cuda);
}

}
