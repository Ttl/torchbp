#include "util.h"

namespace torchbp {

template<typename T>
__global__ void abs_sum_kernel(
          const T *data,
          float *res,
          int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned mask = __ballot_sync(FULL_MASK, idx < N);

    if (idx >= N)
        return;

    float y = abs(data[idbatch * N + idx]);

    for (int offset = 16; offset > 0; offset /= 2) {
        y += __shfl_down_sync(mask, y, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&res[idbatch], y);
    }
}


template<typename T>
__global__ void abs_sum_grad_kernel(
          const T *data,
          const float *grad,
          T *data_grad,
          int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= N)
        return;

    T d = data[idbatch * N + idx];
    T g1 = d / (2.0f * abs(d));

    data_grad[idbatch * N + idx] = 2.0f * grad[idbatch] * g1;
}


template<typename T>
__global__ void entropy_kernel(
          const T *data,
          float *res,
          const float *norm,
          int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned mask = __ballot_sync(FULL_MASK, idx < N);

    if (idx >= N)
        return;

    float xn = abs(data[idbatch * N + idx]) / norm[idbatch];
    float y = 0.0f;
    if (xn != 0.0f) {
        y = -xn * log(xn);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        y += __shfl_down_sync(mask, y, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&res[idbatch], y);
    }
}


template<typename T>
__global__ void entropy_grad_kernel(
          const T *data,
          const float *norm,
          const float *grad,
          T *data_grad,
          float *norm_grad,
          int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned mask = __ballot_sync(FULL_MASK, idx < N);

    if (idx >= N)
        return;

    float n = norm[idbatch];
    T d = data[idbatch * N + idx];
    float x = abs(d);

    T g1 = {0.0f, 0.0f};
    float ng = 0.0f;
    if (x != 0.0f) {
        float lx = 1.0f + log(x / n);
        g1 = -d * lx / (2.0f * n * x);
        ng = x * lx / (n * n);
    }
    data_grad[idbatch * N + idx] = 2.0f * grad[idbatch] * g1;

    for (int offset = 16; offset > 0; offset /= 2) {
        ng += __shfl_down_sync(mask, ng, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&norm_grad[idbatch], ng);
    }
}


at::Tensor entropy_cuda(
          const at::Tensor &data,
          const at::Tensor &norm,
          int64_t nbatch) {
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(norm.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(norm.device().type() == at::DeviceType::CUDA);

	at::Tensor data_contig = data.contiguous();
	at::Tensor norm_contig = norm.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor res = torch::zeros({static_cast<unsigned int>(nbatch)}, options);
    float* res_ptr = res.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
    const float* norm_ptr = norm_contig.data_ptr<float>();

    entropy_kernel<complex64_t>
          <<<block_count, thread_per_block>>>(
                  (complex64_t*)data_ptr,
                  res_ptr,
                  norm_ptr,
                  blocks
                  );

	return res;
}


std::vector<at::Tensor> entropy_grad_cuda(
          const at::Tensor &data,
          const at::Tensor &norm,
          const at::Tensor &grad,
          int64_t nbatch) {
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(norm.dtype() == at::kFloat);
	TORCH_CHECK(grad.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(norm.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);

	at::Tensor data_contig = data.contiguous();
	at::Tensor norm_contig = norm.contiguous();
	at::Tensor grad_contig = grad.contiguous();

	at::Tensor data_grad = torch::zeros_like(data);
	at::Tensor norm_grad = torch::zeros_like(norm);
    c10::complex<float>* data_grad_ptr = data_grad.data_ptr<c10::complex<float>>();
    float* norm_grad_ptr = norm_grad.data_ptr<float>();
    float* grad_ptr = grad.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
    const float* norm_ptr = norm_contig.data_ptr<float>();

    entropy_grad_kernel<complex64_t>
          <<<block_count, thread_per_block>>>(
                  (complex64_t*)data_ptr,
                  norm_ptr,
                  grad_ptr,
                  (complex64_t*)data_grad_ptr,
                  norm_grad_ptr,
                  blocks
                  );

    std::vector<at::Tensor> ret;
    ret.push_back(data_grad);
    ret.push_back(norm_grad);
	return ret;
}


at::Tensor abs_sum_cuda(
          const at::Tensor &data,
          int64_t nbatch) {
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor res = torch::zeros({static_cast<unsigned int>(nbatch), 1}, options);
    float* res_ptr = res.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();

    abs_sum_kernel<complex64_t>
          <<<block_count, thread_per_block>>>(
                  (complex64_t*)data_ptr,
                  res_ptr,
                  blocks
                  );

	return res;
}


at::Tensor abs_sum_grad_cuda(
          const at::Tensor &data,
          const at::Tensor &grad,
          int64_t nbatch) {
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);

	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();

	at::Tensor data_grad = torch::zeros_like(data);
    c10::complex<float>* data_grad_ptr = data_grad.data_ptr<c10::complex<float>>();
    float* grad_ptr = grad.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();

    abs_sum_grad_kernel<complex64_t>
          <<<block_count, thread_per_block>>>(
                  (complex64_t*)data_ptr,
                  grad_ptr,
                  (complex64_t*)data_grad_ptr,
                  blocks
                  );

	return data_grad;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("entropy", &entropy_cuda);
  m.impl("entropy_grad", &entropy_grad_cuda);
  m.impl("abs_sum", &abs_sum_cuda);
  m.impl("abs_sum_grad", &abs_sum_grad_cuda);
}

}
