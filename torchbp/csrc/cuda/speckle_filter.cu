#include "util.h"

namespace torchbp {

template<typename T>
__global__ void lee_filter_kernel(const T *img, float *out, int Nx, int Ny, int wx,
        int wy, float cu) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    int count = 0;
    float mean = 0.0f;
    float m2 = 0.0f;
    for (int i=max(idx - wx, 0); i < min(idx + wx, Nx-1); i++) {
        for (int j=max(idy - wy, 0); j < min(idy + wy, Ny-1); j++) {
            count++;
            T val = img[idbatch * Nx * Ny + i*Ny + j];
            float v;
            if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
                if (isnan(val.real())) {
                    continue;
                }
                v = abs(val);
            } else {
                if (isnan(val)) {
                    continue;
                }
                v = val;
            }
            float delta = v - mean;
            mean += delta / count;
            float delta2 = v - mean;
            m2 += delta * delta2;
        }
    }
    if (count == 0) {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx * Ny + idy] = abs(img[idbatch * Nx * Ny + idx * Ny + idy]);
        } else {
            out[idbatch * Nx * Ny + idx * Ny + idy] = img[idbatch * Nx * Ny + idx * Ny + idy];
        }
    } else {
        float var = m2 / count;
        float ci = sqrtf(var) / mean;
        float w;
        if (ci < cu) {
            w = 0.0f;
        } else {
            w = 1.0f - (cu * cu) / (ci * ci);
        }
        T v = img[idbatch * Nx * Ny + idx * Ny + idy];
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx * Ny + idy] = mean + w * (abs(v) - mean);
        } else {
            out[idbatch * Nx * Ny + idx * Ny + idy] = mean + w * (v - mean);
        }
    }
}


at::Tensor lee_filter_cuda(
          const at::Tensor &img,
          int64_t nbatch,
          int64_t Nx,
          int64_t Ny,
          int64_t wx,
          int64_t wy,
          double cu) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	at::Tensor img_contig = img.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img.device());
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, options);
    float* out_ptr = out.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

	if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        lee_filter_kernel<complex64_t>
              <<<block_count, thread_per_block>>>(
                      (const complex64_t*)img_ptr,
                      out_ptr,
                      Nx,
                      Ny,
                      wx,
                      wy,
                      cu
                      );
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        lee_filter_kernel<float>
              <<<block_count, thread_per_block>>>(
                      img_ptr,
                      out_ptr,
                      Nx,
                      Ny,
                      wx,
                      wy,
                      cu
                      );
    }
	return out;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("lee_filter", &lee_filter_cuda);
}

}
