#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "util.h"

namespace torchbp {

// Generic 2D image resampling with Lanczos interpolation.
// Each output pixel (i, j) reads from input at (i + shift_r[i,j], j + shift_az[i,j]).
template<typename T>
__global__ void resample_2d_lanczos_kernel(
        const T *img,
        const float *shift_r,
        const float *shift_az,
        T *out,
        int Nr, int Naz,
        int order) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int iaz = id1 % Naz;
    const int ir = id1 / Naz;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nr * Naz) {
        return;
    }

    const int shift_idx = ir * Naz + iaz;
    const float src_r = ir + shift_r[shift_idx];
    const float src_az = iaz + shift_az[shift_idx];

    // Check that at least one kernel tap is within bounds
    const float a = 0.5f * order;
    if (src_r >= -a && src_r < Nr + a && src_az >= -a && src_az < Naz + a) {
        T v = lanczos_interp_2d<T, T>(
                &img[idbatch * Nr * Naz], Nr, Naz, src_r, src_az, order);
        out[idbatch * Nr * Naz + shift_idx] = v;
    } else {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nr * Naz + shift_idx] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nr * Naz + shift_idx] = 0.0f;
        }
    }
}

at::Tensor resample_2d_lanczos_cuda(
          const at::Tensor &img,
          const at::Tensor &shift_r,
          const at::Tensor &shift_az,
          int64_t nbatch,
          int64_t Nr,
          int64_t Naz,
          int64_t order) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_2d_lanczos: img must be complex64 or float32");
    TORCH_CHECK(shift_r.dtype() == at::kFloat, "resample_2d_lanczos: shift_r must be float32");
    TORCH_CHECK(shift_az.dtype() == at::kFloat, "resample_2d_lanczos: shift_az must be float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(shift_r.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(shift_az.device().type() == at::DeviceType::CUDA);

    at::Tensor img_contig = img.contiguous();
    at::Tensor shift_r_contig = shift_r.contiguous();
    at::Tensor shift_az_contig = shift_az.contiguous();
    at::Tensor out = torch::empty({nbatch, Nr, Naz}, img_contig.options());

    const float* shift_r_ptr = shift_r_contig.data_ptr<float>();
    const float* shift_az_ptr = shift_az_contig.data_ptr<float>();

    dim3 thread_per_block = {256, 1};
    int blocks = Nr * Naz;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        resample_2d_lanczos_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      (const complex64_t*)img_ptr,
                      shift_r_ptr,
                      shift_az_ptr,
                      (complex64_t*)out_ptr,
                      Nr, Naz,
                      order);
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        resample_2d_lanczos_kernel<float>
              <<<block_count, thread_per_block, 0, stream>>>(
                      img_ptr,
                      shift_r_ptr,
                      shift_az_ptr,
                      out_ptr,
                      Nr, Naz,
                      order);
    }
    return out;
}

// Generic 2D image resampling with Knab interpolation.
// Better accuracy than Lanczos when the oversampling ratio is known.
template<typename T>
__global__ void resample_2d_knab_kernel(
        const T *img,
        const float *shift_r,
        const float *shift_az,
        T *out,
        int Nr, int Naz,
        int order, float v, float norm) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int iaz = id1 % Naz;
    const int ir = id1 / Naz;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nr * Naz) {
        return;
    }

    const int shift_idx = ir * Naz + iaz;
    const float src_r = ir + shift_r[shift_idx];
    const float src_az = iaz + shift_az[shift_idx];

    const float a = 0.5f * order;
    if (src_r >= -a && src_r < Nr + a && src_az >= -a && src_az < Naz + a) {
        T val = knab_interp_2d<T, T>(
                &img[idbatch * Nr * Naz], Nr, Naz, src_r, src_az, order, v, norm);
        out[idbatch * Nr * Naz + shift_idx] = val;
    } else {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nr * Naz + shift_idx] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nr * Naz + shift_idx] = 0.0f;
        }
    }
}

at::Tensor resample_2d_knab_cuda(
          const at::Tensor &img,
          const at::Tensor &shift_r,
          const at::Tensor &shift_az,
          int64_t nbatch,
          int64_t Nr,
          int64_t Naz,
          int64_t order,
          double oversample) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_2d_knab: img must be complex64 or float32");
    TORCH_CHECK(shift_r.dtype() == at::kFloat, "resample_2d_knab: shift_r must be float32");
    TORCH_CHECK(shift_az.dtype() == at::kFloat, "resample_2d_knab: shift_az must be float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(shift_r.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(shift_az.device().type() == at::DeviceType::CUDA);

    at::Tensor img_contig = img.contiguous();
    at::Tensor shift_r_contig = shift_r.contiguous();
    at::Tensor shift_az_contig = shift_az.contiguous();
    at::Tensor out = torch::empty({nbatch, Nr, Naz}, img_contig.options());

    const float* shift_r_ptr = shift_r_contig.data_ptr<float>();
    const float* shift_az_ptr = shift_az_contig.data_ptr<float>();

    // Knab window parameter: v = 1 - 1/oversample
    const float v = 1.0f - 1.0f / static_cast<float>(oversample);
    const float norm = knab_kernel_norm(order, v);

    dim3 thread_per_block = {256, 1};
    int blocks = Nr * Naz;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        resample_2d_knab_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      (const complex64_t*)img_ptr,
                      shift_r_ptr,
                      shift_az_ptr,
                      (complex64_t*)out_ptr,
                      Nr, Naz,
                      order, v, norm);
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        resample_2d_knab_kernel<float>
              <<<block_count, thread_per_block, 0, stream>>>(
                      img_ptr,
                      shift_r_ptr,
                      shift_az_ptr,
                      out_ptr,
                      Nr, Naz,
                      order, v, norm);
    }
    return out;
}

// Generic 1D signal rate change with Lanczos interpolation. Resamples N input
// samples to M output samples along the last axis; everything else is folded
// into nbatch. Output element id1 reads the input at continuous position
// id1 * (N / M), lowpassed to the output rate when M < N (decimation).
template<typename T>
__global__ void resample_1d_lanczos_kernel(
        const T *img,
        T *out,
        int N, int M,
        float step, float cutoff,
        int order) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= M) {
        return;
    }

    const float src = id1 * step;
    out[idbatch * M + id1] = lanczos_resample_1d<T>(
            &img[idbatch * N], N, src, order, cutoff);
}

at::Tensor resample_1d_lanczos_cuda(
          const at::Tensor &img,
          int64_t nbatch,
          int64_t N,
          int64_t M,
          int64_t order) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_1d_lanczos: img must be complex64 or float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);

    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, M}, img_contig.options());

    const float step = (float)N / (float)M;
    const float cutoff = M >= N ? 1.0f : (float)M / (float)N;

    dim3 thread_per_block = {256, 1};
    int blocks = M;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        resample_1d_lanczos_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      (const complex64_t*)img_ptr,
                      (complex64_t*)out_ptr,
                      N, M,
                      step, cutoff,
                      order);
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        resample_1d_lanczos_kernel<float>
              <<<block_count, thread_per_block, 0, stream>>>(
                      img_ptr,
                      out_ptr,
                      N, M,
                      step, cutoff,
                      order);
    }
    return out;
}

// Generic 1D signal rate change with Knab interpolation.
template<typename T>
__global__ void resample_1d_knab_kernel(
        const T *img,
        T *out,
        int N, int M,
        float step, float cutoff,
        int order, float v, float norm) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= M) {
        return;
    }

    const float src = id1 * step;
    out[idbatch * M + id1] = knab_resample_1d<T>(
            &img[idbatch * N], N, src, order, v, norm, cutoff);
}

at::Tensor resample_1d_knab_cuda(
          const at::Tensor &img,
          int64_t nbatch,
          int64_t N,
          int64_t M,
          int64_t order,
          double oversample) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_1d_knab: img must be complex64 or float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);

    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, M}, img_contig.options());

    // Knab window parameter: v = 1 - 1/oversample
    const float v = 1.0f - 1.0f / static_cast<float>(oversample);
    const float norm = knab_kernel_norm(order, v);
    const float step = (float)N / (float)M;
    const float cutoff = M >= N ? 1.0f : (float)M / (float)N;

    dim3 thread_per_block = {256, 1};
    int blocks = M;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        resample_1d_knab_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      (const complex64_t*)img_ptr,
                      (complex64_t*)out_ptr,
                      N, M,
                      step, cutoff,
                      order, v, norm);
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        resample_1d_knab_kernel<float>
              <<<block_count, thread_per_block, 0, stream>>>(
                      img_ptr,
                      out_ptr,
                      N, M,
                      step, cutoff,
                      order, v, norm);
    }
    return out;
}

TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("resample_2d_lanczos", &resample_2d_lanczos_cuda);
  m.impl("resample_2d_knab", &resample_2d_knab_cuda);
  m.impl("resample_1d_lanczos", &resample_1d_lanczos_cuda);
  m.impl("resample_1d_knab", &resample_1d_knab_cuda);
}

}
