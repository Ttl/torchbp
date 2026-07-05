#include "util.h"

// CUDA cfbp merge op. Mirrors cpu/cfbp_merge.cpp.
namespace torchbp {

// One thread per output pixel. Both children and the output share the same
// x and y extents; the y interpolation from child sampling to output
// sampling uses per-output-row weight tables evaluated exactly on the
// Python side (weights depend only on the output y index). Each
// interpolated child is re-referenced from its own demodulation carrier to
// the parent carrier and the two are summed. Taps are contiguous in the
// child rows and start indices are clamped so all reads are in bounds.
__global__ void cfbp_merge2_kernel(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *w0, const int *idx0, const float *w1,
        const int *idx1, int Nx, int Ny0, int Ny1, int Nyout, int order0,
        int order1, float dx, float dy, float ox0, float oy0, float z0,
        float ox1, float oy1, float z1, float oxp, float oyp, float zp,
        float ref_phase) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idx % Nyout;
    const int idxx = idx / Nyout;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= Nx * Nyout) {
        return;
    }

    const float xrp = oxp + idxx * dx;
    const float yrp = oyp + idy * dy;
    const float dp = sqrtf(xrp * xrp + yrp * yrp + zp * zp);

    complex64_t pixel{};

    for (int id = 0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float *w = id == 0 ? w0 : w1;
        const int *tap0 = id == 0 ? idx0 : idx1;
        const int Nyc = id == 0 ? Ny0 : Ny1;
        const int order = id == 0 ? order0 : order1;
        const float xr = (id == 0 ? ox0 : ox1) + idxx * dx;
        const float yr = (id == 0 ? oy0 : oy1) + idy * dy;
        const float z = id == 0 ? z0 : z1;

        const float *row = (const float*)(img + ((size_t)idbatch * Nx + idxx) * Nyc + tap0[idy]);
        const float *wrow = w + (size_t)idy * order;
        float sr = 0.0f, si = 0.0f;
        for (int t = 0; t < order; t++) {
            const float wt = __ldg(&wrow[t]);
            sr += wt * __ldg(&row[2 * t]);
            si += wt * __ldg(&row[2 * t + 1]);
        }

        const float d = sqrtf(xr * xr + yr * yr + z * z);
        float ref_sin, ref_cos;
        sincospif(ref_phase * (d - dp), &ref_sin, &ref_cos);
        pixel += complex64_t(sr * ref_cos - si * ref_sin, sr * ref_sin + si * ref_cos);
    }
    out[((size_t)idbatch * Nx + idxx) * Nyout + idy] = pixel;
}

at::Tensor cfbp_merge2_cuda(
          const at::Tensor &img0,
          const at::Tensor &img1,
          const at::Tensor &w0,
          const at::Tensor &idx0,
          const at::Tensor &w1,
          const at::Tensor &idx1,
          int64_t nbatch,
          int64_t Nx,
          int64_t Ny0,
          int64_t Ny1,
          int64_t Nyout,
          int64_t order0,
          int64_t order1,
          double dx,
          double dy,
          double ox0,
          double oy0,
          double z0,
          double ox1,
          double oy1,
          double z1,
          double oxp,
          double oyp,
          double zp,
          double ref_phase) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_CHECK(w0.dtype() == at::kFloat);
    TORCH_CHECK(w1.dtype() == at::kFloat);
    TORCH_CHECK(idx0.dtype() == at::kInt);
    TORCH_CHECK(idx1.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w0.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w1.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(idx0.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(idx1.device().type() == at::DeviceType::CUDA);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor w0_contig = w0.contiguous();
    at::Tensor w1_contig = w1.contiguous();
    at::Tensor idx0_contig = idx0.contiguous();
    at::Tensor idx1_contig = idx1.contiguous();
    at::Tensor out = torch::empty({nbatch, Nx, Nyout}, img0_contig.options());
    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();
    const float* w0_ptr = w0_contig.data_ptr<float>();
    const float* w1_ptr = w1_contig.data_ptr<float>();
    const int* idx0_ptr = idx0_contig.data_ptr<int>();
    const int* idx1_ptr = idx1_contig.data_ptr<int>();
    complex64_t* out_ptr = (complex64_t*)out.data_ptr<c10::complex<float>>();

    dim3 thread_per_block = {256, 1};
    // Up-rounding division.
    int blocks = Nx * Nyout;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, (unsigned int)nbatch, 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cfbp_merge2_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  img0_ptr,
                  img1_ptr,
                  out_ptr,
                  w0_ptr,
                  idx0_ptr,
                  w1_ptr,
                  idx1_ptr,
                  Nx, Ny0, Ny1, Nyout,
                  order0, order1,
                  dx, dy,
                  ox0, oy0, z0,
                  ox1, oy1, z1,
                  oxp, oyp, zp,
                  ref_phase);
    return out;
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("cfbp_merge2", &cfbp_merge2_cuda);
}

}
