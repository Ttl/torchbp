#include "util.h"

// CPU cfbp merge op. Mirrors cuda/cfbp_merge.cu.
namespace torchbp {

// Tap sum of one child over a chunk: interpolate the child row at the
// precomputed taps and accumulate rotated by the re-reference phasor.
//
// Templated on the tap count: with a constant trip count GCC fully unrolls
// the inner loop and emits contiguous vector loads for the stride-2
// complex data. With a runtime trip count it instead vectorizes the outer
// loop with emulated gathers and a scalar reduction chain, which is about
// 3x slower for the whole kernel.
template <int ORDER>
static void cfbp_merge2_taps_cpu(
          const float* row,
          const float* w,
          const int* idx,
          const float* cs_buf,
          const float* sn_buf,
          float* accr,
          float* acci,
          int yb,
          int nchunk) {
    for (int q = 0; q < nchunk; q++) {
        const float* s = row + 2 * idx[yb + q];
        const float* wq = w + (size_t)(yb + q) * ORDER;
        float sr = 0.0f, si = 0.0f;
        for (int t = 0; t < ORDER; t++) {
            sr += wq[t] * s[2 * t];
            si += wq[t] * s[2 * t + 1];
        }
        accr[q] += sr * cs_buf[q] - si * sn_buf[q];
        acci[q] += sr * sn_buf[q] + si * cs_buf[q];
    }
}

// Runtime tap count fallback for uncommon orders.
static void cfbp_merge2_taps_generic_cpu(
          const float* row,
          const float* w,
          const int* idx,
          const float* cs_buf,
          const float* sn_buf,
          float* accr,
          float* acci,
          int yb,
          int nchunk,
          int order) {
    for (int q = 0; q < nchunk; q++) {
        const float* s = row + 2 * idx[yb + q];
        const float* wq = w + (size_t)(yb + q) * order;
        float sr = 0.0f, si = 0.0f;
        for (int t = 0; t < order; t++) {
            sr += wq[t] * s[2 * t];
            si += wq[t] * s[2 * t + 1];
        }
        accr[q] += sr * cs_buf[q] - si * sn_buf[q];
        acci[q] += sr * sn_buf[q] + si * cs_buf[q];
    }
}

static void cfbp_merge2_taps_dispatch_cpu(
          const float* row,
          const float* w,
          const int* idx,
          const float* cs_buf,
          const float* sn_buf,
          float* accr,
          float* acci,
          int yb,
          int nchunk,
          int order) {
    switch (order) {
    case 1:
        return cfbp_merge2_taps_cpu<1>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 4:
        return cfbp_merge2_taps_cpu<4>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 6:
        return cfbp_merge2_taps_cpu<6>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 8:
        return cfbp_merge2_taps_cpu<8>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 10:
        return cfbp_merge2_taps_cpu<10>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 12:
        return cfbp_merge2_taps_cpu<12>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    case 16:
        return cfbp_merge2_taps_cpu<16>(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk);
    default:
        return cfbp_merge2_taps_generic_cpu(row, w, idx, cs_buf, sn_buf, accr, acci, yb, nchunk, order);
    }
}

// One x-column of the merged image, y pixels processed in chunks.
//
// Structured for SIMD like backprojection_cart_2d_row_cpu: a vectorizable
// geometry + phase pass writes the carrier re-reference phasor of both
// children to small buffers, then a pass per child does the
// y-interpolation tap sum (taps are contiguous in the child row) rotated by
// the phasor. The interpolation weights depend only on the output y index
// (child and parent grids share the same extent), so they are evaluated
// exactly on the Python side and passed in as [Nyout, order] tables with
// clamped start indices.
static void cfbp_merge2_row_cpu(
          const complex64_t* img0,
          const complex64_t* img1,
          const float* w0,
          const int* idx0,
          const float* w1,
          const int* idx1,
          complex64_t* img,
          int Nx,
          int Ny0,
          int Ny1,
          int Nyout,
          int order0,
          int order1,
          float dx,
          float dy,
          float ox0,
          float oy0,
          float z0,
          float ox1,
          float oy1,
          float z1,
          float oxp,
          float oyp,
          float zp,
          float ref_phase,
          int ix,
          int idbatch) {
    constexpr int CHUNK = 256;
    float accr[CHUNK], acci[CHUNK];
    float cs0_buf[CHUNK], sn0_buf[CHUNK];
    float cs1_buf[CHUNK], sn1_buf[CHUNK];

    const float xrp = oxp + ix * dx;
    const float xr0 = ox0 + ix * dx;
    const float xr1 = ox1 + ix * dx;
    const float cp = xrp * xrp + zp * zp;
    const float c0 = xr0 * xr0 + z0 * z0;
    const float c1 = xr1 * xr1 + z1 * z1;

    const float* row0 = (const float*)(img0 + ((size_t)idbatch * Nx + ix) * Ny0);
    const float* row1 = (const float*)(img1 + ((size_t)idbatch * Nx + ix) * Ny1);
    complex64_t* img_row = img + ((size_t)idbatch * Nx + ix) * Nyout;

    for (int yb = 0; yb < Nyout; yb += CHUNK) {
        const int nchunk = std::min(CHUNK, Nyout - yb);

        // Geometry + phase pass. Distances to the pixel from the parent and
        // both child subaperture centers give the demodulation carrier
        // re-reference phase of each child.
#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            const float yrp = oyp + (yb + q) * dy;
            const float yr0 = oy0 + (yb + q) * dy;
            const float yr1 = oy1 + (yb + q) * dy;
            const float dp = sqrtf(cp + yrp * yrp);
            const float d0 = sqrtf(c0 + yr0 * yr0);
            const float d1 = sqrtf(c1 + yr1 * yr1);
            sincospi(ref_phase * (d0 - dp), &sn0_buf[q], &cs0_buf[q]);
            sincospi(ref_phase * (d1 - dp), &sn1_buf[q], &cs1_buf[q]);
        }

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            accr[q] = 0.0f;
            acci[q] = 0.0f;
        }

        cfbp_merge2_taps_dispatch_cpu(
            row0, w0, idx0, cs0_buf, sn0_buf, accr, acci, yb, nchunk, order0);
        cfbp_merge2_taps_dispatch_cpu(
            row1, w1, idx1, cs1_buf, sn1_buf, accr, acci, yb, nchunk, order1);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            img_row[yb + q] = complex64_t(accr[q], acci[q]);
        }
    }
}

at::Tensor cfbp_merge2_cpu(
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
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor w0_contig = w0.contiguous();
    at::Tensor w1_contig = w1.contiguous();
    at::Tensor idx0_contig = idx0.contiguous();
    at::Tensor idx1_contig = idx1.contiguous();
    at::Tensor img = torch::empty({nbatch, Nx, Nyout}, img0_contig.options());
    const complex64_t* img0_ptr = img0_contig.data_ptr<complex64_t>();
    const complex64_t* img1_ptr = img1_contig.data_ptr<complex64_t>();
    const float* w0_ptr = w0_contig.data_ptr<float>();
    const float* w1_ptr = w1_contig.data_ptr<float>();
    const int* idx0_ptr = idx0_contig.data_ptr<int>();
    const int* idx1_ptr = idx1_contig.data_ptr<int>();
    complex64_t* img_ptr = img.data_ptr<complex64_t>();

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int ix = 0; ix < Nx; ix++) {
            cfbp_merge2_row_cpu(
                          img0_ptr,
                          img1_ptr,
                          w0_ptr,
                          idx0_ptr,
                          w1_ptr,
                          idx1_ptr,
                          img_ptr,
                          Nx, Ny0, Ny1, Nyout,
                          order0, order1,
                          dx, dy,
                          ox0, oy0, z0,
                          ox1, oy1, z1,
                          oxp, oyp, zp,
                          ref_phase,
                          ix, idbatch);
        }
    }
    return img;
}

TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("cfbp_merge2", &cfbp_merge2_cpu);
}

}
