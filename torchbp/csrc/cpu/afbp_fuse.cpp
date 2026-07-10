#include "util.h"

// CPU op for the afbp wavenumber-domain spectrum fusion (see ops/afbp.py):
// for each output element of the fine spectrum the placement (x_eq, owning
// subaperture run, raised-cosine region weight) is computed on the fly from
// small 1-D tables and the <= kmax contributing subaperture spectra are
// summed. Replaces the tensor-op path that materializes several
// [nrf, n_fine] intermediates per contribution. CPU only: on CUDA afbp uses
// the batched tensor-op fusion path instead.
namespace torchbp {

// One range-wavenumber row: a vectorizable alias-band recentering pass into
// a small buffer, then a scalar pass with the subaperture search, region
// weights and spectrum gathers.
static void afbp_fuse_row_cpu(
          const complex64_t* S_row,
          const float* nua,
          const float* xs,
          complex64_t* fine_row,
          float inv_kr,
          float x_half,
          float band,
          float x_c,
          float x_taper,
          int nsub,
          int n_c,
          int n_fine,
          size_t s_stride,
          int kmax) {
    constexpr int CHUNK = 256;
    float xeq_buf[CHUNK];

    const float inv_band = 1.0f / band;
    const float off = x_c - 0.5f * band;
    const float span_v = x_half - x_taper;
    const float inv_span = 1.0f / (span_v > 1e-9f ? span_v : 1e-9f);
    const bool taper = x_half > x_taper;

    for (int cb = 0; cb < n_fine; cb += CHUNK) {
        const int nchunk = std::min(CHUNK, n_fine - cb);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            // x_eq recentered on the subaperture positions:
            // remainder(x0 - off, band) + off with band > 0, same as the
            // tensor path.
            const float x = nua[cb + q] * inv_kr - off;
            xeq_buf[q] = x - band * floorf(x * inv_band) + off;
        }

        int c = (cb) % n_c;
        for (int q = 0; q < nchunk; q++) {
            const float xeq = xeq_buf[q];
            // First candidate subaperture: lower bound of x_eq - x_half in
            // the sorted centers, same as torch.bucketize(right=False).
            const float v = xeq - x_half;
            int u0 = 0;
            for (int i = 0; i < nsub; i++)
                u0 += xs[i] < v;
            float accr = 0.0f, acci = 0.0f;
            for (int k = 0; k < kmax; k++) {
                const int u = u0 + k;
                if (u >= nsub)
                    break;
                const float dist = fabsf(xeq - xs[u]);
                float w;
                if (taper) {
                    // Raised cosine roll from x_taper to the region edge
                    // x_half (zero weight beyond via the clamp).
                    float t = (dist - x_taper) * inv_span;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    float sn, cs;
                    sincospi(t, &sn, &cs);
                    w = 0.5f * (1.0f + cs);
                } else {
                    // Region not wider than the taper start: hard cut.
                    w = dist <= x_half ? 1.0f : 0.0f;
                }
                const complex64_t s = S_row[u * s_stride + c];
                accr += w * s.real();
                acci += w * s.imag();
            }
            fine_row[cb + q] = {accr, acci};
            c++;
            if (c == n_c)
                c = 0;
        }
    }
}

at::Tensor afbp_fuse_cpu(
          const at::Tensor &S,
          const at::Tensor &nua,
          const at::Tensor &xs,
          const at::Tensor &inv_kr,
          const at::Tensor &x_half,
          const at::Tensor &band,
          double x_c,
          double x_taper,
          int64_t kmax) {
    TORCH_CHECK(S.dtype() == at::kComplexFloat);
    TORCH_CHECK(nua.dtype() == at::kFloat);
    TORCH_CHECK(xs.dtype() == at::kFloat);
    TORCH_CHECK(inv_kr.dtype() == at::kFloat);
    TORCH_CHECK(x_half.dtype() == at::kFloat);
    TORCH_CHECK(band.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(S.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(S.dim() == 3, "S shape should be [nsub, nrf, n_c]");

    const int nsub = S.size(0);
    const int nrf = S.size(1);
    const int n_c = S.size(2);
    const int n_fine = nua.numel();
    TORCH_CHECK(xs.numel() == nsub);
    TORCH_CHECK(inv_kr.numel() == nrf);
    TORCH_CHECK(x_half.numel() == nrf);
    TORCH_CHECK(band.numel() == nrf);

    at::Tensor S_contig = S.contiguous();
    at::Tensor nua_contig = nua.contiguous();
    at::Tensor xs_contig = xs.contiguous();
    at::Tensor inv_kr_contig = inv_kr.contiguous();
    at::Tensor x_half_contig = x_half.contiguous();
    at::Tensor band_contig = band.contiguous();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(S.device());
    at::Tensor fine = torch::empty({nrf, n_fine}, options);

    const complex64_t* S_ptr = (const complex64_t*)S_contig.data_ptr<c10::complex<float>>();
    const float* nua_ptr = nua_contig.data_ptr<float>();
    const float* xs_ptr = xs_contig.data_ptr<float>();
    const float* inv_kr_ptr = inv_kr_contig.data_ptr<float>();
    const float* x_half_ptr = x_half_contig.data_ptr<float>();
    const float* band_ptr = band_contig.data_ptr<float>();
    complex64_t* fine_ptr = (complex64_t*)fine.data_ptr<c10::complex<float>>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int row = 0; row < nrf; row++) {
        afbp_fuse_row_cpu(
                S_ptr + (size_t)row * n_c, nua_ptr, xs_ptr,
                fine_ptr + (size_t)row * n_fine,
                inv_kr_ptr[row], x_half_ptr[row], band_ptr[row],
                (float)x_c, (float)x_taper,
                nsub, n_c, n_fine, (size_t)nrf * n_c, (int)kmax);
    }
    return fine;
}

TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
    m.impl("afbp_fuse", &afbp_fuse_cpu);
}

}
