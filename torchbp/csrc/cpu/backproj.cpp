#include "util.h"
#include <cstring>

// CPU backprojection, projection and GPGA ops. Mirrors cuda/backproj.cu.
namespace torchbp {

// One range row of the image, theta processed in chunks.
//
// Structured for SIMD like projection_nufft_spread_row_cpu: per sweep, a
// vectorizable geometry + phase pass writes per-pixel sample index, fraction
// and reference phasor to small buffers, a scalar pass copies the
// data-dependent sample (and antenna gain) loads to small buffers, and a
// vectorizable pass interpolates, applies the phasor and accumulates. The
// antenna gain path adds a vectorizable angle pass (asinf_fast/atan2f_fast,
// a few ulp from libm).
template<bool HasDem>
static void backprojection_polar_2d_row_cpu(
          const complex64_t* data,
          const float* pos,
          const float* att,
          complex64_t* img,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          int Nr,
          int Ntheta,
          float d0,
          int dealias,
          float z0,
          const float *g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          const float* dem,
          float dem_r_scale,
          float dem_theta_scale,
          int dem_nr,
          int dem_ntheta,
          float data_fmod,
          float alias_fmod,
          bool normalize,
          int idr,
          int idbatch) {
    constexpr int CHUNK = 256;
    float x_buf[CHUNK], y_buf[CHUNK];
    float accr[CHUNK], acci[CHUNK];
    float cs_buf[CHUNK], sn_buf[CHUNK], frac_buf[CHUNK];
    int idx_buf[CHUNK];
    float vld_buf[CHUNK];
    // (re, im) sample pairs copied as single 8-byte moves; the vector pass
    // deinterleaves them with in-register shuffles.
    double v0_buf[CHUNK], v1_buf[CHUNK];
    float wsum_buf[CHUNK], wsum2_buf[CHUNK];
    float d_buf[CHUNK];
    float ef_buf[CHUNK], af_buf[CHUNK];
    int gi_buf[CHUNK];
    float g00_buf[CHUNK], g01_buf[CHUNK], g10_buf[CHUNK], g11_buf[CHUNK];
    float gvld_buf[CHUNK];
    float z_buf[CHUNK], z2_buf[CHUNK];

    const float r = r0 + idr * dr;
    const float r2 = r * r;
    const complex64_t* data_b = data + (size_t)idbatch * nsweeps * sweep_samples;
    const float* pos_b = pos + (size_t)idbatch * nsweeps * 3;
    complex64_t* img_row = img + ((size_t)idbatch * Nr + idr) * Ntheta;

    for (int tb = 0; tb < Ntheta; tb += CHUNK) {
        const int nchunk = std::min(CHUNK, Ntheta - tb);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            const float theta = theta0 + (tb + q) * dtheta;
            // Guard band support: for |theta| > 1 there is no physical
            // pixel; the grid sample is the smooth continuation of the
            // azimuth signal past the fold (see the distance note below).
            const float ct2 = 1.0f - theta*theta;
            x_buf[q] = r * (ct2 > 0.0f ? sqrtf(ct2) : 0.0f);
            y_buf[q] = r * theta;
            accr[q] = 0.0f;
            acci[q] = 0.0f;
            wsum_buf[q] = 0.0f;
            wsum2_buf[q] = 0.0f;
        }

        // DEM shares the grid extent, so pixel index maps to DEM index by a
        // constant ratio. The r side is fixed for the whole row. Indices are
        // edge-clamped: guard band pixels |theta| > 1 get the DEM edge value.
        if constexpr (HasDem) {
            const float fr = idr * dem_r_scale;
            int ir0 = (int)fr;
            ir0 = ir0 < dem_nr - 1 ? ir0 : dem_nr - 1;
            const int ir1 = ir0 + 1 < dem_nr ? ir0 + 1 : dem_nr - 1;
            const float wr = fr - ir0;
            const float* dem_row0 = dem + (size_t)ir0 * dem_ntheta;
            const float* dem_row1 = dem + (size_t)ir1 * dem_ntheta;
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float ft = (tb + q) * dem_theta_scale;
                int it0 = (int)ft;
                it0 = it0 < dem_ntheta - 1 ? it0 : dem_ntheta - 1;
                const int it1 = it0 + 1 < dem_ntheta ? it0 + 1 : dem_ntheta - 1;
                const float wt = ft - it0;
                const float za = dem_row0[it0] + wt * (dem_row0[it1] - dem_row0[it0]);
                const float zb = dem_row1[it0] + wt * (dem_row1[it1] - dem_row1[it0]);
                const float zq = za + wr * (zb - za);
                z_buf[q] = zq;
                z2_buf[q] = zq * zq;
            }
        }

        for(int i = 0; i < nsweeps; i++) {
            // Sweep reference position.
            const float pos_x = pos_b[i * 3 + 0];
            const float pos_y = pos_b[i * 3 + 1];
            const float pos_z = pos_b[i * 3 + 2];
            const float pz2 = pos_z * pos_z;
            const float r2pn2 = r2 + pos_x * pos_x + pos_y * pos_y + pz2;
            const float* data_row = (const float*)(data_b + (size_t)i * sweep_samples);

            // Geometry + phase pass. idx = -1 marks pixels outside the data
            // range window.
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                // Distance in the polar form d^2 = r^2 + |p|^2
                // - 2*r*(cos(theta)*p_x + sin(theta)*p_y). Inside |theta| <= 1
                // it equals the Cartesian distance to (x, y, 0); with the
                // clamped cosine it stays affine in theta past |theta| = 1,
                // which continues the azimuth chirp smoothly (exactly, for a
                // straight track on the y axis) so that guard band samples
                // give the merge interpolation valid support at the grid
                // edge. Distance from a virtual Cartesian point would instead
                // jump the phase rate to ~2k*r at the fold.
                // With a DEM the pixel is at (x, y, z): d^2 gains
                // z^2 - 2*z*pos_z.
                float d2;
                if constexpr (HasDem) {
                    d2 = r2pn2 + z2_buf[q] - 2.0f * (x_buf[q] * pos_x
                            + y_buf[q] * pos_y + z_buf[q] * pos_z);
                } else {
                    d2 = r2pn2 - 2.0f * (x_buf[q] * pos_x + y_buf[q] * pos_y);
                }
                const float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);
                d_buf[q] = d;

                const float sx = delta_r * (d + d0);
                const int id0 = (int)sx;
                // Float-domain check: (int)sx truncates toward zero, so
                // sx in (-1, 0) would pass an id0 >= 0 check and
                // extrapolate with a negative weight.
                // Out-of-window pixels read sample 0 and are zeroed by the
                // 0/1 mask in the accumulate pass; a skip branch there would
                // keep the accumulation scalar.
                const bool ok = (sx >= 0.0f) & (id0 + 1 < sweep_samples);
                idx_buf[q] = ok ? id0 : 0;
                vld_buf[q] = ok ? 1.0f : 0.0f;
                frac_buf[q] = sx - id0;

                float ref_sin, ref_cos;
                sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
                cs_buf[q] = ref_cos;
                sn_buf[q] = ref_sin;
            }

            if (g != nullptr) {
                const float att0 = att[idbatch * nsweeps * 3 + 3 * i + 0];
                const float att2 = att[idbatch * nsweeps * 3 + 3 * i + 2];
                // Angle pass: bilinear antenna pattern coordinates.
                // Out-of-pattern pixels get gvld = 0 and read pattern
                // element 0.
#pragma omp simd
                for (int q = 0; q < nchunk; q++) {
                    const float px = x_buf[q] - pos_x;
                    const float py = y_buf[q] - pos_y;
                    const float d = d_buf[q];
                    float sin_l;
                    if constexpr (HasDem) {
                        sin_l = (z_buf[q] - pos_z) / d;
                        sin_l = sin_l > 1.0f ? 1.0f : sin_l;
                    } else {
                        sin_l = -pos_z / d;
                    }
                    const float look_angle = asinf_fast(sin_l < -1.0f ? -1.0f : sin_l);
                    const float el_deg = look_angle - att0;
                    const float az_deg = atan2f_fast(py, px) - att2;

                    const float el_idx = (el_deg - g_el0) / g_del;
                    const float az_idx = (az_deg - g_az0) / g_daz;

                    const int el_int = (int)el_idx;
                    const int az_int = (int)az_idx;
                    const bool ok = (el_idx >= 0.0f) & (el_int + 1 < g_nel) &
                                    (az_idx >= 0.0f) & (az_int + 1 < g_naz);
                    // Zero the fractions of invalid pixels: an unbounded
                    // fraction could overflow the masked bilinear to inf and
                    // the 0 * inf mask multiply to NaN.
                    ef_buf[q] = ok ? el_idx - el_int : 0.0f;
                    af_buf[q] = ok ? az_idx - az_int : 0.0f;
                    gi_buf[q] = ok ? el_int * g_naz + az_int : 0;
                    gvld_buf[q] = ok ? 1.0f : 0.0f;
                }
                // Copy pass: only the data-dependent gain and sample loads
                // stay scalar; everything else moves to the vector pass.
                for (int q = 0; q < nchunk; q++) {
                    const int gi = gi_buf[q];
                    g00_buf[q] = g[gi];
                    g01_buf[q] = g[gi + 1];
                    g10_buf[q] = g[gi + g_naz];
                    g11_buf[q] = g[gi + g_naz + 1];
                    const float *s = data_row + 2 * idx_buf[q];
                    memcpy(&v0_buf[q], s, 8);
                    memcpy(&v1_buf[q], s + 2, 8);
                }
                // Vector pass: gain bilinear, sample interpolation, phasor
                // multiply and accumulate, masked by the 0/1 validity
                // products instead of a skip branch.
#pragma omp simd
                for (int q = 0; q < nchunk; q++) {
                    const float ef = ef_buf[q], af = af_buf[q];
                    const float w0 = g00_buf[q] + af * (g01_buf[q] - g00_buf[q]);
                    const float w1 = g10_buf[q] + af * (g11_buf[q] - g10_buf[q]);
                    const float w = (w0 + ef * (w1 - w0)) * vld_buf[q] * gvld_buf[q];

                    const float f = frac_buf[q];
                    const float *v0 = (const float*)v0_buf;
                    const float *v1 = (const float*)v1_buf;
                    const float sr = v0[2*q] + f * (v1[2*q] - v0[2*q]);
                    const float si = v0[2*q+1] + f * (v1[2*q+1] - v0[2*q+1]);
                    accr[q] += w * (sr * cs_buf[q] - si * sn_buf[q]);
                    acci[q] += w * (sr * sn_buf[q] + si * cs_buf[q]);
                    wsum_buf[q] += w;
                    wsum2_buf[q] += w * w;
                }
            } else {
                // Copy pass: only the data-dependent sample load stays
                // scalar. Letting the vectorizer emulate this strided
                // complex gather inside the accumulate loop is slower.
                for (int q = 0; q < nchunk; q++) {
                    const float *s = data_row + 2 * idx_buf[q];
                    memcpy(&v0_buf[q], s, 8);
                    memcpy(&v1_buf[q], s + 2, 8);
                }
                // Vector pass: sample interpolation, phasor multiply and
                // accumulate, masked by the 0/1 validity instead of a skip
                // branch.
#pragma omp simd
                for (int q = 0; q < nchunk; q++) {
                    const float f = frac_buf[q];
                    const float *v0 = (const float*)v0_buf;
                    const float *v1 = (const float*)v1_buf;
                    const float sr = v0[2*q] + f * (v1[2*q] - v0[2*q]);
                    const float si = v0[2*q+1] + f * (v1[2*q+1] - v0[2*q+1]);
                    const float m = vld_buf[q];
                    accr[q] += m * (sr * cs_buf[q] - si * sn_buf[q]);
                    acci[q] += m * (sr * sn_buf[q] + si * cs_buf[q]);
                }
            }
        }
        // Normalize to same average as without antenna pattern.
        // Unweighted: Σs = scene * Σg (signal has g)
        // Weighted: Σ(s * g) = scene * Σg²
        // To match: normalize by Σg / Σg²
        // A denormal wsum2 would blow up the scale (and CUDA flushes it to
        // zero), so require at least the smallest normal float.
        // When normalize=false, skip this normalization (used in FFBP).
        if (g != nullptr && normalize) {
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float scale = wsum2_buf[q] >= 1.17549435e-38f ?
                    wsum_buf[q] / wsum2_buf[q] : 1.0f;
                accr[q] *= scale;
                acci[q] *= scale;
            }
        }
        if (dealias) {
            // Without a DEM the carrier depends only on the range row (also
            // for guard band pixels |theta| > 1 where x^2 + y^2 != r^2).
            // With a DEM the carrier is always referenced to the DEM height
            // instead of the z = 0 plane: a flat-plane carrier would leave a
            // terrain-dependent residual whose local frequency aliases the
            // image spectrum on any significant topography. The carrier then
            // varies with theta, so the phasor is per pixel.
            // bp_polar_range_dealias/alias with the same dem apply/remove
            // the identical carrier.
            if constexpr (HasDem) {
                for (int q = 0; q < nchunk; q++) {
                    const float zz = z0 - z_buf[q];
                    const float dq = sqrtf(r2 + zz*zz);
                    float ref_sin, ref_cos;
                    sincospi(-ref_phase * dq + alias_fmod * idr, &ref_sin, &ref_cos);
                    const float pr = accr[q] * ref_cos - acci[q] * ref_sin;
                    const float pi = accr[q] * ref_sin + acci[q] * ref_cos;
                    accr[q] = pr;
                    acci[q] = pi;
                }
            } else {
                const float d = sqrtf(r2 + z0*z0);
                float ref_sin, ref_cos;
                sincospi(-ref_phase * d + alias_fmod * idr, &ref_sin, &ref_cos);
#pragma omp simd
                for (int q = 0; q < nchunk; q++) {
                    const float pr = accr[q] * ref_cos - acci[q] * ref_sin;
                    const float pi = accr[q] * ref_sin + acci[q] * ref_cos;
                    accr[q] = pr;
                    acci[q] = pi;
                }
            }
        }
#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            img_row[tb + q] = complex64_t(accr[q], acci[q]);
        }
    }
}


at::Tensor backprojection_polar_2d_cpu(
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &att,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double d0,
          int64_t dealias,
          double z0,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          double data_fmod,
          double alias_fmod,
          bool normalize,
          const at::Tensor &dem) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);

    // Match CUDA: att alone (without a gain pattern) is ignored.
    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
    }

	at::Tensor pos_contig = pos.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* pos_ptr = pos_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

    // Keep contiguous copies alive until the kernel has run.
    at::Tensor att_contig;
    at::Tensor g_contig;
    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        att_contig = att.contiguous();
        g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
    }

    const bool has_dem = dem.defined();
    at::Tensor dem_contig;
    const float* dem_ptr = nullptr;
    float dem_r_scale = 0.0f, dem_theta_scale = 0.0f;
    int dem_nr = 0, dem_ntheta = 0;
    if (has_dem) {
        TORCH_CHECK(dem.dtype() == at::kFloat);
        TORCH_CHECK(dem.dim() == 2);
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CPU);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;
    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

    auto row = has_dem ? &backprojection_polar_2d_row_cpu<true>
                       : &backprojection_polar_2d_row_cpu<false>;
#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idr = 0; idr < Nr; idr++) {
            row(
                          data_ptr,
                          pos_ptr,
                          att_ptr,
                          img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0,
                          dealias, z0,
                          g_ptr,
                          g_az0,
                          g_el0,
                          g_daz,
                          g_del,
                          g_naz,
                          g_nel,
                          dem_ptr,
                          dem_r_scale,
                          dem_theta_scale,
                          dem_nr,
                          dem_ntheta,
                          data_fmod/kPI,
                          alias_fmod/kPI,
                          normalize,
                          idr, idbatch);
        }
    }
	return img;
}

static void backprojection_polar_2d_grad_kernel_cpu(
          const complex64_t* data,
          const float* pos,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          int Nr,
          int Ntheta,
          float d0,
          bool dealias,
          float z0,
          float data_fmod,
          float alias_fmod,
          const complex64_t* grad,
          float* pos_grad,
          complex64_t *data_grad,
          int idx,
          int idbatch) {
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    if (idx >= Nr * Ntheta) {
        return;
    }

    bool have_pos_grad = pos_grad != nullptr;
    bool have_data_grad = data_grad != nullptr;

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    // Clamped cosine + polar-form distance below: same guard band
    // continuation past |theta| = 1 as the forward kernel.
    const float ct2 = 1.0f - theta*theta;
    const float x = r * (ct2 > 0.0f ? sqrtf(ct2) : 0.0f);
    const float y = r * theta;

    complex64_t g = grad[idbatch * Nr * Ntheta + idr * Ntheta + idtheta];

    float arg_dealias = 0.0f;
    if (dealias) {
        const float d = sqrtf(r*r + z0*z0);
        arg_dealias = -ref_phase * d + alias_fmod * idr;
        // TODO: Missing z0 gradient.
    }

    complex64_t I = {0.0f, 1.0f};

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        // Image plane is assumed to be at z=0
        float pz2 = pos_z * pos_z;

        // Distance in polar form, matches the forward kernel (also past
        // |theta| = 1). d(pos) derivatives are unchanged: d(d^2)/d(pos_x)
        // = 2*(pos_x - x) with x held constant.
        float d2 = r*r + pos_x*pos_x + pos_y*pos_y + pz2
                 - 2.0f * (x * pos_x + y * pos_y);
        float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);

        float sx = delta_r * (d + d0);

        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;
        complex64_t ds0 = 0.0f;
        complex64_t ds1 = 0.0f;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (sx >= 0.0f && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospi(ref_phase * d - data_fmod * sx + arg_dealias, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * (I * (kPI * (ref_phase - delta_r * data_fmod)) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * std::conj(dout);

                // Take real part
                float gd = std::real(gdout);

                dx = -px / d;
                dy = -py / d;
                // Different from x,y because pos_z is handled differently.
                dz = pos_z / d;
                dx *= gd;
                dy *= gd;
                dz *= gd;
                // Avoid issues with zero range
                if (!std::isfinite(dx)) dx = 0.0f;
                if (!std::isfinite(dy)) dy = 0.0f;
                if (!std::isfinite(dz)) dz = 0.0f;
            }

            if (have_data_grad) {
                ds0 = g * std::conj((1.0f - interp_idx) * ref);
                ds1 = g * std::conj(interp_idx * ref);
            }
        }

        if (have_pos_grad) {
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 0] += dx;
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 1] += dy;
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 2] += dz;
        }

        if (have_data_grad) {
            if (sx >= 0.0f && id1 < sweep_samples) {
                size_t data_idx = idbatch * sweep_samples * nsweeps + i * sweep_samples;
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id0])[0] += std::real(ds0);
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id0])[1] += std::imag(ds0);

                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id1])[0] += std::real(ds1);
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id1])[1] += std::imag(ds1);
            }
        }
    }
}

std::vector<at::Tensor> backprojection_polar_2d_grad_cpu(
          const at::Tensor &grad,
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &att,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double d0,
          int64_t dealias,
          double z0,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          double data_fmod,
          double alias_fmod,
          bool normalize,
          const at::Tensor &dem) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_CHECK(!dem.defined(), "backprojection_polar_2d gradient with dem is not supported");
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
	const c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();

    at::Tensor pos_grad;
    float* pos_grad_ptr = nullptr;
    if (pos.requires_grad()) {
        pos_grad = torch::zeros_like(pos);
        pos_grad_ptr = pos_grad.data_ptr<float>();
    } else {
        pos_grad = torch::Tensor();
    }

    at::Tensor data_grad;
	c10::complex<float>* data_grad_ptr = nullptr;
    if (data.requires_grad()) {
        data_grad = torch::zeros_like(data);
        data_grad_ptr = data_grad.data_ptr<c10::complex<float>>();
    } else {
        data_grad = torch::Tensor();
    }

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nr * Ntheta; idx++) {
            backprojection_polar_2d_grad_kernel_cpu(
                          data_ptr,
                          pos_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0,
                          dealias, z0,
                          data_fmod/kPI,
                          alias_fmod/kPI,
                          grad_ptr,
                          pos_grad_ptr,
                          data_grad_ptr,
                          idx,
                          idbatch
                          );
        }
    }
    std::vector<at::Tensor> ret;
    ret.push_back(data_grad);
    ret.push_back(pos_grad);
	return ret;
}

// One x-column of the image (fixed idx), the y pixels processed in chunks.
//
// Structured for SIMD like backprojection_polar_2d_row_cpu: per sweep, a
// vectorizable geometry + phase pass writes per-pixel sample index, fraction
// and reference phasor to small buffers, then a scalar pass does the
// data-dependent sample gather and accumulates. For a fixed x column, the y
// pixels are contiguous in the output image, so the final store is a plain
// vectorized write.
static void backprojection_cart_2d_row_cpu(
          const complex64_t* data,
          const float* pos,
          complex64_t* img,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          float x0,
          float dx,
          float y0,
          float dy,
          int Nx,
          int Ny,
          float d0,
          float data_fmod,
          int idx,
          int idbatch) {
    constexpr int CHUNK = 256;
    float y_buf[CHUNK];
    float accr[CHUNK], acci[CHUNK];
    float cs_buf[CHUNK], sn_buf[CHUNK], frac_buf[CHUNK];
    int idx_buf[CHUNK];

    const float x = x0 + idx * dx;
    const complex64_t* data_b = data + (size_t)idbatch * nsweeps * sweep_samples;
    const float* pos_b = pos + (size_t)idbatch * nsweeps * 3;
    complex64_t* img_row = img + ((size_t)idbatch * Nx + idx) * Ny;

    for (int yb = 0; yb < Ny; yb += CHUNK) {
        const int nchunk = std::min(CHUNK, Ny - yb);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            y_buf[q] = y0 + (yb + q) * dy;
            accr[q] = 0.0f;
            acci[q] = 0.0f;
        }

        for(int i = 0; i < nsweeps; i++) {
            // Sweep reference position.
            const float pos_x = pos_b[i * 3 + 0];
            const float pos_y = pos_b[i * 3 + 1];
            const float pos_z = pos_b[i * 3 + 2];
            const float pz2 = pos_z * pos_z;
            // x is constant across the column, so px and px*px are too.
            const float px = x - pos_x;
            const float px2 = px * px;
            const float* data_row = (const float*)(data_b + (size_t)i * sweep_samples);

            // Geometry + phase pass. idx = -1 marks pixels outside the data
            // range window.
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float py = y_buf[q] - pos_y;

                // Calculate distance to the pixel.
                const float d = sqrtf(px2 + py * py + pz2);

                const float sx = delta_r * (d + d0);
                const int id0 = (int)sx;
                // Float-domain check: (int)sx truncates toward zero, so
                // sx in (-1, 0) would pass an id0 >= 0 check and
                // extrapolate with a negative weight.
                const bool ok = (sx >= 0.0f) & (id0 + 1 < sweep_samples);
                idx_buf[q] = ok ? id0 : -1;
                frac_buf[q] = sx - id0;

                float ref_sin, ref_cos;
                sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
                cs_buf[q] = ref_cos;
                sn_buf[q] = ref_sin;
            }

            // Scalar pass: data gather (linear interpolation), multiply by
            // reference and accumulate. Forcing this to vectorize with clamped
            // indices + mask is slower: the vectorizer emulates the strided
            // complex gather.
            for (int q = 0; q < nchunk; q++) {
                const int id0 = idx_buf[q];
                if (id0 < 0) {
                    continue;
                }
                const float f = frac_buf[q];
                const float sr = (1.0f - f) * data_row[2*id0]     + f * data_row[2*id0 + 2];
                const float si = (1.0f - f) * data_row[2*id0 + 1] + f * data_row[2*id0 + 3];
                accr[q] += sr * cs_buf[q] - si * sn_buf[q];
                acci[q] += sr * sn_buf[q] + si * cs_buf[q];
            }
        }

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            img_row[yb + q] = complex64_t(accr[q], acci[q]);
        }
    }
}

at::Tensor backprojection_cart_2d_cpu(
          const at::Tensor &data,
          const at::Tensor &pos,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          double x0,
          double dx,
          double y0,
          double dy,
          int64_t Nx,
          int64_t Ny,
          double beamwidth,
          double d0,
          double data_fmod) {
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(data.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);
    at::Tensor pos_contig = pos.contiguous();
    at::Tensor data_contig = data.contiguous();
    at::Tensor img = torch::zeros({nbatch, Nx, Ny}, data_contig.options());
    const float* pos_ptr = pos_contig.data_ptr<float>();
    const complex64_t* data_ptr = data_contig.data_ptr<complex64_t>();
    complex64_t* img_ptr = img.data_ptr<complex64_t>();

    const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;
    // Divide by 2 to get angle from the center.
    const float beamwidth_f = beamwidth / 2.0f;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

    (void)beamwidth_f;

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nx; idx++) {
            backprojection_cart_2d_row_cpu(
                          data_ptr,
                          pos_ptr,
                          img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          d0,
                          data_fmod/kPI,
                          idx, idbatch);
        }
    }
    return img;
}

static void backprojection_cart_2d_grad_kernel_cpu(
          const complex64_t* data,
          const float* pos,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          float x0,
          float dx,
          float y0,
          float dy,
          int Nx,
          int Ny,
          float beamwidth,
          float d0,
          float data_fmod,
          const complex64_t* grad,
          float* pos_grad,
          complex64_t *data_grad,
          int idt,
          int idbatch) {
    const int idy = idt % Ny;
    const int idx = idt / Ny;

    if (idx >= Nx || idy >= Ny) {
        return;
    }

    bool have_pos_grad = pos_grad != nullptr;
    bool have_data_grad = data_grad != nullptr;

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;

    complex64_t g = grad[idbatch * Nx * Ny + idx * Ny + idy];

    complex64_t I = {0.0f, 1.0f};

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz2 = pos_z * pos_z;

        // Calculate distance to the pixel.
        float d = sqrtf(px * px + py * py + pz2);

        float sx = delta_r * (d + d0);

        float gx = 0.0f;
        float gy = 0.0f;
        float gz = 0.0f;
        complex64_t ds0 = 0.0f;
        complex64_t ds1 = 0.0f;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (sx >= 0.0f && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * (I * (kPI * (ref_phase - delta_r * data_fmod)) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * std::conj(dout);

                // Take real part
                float gd = std::real(gdout);

                gx = -px / d;
                gy = -py / d;
                // Different from x,y because pos_z is handled differently.
                gz = pos_z / d;
                gx *= gd;
                gy *= gd;
                gz *= gd;
                // Avoid issues with zero range
                if (!std::isfinite(gx)) gx = 0.0f;
                if (!std::isfinite(gy)) gy = 0.0f;
                if (!std::isfinite(gz)) gz = 0.0f;
            }

            if (have_data_grad) {
                ds0 = g * std::conj((1.0f - interp_idx) * ref);
                ds1 = g * std::conj(interp_idx * ref);
            }
        }

        if (have_pos_grad) {
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 0] += gx;
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 1] += gy;
#pragma omp atomic
            pos_grad[idbatch * nsweeps * 3 + i * 3 + 2] += gz;
        }

        if (have_data_grad) {
            if (sx >= 0.0f && id1 < sweep_samples) {
                size_t data_idx = idbatch * sweep_samples * nsweeps + i * sweep_samples;
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id0])[0] += std::real(ds0);
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id0])[1] += std::imag(ds0);

                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id1])[0] += std::real(ds1);
                #pragma omp atomic
                reinterpret_cast<float*>(&data_grad[data_idx + id1])[1] += std::imag(ds1);
            }
        }
    }
}

std::vector<at::Tensor> backprojection_cart_2d_grad_cpu(
          const at::Tensor &grad,
          const at::Tensor &data,
          const at::Tensor &pos,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          double x0,
          double dx,
          double y0,
          double dy,
          int64_t Nx,
          int64_t Ny,
          double beamwidth,
          double d0,
          double data_fmod) {
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(data.dtype() == at::kComplexFloat);
    TORCH_CHECK(grad.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
    at::Tensor pos_contig = pos.contiguous();
    at::Tensor data_contig = data.contiguous();
    at::Tensor grad_contig = grad.contiguous();
    const float* pos_ptr = pos_contig.data_ptr<float>();
    const complex64_t* data_ptr = data_contig.data_ptr<complex64_t>();
    const complex64_t* grad_ptr = grad_contig.data_ptr<complex64_t>();

    at::Tensor pos_grad;
    float* pos_grad_ptr = nullptr;
    if (pos.requires_grad()) {
        pos_grad = torch::zeros_like(pos);
        pos_grad_ptr = pos_grad.data_ptr<float>();
    } else {
        pos_grad = torch::Tensor();
    }

    at::Tensor data_grad;
    complex64_t* data_grad_ptr = nullptr;
    if (data.requires_grad()) {
        data_grad = torch::zeros_like(data);
        data_grad_ptr = data_grad.data_ptr<complex64_t>();
    } else {
        data_grad = torch::Tensor();
    }

    const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;
    // Divide by 2 to get angle from the center.
    const float beamwidth_f = beamwidth / 2.0f;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idt = 0; idt < Nx * Ny; idt++) {
            backprojection_cart_2d_grad_kernel_cpu(
                          data_ptr,
                          pos_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          beamwidth_f, d0,
                          data_fmod/kPI,
                          grad_ptr,
                          pos_grad_ptr,
                          data_grad_ptr,
                          idt, idbatch);
        }
    }
    std::vector<at::Tensor> ret;
    ret.push_back(data_grad);
    ret.push_back(pos_grad);
    return ret;
}
// One output row of the decimated illumination maps, theta in chunks.
// Structured for SIMD like backprojection_polar_2d_row_cpu: per sweep, a
// vectorizable angle pass (asinf_fast/atan2f_fast, matching the
// backprojection kernel's gain-lookup conventions) writes the bilinear
// pattern coordinates to small buffers, then a scalar pass gathers the
// gain and accumulates the moments. gi = -1 marks out-of-pattern pixels,
// which contribute exactly zero: the merge gating and the Wiener
// normalization rely on W counting exactly the gains that weight A.
static void compute_illumination_row_cpu(
          const float* pos,
          const float* att,
          const float* g,
          float* w1_out,
          float* w2_out,
          int nsweeps,
          float r0, float dr, float theta0, float dtheta,
          float g_el0, float g_del, float g_az0, float g_daz, int g_nel, int g_naz,
          int decimation, int out_ntheta, int out_idr) {
    constexpr int CHUNK = 256;
    float x_buf[CHUNK], y_buf[CHUNK];
    float w1_buf[CHUNK], w2_buf[CHUNK];
    float ef_buf[CHUNK], af_buf[CHUNK];
    int gi_buf[CHUNK];

    const float r = r0 + dr * (out_idr * decimation);
    const float r2 = r * r;
    float* w1_row = w1_out + (size_t)out_idr * out_ntheta;
    float* w2_row = w2_out + (size_t)out_idr * out_ntheta;

    for (int tb = 0; tb < out_ntheta; tb += CHUNK) {
        const int nchunk = std::min(CHUNK, out_ntheta - tb);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            const float t = theta0 + dtheta * ((tb + q) * decimation);  // sin(theta)
            // Clamped cosine + polar-form distance: guard band grids
            // extend past |theta| = 1 (see the backprojection kernel
            // note).
            const float ct2 = 1.0f - t*t;
            x_buf[q] = r * (ct2 > 0.0f ? sqrtf(ct2) : 0.0f);
            y_buf[q] = r * t;
            w1_buf[q] = 0.0f;
            w2_buf[q] = 0.0f;
        }

        for (int i = 0; i < nsweeps; i++) {
            const float pos_x = pos[i * 3 + 0];
            const float pos_y = pos[i * 3 + 1];
            const float pos_z = pos[i * 3 + 2];
            const float r2pn2 = r2 + pos_x*pos_x + pos_y*pos_y + pos_z*pos_z;
            float att_el = 0.0f;
            float att_az = 0.0f;
            if (att != nullptr) {
                att_el = att[i * 3 + 0];
                att_az = att[i * 3 + 2];
            }

            // Angle pass: bilinear antenna pattern coordinates.
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float px = x_buf[q] - pos_x;
                const float py = y_buf[q] - pos_y;
                const float d2 = r2pn2 - 2.0f * (x_buf[q] * pos_x + y_buf[q] * pos_y);
                const float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);
                const float sin_l = -pos_z / d;
                const float look_angle = asinf_fast(sin_l < -1.0f ? -1.0f : sin_l);
                const float el = look_angle - att_el;
                const float az = atan2f_fast(py, px) - att_az;

                const float el_idx = (el - g_el0) / g_del;
                const float az_idx = (az - g_az0) / g_daz;
                const int el_int = (int)el_idx;
                const int az_int = (int)az_idx;
                const bool ok = (el_idx >= 0.0f) & (el_int + 1 < g_nel) &
                                (az_idx >= 0.0f) & (az_int + 1 < g_naz);
                ef_buf[q] = el_idx - el_int;
                af_buf[q] = az_idx - az_int;
                gi_buf[q] = ok ? el_int * g_naz + az_int : -1;
            }
            // Scalar pass: gain gather, accumulate the moments.
            for (int q = 0; q < nchunk; q++) {
                const int gi = gi_buf[q];
                if (gi < 0) {
                    continue;
                }
                const float ef = ef_buf[q], af = af_buf[q];
                const float v00 = g[gi],         v01 = g[gi + 1];
                const float v10 = g[gi + g_naz], v11 = g[gi + g_naz + 1];
                const float w = v00 * (1.0f - ef) * (1.0f - af)
                              + v01 * (1.0f - ef) * af
                              + v10 * ef * (1.0f - af)
                              + v11 * ef * af;
                w1_buf[q] += w;
                w2_buf[q] += w * w;
            }
        }

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            w1_row[tb + q] = w1_buf[q];
            w2_row[tb + q] = w2_buf[q];
        }
    }
}

std::vector<at::Tensor> compute_illumination_cpu(
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t nr,
          int64_t ntheta,
          int64_t decimation) {
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(g.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);

    const int64_t nsweeps = pos.size(0);
    const int64_t g_nel = g.size(0);
    const int64_t g_naz = g.size(1);

    const float* att_ptr = nullptr;
    at::Tensor att_contig;
    if (att.defined() && att.numel() > 0) {
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
        att_contig = att.contiguous();
        att_ptr = att_contig.data_ptr<float>();
    }

    at::Tensor pos_contig = pos.contiguous();
    at::Tensor g_contig = g.contiguous();

    const int64_t dec = decimation > 0 ? decimation : 1;
    const int64_t out_nr = (nr + dec - 1) / dec;
    const int64_t out_ntheta = (ntheta + dec - 1) / dec;

    at::Tensor w1_out = torch::empty({out_nr, out_ntheta},
                                     torch::TensorOptions().dtype(at::kFloat).device(pos.device()));
    at::Tensor w2_out = torch::empty({out_nr, out_ntheta},
                                     torch::TensorOptions().dtype(at::kFloat).device(pos.device()));

    const float* pos_ptr = pos_contig.data_ptr<float>();
    const float* g_ptr = g_contig.data_ptr<float>();
    float* w1_out_ptr = w1_out.data_ptr<float>();
    float* w2_out_ptr = w2_out.data_ptr<float>();

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idr = 0; idr < out_nr; idr++) {
        compute_illumination_row_cpu(
                pos_ptr, att_ptr, g_ptr, w1_out_ptr, w2_out_ptr, nsweeps,
                r0, dr, theta0, dtheta,
                g_el0, g_del, g_az0, g_daz, g_nel, g_naz, dec, out_ntheta, idr);
    }

    std::vector<at::Tensor> ret;
    ret.push_back(w1_out);
    ret.push_back(w2_out);
    return ret;
}

/*
NUFFT-based forward projection (vel=None path). CPU analogue of
projection_cart_2d_nufft_cuda in cuda/backproj.cu.

Reformulates the direct O(N*M) sum as a Type-1 NUFFT:
  A_p  = w_p exp(-j 2pi fc tau_p)          complex amplitude per pixel
  nu_p = gamma tau_p / fs                  normalised frequency
  data[k] = Sum_p A_p exp(-j 2pi nu_p k)   = IFFT of the KB-spread grid
Cost is O(N*W) spreading + O(M log M) IFFT per sweep instead of O(N*M).

The KB look-up table and deconvolution window are bit-identical to the CUDA
build (same host-side formulas), so CPU and CUDA outputs match to float
precision.
*/

// Kaiser-Bessel helper (matches bessel_i0_host in cuda/backproj.cu).
static float bessel_i0_cpu(float x) {
    if (x < 3.75f) {
        float t = (x / 3.75f) * (x / 3.75f);
        return 1.0f + t*(3.5156229f + t*(3.0899424f + t*(1.2067492f
             + t*(0.2659732f + t*(0.0360768f + t*0.0045813f)))));
    }
    float t = 3.75f / x;
    return (expf(x) / sqrtf(x)) * (0.39894228f + t*(0.01328592f
         + t*(0.00225319f + t*(-0.00157565f + t*(0.00916281f
         + t*(-0.02057706f + t*(0.02635537f + t*(-0.01647633f
         + t*0.00392377f))))))));
}

// KB LUT: N_LUT samples of psi_KB(t) for t in [0, W/2].
static at::Tensor make_kb_lut_cpu(int N_LUT, float W, float beta) {
    at::Tensor lut = at::zeros({N_LUT},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    float* p = lut.data_ptr<float>();
    float i0b = bessel_i0_cpu(beta);
    for (int i = 0; i < N_LUT; ++i) {
        float t  = (float)i / (float)(N_LUT - 1) * (W * 0.5f);  // [0, W/2]
        float u  = 2.0f * t / W;                                // [0, 1]
        float sq = 1.0f - u * u;
        p[i] = (sq > 0.0f) ? bessel_i0_cpu(beta * sqrtf(sq)) / i0b : 0.0f;
    }
    return lut;
}

// Deconvolution window for centered extraction (see cuda/backproj.cu for the
// derivation): correction[j] = M_ext / conj(phi_fwd[k]), k = j - M/2.
static at::Tensor make_deconv_window_cpu(int M, int M_ext, float W, float beta) {
    int half_W = (int)(W * 0.5f);
    float i0b  = bessel_i0_cpu(beta);
    std::vector<float> psi(half_W + 1);
    for (int d = 0; d <= half_W; ++d) {
        float u  = (float)d / (float)half_W;
        float sq = 1.0f - u * u;
        psi[d] = (sq > 0.0f) ? bessel_i0_cpu(beta * sqrtf(sq)) / i0b : 0.0f;
    }
    at::Tensor win = at::zeros({M},
        at::TensorOptions().dtype(at::kComplexFloat).device(at::kCPU));
    complex64_t* w = win.data_ptr<complex64_t>();
    for (int j = 0; j < M; ++j) {
        int k = j - M / 2;
        float re = 0.0f, im = 0.0f;
        for (int d = -(half_W - 1); d <= half_W; ++d) {
            float angle = -2.0f * (float)M_PI * (float)k * (float)d / (float)M_ext;
            float kb    = psi[abs(d)];
            re += kb * cosf(angle);
            im += kb * sinf(angle);
        }
        float denom = re * re + im * im;
        float scale = (float)M_ext / denom;
        w[j] = complex64_t(re * scale, +im * scale);
    }
    return win;
}

// Static cache keyed on (sweep_samples, M_ext); W/beta/N_LUT are fixed.
struct NufftCacheCpu { at::Tensor kb_lut, deconv_win; };
static std::mutex                               g_nufft_cpu_cache_mutex;
static std::map<std::tuple<int,int>, NufftCacheCpu> g_nufft_cpu_cache;

static std::pair<at::Tensor, at::Tensor>
get_nufft_tables_cpu(int sweep_samples, int M_ext, float W, float beta, int N_LUT) {
    std::lock_guard<std::mutex> lock(g_nufft_cpu_cache_mutex);
    auto key = std::make_tuple(sweep_samples, M_ext);
    auto it  = g_nufft_cpu_cache.find(key);
    if (it != g_nufft_cpu_cache.end())
        return {it->second.kb_lut, it->second.deconv_win};
    NufftCacheCpu c;
    c.kb_lut     = make_kb_lut_cpu(N_LUT, W, beta);
    c.deconv_win = make_deconv_window_cpu(sweep_samples, M_ext, W, beta);
    g_nufft_cpu_cache[key] = c;
    return {c.kb_lut, c.deconv_win};
}

// Geometry + KB spreading for one sweep row. This thread owns grid_row
// (M_ext bins) exclusively, so the scatter needs no atomics.
//
// Structured for SIMD: pixels are processed in chunks, with geometry+phase pass
// writing per-pixel amplitude and grid position to small buffers, followed by
// a scatter pass into a pad-extended scratch row. The padding makes every tap
// index in [j0*(half_W-1), j0+half_W] valid without the per-tap modulo of the
// direct formulation, the pads are folded back once per row.  The antenna gain
// uses asinf_fast/atan2f_fast, differing by a few ulp for libm.
static void projection_nufft_spread_row_cpu(
          const complex64_t* img,   // [N] this batch's image
          const float* dem,         // [N] or nullptr
          const float* pos_s,       // [3] this sweep
          const float* att_s,       // [3] or nullptr
          complex64_t* grid_row,    // [M_ext] zeroed
          const float* kb_lut,
          int N, int Nx, int Ny, int M_ext, int N_LUT,
          float fc, float gamma, float fs, float d0,
          float x0, float dx, float y0, float dy,
          float W, bool use_rvp,
          const float* g,
          float g_az0, float g_el0, float g_daz, float g_del,
          int g_naz, int g_nel, int normalization) {

    const float M_ext_f   = (float)M_ext;
    const float lut_scale = (float)(N_LUT - 1) * 2.0f / W;
    const int   half_W    = (int)(W * 0.5f);

    // Scratch row with half_W pad on both sides (tap indices span
    // [-(half_W-1), M_ext+half_W]), sc indexes the unpadded part.
    std::vector<complex64_t> scratch(M_ext + 2 * half_W + 1,
                                     complex64_t(0.0f, 0.0f));
    complex64_t* sc = scratch.data() + half_W;

    constexpr int CHUNK = 256;
    float ar_buf[CHUNK], ai_buf[CHUNK], up_buf[CHUNK];
    float gain_buf[CHUNK], ef_buf[CHUNK], af_buf[CHUNK];
    int gi_buf[CHUNK];
    float zero_dem[CHUNK] = {};

    const float pos_y = pos_s[1], pos_z = pos_s[2];
    const float rvp_c = use_rvp ? gamma : 0.0f;

    for (int px_x = 0; px_x < Nx; px_x++) {
        const float x = x0 + px_x * dx;
        const float dpx  = x - pos_s[0];
        const float dpx2 = dpx * dpx;

        for (int yb = 0; yb < Ny; yb += CHUNK) {
            const int nchunk = std::min(CHUNK, Ny - yb);
            const float* img_f = (const float*)(img + (size_t)px_x * Ny + yb);
            const float* dem_row = dem ? dem + (size_t)px_x * Ny + yb : nullptr;
            const float* __restrict dem_q = dem_row ? dem_row : zero_dem;
            const float ybase = y0 + yb * dy;

            if (g == nullptr) {
                for (int q = 0; q < nchunk; q++) gain_buf[q] = 1.0f;
            } else {
                const float att0 = att_s[0];
                const float att2 = att_s[2];
                float* __restrict ef_q = ef_buf;
                float* __restrict af_q = af_buf;
                int* __restrict gi_q = gi_buf;
                // Vectorizable index pass: look/azimuth angles and bilinear
                // coordinates. gi = -1 marks out-of-pattern pixels (gain 0);
                // NaN angles (pixel exactly at or directly below the radar)
                // convert to INT_MIN and take the same path.
#pragma omp simd
                for (int q = 0; q < nchunk; q++) {
                    const float y = ybase + q * dy;
                    const float z = dem_q[q];
                    const float dpy = y - pos_y;
                    const float dpz = z - pos_z;
                    const float d = sqrtf(dpx2 + dpy*dpy + dpz*dpz);
                    const float sin_l  = dpz / d;
                    const float look   = asinf_fast(sin_l < -1.0f ? -1.0f : sin_l);
                    const float el_deg = look - att0;
                    const float az_deg = atan2f_fast(dpy, dpx) - att2;
                    const float el_idx = (el_deg - g_el0) / g_del;
                    const float az_idx = (az_deg - g_az0) / g_daz;
                    const int el_int = (int)el_idx, az_int = (int)az_idx;
                    const bool ok = (el_idx >= 0.0f) & (el_int + 1 < g_nel) &
                                    (az_idx >= 0.0f) & (az_int + 1 < g_naz);
                    ef_q[q] = el_idx - el_int;
                    af_q[q] = az_idx - az_int;
                    gi_q[q] = ok ? el_int * g_naz + az_int : -1;
                }
                // Scalar pass for the bilinear pattern lookup (gathers, which
                // the vectorizer doesn't handle here).
                for (int q = 0; q < nchunk; q++) {
                    const int gi = gi_buf[q];
                    if (gi < 0) { gain_buf[q] = 0.0f; continue; }
                    const float ef = ef_buf[q], af = af_buf[q];
                    const float v00 = g[gi],         v01 = g[gi + 1];
                    const float v10 = g[gi + g_naz], v11 = g[gi + g_naz + 1];
                    gain_buf[q] = v00 * (1.0f - ef) * (1.0f - af)
                                + v01 * (1.0f - ef) * af
                                + v10 * ef * (1.0f - af)
                                + v11 * ef * af;
                }
            }

            const float* __restrict img_q = img_f;
            const float* __restrict gain_q = gain_buf;
            float* __restrict ar_q = ar_buf;
            float* __restrict ai_q = ai_buf;
            float* __restrict up_q = up_buf;
            const bool gamma_norm = normalization == 1;

            // Geometry + phase pass.
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float y = ybase + q * dy;
                const float z = dem_q[q];
                const float dpy = y - pos_y;
                const float dpz = z - pos_z;
                const float d2  = dpx2 + dpy*dpy + dpz*dpz;
                const float d   = sqrtf(d2);

                float norm = gain_q[q] / d2;
                norm *= gamma_norm ? sqrtf(-dpz / d) : 1.0f;

                const float wr = img_q[2*q] * norm;
                const float wi = img_q[2*q+1] * norm;

                const float tau  = 2.0f * (d + d0) / kC0;
                const float nu_p = gamma * tau / fs;

                // Carrier phase + centering term exp(-j pi nu_p M).
                float phase = -2.0f * fc * tau;
                phase += rvp_c * tau * tau;
                phase -= nu_p * M_ext_f * 0.5f;
                float sn, cs;
                sincospi(phase, &sn, &cs);

                ar_q[q] = wr * cs - wi * sn;
                ai_q[q] = wr * sn + wi * cs;

                // exp(+j 2pi u_p k/M_ext) = exp(-j 2pi nu_p k) for integer k.
                // Wrapping nu_p to [0, 1) puts j0 in [0, M_ext] so the tap
                // loop needs no modulo (pads + fold handle the edges).
                const float nu_w = nu_p - floorf(nu_p);
                up_q[q] = M_ext_f - nu_w * M_ext_f;
            }

            // Scatter pass. ar = ai = 0 means a masked (or zero) pixel:
            // amplitude is a pure rotation of w, so it vanishes only when w
            // does, and spreading an exact zero is a no-op.
            for (int q = 0; q < nchunk; q++) {
                const float ar = ar_buf[q], ai = ai_buf[q];
                if (ar == 0.0f && ai == 0.0f) continue;
                const float up = up_buf[q];
                const int   j0  = (int)rintf(up);
                const float frc = up - (float)j0;
                complex64_t* dst = sc + j0;
                for (int delta = -(half_W - 1); delta <= half_W; ++delta) {
                    float dist    = fabsf(frc - (float)delta);
                    int   lut_idx = (int)(dist * lut_scale);
                    if (lut_idx > N_LUT - 1) lut_idx = N_LUT - 1;
                    float kb = kb_lut[lut_idx];
                    dst[delta] += complex64_t(ar * kb, ai * kb);
                }
            }
        }
    }

    // Fold the pads back into the periodic grid row.
    for (int k = 0; k < M_ext; k++)
        grid_row[k] += sc[k];
    for (int k = -half_W; k < 0; k++)
        grid_row[k + M_ext] += sc[k];
    for (int k = M_ext; k <= M_ext + half_W; k++)
        grid_row[k - M_ext] += sc[k];
}

at::Tensor projection_cart_2d_nufft_cpu(
          const at::Tensor &img,
          const at::Tensor &dem,
          const at::Tensor &pos,
          const at::Tensor &att,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double fs,
          double gamma,
          double x0, double dx, double y0, double dy,
          int64_t Nx, int64_t Ny, double d0,
          const at::Tensor &g,
          double g_az0, double g_el0, double g_daz, double g_del,
          int64_t g_naz, int64_t g_nel,
          int64_t use_rvp, int64_t normalization) {
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(img.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
    }
    if (dem.defined()) {
        TORCH_CHECK(dem.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CPU);
    }

    const int N     = (int)(Nx * Ny);
    const int M     = (int)sweep_samples;
    const int M_ext = 2 * M;            // oversampled grid size
    const float W   = 6.0f;
    const float beta = 2.34f * W;       // KB parameter
    const int N_LUT = 8192;

    auto [kb_lut, deconv_win] = get_nufft_tables_cpu(M, M_ext, W, beta, N_LUT);

    at::Tensor img_contig = img.contiguous();
    at::Tensor pos_contig = pos.contiguous();
    auto cplx_opts = at::TensorOptions().dtype(at::kComplexFloat).device(at::kCPU);
    at::Tensor grid = at::zeros({nbatch, nsweeps, M_ext}, cplx_opts);

    at::Tensor dem_contig, att_contig, g_contig;
    const float* dem_ptr = nullptr;
    if (dem.defined()) { dem_contig = dem.contiguous(); dem_ptr = dem_contig.data_ptr<float>(); }
    const float* att_ptr = nullptr;
    const float* g_ptr = nullptr;
    if (antenna_pattern) {
        att_contig = att.contiguous();
        g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
    }

    const complex64_t* img_ptr = img_contig.data_ptr<complex64_t>();
    const float*       pos_ptr = pos_contig.data_ptr<float>();
    const float*       lut_ptr = kb_lut.data_ptr<float>();
    complex64_t*       grid_ptr = grid.data_ptr<complex64_t>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    // Each (batch, sweep) spreads into its own grid row -> no atomics.
#pragma omp parallel for collapse(2)
    for (int b = 0; b < nbatch; b++) {
        for (int i = 0; i < nsweeps; i++) {
            const float* pos_si = pos_ptr + (b * nsweeps + i) * 3;
            const float* att_si = att_ptr ? att_ptr + (b * nsweeps + i) * 3 : nullptr;
            projection_nufft_spread_row_cpu(
                img_ptr + b * N, dem_ptr, pos_si, att_si,
                grid_ptr + (b * nsweeps + (size_t)i) * M_ext, lut_ptr,
                N, Nx, Ny, M_ext, N_LUT,
                fc, gamma, fs, d0, x0, dx, y0, dy, W, (bool)use_rvp,
                g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, normalization);
        }
    }

    // Batched IFFT over the last dimension (same call/normalization as CUDA).
    at::Tensor Z = at::fft_ifft(grid, std::optional<int64_t>(M_ext), -1);
    at::Tensor Z_contig = Z.contiguous();

    at::Tensor data = at::empty({nbatch, nsweeps, M}, cplx_opts);
    const complex64_t* Z_ptr  = Z_contig.data_ptr<complex64_t>();
    const complex64_t* dw_ptr = deconv_win.data_ptr<complex64_t>();
    complex64_t*       data_ptr = data.data_ptr<complex64_t>();

    // Deconvolve + centered extraction: out[j] = Z[(j - M/2) mod M_ext] * deconv[j].
#pragma omp parallel for collapse(2)
    for (int b = 0; b < nbatch; b++) {
        for (int i = 0; i < nsweeps; i++) {
            const complex64_t* z_row = Z_ptr + (b * nsweeps + (size_t)i) * M_ext;
            complex64_t* out_row = data_ptr + (b * nsweeps + (size_t)i) * M;
            for (int j = 0; j < M; j++) {
                int k = ((j - M / 2) % M_ext + M_ext) % M_ext;
                out_row[j] = z_row[k] * dw_ptr[j];
            }
        }
    }

    return data;
}

// Number of samples between exact (sincospi) re-anchors of the chirp
// recurrence below. The float32 phasor magnitude drifts by ~1e-6 per step, so
// re-pinning every RESYNC samples keeps the accumulated error well under the
// kernel's inherent float32 precision (~1e-3 relative).
#define PROJ_RESYNC 256

// Computes one full output row data[idbatch, i, :] (all sweep_samples of sweep
// i) by accumulating the contribution of every image pixel. CPU analogue of
// projection_cart_2d_kernel in cuda/backproj.cu.
//
// For a fixed (sweep, pixel) the phase is exactly quadratic in the sample index
// j:  phase(j) = (a0 + a1*j) * (s0 + s1*j) [+ gamma*(a0+a1*j)^2 for RVP], where
// tau_s(j) = a0 + a1*j and the IF slope is s0 + s1*j. So exp(i*pi*phase(j)) is a
// chirp generated by a two-multiply recurrence (one multiply when a1 == 0, i.e.
// no velocity). This computes the per-pixel geometry (sqrt, antenna pattern,
// normalization) once instead of once per sample, and replaces the per-sample
// sincospi with a complex multiply — the slow per-sample form recomputed all of
// that sweep_samples times over.
static void projection_cart_2d_kernel_cpu(
          const complex64_t* img,
          const float* dem,
          const float* pos,
          const float* vel,   // nullptr when !has_vel
          const float* att,   // nullptr when g == nullptr
          complex64_t* data,
          int sweep_samples,
          int nsweeps,
          float fc,
          float fs,
          float gamma,
          float x0,
          float dx,
          float y0,
          float dy,
          int Nx,
          int Ny,
          float d0,
          const float* g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          bool use_rvp,
          int normalization,
          int i,
          int idbatch) {
    if (i >= nsweeps) {
        return;
    }

    const bool has_vel = (vel != nullptr);
    const int N = sweep_samples;

    // IF slope coefficients: phase_slope(j) = s0 + s1*j.
    const float s0 = -2.0f * fc;
    const float s1 = -2.0f * gamma / fs;

    const float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
    const float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
    const float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
    const float att_roll = (g != nullptr) ? att[idbatch * nsweeps * 3 + 3*i + 0] : 0.0f;
    const float att_yaw  = (g != nullptr) ? att[idbatch * nsweeps * 3 + 3*i + 2] : 0.0f;

    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    if (has_vel) {
        vx = vel[idbatch * nsweeps * 3 + i * 3 + 0];
        vy = vel[idbatch * nsweeps * 3 + i * 3 + 1];
        vz = vel[idbatch * nsweeps * 3 + i * 3 + 2];
    }

    const int total_pixels = Nx * Ny;
    // This thread owns the whole row, so it accumulates directly (no atomics).
    complex64_t* row = data + (idbatch * nsweeps + (size_t)i) * N;

    for (int p_idx = 0; p_idx < total_pixels; p_idx++) {
        const int px_y = p_idx % Ny;
        const int px_x = p_idx / Ny;
        const float x = x0 + px_x * dx;
        const float y = y0 + px_y * dy;
        const float z = (dem != nullptr) ? dem[px_x * Ny + px_y] : 0.0f;

        const complex64_t w_img = img[idbatch * total_pixels + p_idx];

        const float dpx = x - pos_x;
        const float dpy = y - pos_y;
        const float dpz = z - pos_z;
        const float d2  = dpx*dpx + dpy*dpy + dpz*dpz;
        const float d   = sqrtf(d2);

        float norm = 1.0f / d2;
        if (g != nullptr) {
            const float look   = asinf(fmaxf(dpz / d, -1.0f));
            const float el_deg = look - att_roll;
            const float az_deg = atan2f(dpy, dpx) - att_yaw;
            const float el_idx = (el_deg - g_el0) / g_del;
            const float az_idx = (az_deg - g_az0) / g_daz;
            const int el_int = (int)el_idx, az_int = (int)az_idx;
            if (el_idx < 0.0f || el_int+1 >= g_nel || az_idx < 0.0f || az_int+1 >= g_naz)
                norm = 0.0f;
            else
                norm *= interp2d<float>(g, g_nel, g_naz,
                                        el_int, el_idx - el_int,
                                        az_int, az_idx - az_int);
        }
        if (normalization == 1) norm *= sqrtf(-dpz / d);

        // Out-of-pattern pixels contribute nothing; skip the chirp entirely.
        if (norm == 0.0f) continue;

        const complex64_t w(w_img.real() * norm, w_img.imag() * norm);

        // tau_s(j) = a0 + a1*j. Without velocity a1 = 0 (tau constant per pixel).
        float a0, a1 = 0.0f;
        if (has_vel) {
            const float vel_proj = (dpx * vx + dpy * vy + dpz * vz) / d / fs;
            a1 = 2.0f * vel_proj / kC0;
            // j_half = j - N/2, so a0 absorbs the -N/2 offset.
            a0 = 2.0f * (d + d0) / kC0 - a1 * (0.5f * (float)N);
        } else {
            a0 = 2.0f * (d + d0) / kC0;
        }

        // phase(j) = A + B*j + C*j^2. B and C are small and computed directly so
        // the per-step phase increment stays accurate (no cancellation of the
        // huge absolute phase). The anchor phase itself is evaluated from the
        // factored form below to match the CUDA kernel bit-for-bit.
        float B = a0 * s1 + a1 * s0;
        float C = a1 * s1;
        if (use_rvp) { B += 2.0f * gamma * a0 * a1; C += gamma * a1 * a1; }

        // Second-difference phasor: d1 *= d2 each step (d2 == 1 when C == 0).
        float d2s, d2c;
        sincospi(2.0f * C, &d2s, &d2c);
        const complex64_t d2c1(d2c, d2s);

        for (int j0 = 0; j0 < N; j0 += PROJ_RESYNC) {
            const int jend = (j0 + PROJ_RESYNC < N) ? j0 + PROJ_RESYNC : N;

            // Exact anchor phase at j0 (same expression as the per-sample form).
            const float tau_s = a0 + a1 * (float)j0;
            const float slope = s0 + s1 * (float)j0;
            float phase0 = tau_s * slope;
            if (use_rvp) phase0 += gamma * tau_s * tau_s;
            float p0s, p0c;
            sincospi(phase0, &p0s, &p0c);
            complex64_t P = w * complex64_t(p0c, p0s);  // weight folded in

            // First-difference phasor at j0: exp(i*pi*(phase(j0+1)-phase(j0))).
            const float g0 = B + C * (2.0f * (float)j0 + 1.0f);
            float g0s, g0c;
            sincospi(g0, &g0s, &g0c);
            complex64_t d1(g0c, g0s);

            if (has_vel) {
                for (int j = j0; j < jend; j++) {
                    row[j] += P;
                    P *= d1;
                    d1 *= d2c1;
                }
            } else {
                // C == 0: d1 is constant. The bare recurrence P *= d1 is a
                // latency-bound serial chain; run 4 independent phasor lanes
                // (offset by one sample, stepping by d1^4) so the multiply
                // latency is hidden by instruction-level parallelism.
                complex64_t P0 = P;
                complex64_t P1 = P0 * d1;
                complex64_t P2 = P1 * d1;
                complex64_t P3 = P2 * d1;
                const complex64_t d1_2 = d1 * d1;
                const complex64_t step = d1_2 * d1_2;  // d1^4
                int j = j0;
                for (; j + 4 <= jend; j += 4) {
                    row[j + 0] += P0;
                    row[j + 1] += P1;
                    row[j + 2] += P2;
                    row[j + 3] += P3;
                    P0 *= step;
                    P1 *= step;
                    P2 *= step;
                    P3 *= step;
                }
                // P0 now tracks phase(j); finish the 0-3 sample tail serially.
                for (; j < jend; j++) {
                    row[j] += P0;
                    P0 *= d1;
                }
            }
        }
    }
}

at::Tensor projection_cart_2d_cpu(
          const at::Tensor &img,
          const at::Tensor &dem,
          const at::Tensor &pos,
          const at::Tensor &vel,
          const at::Tensor &att,
          int64_t nbatch,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double fs,
          double gamma,
          double x0,
          double dx,
          double y0,
          double dy,
          int64_t Nx,
          int64_t Ny,
          double d0,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t use_rvp,
          int64_t normalization) {
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(img.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
    }
    if (vel.defined()) {
        TORCH_CHECK(vel.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CPU);
    }
    if (dem.defined()) {
        TORCH_CHECK(dem.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CPU);
    }

    at::Tensor pos_contig = pos.contiguous();
    at::Tensor img_contig = img.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(img.device());
    at::Tensor data = torch::zeros({nbatch, nsweeps, sweep_samples}, options);
    const float* pos_ptr = pos_contig.data_ptr<float>();
    const complex64_t* img_ptr = img_contig.data_ptr<complex64_t>();
    complex64_t* data_ptr = data.data_ptr<complex64_t>();

    at::Tensor dem_contig, vel_contig, att_contig, g_contig;
    const float* dem_ptr = nullptr;
    if (dem.defined()) {
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
    }
    const float* vel_ptr = nullptr;
    if (vel.defined()) {
        vel_contig = vel.contiguous();
        vel_ptr = vel_contig.data_ptr<float>();
    }
    const float* att_ptr = nullptr;
    const float* g_ptr = nullptr;
    if (antenna_pattern) {
        att_contig = att.contiguous();
        g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
    }

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    // Each (batch, sweep) owns one output row, so parallelize over rows: every
    // thread accumulates all pixels into its own row without contention.
#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int i = 0; i < nsweeps; i++) {
            projection_cart_2d_kernel_cpu(
                          img_ptr, dem_ptr, pos_ptr, vel_ptr, att_ptr,
                          data_ptr, sweep_samples, nsweeps,
                          fc, fs, gamma, x0, dx, y0, dy, Nx, Ny, d0,
                          g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
                          (bool)use_rvp, normalization,
                          i, idbatch);
        }
    }
    return data;
}
// Generalized phase gradient autofocus: demodulated phase of each target for
// every sweep. Same math as gpga_backprojection_2d_kernel in cuda/backproj.cu
// (complex64 data, linear range interpolation), restructured per sweep for
// memory locality: all targets of one sweep gather from the same data row
// (targets arrive sorted by range, so the row accesses are nearly
// sequential), where a target-outer order would touch a different
// ~100 KB-distant row on every iteration and be DRAM latency bound.
// Structured like backprojection_polar_2d_row_cpu: a vectorizable geometry
// + phase pass into small buffers, then a scalar gather pass.
static void gpga_backprojection_2d_sweep_cpu(
          const float* tx, const float* ty, const float* tz,
          const complex64_t* data, const float* pos,
          complex64_t* data_out, int sweep_samples, int nsweeps,
          float ref_phase, float delta_r, int Ntarget, float d0, float data_fmod,
          int idsweep) {
    constexpr int CHUNK = 256;
    float frac_buf[CHUNK], cs_buf[CHUNK], sn_buf[CHUNK];
    int idx_buf[CHUNK];

    const float pos_x = pos[idsweep * 3 + 0];
    const float pos_y = pos[idsweep * 3 + 1];
    const float pos_z = pos[idsweep * 3 + 2];
    const float* data_row = (const float*)(data + (size_t)idsweep * sweep_samples);

    for (int tb = 0; tb < Ntarget; tb += CHUNK) {
        const int nchunk = std::min(CHUNK, Ntarget - tb);

#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            const float px = tx[tb + q] - pos_x;
            const float py = ty[tb + q] - pos_y;
            const float pz = tz[tb + q] - pos_z;
            const float d = sqrtf(px * px + py * py + pz * pz);

            const float sx = delta_r * (d + d0);
            const int id0 = (int)sx;
            // Float-domain check: (int)sx truncates toward zero, so
            // sx in (-1, 0) would pass an id0 >= 0 check and
            // extrapolate with a negative weight.
            const bool ok = (sx >= 0.0f) & (id0 + 1 < sweep_samples);
            idx_buf[q] = ok ? id0 : -1;
            frac_buf[q] = sx - id0;

            float ref_sin, ref_cos;
            sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            cs_buf[q] = ref_cos;
            sn_buf[q] = ref_sin;
        }

        // Scalar pass: data gather (linear interpolation), multiply by
        // reference and store.
        for (int q = 0; q < nchunk; q++) {
            const int id0 = idx_buf[q];
            complex64_t out = {0.0f, 0.0f};
            if (id0 >= 0) {
                const float f = frac_buf[q];
                const float sr = (1.0f - f) * data_row[2*id0]     + f * data_row[2*id0 + 2];
                const float si = (1.0f - f) * data_row[2*id0 + 1] + f * data_row[2*id0 + 3];
                out = {sr * cs_buf[q] - si * sn_buf[q],
                       sr * sn_buf[q] + si * cs_buf[q]};
            }
            data_out[(size_t)(tb + q) * nsweeps + idsweep] = out;
        }
    }
}

at::Tensor gpga_backprojection_2d_cpu(
          const at::Tensor &target_pos,
          const at::Tensor &data,
          const at::Tensor &pos,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          int64_t Ntarget,
          double d0,
          double data_fmod) {
    TORCH_CHECK(target_pos.dtype() == at::kFloat);
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(data.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(target_pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);

    at::Tensor target_pos_contig = target_pos.contiguous();
    at::Tensor pos_contig = pos.contiguous();
    at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
    at::Tensor data_out = torch::zeros({Ntarget, nsweeps}, options);

    const float* target_pos_ptr = target_pos_contig.data_ptr<float>();
    const float* pos_ptr = pos_contig.data_ptr<float>();
    const complex64_t* data_ptr = (const complex64_t*)data_contig.data_ptr<c10::complex<float>>();
    complex64_t* data_out_ptr = (complex64_t*)data_out.data_ptr<c10::complex<float>>();

    const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

    // Unit-stride target coordinates for the vectorized geometry pass.
    std::vector<float> tx(Ntarget), ty(Ntarget), tz(Ntarget);
    for (int t = 0; t < Ntarget; t++) {
        tx[t] = target_pos_ptr[t * 3 + 0];
        ty[t] = target_pos_ptr[t * 3 + 1];
        tz[t] = target_pos_ptr[t * 3 + 2];
    }

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idsweep = 0; idsweep < nsweeps; idsweep++) {
        gpga_backprojection_2d_sweep_cpu(
                tx.data(), ty.data(), tz.data(), data_ptr, pos_ptr,
                data_out_ptr, sweep_samples, nsweeps, ref_phase, delta_r,
                Ntarget, d0, data_fmod / kPI, idsweep);
    }
    return data_out;
}

// Inner product of one image block against one sweep's backprojection
// footprint: alpha[b, m] = sum over the block's pixels of
// conj(img[pix]) * data[m, interp] * exp(j (ref_phase d - data_fmod sx)).
// Per-pixel math matches gpga_backprojection_2d_kernel_cpu (z=0 pixel
// plane, linear range interpolation); the master image acts as the pixel
// weighting so the [npix, nsweeps] footprint matrix is never materialized.
// Mirrors blocksvd_alpha_kernel in cuda/backproj.cu; structured like
// backprojection_polar_2d_row_cpu (SIMD geometry + phase pass into chunk
// buffers, scalar gather + accumulate pass).
static void blocksvd_alpha_kernel_cpu(
          const complex64_t* img, const complex64_t* data, const float* pos,
          const int32_t* blocks, complex64_t* alpha, int sweep_samples,
          int nsweeps, int Ntheta, float ref_phase, float delta_r,
          float r0, float dr, float theta0, float dtheta, float d0,
          float data_fmod, int idblock, int idsweep) {
    constexpr int CHUNK = 256;
    float cs_buf[CHUNK], sn_buf[CHUNK], frac_buf[CHUNK];
    int idx_buf[CHUNK];

    const int ri0 = blocks[idblock * 6 + 0];
    const int ri1 = blocks[idblock * 6 + 1];
    const int ti0 = blocks[idblock * 6 + 2];
    const int ti1 = blocks[idblock * 6 + 3];
    const int sweep_lo = blocks[idblock * 6 + 4];
    const int sweep_hi = blocks[idblock * 6 + 5];
    if (idsweep < sweep_lo || idsweep >= sweep_hi) {
        return;
    }

    const float pos_x = pos[idsweep * 3 + 0];
    const float pos_y = pos[idsweep * 3 + 1];
    const float pos_z = pos[idsweep * 3 + 2];
    const float pz2 = pos_z * pos_z;
    const float* data_row = (const float*)(data + (size_t)idsweep * sweep_samples);

    // Theta in the outer loop, SIMD over range rows: blocksvd blocks are
    // tall (tens to hundreds of range rows, ~10-20 theta columns), so the
    // vectorized pass runs at full width along r.
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    for (int j = ti0; j < ti1; j++) {
        const float theta = theta0 + j * dtheta;
        const float ct = sqrtf(fmaxf(0.0f, 1.0f - theta * theta));

        for (int rb = ri0; rb < ri1; rb += CHUNK) {
            const int nchunk = std::min(CHUNK, ri1 - rb);

            // Geometry + phase pass. idx = -1 marks pixels outside the
            // data range window.
#pragma omp simd
            for (int q = 0; q < nchunk; q++) {
                const float r = r0 + (rb + q) * dr;
                const float px = r * ct - pos_x;
                const float py = r * theta - pos_y;

                const float d = sqrtf(px * px + py * py + pz2);

                const float sx = delta_r * (d + d0);
                const int id0 = (int)sx;
                // Float-domain check: (int)sx truncates toward zero, so
                // sx in (-1, 0) would pass an id0 >= 0 check and
                // extrapolate with a negative weight.
                const bool ok = (sx >= 0.0f) & (id0 + 1 < sweep_samples);
                idx_buf[q] = ok ? id0 : -1;
                frac_buf[q] = sx - id0;

                float ref_sin, ref_cos;
                sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
                cs_buf[q] = ref_cos;
                sn_buf[q] = ref_sin;
            }

            // Scalar pass: data gather (linear interpolation), multiply by
            // reference, accumulate against the conjugated image pixel.
            // The image patch is small enough to stay cached despite the
            // Ntheta-strided access.
            const float* img_col = (const float*)(img + (size_t)rb * Ntheta + j);
            for (int q = 0; q < nchunk; q++) {
                const int id0 = idx_buf[q];
                if (id0 < 0) {
                    continue;
                }
                const float f = frac_buf[q];
                const float sr = (1.0f - f) * data_row[2*id0]     + f * data_row[2*id0 + 2];
                const float si = (1.0f - f) * data_row[2*id0 + 1] + f * data_row[2*id0 + 3];
                const float vr = sr * cs_buf[q] - si * sn_buf[q];
                const float vi = sr * sn_buf[q] + si * cs_buf[q];
                const float wr = img_col[2*(size_t)q*Ntheta];
                const float wi = img_col[2*(size_t)q*Ntheta + 1];
                // conj(w) * v
                acc_r += wr * vr + wi * vi;
                acc_i += wr * vi - wi * vr;
            }
        }
    }
    alpha[(size_t)idblock * nsweeps + idsweep] = complex64_t(acc_r, acc_i);
}

at::Tensor blocksvd_alpha_cpu(
          const at::Tensor &img,
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &blocks,
          int64_t sweep_samples,
          int64_t nsweeps,
          int64_t nblocks,
          int64_t Ntheta,
          double fc,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          double d0,
          double data_fmod) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat);
    TORCH_CHECK(data.dtype() == at::kComplexFloat);
    TORCH_CHECK(pos.dtype() == at::kFloat);
    TORCH_CHECK(blocks.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(blocks.device().type() == at::DeviceType::CPU);

    at::Tensor img_contig = img.contiguous();
    at::Tensor data_contig = data.contiguous();
    at::Tensor pos_contig = pos.contiguous();
    at::Tensor blocks_contig = blocks.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
    at::Tensor alpha = torch::zeros({nblocks, nsweeps}, options);

    const complex64_t* img_ptr = (const complex64_t*)img_contig.data_ptr<c10::complex<float>>();
    const complex64_t* data_ptr = (const complex64_t*)data_contig.data_ptr<c10::complex<float>>();
    const float* pos_ptr = pos_contig.data_ptr<float>();
    const int32_t* blocks_ptr = blocks_contig.data_ptr<int32_t>();
    complex64_t* alpha_ptr = (complex64_t*)alpha.data_ptr<c10::complex<float>>();

    const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    // Dynamic schedule: sweeps outside a block's aperture window return
    // immediately, so per-(block, sweep) work is very uneven.
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int idblock = 0; idblock < nblocks; idblock++) {
        for (int idsweep = 0; idsweep < nsweeps; idsweep++) {
            blocksvd_alpha_kernel_cpu(
                    img_ptr, data_ptr, pos_ptr, blocks_ptr, alpha_ptr,
                    sweep_samples, nsweeps, Ntheta, ref_phase, delta_r,
                    r0, dr, theta0, dtheta, d0, data_fmod / kPI,
                    idblock, idsweep);
        }
    }
    return alpha;
}

static void backprojection_polar_2d_tx_power_kernel_cpu(
          const float* wa,
          const float* pos,
          const float* att,
          const float* g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          float* img,
          int nsweeps,
          float delta_r,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          int Nr,
          int Ntheta,
          int normalization,
          int azimuth_resolution,
          float altitude,
          int idx,
          int idbatch) {
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    if (idr >= Nr || idtheta >= Ntheta) {
        return;
    }

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    const float cos2 = 1.0f - theta * theta;

    // Pixel ground position and effective altitude for angle/distance computation.
    // altitude > 0: slant-range grid (BP origin at sensor altitude, pos z ≈ 0).
    // altitude == 0: ground-range grid (pos z = real altitude).
    float px_base, py_base, z_eff;
    if (altitude > 0.0f) {
        float r2cos2 = r * r * cos2;
        float H2 = altitude * altitude;
        if (r2cos2 < H2) {
            // No ground intersection (shadow zone below nadir).
            img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = 0.0f;
            return;
        }
        px_base = sqrtf(r2cos2 - H2);
        py_base = r * theta;
        z_eff = altitude;
    } else {
        px_base = r * sqrtf(cos2);
        py_base = r * theta;
        z_eff = 0.0f;  // will use per-sweep pos_z
    }

    // Squared sine of the angle subtended by one range cell at nadir, used
    // as a floor on sin^2 of the look angle.
    float h_ref = (altitude > 0.0f) ? altitude
                  : pos[idbatch * nsweeps * 3 + (nsweeps/2) * 3 + 2];
    const float min_sin2_look = 2.0f * dr / h_ref;

    // Per-sweep accumulation (shared helper). nbatch is handled by offsetting
    // the pointers. Slant grids use a fixed reference height (z_eff = altitude),
    // ground grids the per-sweep platform z.
    float pixel, m_w, m_mean, m_s;
    tx_power_pixel_moments(px_base, py_base, /*use_h_fixed=*/altitude > 0.0f, z_eff,
            pos + (size_t)idbatch * nsweeps * 3, att + (size_t)idbatch * nsweeps * 3,
            nsweeps, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            wa + (size_t)idbatch * nsweeps, normalization, min_sin2_look,
            &pixel, &m_w, &m_mean, &m_s);

    if (azimuth_resolution) {
        const float Rg = sqrtf(px_base * px_base + py_base * py_base);
        const float var = (m_w > 0.0f) ? m_s / m_w : 0.0f;
        const float sigma = sqrtf(fmaxf(var, 0.0f));
        if (sigma > 0.0f && Rg > 0.0f) {
            pixel = pixel / (sigma * Rg);
        } else {
            // No measurable azimuth aperture (<=1 contributing sweep)
            pixel = INFINITY;
        }
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = sqrtf(pixel);
}

static at::Tensor backprojection_polar_2d_tx_power_impl_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          int64_t nbatch,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          int64_t normalization,
          int64_t azimuth_resolution,
          double altitude) {
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(g.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);

	at::Tensor wa_contig = wa.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(wa.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* wa_ptr = wa_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const float* g_ptr = g_contig.data_ptr<float>();
	float* img_ptr = img.data_ptr<float>();

	const float delta_r = 1.0f / r_res;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nr * Ntheta; idx++) {
            backprojection_polar_2d_tx_power_kernel_cpu(
                          wa_ptr,
                          pos_ptr,
                          att_ptr,
                          g_ptr,
                          g_az0,
                          g_el0,
                          g_daz,
                          g_del,
                          g_naz,
                          g_nel,
                          img_ptr,
                          nsweeps,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          normalization,
                          azimuth_resolution,
                          static_cast<float>(altitude),
                          idx, idbatch);
        }
    }
	return img;
}

at::Tensor backprojection_polar_2d_tx_power_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          int64_t nbatch,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          int64_t normalization,
          int64_t azimuth_resolution) {
    return backprojection_polar_2d_tx_power_impl_cpu(
            wa, pos, att, g, nbatch, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            nsweeps, r_res, r0, dr, theta0, dtheta, Nr, Ntheta, normalization,
            azimuth_resolution, 0.0);
}

at::Tensor backprojection_polar_2d_tx_power_slant_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          int64_t nbatch,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double r_res,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          int64_t normalization,
          int64_t azimuth_resolution,
          double altitude) {
    return backprojection_polar_2d_tx_power_impl_cpu(
            wa, pos, att, g, nbatch, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            nsweeps, r_res, r0, dr, theta0, dtheta, Nr, Ntheta, normalization,
            azimuth_resolution, altitude);
}

// Cartesian analog of backprojection_polar_2d_tx_power_kernel_cpu. The only
// difference is that the pixel ground position comes straight from the
// Cartesian image grid (z = 0 ground plane) instead of a polar (r, theta)
// cell; the platform altitude is always the per-sweep pos_z.
static void backprojection_cart_2d_tx_power_kernel_cpu(
          const float* wa,
          const float* pos,
          const float* att,
          const float* g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          float* img,
          int nsweeps,
          float delta_r,
          float x0,
          float dx,
          float y0,
          float dy,
          int Nx,
          int Ny,
          int normalization,
          int azimuth_resolution,
          int idx,
          int idbatch) {
    const int idy = idx % Ny;
    const int idx_x = idx / Ny;
    if (idx_x >= Nx || idy >= Ny) {
        return;
    }

    // Pixel ground position (z = 0 ground plane).
    const float px_base = x0 + idx_x * dx;
    const float py_base = y0 + idy * dy;

    // Squared sine of the angle subtended by one ground-range cell at nadir,
    // used as a floor on sin^2 of the look angle. Uses the mid-sweep altitude.
    const float h_ref = pos[idbatch * nsweeps * 3 + (nsweeps/2) * 3 + 2];
    const float min_sin2_look = 2.0f * dx / h_ref;

    // Per-sweep accumulation (shared helper). nbatch is handled by offsetting
    // the pointers. Cartesian ground grid: per-sweep height is the platform z.
    float pixel, m_w, m_mean, m_s;
    tx_power_pixel_moments(px_base, py_base, /*use_h_fixed=*/false, 0.0f,
            pos + (size_t)idbatch * nsweeps * 3, att + (size_t)idbatch * nsweeps * 3,
            nsweeps, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            wa + (size_t)idbatch * nsweeps, normalization, min_sin2_look,
            &pixel, &m_w, &m_mean, &m_s);

    if (azimuth_resolution) {
        const float Rg = sqrtf(px_base * px_base + py_base * py_base);
        const float var = (m_w > 0.0f) ? m_s / m_w : 0.0f;
        const float sigma = sqrtf(fmaxf(var, 0.0f));
        if (sigma > 0.0f && Rg > 0.0f) {
            pixel = pixel / (sigma * Rg);
        } else {
            // No measurable azimuth aperture (<=1 contributing sweep)
            pixel = INFINITY;
        }
    }
    img[idbatch * Nx * Ny + idx_x * Ny + idy] = sqrtf(pixel);
}

at::Tensor backprojection_cart_2d_tx_power_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          int64_t nbatch,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double r_res,
          double x0,
          double dx,
          double y0,
          double dy,
          int64_t Nx,
          int64_t Ny,
          int64_t normalization,
          int64_t azimuth_resolution) {
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(g.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);

	at::Tensor wa_contig = wa.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(wa.device());
	at::Tensor img = torch::zeros({nbatch, Nx, Ny}, options);
	const float* wa_ptr = wa_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const float* g_ptr = g_contig.data_ptr<float>();
	float* img_ptr = img.data_ptr<float>();

	const float delta_r = 1.0f / r_res;

    // See backprojection_polar_2d_tx_power_impl_cpu for why the team size is
    // set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nx * Ny; idx++) {
            backprojection_cart_2d_tx_power_kernel_cpu(
                          wa_ptr,
                          pos_ptr,
                          att_ptr,
                          g_ptr,
                          g_az0,
                          g_el0,
                          g_daz,
                          g_del,
                          g_naz,
                          g_nel,
                          img_ptr,
                          nsweeps,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          normalization,
                          azimuth_resolution,
                          idx, idbatch);
        }
    }
	return img;
}

static void backprojection_polar_2d_tx_power_accum_kernel_cpu(
          const float* wa,
          const float* pos,
          const float* att,
          const float* g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          float* img,
          int nsweeps,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          int Nr,
          int Ntheta,
          int normalization,
          float dr_ref,
          float h_ref,
          float altitude,
          int theta_psi,
          int idx) {
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    if (idr >= Nr || idtheta >= Ntheta) {
        return;
    }
    const size_t np = (size_t)Nr * Ntheta;
    float* out = &img[idr * Ntheta + idtheta];

    const float r = r0 + idr * dr;
    // theta_psi: grid is sampled uniformly in psi = asin(theta). This
    // resolves the antenna pattern features with a constant number of cells
    // per beamwidth; a uniform theta grid compresses them by cos(psi) near
    // |theta| = 1. Used for the backprojection_polar_2d_tx_power_ffbp subaperture maps.
    const float tc = theta0 + idtheta * dtheta;
    const float theta = theta_psi ? sinf(tc) : tc;
    const float cos2 = 1.0f - theta * theta;

    // Pixel ground position and effective altitude for angle/distance computation.
    // altitude > 0: slant-range grid (BP origin at sensor altitude, pos z ≈ 0).
    // altitude == 0: ground-range grid (pos z = real altitude).
    float px_base, py_base, z_eff;
    if (altitude > 0.0f) {
        float r2cos2 = r * r * cos2;
        float H2 = altitude * altitude;
        if (r2cos2 < H2) {
            // No ground intersection (shadow zone below nadir).
            for (int c = 0; c < 4; c++) {
                out[c * np] = 0.0f;
            }
            return;
        }
        px_base = sqrtf(r2cos2 - H2);
        py_base = r * theta;
        z_eff = altitude;
    } else {
        px_base = r * sqrtf(cos2);
        py_base = r * theta;
        z_eff = 0.0f;  // will use per-sweep pos_z
    }

    // Angular size of resolution cell at nadir. dr_ref and h_ref refer to the
    // final output grid so that subaperture grids use the same clamp floor.
    const float min_look_angle = sqrtf(2.0f * dr_ref / h_ref);

    // Polar grid: slant grids use a fixed reference height (z_eff = altitude),
    // ground grids the per-sweep platform z.
    float pixel, m_w, m_mean, m_s;
    tx_power_pixel_moments(px_base, py_base, /*use_h_fixed=*/altitude > 0.0f, z_eff,
            pos, att, nsweeps, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            wa, normalization, min_look_angle,
            &pixel, &m_w, &m_mean, &m_s);

    // Unfinished accumulators for factorized (ffbp style) processing:
    // channel 0: S = sum wi/sinl, 1: W = sum wi, 2: P1 = W*mean(psi),
    // 3: M2 = weighted sum of squared deviations of psi.
    // P1 is premultiplied by W so that bilinear interpolation of the
    // channels stays valid where W approaches zero.
    out[0 * np] = pixel;
    out[1 * np] = m_w;
    out[2 * np] = m_w * m_mean;
    out[3 * np] = m_s;
}

at::Tensor backprojection_polar_2d_tx_power_accum_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          int64_t normalization,
          double dr_ref,
          double h_ref,
          double altitude,
          int64_t theta_psi) {
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(g.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);

	at::Tensor wa_contig = wa.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(wa.device());
	at::Tensor img = torch::zeros({4, Nr, Ntheta}, options);
	const float* wa_ptr = wa_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const float* g_ptr = g_contig.data_ptr<float>();
	float* img_ptr = img.data_ptr<float>();

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for(int idx = 0; idx < Nr * Ntheta; idx++) {
        backprojection_polar_2d_tx_power_accum_kernel_cpu(
                      wa_ptr,
                      pos_ptr,
                      att_ptr,
                      g_ptr,
                      g_az0,
                      g_el0,
                      g_daz,
                      g_del,
                      g_naz,
                      g_nel,
                      img_ptr,
                      nsweeps,
                      r0, dr,
                      theta0, dtheta,
                      Nr, Ntheta,
                      normalization,
                      static_cast<float>(dr_ref),
                      static_cast<float>(h_ref),
                      static_cast<float>(altitude),
                      static_cast<int>(theta_psi),
                      idx);
    }
	return img;
}

static void backprojection_cart_2d_tx_power_accum_kernel_cpu(
          const float* wa,
          const float* pos,
          const float* att,
          const float* g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          float* img,
          int nsweeps,
          float x0,
          float dx,
          float y0,
          float dy,
          int Nx,
          int Ny,
          int normalization,
          float dx_ref,
          float h_ref,
          int idx) {
    const int idy = idx % Ny;
    const int idx_x = idx / Ny;
    if (idx_x >= Nx || idy >= Ny) {
        return;
    }
    const size_t np = (size_t)Nx * Ny;
    float* out = &img[idx_x * Ny + idy];

    // Pixel ground position (z = 0 ground plane).
    const float px_base = x0 + idx_x * dx;
    const float py_base = y0 + idy * dy;

    // Squared sine of the angle subtended by one ground-range cell at nadir,
    // used as a floor on sin^2 of the look angle. dx_ref and h_ref refer to the
    // final output grid so that all subaperture maps use the same clamp floor.
    const float min_sin2_look = 2.0f * dx_ref / h_ref;

    // Cartesian ground grid: per-sweep height is the platform z.
    float pixel, m_w, m_mean, m_s;
    tx_power_pixel_moments(px_base, py_base, /*use_h_fixed=*/false, 0.0f,
            pos, att, nsweeps, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
            wa, normalization, min_sin2_look,
            &pixel, &m_w, &m_mean, &m_s);

    // Unfinished accumulators for factorized (cfbp style) processing:
    // channel 0: S = sum wi/sinl, 1: W = sum wi, 2: P1 = W*mean(psi),
    // 3: M2 = weighted sum of squared deviations of psi.
    // P1 is premultiplied by W so that bilinear interpolation of the
    // channels stays valid where W approaches zero.
    out[0 * np] = pixel;
    out[1 * np] = m_w;
    out[2 * np] = m_w * m_mean;
    out[3 * np] = m_s;
}

at::Tensor backprojection_cart_2d_tx_power_accum_cpu(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &g,
          double g_az0,
          double g_el0,
          double g_daz,
          double g_del,
          int64_t g_naz,
          int64_t g_nel,
          int64_t nsweeps,
          double x0,
          double dx,
          double y0,
          double dy,
          int64_t Nx,
          int64_t Ny,
          int64_t normalization,
          double dx_ref,
          double h_ref) {
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(g.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CPU);

	at::Tensor wa_contig = wa.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(wa.device());
	at::Tensor img = torch::zeros({4, Nx, Ny}, options);
	const float* wa_ptr = wa_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const float* g_ptr = g_contig.data_ptr<float>();
	float* img_ptr = img.data_ptr<float>();

    // See backprojection_polar_2d_tx_power_accum_cpu for why the team size is
    // set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for(int idx = 0; idx < Nx * Ny; idx++) {
        backprojection_cart_2d_tx_power_accum_kernel_cpu(
                      wa_ptr,
                      pos_ptr,
                      att_ptr,
                      g_ptr,
                      g_az0,
                      g_el0,
                      g_daz,
                      g_del,
                      g_naz,
                      g_nel,
                      img_ptr,
                      nsweeps,
                      x0, dx,
                      y0, dy,
                      Nx, Ny,
                      normalization,
                      static_cast<float>(dx_ref),
                      static_cast<float>(h_ref),
                      idx);
    }
	return img;
}

static void cart_tx_power_merge2_kernel_cpu(
          const float* acc0,
          const float* acc1,
          float* out,
          float x0_0, float dx_0, float y0_0, float dy_0, int Nx_0, int Ny_0,
          float x0_1, float dx_1, float y0_1, float dy_1, int Nx_1, int Ny_1,
          float x1, float dx1, float y1, float dy1, int Nx1, int Ny1,
          int idx) {
    const int idy = idx % Ny1;
    const int idx_x = idx / Ny1;
    if (idx >= Nx1 * Ny1) {
        return;
    }

    // Output pixel absolute ground position.
    const float X = x1 + dx1 * idx_x;
    const float Y = y1 + dy1 * idy;

    const float* accs[2] = {acc0, acc1};
    const float ix0[2] = {x0_0, x0_1};
    const float idxs[2] = {dx_0, dx_1};
    const float iy0[2] = {y0_0, y0_1};
    const float idys[2] = {dy_0, dy_1};
    const int inx[2] = {Nx_0, Nx_1};
    const int iny[2] = {Ny_0, Ny_1};

    float S = 0.0f, W = 0.0f, P1 = 0.0f, M2 = 0.0f;
    for (int id = 0; id < 2; id++) {
        const float* acc = accs[id];
        if (acc == nullptr) {
            continue;
        }
        const int nxi = inx[id];
        const int nyi = iny[id];
        // Map the absolute output position into this input's local index.
        const float dxi = (X - ix0[id]) / idxs[id];
        const float dyi = (Y - iy0[id]) / idys[id];
        if (!(dxi >= 0.0f && dxi < nxi - 1 && dyi >= 0.0f && dyi < nyi - 1)) {
            continue;
        }
        const int xi_int = dxi;
        const int yi_int = dyi;
        const float xi_frac = dxi - xi_int;
        const float yi_frac = dyi - yi_int;
        tx_power_merge_sample(acc, nxi, nyi, xi_int, xi_frac, yi_int, yi_frac,
                              &S, &W, &P1, &M2);
    }
    // Float cancellation could leave a small negative value which would give
    // NaN in the final sqrt of the variance.
    M2 = fmaxf(M2, 0.0f);

    const size_t np_out = (size_t)Nx1 * Ny1;
    float* o = &out[idx_x * Ny1 + idy];
    o[0 * np_out] = S;
    o[1 * np_out] = W;
    o[2 * np_out] = P1;
    o[3 * np_out] = M2;
}

at::Tensor cart_tx_power_merge2_cpu(
          const at::Tensor &acc0,
          const at::Tensor &acc1,
          double x0_0, double dx_0, double y0_0, double dy_0, int64_t Nx_0, int64_t Ny_0,
          double x0_1, double dx_1, double y0_1, double dy_1, int64_t Nx_1, int64_t Ny_1,
          double x1, double dx1, double y1, double dy1, int64_t Nx1, int64_t Ny1) {
    TORCH_CHECK(acc0.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(acc0.device().type() == at::DeviceType::CPU);

    bool has1 = acc1.defined() && acc1.numel() > 0;
    if (has1) {
        TORCH_CHECK(acc1.dtype() == at::kFloat);
    }

    at::Tensor acc0_contig = acc0.contiguous();
    at::Tensor acc1_contig;
    at::Tensor out = torch::zeros({4, Nx1, Ny1}, acc0_contig.options());
    const float* acc0_ptr = acc0_contig.data_ptr<float>();
    const float* acc1_ptr = nullptr;
    if (has1) {
        acc1_contig = acc1.contiguous();
        acc1_ptr = acc1_contig.data_ptr<float>();
    }
    float* out_ptr = out.data_ptr<float>();

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for(int idx = 0; idx < Nx1 * Ny1; idx++) {
        cart_tx_power_merge2_kernel_cpu(
                acc0_ptr, acc1_ptr, out_ptr,
                x0_0, dx_0, y0_0, dy_0, Nx_0, Ny_0,
                x0_1, dx_1, y0_1, dy_1, Nx_1, Ny_1,
                x1, dx1, y1, dy1, Nx1, Ny1,
                idx);
    }
    return out;
}


// Multiply one range row of a polar SAR image by the range-dealias carrier.
// Same carrier as the dealias option of backprojection_polar_2d: distance
// from origin to the pixel at the DEM height (z=0 without a DEM), in the
// polar form that matches the backprojection kernels also on guard band
// pixels |theta| > 1. fc < 0 applies the conjugate carrier (re-alias).
// Single pass over the image, per-pixel bilinear DEM lookup: no full-size
// temporaries beyond the output.
template<bool HasDem>
static void polar_range_dealias_row_cpu(
          const complex64_t* img,
          complex64_t* out,
          float ref_phase,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          int Nr,
          int Ntheta,
          float ox,
          float oy,
          float oz,
          const float* dem,
          float dem_r_scale,
          float dem_theta_scale,
          int dem_nr,
          int dem_ntheta,
          float alias_fmod,
          int idr,
          int idbatch) {
    const float r = r0 + idr * dr;
    const complex64_t* img_row = img + ((size_t)idbatch * Nr + idr) * Ntheta;
    complex64_t* out_row = out + ((size_t)idbatch * Nr + idr) * Ntheta;

    // DEM row interpolation setup, same pixel-index-ratio convention as the
    // backprojection kernels.
    float wr = 0.0f;
    const float* dem_row0 = nullptr;
    const float* dem_row1 = nullptr;
    if constexpr (HasDem) {
        const float fr = idr * dem_r_scale;
        int ir0 = (int)fr;
        ir0 = ir0 < dem_nr - 1 ? ir0 : dem_nr - 1;
        const int ir1 = ir0 + 1 < dem_nr ? ir0 + 1 : dem_nr - 1;
        wr = fr - ir0;
        dem_row0 = dem + (size_t)ir0 * dem_ntheta;
        dem_row1 = dem + (size_t)ir1 * dem_ntheta;
    }

    const float oxy2 = ox * ox + oy * oy;
#pragma omp simd
    for (int q = 0; q < Ntheta; q++) {
        const float theta = theta0 + q * dtheta;
        const float ct2 = 1.0f - theta * theta;
        const float x = r * (ct2 > 0.0f ? sqrtf(ct2) : 0.0f);
        const float y = r * theta;
        float z = 0.0f;
        if constexpr (HasDem) {
            const float ft = q * dem_theta_scale;
            int it0 = (int)ft;
            it0 = it0 < dem_ntheta - 1 ? it0 : dem_ntheta - 1;
            const int it1 = it0 + 1 < dem_ntheta ? it0 + 1 : dem_ntheta - 1;
            const float wt = ft - it0;
            const float za = dem_row0[it0] + wt * (dem_row0[it1] - dem_row0[it0]);
            const float zb = dem_row1[it0] + wt * (dem_row1[it1] - dem_row1[it0]);
            z = za + wr * (zb - za);
        }
        // Polar-form distance r^2 - 2*(x*ox + y*oy) + |o_xy|^2 + (oz-z)^2:
        // equals the Cartesian distance to the origin for |theta| <= 1 and
        // continues the backprojection kernels' clamped-cosine convention
        // past |theta| = 1.
        const float zz = oz - z;
        const float d2 = r * r - 2.0f * (x * ox + y * oy) + oxy2 + zz * zz;
        const float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);
        float ref_sin, ref_cos;
        sincospi(-ref_phase * d + alias_fmod * idr, &ref_sin, &ref_cos);
        const float vr = img_row[q].real();
        const float vi = img_row[q].imag();
        out_row[q] = complex64_t(vr * ref_cos - vi * ref_sin,
                                 vr * ref_sin + vi * ref_cos);
    }
}

at::Tensor polar_range_dealias_cpu(
          const at::Tensor &img,
          const at::Tensor &dem,
          int64_t nbatch,
          int64_t Nr,
          int64_t Ntheta,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          double ox,
          double oy,
          double oz,
          double alias_fmod) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat,
                "polar_range_dealias: img must be complex64");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);

    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty_like(img_contig);
    const complex64_t* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    complex64_t* out_ptr = out.data_ptr<c10::complex<float>>();

    const bool has_dem = dem.defined();
    at::Tensor dem_contig;
    const float* dem_ptr = nullptr;
    float dem_r_scale = 0.0f, dem_theta_scale = 0.0f;
    int dem_nr = 0, dem_ntheta = 0;
    if (has_dem) {
        TORCH_CHECK(dem.dtype() == at::kFloat, "polar_range_dealias: dem must be float32");
        TORCH_CHECK(dem.dim() == 2, "polar_range_dealias: dem must be 2D");
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CPU);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

    const float ref_phase = 4.0f * fc / kC0;

    omp_set_num_threads(omp_get_num_procs());

    auto row = has_dem ? &polar_range_dealias_row_cpu<true>
                       : &polar_range_dealias_row_cpu<false>;
#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int idr = 0; idr < Nr; idr++) {
            row(img_ptr, out_ptr, ref_phase, r0, dr, theta0, dtheta,
                Nr, Ntheta, ox, oy, oz,
                dem_ptr, dem_r_scale, dem_theta_scale, dem_nr, dem_ntheta,
                alias_fmod / kPI, idr, idbatch);
        }
    }
    return out;
}


// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cpu);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cpu);
  m.impl("backprojection_polar_2d_tx_power", &backprojection_polar_2d_tx_power_cpu);
  m.impl("backprojection_polar_2d_tx_power_slant", &backprojection_polar_2d_tx_power_slant_cpu);
  m.impl("backprojection_polar_2d_tx_power_accum", &backprojection_polar_2d_tx_power_accum_cpu);
  m.impl("backprojection_cart_2d_tx_power", &backprojection_cart_2d_tx_power_cpu);
  m.impl("backprojection_cart_2d_tx_power_accum", &backprojection_cart_2d_tx_power_accum_cpu);
  m.impl("cart_tx_power_merge2", &cart_tx_power_merge2_cpu);
  m.impl("backprojection_cart_2d", &backprojection_cart_2d_cpu);
  m.impl("backprojection_cart_2d_grad", &backprojection_cart_2d_grad_cpu);
  m.impl("compute_illumination", &compute_illumination_cpu);
  m.impl("projection_cart_2d", &projection_cart_2d_cpu);
  m.impl("projection_cart_2d_nufft", &projection_cart_2d_nufft_cpu);
  m.impl("gpga_backprojection_2d", &gpga_backprojection_2d_cpu);
  m.impl("blocksvd_alpha", &blocksvd_alpha_cpu);
  m.impl("polar_range_dealias", &polar_range_dealias_cpu);
}

}
