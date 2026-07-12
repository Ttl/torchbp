#include "util.h"
#include <ATen/ops/fft_ifft.h>
#include <map>
#include <mutex>
#include <tuple>

namespace torchbp {

// Interpolation methods for backprojection kernels
enum class InterpMethod {
    LINEAR,
    LANCZOS,
    KNAB
};

template<typename T, bool HasAntennaPattern, bool Normalize = true, InterpMethod Method = InterpMethod::LINEAR, bool HasDem = false>
__global__ void backprojection_polar_2d_kernel(
          const T* __restrict__ data,
          const float* __restrict__ pos,
          const float* __restrict__ att,
          complex64_t* __restrict__ img,
          int sweep_samples,
          int nsweeps,
          float phase_coef,
          float phase_offset,
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
          float dealias_coef,
          float dealias_fmod,
          const float* __restrict__ g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          const float* __restrict__ dem,
          float dem_r_scale,
          float dem_theta_scale,
          int dem_nr,
          int dem_ntheta,
          int interp_order = 2,
          float knab_v = 0.0f) {

    // Process multiple pixels per thread to amortize pos loads
    const int PIXELS_PER_THREAD = 4;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta;
    const int idr_base = (idx / Ntheta) * PIXELS_PER_THREAD;
    const int idbatch = blockIdx.y;

    if (idr_base >= Nr || idtheta >= Ntheta) return;

    // Precompute theta sin/cos once (shared by all pixels in this thread).
    // The cosine is clamped: guard band grids extend past |theta| = 1 where
    // the pixel is the smooth continuation of the azimuth signal past the
    // fold (see the polar-form distance note below).
    const float sin_theta = theta0 + idtheta * dtheta;  // theta IS sin(angle) in this coord system
    const float cos_theta = sqrtf(fmaxf(0.0f, 1.0f - sin_theta * sin_theta));

    // Coordinate storage for the pixels
    float r[PIXELS_PER_THREAD];
    float r2[PIXELS_PER_THREAD];
    float x[PIXELS_PER_THREAD];
    float y[PIXELS_PER_THREAD];
    float z[PIXELS_PER_THREAD];
    float rz2[PIXELS_PER_THREAD];
    float pixel_re[PIXELS_PER_THREAD] = {0};
    float pixel_im[PIXELS_PER_THREAD] = {0};
    float w_sum2[PIXELS_PER_THREAD] = {0};
    float w_sum1[PIXELS_PER_THREAD] = {0};

    // DEM shares the grid extent, so pixel index maps to DEM index by a
    // constant ratio. Theta side is fixed for the whole thread. Indices are
    // edge-clamped: guard band pixels |theta| > 1 get the DEM edge value.
    int dem_it0, dem_it1;
    float dem_wt;
    if constexpr (HasDem) {
        const float ft = idtheta * dem_theta_scale;
        dem_it0 = min((int)ft, dem_ntheta - 1);
        dem_it1 = min(dem_it0 + 1, dem_ntheta - 1);
        dem_wt = ft - dem_it0;
    }

    #pragma unroll
    for(int k=0; k<PIXELS_PER_THREAD; ++k) {
        if(idr_base + k < Nr) {
            r[k] = r0 + (idr_base + k) * dr;
            r2[k] = r[k] * r[k];
            x[k] = r[k] * cos_theta;
            y[k] = r[k] * sin_theta;
            if constexpr (HasDem) {
                const float fr = (idr_base + k) * dem_r_scale;
                const int ir0 = min((int)fr, dem_nr - 1);
                const int ir1 = min(ir0 + 1, dem_nr - 1);
                const float wr = fr - ir0;
                const float z00 = __ldg(&dem[ir0 * dem_ntheta + dem_it0]);
                const float z01 = __ldg(&dem[ir0 * dem_ntheta + dem_it1]);
                const float z10 = __ldg(&dem[ir1 * dem_ntheta + dem_it0]);
                const float z11 = __ldg(&dem[ir1 * dem_ntheta + dem_it1]);
                const float za = fmaf(dem_wt, z01 - z00, z00);
                const float zb = fmaf(dem_wt, z11 - z10, z10);
                z[k] = fmaf(wr, zb - za, za);
                rz2[k] = fmaf(z[k], z[k], r2[k]);
            }
        }
    }

    const int pos_batch_offset = idbatch * nsweeps * 3;
    const int data_batch_stride = idbatch * sweep_samples * nsweeps;
    const int max_id0 = sweep_samples - 2;

    // Fused phase coefficient: phase = phase_coef * (d + d0) + phase_offset2
    // where phase_offset2 = phase_offset - phase_coef * d0
    // This way we compute d_eff = d + d0 once and use it for both sx and phase
    const float phase_offset2 = phase_offset - phase_coef * d0;

    for (int i = 0; i < nsweeps; i++) {
        // Load pos directly via __ldg (L1 cached, broadcast across warp)
        const int pos_idx = pos_batch_offset + i * 3;
        float pos_x = __ldg(&pos[pos_idx + 0]);
        float pos_y = __ldg(&pos[pos_idx + 1]);
        float pos_z = __ldg(&pos[pos_idx + 2]);

        float pz2 = pos_z * pos_z;
        float pn2 = fmaf(pos_x, pos_x, fmaf(pos_y, pos_y, pz2));
        int sweep_offset = data_batch_stride + i * sweep_samples;

        float att_el, att_az;
        if constexpr (HasAntennaPattern) {
            att_el = __ldg(&att[pos_idx + 0]);
            att_az = __ldg(&att[pos_idx + 2]);
        }

        #pragma unroll
        for(int k=0; k<PIXELS_PER_THREAD; ++k) {
            if(idr_base + k >= Nr) continue;

            float px = x[k] - pos_x;
            float py = y[k] - pos_y;
            // Distance in the polar form d^2 = r^2 + |p|^2
            // - 2*r*(cos(theta)*p_x + sin(theta)*p_y). Inside |theta| <= 1 it
            // equals the Cartesian distance to (x, y, 0); with the clamped
            // cosine it stays affine in theta past |theta| = 1, which
            // continues the azimuth chirp smoothly (exactly, for a straight
            // track on the y axis) so that guard band samples give the FFBP
            // merge interpolation valid support at the grid edge. Distance
            // from a virtual Cartesian point would instead jump the phase
            // rate to ~2k*r at the fold.
            // With a DEM the pixel is at (x, y, z): d^2 gains z^2 - 2*z*p_z.
            float d_sq;
            if constexpr (HasDem) {
                d_sq = fmaf(-2.0f * x[k], pos_x, fmaf(-2.0f * y[k], pos_y,
                            fmaf(-2.0f * z[k], pos_z, rz2[k] + pn2)));
            } else {
                d_sq = fmaf(-2.0f * x[k], pos_x, fmaf(-2.0f * y[k], pos_y, r2[k] + pn2));
            }
            float d = sqrtf(fmaxf(d_sq, 0.0f));

            // Compute d_eff once, use for both sx and phase
            float d_eff = d + d0;
            float sx = delta_r * d_eff;

            // Bounds check and interpolation
            float s_re, s_im;
            bool valid_sample;

            if constexpr (Method == InterpMethod::LINEAR) {
                int id0 = (int)sx;
                // Float-domain check: (int)sx truncates toward zero, so
                // sx in (-1, 0) would pass an id0 >= 0 check and
                // extrapolate with a negative weight.
                valid_sample = (sx >= 0.0f && id0 <= max_id0);

                if (valid_sample) {
                    int data_idx = sweep_offset + id0;
                    float s0_re, s0_im, s1_re, s1_im;

                    if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
                        float2 s0f = __ldg(&((const float2*)data)[data_idx]);
                        float2 s1f = __ldg(&((const float2*)data)[data_idx + 1]);
                        s0_re = s0f.x; s0_im = s0f.y;
                        s1_re = s1f.x; s1_im = s1f.y;
                    } else {
                        half2 s0h = __ldg(&((const half2*)data)[data_idx]);
                        half2 s1h = __ldg(&((const half2*)data)[data_idx + 1]);
                        s0_re = __half2float(s0h.x); s0_im = __half2float(s0h.y);
                        s1_re = __half2float(s1h.x); s1_im = __half2float(s1h.y);
                    }

                    float interp_idx = sx - id0;
                    s_re = fmaf(interp_idx, s1_re - s0_re, s0_re);
                    s_im = fmaf(interp_idx, s1_im - s0_im, s0_im);
                }
            } else if constexpr (Method == InterpMethod::LANCZOS) {
                valid_sample = (sx >= 0.0f && sx < sweep_samples - 1);
                if (valid_sample) {
                    complex64_t s = lanczos_interp_1d<complex64_t, T>(
                        &data[sweep_offset], sweep_samples, sx, interp_order);
                    s_re = s.real();
                    s_im = s.imag();
                }
            } else if constexpr (Method == InterpMethod::KNAB) {
                valid_sample = (sx >= 0.0f && sx < sweep_samples - 1);
                if (valid_sample) {
                    const float knab_norm = knab_kernel_norm(interp_order, knab_v);
                    complex64_t s = knab_interp_1d<complex64_t, T>(
                        &data[sweep_offset], sweep_samples, sx, interp_order, knab_v, knab_norm);
                    s_re = s.real();
                    s_im = s.imag();
                }
            }

            if (valid_sample) {

                float ref_sin, ref_cos;
                __sincosf(fmaf(phase_coef, d_eff, phase_offset2), &ref_sin, &ref_cos);

                if constexpr (HasAntennaPattern) {
                    float look_angle;
                    if constexpr (HasDem) {
                        look_angle = asinf(fminf(fmaxf((z[k] - pos_z) / d, -1.0f), 1.0f));
                    } else {
                        look_angle = asinf(fmaxf(-pos_z / d, -1.0f));
                    }
                    const float el_deg = look_angle - att_el;
                    const float az_deg = atan2f(py, px) - att_az;

                    const float el_idx = (el_deg - g_el0) / g_del;
                    const float az_idx = (az_deg - g_az0) / g_daz;

                    const int el_int = (int)el_idx;
                    const int az_int = (int)az_idx;

                    if (el_idx >= 0.0f && el_int + 1 < g_nel && az_idx >= 0.0f && az_int + 1 < g_naz) {
                        const float el_frac = el_idx - el_int;
                        const float az_frac = az_idx - az_int;
                        const float w = interp2d<float>(g, g_nel, g_naz, el_int, el_frac, az_int, az_frac);

                        const float ws_re = w * s_re;
                        const float ws_im = w * s_im;
                        pixel_re[k] = fmaf(ws_re, ref_cos, fmaf(-ws_im, ref_sin, pixel_re[k]));
                        pixel_im[k] = fmaf(ws_re, ref_sin, fmaf(ws_im, ref_cos, pixel_im[k]));
                        w_sum2[k] = fmaf(w, w, w_sum2[k]);
                        w_sum1[k] += w;
                    }
                } else {
                    pixel_re[k] = fmaf(s_re, ref_cos, fmaf(-s_im, ref_sin, pixel_re[k]));
                    pixel_im[k] = fmaf(s_re, ref_sin, fmaf(s_im, ref_cos, pixel_im[k]));
                }
            }
        }
    }

    #pragma unroll
    for(int k=0; k<PIXELS_PER_THREAD; ++k) {
        if(idr_base + k < Nr) {
            complex64_t pixel = {pixel_re[k], pixel_im[k]};

            // Normalize to same average as without antenna pattern.
            // Unweighted: Σs = scene * Σg (signal has g)
            // Weighted: Σ(s * g) = scene * Σg²
            // To match: normalize by Σg / Σg²
            // A denormal w_sum2 would blow up the scale, so require at
            // least the smallest normal float (with FTZ it flushes to zero
            // anyway; this keeps CPU and CUDA consistent).
            // When Normalize=false, skip this normalization (used in FFBP)
            if constexpr (HasAntennaPattern && Normalize) {
                if (w_sum2[k] >= 1.17549435e-38f) {
                    pixel *= w_sum1[k] / w_sum2[k];
                }
            }

            if (dealias) {
                // Without a DEM: range-row-only carrier (also for guard band
                // pixels |theta| > 1 where x^2 + y^2 != r^2). With a DEM the
                // carrier is always referenced to the DEM height instead of
                // the z = 0 plane: a flat-plane carrier would leave a
                // terrain-dependent residual whose local frequency aliases
                // the image spectrum on any significant topography.
                // bp_polar_range_dealias/alias with the same dem apply/remove
                // the identical carrier.
                float dd;
                if constexpr (HasDem) {
                    const float zz = z0 - z[k];
                    dd = sqrtf(fmaf(zz, zz, r2[k]));
                } else {
                    dd = sqrtf(r2[k] + z0*z0);
                }
                float ref_sin, ref_cos;
                __sincosf(fmaf(dealias_coef, dd, dealias_fmod * (idr_base + k)), &ref_sin, &ref_cos);
                complex64_t ref = {ref_cos, ref_sin};
                pixel *= ref;
            }

            img[idbatch * Nr * Ntheta + (idr_base + k) * Ntheta + idtheta] = pixel;
        }
    }
}

template<typename T, bool have_pos_grad, bool have_data_grad>
__global__ void backprojection_polar_2d_grad_kernel(
          const T* data,
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
          complex64_t *data_grad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    // Out-of-range threads can't return early because every lane in the warp
    // must participate in the shuffle reduction below. They stay alive with
    // zero contributions instead.
    const bool active = idx < Nr * Ntheta;

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    // Clamped cosine + polar-form distance below: same guard band
    // continuation past |theta| = 1 as the forward kernel.
    const float x = r * sqrtf(fmaxf(0.0f, 1.0f - theta*theta));
    const float y = r * theta;

    complex64_t g = {0.0f, 0.0f};
    if (active) {
        g = grad[idbatch * Nr * Ntheta + idr * Ntheta + idtheta];
    }

    float arg_dealias = 0.0f;
    if (dealias) {
        const float d = sqrtf(r*r + z0*z0);
        arg_dealias = -ref_phase * d + alias_fmod * idr;
        // TODO: Missing z0 gradient.
    }

    const complex64_t I = {0.0f, 1.0f};

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
        const float d_sq = r*r + pos_x*pos_x + pos_y*pos_y + pz2
                         - 2.0f * (x * pos_x + y * pos_y);
        const float d = sqrtf(fmaxf(d_sq, 0.0f));

        float sx = delta_r * (d + d0);

        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;
        complex64_t ds0 = 0.0f;
        complex64_t ds1 = 0.0f;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (active && sx >= 0.0f && id1 < sweep_samples) {
            complex64_t s0, s1;
            if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
                s0 = ((complex64_t*)data)[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
                s1 = ((complex64_t*)data)[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];
            } else {
                half2 s0h = ((half2*)data)[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
                half2 s1h = ((half2*)data)[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];
                s0 = {__half2float(s0h.x), __half2float(s0h.y)};
                s1 = {__half2float(s1h.x), __half2float(s1h.y)};
            }

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospif(ref_phase * d - data_fmod * sx + arg_dealias, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * (I * (kPI * (ref_phase - delta_r * data_fmod)) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * cuda::std::conj(dout);

                // Take real part
                float gd = cuda::std::real(gdout);

                dx = -px / d;
                dy = -py / d;
                // Different from x,y because pos_z is handled differently.
                dz = pos_z / d;
                dx *= gd;
                dy *= gd;
                dz *= gd;
                // Avoid issues with zero range
                if (!isfinite(dx)) dx = 0.0f;
                if (!isfinite(dy)) dy = 0.0f;
                if (!isfinite(dz)) dz = 0.0f;
            }

            if (have_data_grad) {
                ds0 = g * cuda::std::conj((1.0f - interp_idx) * ref);
                ds1 = g * cuda::std::conj(interp_idx * ref);
            }
        }

        if (have_pos_grad) {
            for (int offset = 16; offset > 0; offset /= 2) {
                dx += __shfl_down_sync(FULL_MASK, dx, offset);
                dy += __shfl_down_sync(FULL_MASK, dy, offset);
                dz += __shfl_down_sync(FULL_MASK, dz, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 0]), dx);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 1]), dy);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 2]), dz);
            }
        }

        if (have_data_grad) {
            if (active && sx >= 0.0f && id1 < sweep_samples) {
                float2 *x0 = (float2*)&data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
                float2 *x1 = (float2*)&data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];
                atomicAdd(&x0->x, ds0.real());
                atomicAdd(&x0->y, ds0.imag());
                atomicAdd(&x1->x, ds1.real());
                atomicAdd(&x1->y, ds1.imag());
            }
        }
    }
}

template<typename T>
__global__ void gpga_backprojection_2d_kernel(
          const float* target_pos,
          const T* data,
          const float* pos,
          complex64_t* data_out,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          int Ntarget,
          float d0,
          float data_fmod) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idsweep = idx % nsweeps;
    const int idtarget = idx / nsweeps;

    if (idtarget >= Ntarget || idsweep >= nsweeps) {
        return;
    }

    const float x = target_pos[idtarget * 3 + 0];
    const float y = target_pos[idtarget * 3 + 1];
    const float z = target_pos[idtarget * 3 + 2];

    // Sweep reference position.
    float pos_x = pos[idsweep * 3 + 0];
    float pos_y = pos[idsweep * 3 + 1];
    float pos_z = pos[idsweep * 3 + 2];
    float px = (x - pos_x);
    float py = (y - pos_y);
    float pz = (z - pos_z);

    // Calculate distance to the pixel.
    const float d = sqrtf(px * px + py * py + pz * pz);

    float sx = delta_r * (d + d0);

    // Linear interpolation.
    int id0 = sx;
    int id1 = id0 + 1;
    if (sx < 0.0f || id1 >= sweep_samples) {
        data_out[idtarget * nsweeps + idsweep] = {0.0f, 0.0f};
    } else {
        complex64_t s0, s1;
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            s0 = ((complex64_t*)data)[idsweep * sweep_samples + id0];
            s1 = ((complex64_t*)data)[idsweep * sweep_samples + id1];
        } else {
            half2 s0h = ((half2*)data)[idsweep * sweep_samples + id0];
            half2 s1h = ((half2*)data)[idsweep * sweep_samples + id1];
            s0 = {__half2float(s0h.x), __half2float(s0h.y)};
            s1 = {__half2float(s1h.x), __half2float(s1h.y)};
        }
        float interp_idx = sx - id0;
        complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

        float ref_sin, ref_cos;
        sincospif(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        data_out[idtarget * nsweeps + idsweep] = s * ref;
    }
}

// Inner product of one image block against one sweep's backprojection
// footprint: alpha[b, m] = sum over the block's pixels of
// conj(img[pix]) * data[m, interp] * exp(j (ref_phase d - data_fmod sx)).
// Per-pixel math matches gpga_backprojection_2d_kernel (z=0 pixel plane,
// linear range interpolation); the master image acts as the pixel
// weighting so the [npix, nsweeps] footprint matrix is never
// materialized. Mirrors blocksvd_alpha_kernel_cpu in cpu/backproj.cpp.
__global__ void blocksvd_alpha_kernel(
          const complex64_t* img,
          const complex64_t* data,
          const float* pos,
          const int32_t* blocks,
          complex64_t* alpha,
          int sweep_samples,
          int nsweeps,
          int nblocks,
          int Ntheta,
          float ref_phase,
          float delta_r,
          float r0,
          float dr,
          float theta0,
          float dtheta,
          float d0,
          float data_fmod) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idsweep = idx % nsweeps;
    const int idblock = idx / nsweeps;

    if (idblock >= nblocks || idsweep >= nsweeps) {
        return;
    }

    const int ri0 = blocks[idblock * 6 + 0];
    const int ri1 = blocks[idblock * 6 + 1];
    const int ti0 = blocks[idblock * 6 + 2];
    const int ti1 = blocks[idblock * 6 + 3];
    const int sweep_lo = blocks[idblock * 6 + 4];
    const int sweep_hi = blocks[idblock * 6 + 5];
    // Output is pre-zeroed.
    if (idsweep < sweep_lo || idsweep >= sweep_hi) {
        return;
    }

    const float pos_x = pos[idsweep * 3 + 0];
    const float pos_y = pos[idsweep * 3 + 1];
    const float pos_z = pos[idsweep * 3 + 2];
    const float pz2 = pos_z * pos_z;
    const complex64_t* data_row = data + (size_t)idsweep * sweep_samples;

    // Theta in the outer loop, range rows inner, mirroring
    // blocksvd_alpha_kernel_cpu.
    complex64_t acc = {0.0f, 0.0f};
    for (int j = ti0; j < ti1; j++) {
        const float theta = theta0 + j * dtheta;
        const float ct = sqrtf(fmaxf(0.0f, 1.0f - theta * theta));

        for (int i = ri0; i < ri1; i++) {
            const float r = r0 + i * dr;
            const float px = r * ct - pos_x;
            const float py = r * theta - pos_y;

            // Calculate distance to the pixel.
            const float d = sqrtf(px * px + py * py + pz2);

            const float sx = delta_r * (d + d0);

            // Linear interpolation.
            const int id0 = sx;
            const int id1 = id0 + 1;
            if (sx < 0.0f || id1 >= sweep_samples) {
                continue;
            }
            const complex64_t s0 = data_row[id0];
            const complex64_t s1 = data_row[id1];
            const float interp_idx = sx - id0;
            const complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospif(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            const complex64_t ref = {ref_cos, ref_sin};
            acc += cuda::std::conj(img[(size_t)i * Ntheta + j]) * (s * ref);
        }
    }
    alpha[(size_t)idblock * nsweeps + idsweep] = acc;
}

template<typename T>
__global__ void gpga_backprojection_2d_lanczos_kernel(
          const float* target_pos,
          const T* data,
          const float* pos,
          complex64_t* data_out,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          int Ntarget,
          float d0,
          int order,
          float data_fmod) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idsweep = idx % nsweeps;
    const int idtarget = idx / nsweeps;

    if (idtarget >= Ntarget || idsweep >= nsweeps) {
        return;
    }

    const float x = target_pos[idtarget * 3 + 0];
    const float y = target_pos[idtarget * 3 + 1];
    const float z = target_pos[idtarget * 3 + 2];

    // Sweep reference position.
    float pos_x = pos[idsweep * 3 + 0];
    float pos_y = pos[idsweep * 3 + 1];
    float pos_z = pos[idsweep * 3 + 2];
    float px = (x - pos_x);
    float py = (y - pos_y);
    float pz = (z - pos_z);

    // Calculate distance to the pixel.
    const float d = sqrtf(px * px + py * py + pz * pz);

    float sx = delta_r * (d + d0);

    // Linear interpolation.
    int id0 = sx;
    int id1 = id0 + 1;
    if (sx < 0.0f || id1 >= sweep_samples) {
        data_out[idtarget * nsweeps + idsweep] = {0.0f, 0.0f};
    } else {
        complex64_t s = lanczos_interp_1d<complex64_t, T>(
                &data[idsweep * sweep_samples],
                sweep_samples, sx, order);

        float ref_sin, ref_cos;
        sincospif(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        data_out[idtarget * nsweeps + idsweep] = s * ref;
    }
}


__global__ void backprojection_polar_2d_tx_power_kernel(
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
          float altitude) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

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

// Cartesian analog of backprojection_polar_2d_tx_power_kernel. The pixel
// ground position comes straight from the Cartesian image grid (z = 0 ground
// plane) instead of a polar (r, theta) cell; the platform altitude is always
// the per-sweep pos_z.
__global__ void backprojection_cart_2d_tx_power_kernel(
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
          int azimuth_resolution) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idx % Ny;
    const int idx_x = idx / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void backprojection_polar_2d_tx_power_accum_kernel(
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
          int theta_psi) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void backprojection_cart_2d_tx_power_accum_kernel(
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
          float h_ref) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void cart_tx_power_merge2_kernel(
          const float* acc0,
          const float* acc1,
          float* out,
          float x0_0, float dx_0, float y0_0, float dy_0, int Nx_0, int Ny_0,
          float x0_1, float dx_1, float y0_1, float dy_1, int Nx_1, int Ny_1,
          float x1, float dx1, float y1, float dy1, int Nx1, int Ny1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void backprojection_cart_2d_kernel(
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
          float beamwidth,
          float d0,
          float data_fmod) {
    const int idt = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idt % Ny;
    const int idx = idt / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= Nx || idy >= Ny) {
        return;
    }

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;

    complex64_t pixel = {0, 0};

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz2 = pos_z * pos_z;

        //float target_angle = atan2(px, py);
        //if (fabsf(target_angle - att[i*3 + 2]) > beamwidth) {
        //    // Pixel outside of the beam.
        //    continue;
        //}
        // Calculate distance to the pixel.

        float d = sqrtf(px * px + py * py + pz2);

        float sx = delta_r * (d + d0);

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (sx >= 0.0f && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospif(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += s * ref;
        }
    }
    img[idbatch * Nx * Ny + idx * Ny + idy] = pixel;
}

template<bool have_pos_grad, bool have_data_grad>
__global__ void backprojection_cart_2d_grad_kernel(
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
          complex64_t *data_grad) {
    const int idt = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idt % Ny;
    const int idx = idt / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    // Out-of-range threads can't return early because every lane in the warp
    // must participate in the shuffle reduction below. They stay alive with
    // zero contributions instead.
    const bool active = idx < Nx && idy < Ny;

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;

    complex64_t g = {0.0f, 0.0f};
    if (active) {
        g = grad[idbatch * Nx * Ny + idx * Ny + idy];
    }

    const complex64_t I = {0.0f, 1.0f};

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz2 = pos_z * pos_z;

        //float target_angle = atan2(px, py);
        //if (fabsf(target_angle - att[i*3 + 2]) > beamwidth) {
        //    // Pixel outside of the beam.
        //    continue;
        //}
        // Calculate distance to the pixel.

        float d = sqrtf(px * px + py * py + pz2);

        float sx = delta_r * (d + d0);

        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;
        complex64_t ds0 = 0.0f;
        complex64_t ds1 = 0.0f;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (active && sx >= 0.0f && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospif(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * ((I * kPI * (ref_phase - delta_r * data_fmod)) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * cuda::std::conj(dout);

                float gd = cuda::std::real(gdout);

                dx = -px / d;
                dy = -py / d;
                // Different from x,y because pos_z is handled differently.
                dz = pos_z / d;
                dx *= gd;
                dy *= gd;
                dz *= gd;
                // Avoid issues with zero range
                if (!isfinite(dx)) dx = 0.0f;
                if (!isfinite(dy)) dy = 0.0f;
                if (!isfinite(dz)) dz = 0.0f;
            }

            if (have_data_grad) {
                ds0 = g * cuda::std::conj((1.0f - interp_idx) * ref);
                ds1 = g * cuda::std::conj(interp_idx * ref);
            }
        }

        if (have_pos_grad) {
            for (int offset = 16; offset > 0; offset /= 2) {
                dx += __shfl_down_sync(FULL_MASK, dx, offset);
                dy += __shfl_down_sync(FULL_MASK, dy, offset);
                dz += __shfl_down_sync(FULL_MASK, dz, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 0]), dx);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 1]), dy);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 2]), dz);
            }
        }

        if (have_data_grad) {
            if (active && sx >= 0.0f && id1 < sweep_samples) {
                float2 *x0 = (float2*)&data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
                float2 *x1 = (float2*)&data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];
                atomicAdd(&x0->x, ds0.real());
                atomicAdd(&x0->y, ds0.imag());
                atomicAdd(&x1->x, ds1.real());
                atomicAdd(&x1->y, ds1.imag());
            }
        }
    }
}


// Forward projection kernel.
//
// Grid:  (ceil(sweep_samples / (blockDim.x * NSAMP)), nsweeps, nbatch)
// Block: (256, 1, 1)
// Shared memory (bytes): (HAS_VEL ? 4 : 3) * blockDim.x * sizeof(float)
//
// Each thread owns NSAMP consecutive output samples and accumulates
// contributions from all pixels via cooperative tiling.
//
// Template parameters:
//   NSAMP:     output samples per thread (4 is optimal: fills the 16-cycle
//              SFU latency window with independent sincospif calls)
//   HAS_VEL:   enables per-sample Doppler range correction; requires
//              `vel` pointer and stores (d, vel_proj) per pixel in smem
//   USE_RVP:   adds the residual video phase term gamma*tau^2
template <int NSAMP, bool HAS_VEL, bool USE_RVP>
__global__ void projection_cart_2d_kernel(
          const complex64_t* __restrict__ img,
          const float* __restrict__ dem,
          const float* __restrict__ pos,
          const float* __restrict__ vel,   // may be nullptr when !HAS_VEL
          const float* __restrict__ att,   // may be nullptr when g == nullptr
          complex64_t* __restrict__ data,
          int sweep_samples, int nsweeps,
          float fc, float fs, float gamma,
          float x0, float dx, float y0, float dy, int Nx, int Ny, float d0,
          const float* __restrict__ g,
          float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel,
          int normalization) {
    const int j_base   = (blockIdx.x * blockDim.x + threadIdx.x) * NSAMP;
    const int i        = blockIdx.y;
    const int idbatch  = blockIdx.z;
    const int tid      = threadIdx.x;
    const int TILE     = blockDim.x;

    // Shared memory layout (per block):
    //   [0..TILE)       sh_w_re  – pixel weight real part
    //   [TILE..2TILE)   sh_w_im  – pixel weight imaginary part
    //   [2TILE..3TILE)  sh_tau (HAS_VEL=false) or sh_d (HAS_VEL=true)
    //   [3TILE..4TILE)  sh_vel_proj (HAS_VEL=true only)
    extern __shared__ float smem[];
    float* sh_w_re = smem;
    float* sh_w_im = smem + TILE;
    float* sh_a    = smem + 2 * TILE;  // tau (!HAS_VEL) or d (HAS_VEL)
    float* sh_b    = smem + 3 * TILE;  // vel_proj (HAS_VEL only)

    // phase = tau * phase_slope  where  phase_slope = -2*gamma*j/fs - 2*fc
    // Works for both HAS_VEL=false (tau constant per pixel) and HAS_VEL=true
    // (tau_s computed from sh_a/sh_b in Phase 2 but multiplied by same slope).
    float phase_slopes[NSAMP];
    float s_re[NSAMP], s_im[NSAMP];
    #pragma unroll
    for (int s = 0; s < NSAMP; s++) {
        const int j = j_base + s;
        phase_slopes[s] = (j < sweep_samples)
            ? (-2.0f * gamma * (float)j / fs - 2.0f * fc) : 0.0f;
        s_re[s] = s_im[s] = 0.0f;
    }

    // j - sweep_samples/2: the Doppler range correction uses this offset.
    // Declared unconditionally; only read inside if constexpr (HAS_VEL).
    float j_half[NSAMP];
    #pragma unroll
    for (int s = 0; s < NSAMP; s++)
        j_half[s] = (float)(j_base + s) - 0.5f * (float)sweep_samples;

    const float pos_x    = pos[idbatch * nsweeps * 3 + i * 3 + 0];
    const float pos_y    = pos[idbatch * nsweeps * 3 + i * 3 + 1];
    const float pos_z    = pos[idbatch * nsweeps * 3 + i * 3 + 2];
    const float att_roll = (g != nullptr) ? att[idbatch * nsweeps * 3 + 3*i + 0] : 0.0f;
    const float att_yaw  = (g != nullptr) ? att[idbatch * nsweeps * 3 + 3*i + 2] : 0.0f;

    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    if constexpr (HAS_VEL) {
        vx = vel[idbatch * nsweeps * 3 + i * 3 + 0];
        vy = vel[idbatch * nsweeps * 3 + i * 3 + 1];
        vz = vel[idbatch * nsweeps * 3 + i * 3 + 2];
    }

    const int total_pixels = Nx * Ny;

    for (int p_base = 0; p_base < total_pixels; p_base += TILE) {
        // Phase 1: each thread computes geometry for one pixel in this tile
        const int p_idx = p_base + tid;
        float my_w_re = 0.0f, my_w_im = 0.0f, my_a = 0.0f, my_b = 0.0f;

        if (p_idx < total_pixels) {
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

            my_w_re = w_img.real() * norm;
            my_w_im = w_img.imag() * norm;

            if constexpr (HAS_VEL) {
                my_a = d;
                my_b = (dpx * vx + dpy * vy + dpz * vz) / d / fs;
            } else {
                my_a = 2.0f * (d + d0) / kC0;  // tau
            }
        }

        __syncthreads();
        sh_w_re[tid] = my_w_re;
        sh_w_im[tid] = my_w_im;
        sh_a[tid]    = my_a;
        if constexpr (HAS_VEL) sh_b[tid] = my_b;
        __syncthreads();

        // Phase 2: accumulate TILE pixels into each of NSAMP sample outputs
        // NSAMP independent sincospif calls per pixel fill the 16-cycle SFU
        // latency window.
        const int limit = min(TILE, total_pixels - p_base);
        #pragma unroll 4
        for (int k = 0; k < limit; k++) {
            const float wr = sh_w_re[k];
            const float wi = sh_w_im[k];

            // RVP term: gamma*tau^2. For HAS_VEL=false, tau is constant per
            // pixel so it can be hoisted out of the s-loop.
            float rvp_nv = 0.0f;
            if constexpr (!HAS_VEL && USE_RVP) rvp_nv = gamma * sh_a[k] * sh_a[k];

            float ss[NSAMP], cs[NSAMP];
            #pragma unroll
            for (int s = 0; s < NSAMP; s++) {
                float phase;
                if constexpr (HAS_VEL) {
                    const float tau_s =
                        2.0f * (sh_a[k] + d0 + sh_b[k] * j_half[s]) / kC0;
                    phase = tau_s * phase_slopes[s];
                    if constexpr (USE_RVP) phase += gamma * tau_s * tau_s;
                } else {
                    phase = sh_a[k] * phase_slopes[s] + rvp_nv;
                }
                sincospif(phase, &ss[s], &cs[s]);
            }

            #pragma unroll
            for (int s = 0; s < NSAMP; s++) {
                s_re[s] += wr * cs[s] - wi * ss[s];
                s_im[s] += wr * ss[s] + wi * cs[s];
            }
        }
    }

    // One direct write per sample
    #pragma unroll
    for (int s = 0; s < NSAMP; s++) {
        const int j = j_base + s;
        if (j < sweep_samples)
            data[idbatch * sweep_samples * nsweeps + i * sweep_samples + j] =
                complex64_t(s_re[s], s_im[s]);
    }
}


/*
NUFFT-based forward projection (vel=None path)

Reformulates the direct sum as a Type-1 NUFFT:
  A_p = w_p exp(-j 2pi fc tau_p)          Complex amplitude
  nu_p = gamma tau_p / fs                 Normalised frequency
  data[k] = Sum_p A_p exp(-j 2pi nu_p k)  IFFT of spread grid

Three kernels: geometry, spread, deconvolve (after IFFT via PyTorch).
*/

// Kaiser-Bessel helper (host side)
static float bessel_i0_host(float x) {
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

// KB LUT: N_LUT samples of psi_KB(t) for t in [0, W/2]
static at::Tensor make_kb_lut(int N_LUT, float W, float beta, at::Device dev) {
    at::Tensor lut = at::zeros({N_LUT},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    float* p = lut.data_ptr<float>();
    float i0b = bessel_i0_host(beta);
    for (int i = 0; i < N_LUT; ++i) {
        float t  = (float)i / (float)(N_LUT - 1) * (W * 0.5f);  // [0, W/2]
        float u  = 2.0f * t / W;                                // [0, 1]
        float sq = 1.0f - u * u;
        p[i] = (sq > 0.0f) ? bessel_i0_host(beta * sqrtf(sq)) / i0b : 0.0f;
    }
    return lut.to(dev);
}

// Deconvolution window for centered extraction: correction[j] = M_ext / conj(phi_fwd[k])
// where k = j - M/2  (centered bin index, k in [-M/2, M/2-1]).
// Using centered bins keeps |k/M_ext| <= 0.25, well within the KB passband
// where |phi_fwd[k]| >= 0.93 (vs ~0.15 at the one-sided edge j=M-1).
// phi_fwd[k] = Sum_{delta=-(W/2-1)}^{W/2} psi_KB(|delta|) * exp(-j 2pi k delta/M_ext)
static at::Tensor make_deconv_window(int M, int M_ext, float W, float beta,
                                     at::Device dev) {
    int half_W = (int)(W * 0.5f);          // = 3
    float i0b  = bessel_i0_host(beta);
    // psi_KB values for delta=0..half_W
    std::vector<float> psi(half_W + 1);
    for (int d = 0; d <= half_W; ++d) {
        float u  = (float)d / (float)half_W;
        float sq = 1.0f - u * u;
        psi[d] = (sq > 0.0f) ? bessel_i0_host(beta * sqrtf(sq)) / i0b : 0.0f;
    }
    at::Tensor win = at::zeros({M},
        at::TensorOptions().dtype(at::kComplexFloat).device(at::kCPU));
    c10::complex<float>* w = win.data_ptr<c10::complex<float>>();
    for (int j = 0; j < M; ++j) {
        // Centered bin: k = j - M/2, evaluated at k for deconvolution.
        int k = j - M / 2;
        float re = 0.0f, im = 0.0f;
        // delta in {-(half_W-1), ..., half_W}  = {-2,-1,0,1,2,3} for W=6
        for (int d = -(half_W - 1); d <= half_W; ++d) {
            float angle = -2.0f * (float)M_PI * (float)k * (float)d / (float)M_ext;
            float kb    = psi[abs(d)];
            re += kb * cosf(angle);
            im += kb * sinf(angle);
        }
        // correction = M_ext / conj(phi_fwd[k]) = M_ext phi_fwd[k] / |phi_fwd[k]|^2
        float denom = re * re + im * im;
        float scale = (float)M_ext / denom;
        w[j] = c10::complex<float>(re * scale, +im * scale);
    }
    return win.to(dev);
}

// Static cache: keyed on (sweep_samples, M_ext, device_index)
struct NufftCache { at::Tensor kb_lut, deconv_win; };
static std::mutex                                       g_nufft_cache_mutex;
static std::map<std::tuple<int,int,int>, NufftCache>   g_nufft_cache;

static std::pair<at::Tensor, at::Tensor>
get_nufft_tables(int sweep_samples, int M_ext, float W, float beta,
                 int N_LUT, at::Device dev) {
    std::lock_guard<std::mutex> lock(g_nufft_cache_mutex);
    auto key = std::make_tuple(sweep_samples, M_ext, dev.index());
    auto it  = g_nufft_cache.find(key);
    if (it != g_nufft_cache.end())
        return {it->second.kb_lut, it->second.deconv_win};
    NufftCache c;
    c.kb_lut    = make_kb_lut(N_LUT, W, beta, dev);
    c.deconv_win = make_deconv_window(sweep_samples, M_ext, W, beta, dev);
    g_nufft_cache[key] = c;
    return {c.kb_lut, c.deconv_win};
}

// Kernel 1: geometry
// Computes complex amplitude A_p and
// the unsigned grid position u_p = M_ext - gamma tau / fs for each pixel.
template <bool USE_RVP>
__global__ void projection_nufft_geometry_kernel(
          const complex64_t* __restrict__ img,      // [N]
          const float*       __restrict__ dem,      // [Nx*Ny] or nullptr
          const float*       __restrict__ pos_s,    // [3] this sweep pos
          const float*       __restrict__ att_s,    // [3] or nullptr
          float* __restrict__ A_re,
          float* __restrict__ A_im,
          float* __restrict__ u_out,                // unsigned grid position
          int N, float fc, float gamma_f, float fs_f, float d0,
          float x0, float dx, float y0, float dy, int Nx, int Ny,
          float M_ext_f,
          const float* __restrict__ g,
          float g_az0, float g_el0, float g_daz, float g_del,
          int g_naz, int g_nel, int normalization) {

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;

    const int px_y = p % Ny;
    const int px_x = p / Ny;
    const float x = x0 + px_x * dx;
    const float y = y0 + px_y * dy;
    const float z = (dem != nullptr) ? dem[p] : 0.0f;

    const float dpx = x - pos_s[0];
    const float dpy = y - pos_s[1];
    const float dpz = z - pos_s[2];
    const float d2  = dpx*dpx + dpy*dpy + dpz*dpz;
    const float d   = sqrtf(d2);

    float norm = 1.0f / d2;
    if (g != nullptr) {
        const float look   = asinf(fmaxf(dpz / d, -1.0f));
        const float el_deg = look - att_s[0];
        const float az_deg = atan2f(dpy, dpx) - att_s[2];
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

    const complex64_t w_img = img[p];
    const float wr = w_img.real() * norm;
    const float wi = w_img.imag() * norm;

    const float tau   = 2.0f * (d + d0) / kC0;
    const float nu_p  = gamma_f * tau / fs_f;

    // Grid position: u_p = M_ext - nu_p * M_ext
    // exp(+j*2pi*u_p*k/M_ext) = exp(-j*2pi*nu_p*k) for integer k (via periodicity)
    u_out[p] = M_ext_f - nu_p * M_ext_f;

    // Carrier phase: -2fc*tau [+ gamma*tau^2 if USE_RVP]
    // Centered extraction adds exp(-j*pi*nu_p*M) to amplitude for spectral centering
    float phase = -2.0f * fc * tau;
    if constexpr (USE_RVP) phase += gamma_f * tau * tau;
    phase -= nu_p * M_ext_f * 0.5f;  // add exp(-j*pi*nu_p*M), M = M_ext/2
    float cs, sn;
    sincospif(phase, &sn, &cs);
    A_re[p] = wr * cs - wi * sn;
    A_im[p] = wr * sn + wi * cs;
}

// Kernel 2: spreading
// Scatter each pixel's amplitude onto the oversampled grid using
// W=6 Kaiser-Bessel taps.
__global__ void projection_nufft_spread_kernel(
          const float* __restrict__ A_re,
          const float* __restrict__ A_im,
          const float* __restrict__ u,
          float2*      __restrict__ grid,   // [M_ext], zeroed externally
          const float* __restrict__ kb_lut,
          int N, int M_ext, int N_LUT, float W) {

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;

    const float ar  = A_re[p];
    const float ai  = A_im[p];
    if (ar == 0.0f && ai == 0.0f) return;

    const float up  = u[p];
    const int   j0  = (int)rintf(up);              // nearest grid bin
    const float frc = up - (float)j0;              // fractional offset
    const float lut_scale = (float)(N_LUT - 1) * 2.0f / W;

    // delta in {-W/2+1, ..., W/2} = {-2,-1,0,1,2,3} for W=6
    const int half_W = (int)(W * 0.5f);
    #pragma unroll
    for (int delta = -(half_W - 1); delta <= half_W; ++delta) {
        float dist    = fabsf(frc - (float)delta);
        int   lut_idx = min((int)(dist * lut_scale), N_LUT - 1);
        float kb      = kb_lut[lut_idx];

        int gidx = ((j0 + delta) % M_ext + M_ext) % M_ext;
        atomicAdd(&grid[gidx].x, ar * kb);
        atomicAdd(&grid[gidx].y, ai * kb);
    }
}

// Kernel 3: deconvolve + extract
// Multiply IFFT output by precomputed correction and copy the
// first sweep_samples bins to the output tensor.
__global__ void projection_nufft_deconvolve_kernel(
          const complex64_t* __restrict__ Z,       // [nbatch, nsweeps, M_ext]
          const complex64_t* __restrict__ deconv,  // [sweep_samples]
          complex64_t*       __restrict__ data,    // [nbatch, nsweeps, sweep_samples]
          int sweep_samples, int nsweeps, int M_ext) {

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;   // sweep index
    const int b = blockIdx.z;   // batch index
    if (j >= sweep_samples) return;

    // Centered extraction: IFFT bin k = j - M/2 (mapped into [0, M_ext-1]).
    // This keeps all bins in the flat KB passband (|k/M_ext| <= 0.25).
    const int k = ((j - sweep_samples / 2) % M_ext + M_ext) % M_ext;
    const int z_idx   = (b * nsweeps + i) * M_ext + k;
    const int out_idx = (b * nsweeps + i) * sweep_samples + j;
    const complex64_t z = Z[z_idx];
    const complex64_t w = deconv[j];
    data[out_idx] =
        complex64_t(z.real()*w.real() - z.imag()*w.imag(),
                    z.real()*w.imag() + z.imag()*w.real());
}

// NUFFT launcher
at::Tensor projection_cart_2d_nufft_cuda(
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

    const int N      = (int)(Nx * Ny);
    const int M      = (int)sweep_samples;
    const int M_ext  = 2 * M;           // oversampled grid size
    const float W    = 6.0f;
    const float beta = 2.34f * W;       // KB parameter
    const int N_LUT  = 8192;

    auto dev  = img.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Retrieve or build KB LUT and deconvolution window
    auto [kb_lut, deconv_win] = get_nufft_tables(M, M_ext, W, beta, N_LUT, dev);

    // Allocate buffers
    auto float_opts = at::TensorOptions().dtype(at::kFloat).device(dev);
    auto cplx_opts  = at::TensorOptions().dtype(at::kComplexFloat).device(dev);

    at::Tensor A_re  = at::empty({N}, float_opts);
    at::Tensor A_im  = at::empty({N}, float_opts);
    at::Tensor u_buf = at::empty({N}, float_opts);
    at::Tensor grid  = at::zeros({nbatch, nsweeps, M_ext}, cplx_opts);
    at::Tensor data  = at::empty({nbatch, nsweeps, M},     cplx_opts);

    // Keep contiguous copies alive until the kernels have run.
    at::Tensor pos_contig = pos.contiguous();
    at::Tensor dem_contig = dem.defined() ? dem.contiguous() : at::Tensor();
    at::Tensor att_contig = (g.defined() && att.defined()) ? att.contiguous() : at::Tensor();
    at::Tensor g_contig   = g.defined() ? g.contiguous() : at::Tensor();

    const complex64_t* img_raw  = (const complex64_t*)img.data_ptr<c10::complex<float>>();
    const float*       dem_raw  = dem_contig.defined() ? dem_contig.data_ptr<float>() : nullptr;
    const float*       pos_raw  = pos_contig.data_ptr<float>();
    const float*       att_raw  = att_contig.defined() ? att_contig.data_ptr<float>() : nullptr;
    const float*       g_raw    = g_contig.defined() ? g_contig.data_ptr<float>() : nullptr;
    const float*       lut_raw  = kb_lut.data_ptr<float>();
    float2*            grid_raw = (float2*)grid.data_ptr<c10::complex<float>>();
    float*             ar       = A_re.data_ptr<float>();
    float*             ai       = A_im.data_ptr<float>();
    float*             up       = u_buf.data_ptr<float>();

    const int BLOCK = 256;
    dim3 geom_tpb  = {BLOCK, 1, 1};
    dim3 geom_grid = {(unsigned)((N + BLOCK - 1) / BLOCK), 1, 1};
    dim3 sprd_grid = {(unsigned)((N + BLOCK - 1) / BLOCK), 1, 1};

    // Per-sweep geometry + spreading
    for (int64_t b = 0; b < nbatch; ++b) {
        const complex64_t* img_b  = img_raw  + b * N;
        float2*            grid_b = grid_raw + (b * nsweeps) * M_ext;

        for (int64_t i = 0; i < nsweeps; ++i) {
            const float* pos_si = pos_raw + (b * nsweeps + i) * 3;
            const float* att_si = att_raw ? att_raw + (b * nsweeps + i) * 3 : nullptr;

            // Geometry
            if (use_rvp) {
                projection_nufft_geometry_kernel<true>
                    <<<geom_grid, geom_tpb, 0, stream>>>(
                    img_b, dem_raw, pos_si, att_si,
                    ar, ai, up, N,
                    (float)fc, (float)gamma, (float)fs, (float)d0,
                    (float)x0, (float)dx, (float)y0, (float)dy, (int)Nx, (int)Ny,
                    (float)M_ext,
                    g_raw, (float)g_az0, (float)g_el0, (float)g_daz, (float)g_del,
                    (int)g_naz, (int)g_nel, (int)normalization);
            } else {
                projection_nufft_geometry_kernel<false>
                    <<<geom_grid, geom_tpb, 0, stream>>>(
                    img_b, dem_raw, pos_si, att_si,
                    ar, ai, up, N,
                    (float)fc, (float)gamma, (float)fs, (float)d0,
                    (float)x0, (float)dx, (float)y0, (float)dy, (int)Nx, (int)Ny,
                    (float)M_ext,
                    g_raw, (float)g_az0, (float)g_el0, (float)g_daz, (float)g_del,
                    (int)g_naz, (int)g_nel, (int)normalization);
            }

            // Spreading into the grid slice for this sweep
            projection_nufft_spread_kernel
                <<<sprd_grid, geom_tpb, 0, stream>>>(
                ar, ai, up, grid_b + i * M_ext, lut_raw,
                N, M_ext, N_LUT, W);
        }
    }

    // Batched IFFT over last dimension
    at::Tensor Z = at::fft_ifft(grid, std::optional<int64_t>(M_ext), -1);

    // Deconvolve + extract sweep_samples bins per row
    {
        const complex64_t* Z_raw    = (const complex64_t*)Z.data_ptr<c10::complex<float>>();
        const complex64_t* dw_raw   = (const complex64_t*)deconv_win.data_ptr<c10::complex<float>>();
        complex64_t*       data_raw = (complex64_t*)data.data_ptr<c10::complex<float>>();

        dim3 dconv_tpb  = {BLOCK, 1, 1};
        dim3 dconv_grid = {
            (unsigned)((M + BLOCK - 1) / BLOCK),
            (unsigned)nsweeps,
            (unsigned)nbatch
        };
        projection_nufft_deconvolve_kernel
            <<<dconv_grid, dconv_tpb, 0, stream>>>(
            Z_raw, dw_raw, data_raw, M, (int)nsweeps, M_ext);
    }

    return data;
}


at::Tensor projection_cart_2d_cuda(
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
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
    }
    if (vel.defined()) {
        TORCH_CHECK(vel.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
    }

    if (dem.defined()) {
        TORCH_CHECK(dem.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CUDA);
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
    // Keep contiguous copies alive until the kernel has run.
    at::Tensor dem_contig;
    at::Tensor vel_contig;
    at::Tensor att_contig;
    at::Tensor g_contig;
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
    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        att_contig = att.contiguous();
        g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
    }

    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();
    c10::complex<float>* data_ptr = data.data_ptr<c10::complex<float>>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Direct kernel: templated on (NSAMP, HAS_VEL, USE_RVP).
    // One kernel for all cases, templated on (NSAMP, HAS_VEL, USE_RVP).
    // NSAMP=4: 4 independent sincospif per pixel fills the 16-cycle SFU latency
    // window
    const int  BLOCK = 256;
    const int  NSAMP = 4;
    const bool hv    = (vel_ptr != nullptr);
    const bool rv    = (bool)use_rvp;

    // Shared memory: 3 arrays of BLOCK floats (!HAS_VEL) or 4 (HAS_VEL).
    const size_t smem = (hv ? 4 : 3) * BLOCK * sizeof(float);

    dim3 tpb  = {static_cast<unsigned int>(BLOCK), 1, 1};
    dim3 grid = {
        static_cast<unsigned int>((sweep_samples + BLOCK * NSAMP - 1) / (BLOCK * NSAMP)),
        static_cast<unsigned int>(nsweeps),
        static_cast<unsigned int>(nbatch)
    };

#define LAUNCH(HV, URV) \
    projection_cart_2d_kernel<NSAMP, HV, URV> \
        <<<grid, tpb, smem, stream>>>( \
            (complex64_t*)img_ptr, dem_ptr, pos_ptr, vel_ptr, att_ptr, \
            (complex64_t*)data_ptr, sweep_samples, nsweeps, \
            fc, fs, gamma, x0, dx, y0, dy, Nx, Ny, d0, \
            g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, normalization)

    if      (!hv && !rv) { LAUNCH(false, false); }
    else if (!hv &&  rv) { LAUNCH(false, true);  }
    else if ( hv && !rv) { LAUNCH(true,  false); }
    else                 { LAUNCH(true,  true);  }

#undef LAUNCH

    return data;
}


at::Tensor backprojection_polar_2d_cuda(
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
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
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CUDA);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

    // Precompute phase coefficients with PI multiplied in
    const float phase_coef = (ref_phase - (data_fmod / kPI) * delta_r) * kPI;
    const float phase_offset = -(data_fmod / kPI) * delta_r * d0 * kPI;
    const float dealias_coef = -ref_phase * kPI;
    const float dealias_fmod = alias_fmod;  // already divided by kPI in caller

	dim3 thread_per_block = {256, 1};
    constexpr int pixels_per_thread = 4;
    int r_groups = (Nr + pixels_per_thread - 1) / pixels_per_thread;
    int total_work_items = r_groups * Ntheta;
    unsigned int block_x = (total_work_items + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use template specialization to eliminate antenna pattern, normalize and DEM branches
    #define LAUNCH_KERNEL(T, has_antenna, do_normalize, use_dem) \
        backprojection_polar_2d_kernel<T, has_antenna, do_normalize, InterpMethod::LINEAR, use_dem> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, \
                      dem_ptr, dem_r_scale, dem_theta_scale, dem_nr, dem_ntheta)
    #define LAUNCH_KERNEL_DEM(T, has_antenna, do_normalize) \
        do { \
            if (has_dem) { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, true); \
            } else { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, false); \
            } \
        } while (0)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, false);
        } else {
            LAUNCH_KERNEL_DEM(complex64_t, false, true);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(half2, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(half2, true, false);
        } else {
            LAUNCH_KERNEL_DEM(half2, false, true);
        }
    }

    #undef LAUNCH_KERNEL_DEM
    #undef LAUNCH_KERNEL

	return img;
}

std::vector<at::Tensor> backprojection_polar_2d_grad_cuda(
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_CHECK(!dem.defined(), "backprojection_polar_2d gradient with dem is not supported");
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();

    //TODO: Handle antenna pattern

    const bool have_pos_grad = pos.requires_grad();
    const bool have_data_grad = data.requires_grad();

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

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (have_pos_grad) {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<complex64_t, true, true>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (complex64_t*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                backprojection_polar_2d_grad_kernel<complex64_t, true, false>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (complex64_t*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );

            }
        } else {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<complex64_t, false, true>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (complex64_t*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                // Nothing to do
            }
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (have_pos_grad) {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<half2, true, true>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (half2*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                backprojection_polar_2d_grad_kernel<half2, true, false>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (half2*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );

            }
        } else {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<half2, false, true>
                      <<<block_count, thread_per_block, 0, stream>>>(
                              (half2*)data_ptr,
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
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                // Nothing to do
            }
        }
    }
    std::vector<at::Tensor> ret;
    ret.push_back(data_grad);
    ret.push_back(pos_grad);
	return ret;
}

at::Tensor backprojection_polar_2d_lanczos_cuda(
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
          int64_t order,
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
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
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CUDA);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

    // Precompute phase coefficients with PI multiplied in
    const float phase_coef = (ref_phase - (data_fmod / kPI) * delta_r) * kPI;
    const float phase_offset = -(data_fmod / kPI) * delta_r * d0 * kPI;
    const float dealias_coef = -ref_phase * kPI;
    const float dealias_fmod = alias_fmod;  // already divided by kPI in caller

	dim3 thread_per_block = {256, 1};
    constexpr int pixels_per_thread = 4;
    int r_groups = (Nr + pixels_per_thread - 1) / pixels_per_thread;
    int total_work_items = r_groups * Ntheta;
    unsigned int block_x = (total_work_items + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use template specialization to eliminate antenna pattern, normalize and DEM branches
    #define LAUNCH_KERNEL(T, has_antenna, do_normalize, use_dem) \
        backprojection_polar_2d_kernel<T, has_antenna, do_normalize, InterpMethod::LANCZOS, use_dem> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, \
                      dem_ptr, dem_r_scale, dem_theta_scale, dem_nr, dem_ntheta, \
                      order)
    #define LAUNCH_KERNEL_DEM(T, has_antenna, do_normalize) \
        do { \
            if (has_dem) { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, true); \
            } else { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, false); \
            } \
        } while (0)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, false);
        } else {
            LAUNCH_KERNEL_DEM(complex64_t, false, true);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(half2, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(half2, true, false);
        } else {
            LAUNCH_KERNEL_DEM(half2, false, true);
        }
    }

    #undef LAUNCH_KERNEL_DEM
    #undef LAUNCH_KERNEL

	return img;
}

at::Tensor backprojection_polar_2d_knab_cuda(
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
          int64_t order,
          double oversample,
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
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
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CUDA);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;
    const float v = 1.0f - 1.0f / oversample;

    // Precompute phase coefficients with PI multiplied in
    const float phase_coef = (ref_phase - (data_fmod / kPI) * delta_r) * kPI;
    const float phase_offset = -(data_fmod / kPI) * delta_r * d0 * kPI;
    const float dealias_coef = -ref_phase * kPI;
    const float dealias_fmod = alias_fmod;  // already divided by kPI in caller

	dim3 thread_per_block = {256, 1};
    constexpr int pixels_per_thread = 4;
    int r_groups = (Nr + pixels_per_thread - 1) / pixels_per_thread;
    int total_work_items = r_groups * Ntheta;
    unsigned int block_x = (total_work_items + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use template specialization to eliminate antenna pattern, normalize and DEM branches
    #define LAUNCH_KERNEL(T, has_antenna, do_normalize, use_dem) \
        backprojection_polar_2d_kernel<T, has_antenna, do_normalize, InterpMethod::KNAB, use_dem> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, \
                      dem_ptr, dem_r_scale, dem_theta_scale, dem_nr, dem_ntheta, \
                      order, v)
    #define LAUNCH_KERNEL_DEM(T, has_antenna, do_normalize) \
        do { \
            if (has_dem) { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, true); \
            } else { \
                LAUNCH_KERNEL(T, has_antenna, do_normalize, false); \
            } \
        } while (0)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(complex64_t, true, false);
        } else {
            LAUNCH_KERNEL_DEM(complex64_t, false, true);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern && normalize) {
            LAUNCH_KERNEL_DEM(half2, true, true);
        } else if (antenna_pattern && !normalize) {
            LAUNCH_KERNEL_DEM(half2, true, false);
        } else {
            LAUNCH_KERNEL_DEM(half2, false, true);
        }
    }

    #undef LAUNCH_KERNEL_DEM
    #undef LAUNCH_KERNEL

	return img;
}

at::Tensor gpga_backprojection_2d_cuda(
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(target_pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

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
    c10::complex<float>* data_out_ptr = data_out.data_ptr<c10::complex<float>>();

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

	dim3 thread_per_block = {256};
	// Up-rounding division.
    int blocks = Ntarget * nsweeps;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        gpga_backprojection_2d_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      target_pos_ptr,
                      (complex64_t*)data_ptr,
                      pos_ptr,
                      (complex64_t*)data_out_ptr,
                      sweep_samples,
                      nsweeps,
                      ref_phase,
                      delta_r,
                      Ntarget,
                      d0,
                      data_fmod/kPI);
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        gpga_backprojection_2d_kernel<half2>
              <<<block_count, thread_per_block, 0, stream>>>(
                      target_pos_ptr,
                      (half2*)data_ptr,
                      pos_ptr,
                      (complex64_t*)data_out_ptr,
                      sweep_samples,
                      nsweeps,
                      ref_phase,
                      delta_r,
                      Ntarget,
                      d0,
                      data_fmod/kPI);
    }
	return data_out;
}

at::Tensor blocksvd_alpha_cuda(
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
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(blocks.device().type() == at::DeviceType::CUDA);

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
    const c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const int32_t* blocks_ptr = blocks_contig.data_ptr<int32_t>();
    c10::complex<float>* alpha_ptr = alpha.data_ptr<c10::complex<float>>();

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

	dim3 thread_per_block = {256};
	// Up-rounding division.
    int threads = nblocks * nsweeps;
	unsigned int block_x = (threads + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    blocksvd_alpha_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  (const complex64_t*)img_ptr,
                  (const complex64_t*)data_ptr,
                  pos_ptr,
                  blocks_ptr,
                  (complex64_t*)alpha_ptr,
                  sweep_samples,
                  nsweeps,
                  nblocks,
                  Ntheta,
                  ref_phase,
                  delta_r,
                  r0,
                  dr,
                  theta0,
                  dtheta,
                  d0,
                  data_fmod/kPI);
	return alpha;
}

at::Tensor gpga_backprojection_2d_lanczos_cuda(
          const at::Tensor &target_pos,
          const at::Tensor &data,
          const at::Tensor &pos,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          int64_t Ntarget,
          double d0,
          int64_t order,
          double data_fmod) {
	TORCH_CHECK(target_pos.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(target_pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

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
    c10::complex<float>* data_out_ptr = data_out.data_ptr<c10::complex<float>>();

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

	dim3 thread_per_block = {256};
	// Up-rounding division.
    int blocks = Ntarget * nsweeps;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        gpga_backprojection_2d_lanczos_kernel<complex64_t>
              <<<block_count, thread_per_block, 0, stream>>>(
                      target_pos_ptr,
                      (complex64_t*)data_ptr,
                      pos_ptr,
                      (complex64_t*)data_out_ptr,
                      sweep_samples,
                      nsweeps,
                      ref_phase,
                      delta_r,
                      Ntarget,
                      d0,
                      order,
                      data_fmod/kPI);
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        gpga_backprojection_2d_lanczos_kernel<half2>
              <<<block_count, thread_per_block, 0, stream>>>(
                      target_pos_ptr,
                      (half2*)data_ptr,
                      pos_ptr,
                      (complex64_t*)data_out_ptr,
                      sweep_samples,
                      nsweeps,
                      ref_phase,
                      delta_r,
                      Ntarget,
                      d0,
                      order,
                      data_fmod/kPI);
    }
	return data_out;
}

at::Tensor backprojection_polar_2d_tx_power_cuda(
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
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(g.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

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

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    backprojection_polar_2d_tx_power_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
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
                  0.0f);
	return img;
}

at::Tensor backprojection_polar_2d_tx_power_slant_cuda(
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
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

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

	dim3 thread_per_block = {256, 1};
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    backprojection_polar_2d_tx_power_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
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
                  static_cast<float>(altitude));
	return img;
}

at::Tensor backprojection_cart_2d_tx_power_cuda(
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
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

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

	dim3 thread_per_block = {256, 1};
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    backprojection_cart_2d_tx_power_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
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
                  azimuth_resolution);
	return img;
}

at::Tensor backprojection_polar_2d_tx_power_accum_cuda(
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
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

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

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, 1};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    backprojection_polar_2d_tx_power_accum_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
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
                  static_cast<int>(theta_psi));
	return img;
}

at::Tensor backprojection_cart_2d_tx_power_accum_cuda(
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
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

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

	dim3 thread_per_block = {256, 1};
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, 1};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    backprojection_cart_2d_tx_power_accum_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
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
                  static_cast<float>(h_ref));
	return img;
}

at::Tensor cart_tx_power_merge2_cuda(
          const at::Tensor &acc0,
          const at::Tensor &acc1,
          double x0_0, double dx_0, double y0_0, double dy_0, int64_t Nx_0, int64_t Ny_0,
          double x0_1, double dx_1, double y0_1, double dy_1, int64_t Nx_1, int64_t Ny_1,
          double x1, double dx1, double y1, double dy1, int64_t Nx1, int64_t Ny1) {
    TORCH_CHECK(acc0.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(acc0.device().type() == at::DeviceType::CUDA);

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

    dim3 thread_per_block = {256, 1};
    unsigned int block_x = (Nx1 * Ny1 + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, 1};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cart_tx_power_merge2_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  acc0_ptr, acc1_ptr, out_ptr,
                  x0_0, dx_0, y0_0, dy_0, Nx_0, Ny_0,
                  x0_1, dx_1, y0_1, dy_1, Nx_1, Ny_1,
                  x1, dx1, y1, dy1, Nx1, Ny1);
    return out;
}

at::Tensor backprojection_cart_2d_cuda(
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
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor img = torch::zeros({nbatch, Nx, Ny}, data_contig.options());
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

	const float delta_r = 1.0f / r_res;
    const float ref_phase = 4.0f * fc / kC0;

	// Divide by 2 to get angle from the center.
	const float beamwidth_f = beamwidth / 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (Nx * Ny + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	backprojection_cart_2d_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  (complex64_t*)data_ptr,
                  pos_ptr,
                  (complex64_t*)img_ptr,
                  sweep_samples,
                  nsweeps,
                  ref_phase,
                  delta_r,
                  x0, dx,
                  y0, dy,
                  Nx, Ny,
                  beamwidth_f, d0,
                  data_fmod/kPI);
	return img;
}

std::vector<at::Tensor> backprojection_cart_2d_grad_cuda(
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
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
	const c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();

    const bool have_pos_grad = pos.requires_grad();
    const bool have_data_grad = data.requires_grad();

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

	// Divide by 2 to get angle from the center.
	const float beamwidth_f = beamwidth / 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (Nx * Ny + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (have_pos_grad) {
        if (have_data_grad) {
            backprojection_cart_2d_grad_kernel<true, true>
                  <<<block_count, thread_per_block, 0, stream>>>(
                          (complex64_t*)data_ptr,
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
                          (complex64_t*)grad_ptr,
                          pos_grad_ptr,
                          (complex64_t*)data_grad_ptr
                          );
        } else {
            backprojection_cart_2d_grad_kernel<true, false>
                  <<<block_count, thread_per_block, 0, stream>>>(
                          (complex64_t*)data_ptr,
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
                          (complex64_t*)grad_ptr,
                          pos_grad_ptr,
                          (complex64_t*)data_grad_ptr
                          );
        }
    } else {
        if (have_data_grad) {
            backprojection_cart_2d_grad_kernel<false, true>
                  <<<block_count, thread_per_block, 0, stream>>>(
                          (complex64_t*)data_ptr,
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
                          (complex64_t*)grad_ptr,
                          pos_grad_ptr,
                          (complex64_t*)data_grad_ptr
                          );
        } else {
            // Nothing to do
        }
    }
    std::vector<at::Tensor> ret;
    ret.push_back(data_grad);
    ret.push_back(pos_grad);
	return ret;
}


// CUDA kernel to compute subaperture illumination weight maps (W1, W2)
__global__ void compute_illumination_kernel(
          const float* __restrict__ pos,  // [nsweeps, 3]
          const float* __restrict__ att,  // [nsweeps, 3] or nullptr
          const float* __restrict__ g,    // [g_nel, g_naz] antenna pattern
          float* __restrict__ w1_out,     // [nr, ntheta] output W1 map
          float* __restrict__ w2_out,     // [nr, ntheta] output W2 map
          int nsweeps,
          // Grid parameters
          float r0, float dr, float theta0, float dtheta, int nr, int ntheta,
          // Antenna pattern parameters
          float g_el0, float g_del, float g_az0, float g_daz, int g_nel, int g_naz,
          // Decimation factor (1 = no decimation)
          int decimation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Output dimensions are decimated
    const int out_ntheta = (ntheta + decimation - 1) / decimation;
    const int out_nr = (nr + decimation - 1) / decimation;

    if (idx >= out_nr * out_ntheta) return;

    const int out_idtheta = idx % out_ntheta;
    const int out_idr = idx / out_ntheta;

    // Map to full grid coordinates (sample center of decimated cell)
    const int full_idr = out_idr * decimation;
    const int full_idtheta = out_idtheta * decimation;

    // Compute pixel coordinates. Clamped cosine + polar-form distance:
    // guard band grids extend past |theta| = 1 (see the backprojection
    // kernel note).
    const float r = r0 + dr * full_idr;
    const float t = theta0 + dtheta * full_idtheta;  // t is sin(theta)
    const float cost = sqrtf(fmaxf(0.0f, 1.0f - t*t));
    const float x = r * cost;
    const float y = r * t;

    float w1 = 0.0f;
    float w2 = 0.0f;

    for (int i = 0; i < nsweeps; i++) {
        const float pos_x = pos[i * 3 + 0];
        const float pos_y = pos[i * 3 + 1];
        const float pos_z = pos[i * 3 + 2];

        const float px = x - pos_x;
        const float py = y - pos_y;
        const float d_sq = r*r + pos_x*pos_x + pos_y*pos_y + pos_z*pos_z
                         - 2.0f * (x * pos_x + y * pos_y);
        const float d = sqrtf(fmaxf(d_sq, 0.0f));

        const float look_angle = asinf(fmaxf(-1.0f, fminf(1.0f, -pos_z / d)));

        float att_el = 0.0f;
        float att_az = 0.0f;
        if (att != nullptr) {
            att_el = att[i * 3 + 0];
            att_az = att[i * 3 + 2];
        }

        // Antenna-relative angles
        const float el = look_angle - att_el;
        const float az = atan2f(py, px) - att_az;

        // Antenna pattern interpolation
        const float el_idx = (el - g_el0) / g_del;
        const float az_idx = (az - g_az0) / g_daz;

        if (el_idx >= 0 && el_idx < g_nel - 1 && az_idx >= 0 && az_idx < g_naz - 1) {
            // Bilinear interpolation
            const int el_int = (int)el_idx;
            const int az_int = (int)az_idx;
            const float el_frac = el_idx - el_int;
            const float az_frac = az_idx - az_int;

            const float gain = interp2d<float>(g, g_nel, g_naz, el_int, el_frac, az_int, az_frac);

            w1 += gain;
            w2 += gain * gain;
        }
    }

    w1_out[idx] = w1;
    w2_out[idx] = w2;
}


std::vector<at::Tensor> compute_illumination_cuda(
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
    TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);

    const int64_t nsweeps = pos.size(0);
    const int64_t g_nel = g.size(0);
    const int64_t g_naz = g.size(1);

    // Check att tensor (can be undefined)
    const float* att_ptr = nullptr;
    at::Tensor att_contig;
    if (att.defined() && att.numel() > 0) {
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
        att_contig = att.contiguous();
        att_ptr = att_contig.data_ptr<float>();
    }

    at::Tensor pos_contig = pos.contiguous();
    at::Tensor g_contig = g.contiguous();

    // Compute decimated output dimensions
    const int64_t dec = decimation > 0 ? decimation : 1;
    const int64_t out_nr = (nr + dec - 1) / dec;
    const int64_t out_ntheta = (ntheta + dec - 1) / dec;

    // Allocate output tensors at decimated resolution
    at::Tensor w1_out = torch::empty({out_nr, out_ntheta},
                                     torch::TensorOptions().dtype(at::kFloat).device(pos.device()));
    at::Tensor w2_out = torch::empty({out_nr, out_ntheta},
                                     torch::TensorOptions().dtype(at::kFloat).device(pos.device()));

    dim3 thread_per_block = {256, 1};
    int blocks = out_nr * out_ntheta;
    unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, 1, 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    compute_illumination_kernel<<<block_count, thread_per_block, 0, stream>>>(
        pos_contig.data_ptr<float>(),
        att_ptr,
        g_contig.data_ptr<float>(),
        w1_out.data_ptr<float>(),
        w2_out.data_ptr<float>(),
        nsweeps,
        r0, dr, theta0, dtheta, nr, ntheta,
        g_el0, g_del, g_az0, g_daz, g_nel, g_naz,
        dec
    );

    std::vector<at::Tensor> ret;
    ret.push_back(w1_out);
    ret.push_back(w2_out);
    return ret;
}


// Registers CUDA implementations
// Multiply a polar SAR image by the range-dealias carrier. Same carrier as
// the dealias option of backprojection_polar_2d: distance from origin to the
// pixel at the DEM height (z=0 without a DEM), in the polar form that matches
// the backprojection kernels also on guard band pixels |theta| > 1. fc < 0
// applies the conjugate carrier (re-alias). Mirrors
// polar_range_dealias_row_cpu in cpu/backproj.cpp. Single elementwise pass,
// per-pixel bilinear DEM lookup: no full-size temporaries beyond the output.
template<bool HasDem>
__global__ void polar_range_dealias_kernel(
          const complex64_t* __restrict__ img,
          complex64_t* __restrict__ out,
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
          const float* __restrict__ dem,
          float dem_r_scale,
          float dem_theta_scale,
          int dem_nr,
          int dem_ntheta,
          float alias_fmod) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= Nr * Ntheta) {
        return;
    }
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    const float x = r * sqrtf(fmaxf(0.0f, 1.0f - theta*theta));
    const float y = r * theta;

    float z = 0.0f;
    if constexpr (HasDem) {
        // Same pixel-index-ratio bilinear convention as the backprojection
        // kernels.
        const float fr = idr * dem_r_scale;
        int ir0 = (int)fr;
        ir0 = min(ir0, dem_nr - 1);
        const int ir1 = min(ir0 + 1, dem_nr - 1);
        const float wr = fr - ir0;
        const float ft = idtheta * dem_theta_scale;
        int it0 = (int)ft;
        it0 = min(it0, dem_ntheta - 1);
        const int it1 = min(it0 + 1, dem_ntheta - 1);
        const float wt = ft - it0;
        const float* dem_row0 = dem + (size_t)ir0 * dem_ntheta;
        const float* dem_row1 = dem + (size_t)ir1 * dem_ntheta;
        const float za = dem_row0[it0] + wt * (dem_row0[it1] - dem_row0[it0]);
        const float zb = dem_row1[it0] + wt * (dem_row1[it1] - dem_row1[it0]);
        z = za + wr * (zb - za);
    }

    // Polar-form distance r^2 - 2*(x*ox + y*oy) + |o_xy|^2 + (oz-z)^2:
    // equals the Cartesian distance to the origin for |theta| <= 1 and
    // continues the backprojection kernels' clamped-cosine convention past
    // |theta| = 1.
    const float zz = oz - z;
    const float d2 = r*r - 2.0f * (x * ox + y * oy) + ox*ox + oy*oy + zz*zz;
    const float d = sqrtf(fmaxf(d2, 0.0f));
    float ref_sin, ref_cos;
    sincospif(-ref_phase * d + alias_fmod * idr, &ref_sin, &ref_cos);
    const complex64_t ref = {ref_cos, ref_sin};
    out[(size_t)idbatch * Nr * Ntheta + idx] =
        img[(size_t)idbatch * Nr * Ntheta + idx] * ref;
}

at::Tensor polar_range_dealias_cuda(
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
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);

    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty_like(img_contig);
    const complex64_t* img_ptr = (const complex64_t*)img_contig.data_ptr<c10::complex<float>>();
    complex64_t* out_ptr = (complex64_t*)out.data_ptr<c10::complex<float>>();

    const bool has_dem = dem.defined();
    at::Tensor dem_contig;
    const float* dem_ptr = nullptr;
    float dem_r_scale = 0.0f, dem_theta_scale = 0.0f;
    int dem_nr = 0, dem_ntheta = 0;
    if (has_dem) {
        TORCH_CHECK(dem.dtype() == at::kFloat, "polar_range_dealias: dem must be float32");
        TORCH_CHECK(dem.dim() == 2, "polar_range_dealias: dem must be 2D");
        TORCH_INTERNAL_ASSERT(dem.device().type() == at::DeviceType::CUDA);
        dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
        dem_nr = dem_contig.size(0);
        dem_ntheta = dem_contig.size(1);
        dem_r_scale = (float)dem_nr / Nr;
        dem_theta_scale = (float)dem_ntheta / Ntheta;
    }

    const float ref_phase = 4.0f * fc / kC0;

    dim3 thread_per_block = {256, 1};
    unsigned int block_x = (Nr * Ntheta + thread_per_block.x - 1) / thread_per_block.x;
    dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (has_dem) {
        polar_range_dealias_kernel<true><<<block_count, thread_per_block, 0, stream>>>(
            img_ptr, out_ptr, ref_phase, r0, dr, theta0, dtheta, Nr, Ntheta,
            ox, oy, oz, dem_ptr, dem_r_scale, dem_theta_scale, dem_nr,
            dem_ntheta, alias_fmod / kPI);
    } else {
        polar_range_dealias_kernel<false><<<block_count, thread_per_block, 0, stream>>>(
            img_ptr, out_ptr, ref_phase, r0, dr, theta0, dtheta, Nr, Ntheta,
            ox, oy, oz, dem_ptr, dem_r_scale, dem_theta_scale, dem_nr,
            dem_ntheta, alias_fmod / kPI);
    }
    return out;
}

TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("polar_range_dealias", &polar_range_dealias_cuda);
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cuda);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cuda);
  m.impl("backprojection_polar_2d_lanczos", &backprojection_polar_2d_lanczos_cuda);
  m.impl("backprojection_polar_2d_knab", &backprojection_polar_2d_knab_cuda);
  m.impl("backprojection_cart_2d", &backprojection_cart_2d_cuda);
  m.impl("backprojection_cart_2d_grad", &backprojection_cart_2d_grad_cuda);
  m.impl("gpga_backprojection_2d", &gpga_backprojection_2d_cuda);
  m.impl("gpga_backprojection_2d_lanczos", &gpga_backprojection_2d_lanczos_cuda);
  m.impl("blocksvd_alpha", &blocksvd_alpha_cuda);
  m.impl("backprojection_polar_2d_tx_power", &backprojection_polar_2d_tx_power_cuda);
  m.impl("backprojection_polar_2d_tx_power_slant", &backprojection_polar_2d_tx_power_slant_cuda);
  m.impl("backprojection_polar_2d_tx_power_accum", &backprojection_polar_2d_tx_power_accum_cuda);
  m.impl("backprojection_cart_2d_tx_power", &backprojection_cart_2d_tx_power_cuda);
  m.impl("backprojection_cart_2d_tx_power_accum", &backprojection_cart_2d_tx_power_accum_cuda);
  m.impl("cart_tx_power_merge2", &cart_tx_power_merge2_cuda);
  m.impl("projection_cart_2d", &projection_cart_2d_cuda);
  m.impl("projection_cart_2d_nufft", &projection_cart_2d_nufft_cuda);
  m.impl("compute_illumination", &compute_illumination_cuda);
}

}
