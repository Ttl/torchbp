#include "util.h"

namespace torchbp {

// Interpolation methods for backprojection kernels
enum class InterpMethod {
    LINEAR,
    LANCZOS,
    KNAB
};

template<typename T, bool HasAntennaPattern, InterpMethod Method = InterpMethod::LINEAR>
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
          bool dealias,
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
          int interp_order = 2,
          float knab_v = 0.0f) {

    // Process multiple pixels per thread to amortize pos loads
    const int PIXELS_PER_THREAD = 4;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta;
    const int idr_base = (idx / Ntheta) * PIXELS_PER_THREAD;
    const int idbatch = blockIdx.y;

    // Bounds check
    if (idr_base >= Nr || idtheta >= Ntheta) return;

    // Precompute theta sin/cos once (shared by all pixels in this thread)
    const float sin_theta = theta0 + idtheta * dtheta;  // theta IS sin(angle) in this coord system
    const float cos_theta = sqrtf(1.0f - sin_theta * sin_theta);

    // Coordinate storage for the pixels
    float r[PIXELS_PER_THREAD];
    float x[PIXELS_PER_THREAD];
    float y[PIXELS_PER_THREAD];
    float pixel_re[PIXELS_PER_THREAD] = {0};
    float pixel_im[PIXELS_PER_THREAD] = {0};
    float w_sum[PIXELS_PER_THREAD] = {0};

    #pragma unroll
    for(int k=0; k<PIXELS_PER_THREAD; ++k) {
        if(idr_base + k < Nr) {
            r[k] = r0 + (idr_base + k) * dr;
            x[k] = r[k] * cos_theta;
            y[k] = r[k] * sin_theta;
        }
    }

    const int pos_batch_offset = idbatch * nsweeps * 3;
    const int data_batch_stride = idbatch * sweep_samples * nsweeps;
    const int max_id0 = sweep_samples - 2;

    // Fused phase coefficient: phase = phase_coef * (d + d0) + phase_offset2
    // where phase_offset2 = phase_offset - phase_coef * d0
    // This way we compute d_eff = d + d0 once and use it for both sx and phase
    const float phase_offset2 = phase_offset - phase_coef * d0;

    // Direct L1 cache path - no shared memory overhead
    for (int i = 0; i < nsweeps; i++) {
        // Load pos directly via __ldg (L1 cached, broadcast across warp)
        const int pos_idx = pos_batch_offset + i * 3;
        float pos_x = __ldg(&pos[pos_idx + 0]);
        float pos_y = __ldg(&pos[pos_idx + 1]);
        float pos_z = __ldg(&pos[pos_idx + 2]);

        // Precompute pos_z² outside pixel loop
        float pz2 = pos_z * pos_z;
        int sweep_offset = data_batch_stride + i * sweep_samples;

        #pragma unroll
        for(int k=0; k<PIXELS_PER_THREAD; ++k) {
            if(idr_base + k >= Nr) continue;

            float px = x[k] - pos_x;
            float py = y[k] - pos_y;
            float d_sq = fmaf(px, px, fmaf(py, py, pz2));
            float d = sqrtf(d_sq);

            // Compute d_eff once, use for both sx and phase
            float d_eff = d + d0;
            float sx = delta_r * d_eff;

            // Bounds check and interpolation
            float s_re, s_im;
            bool valid_sample;

            if constexpr (Method == InterpMethod::LINEAR) {
                int id0 = (int)sx;
                valid_sample = (id0 >= 0 && id0 <= max_id0);

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
                    const float look_angle = asinf(fmaxf(-pos_z / d, -1.0f));
                    const float el_deg = look_angle - __ldg(&att[pos_idx + 0]);
                    const float az_deg = atan2f(py, px) - __ldg(&att[pos_idx + 2]);

                    const float el_idx = (el_deg - g_el0) / g_del;
                    const float az_idx = (az_deg - g_az0) / g_daz;

                    const int el_int = (int)el_idx;
                    const int az_int = (int)az_idx;

                    if (el_int >= 0 && el_int + 1 < g_nel && az_int >= 0 && az_int + 1 < g_naz) {
                        const float el_frac = el_idx - el_int;
                        const float az_frac = az_idx - az_int;
                        const float w = interp2d<float>(g, g_naz, g_nel, az_int, az_frac, el_int, el_frac);

                        float ws_re = w * s_re;
                        float ws_im = w * s_im;
                        pixel_re[k] = fmaf(ws_re, ref_cos, fmaf(-ws_im, ref_sin, pixel_re[k]));
                        pixel_im[k] = fmaf(ws_re, ref_sin, fmaf(ws_im, ref_cos, pixel_im[k]));
                        w_sum[k] += w * w;
                    }
                } else {
                    pixel_re[k] = fmaf(s_re, ref_cos, fmaf(-s_im, ref_sin, pixel_re[k]));
                    pixel_im[k] = fmaf(s_re, ref_sin, fmaf(s_im, ref_cos, pixel_im[k]));
                }
            }
        }
    }

    // Write output
    #pragma unroll
    for(int k=0; k<PIXELS_PER_THREAD; ++k) {
        if(idr_base + k < Nr) {
            complex64_t pixel = {pixel_re[k], pixel_im[k]};

            if constexpr (HasAntennaPattern) {
                if (w_sum[k] > 0.0f) {
                    pixel *= nsweeps / sqrtf(w_sum[k]);
                }
            }

            if (dealias) {
                const float dd = sqrtf(x[k]*x[k] + y[k]*y[k] + z0*z0);
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

    unsigned mask = __ballot_sync(FULL_MASK, idx < Nr * Ntheta);

    if (idx >= Nr * Ntheta) {
        return;
    }

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    const float x = r * sqrtf(1.0f - theta*theta);
    const float y = r * theta;

    complex64_t g = grad[idbatch * Nr * Ntheta + idr * Ntheta + idtheta];

    float arg_dealias = 0.0f;
    if (dealias) {
        const float d = sqrtf(x*x + y*y + z0*z0);
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

        // Calculate distance to the pixel.
        const float d = sqrtf(px * px + py * py + pz2);

        float sx = delta_r * (d + d0);

        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;
        complex64_t ds0 = 0.0f;
        complex64_t ds1 = 0.0f;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (id0 >= 0 && id1 < sweep_samples) {
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
                dx += __shfl_down_sync(mask, dx, offset);
                dy += __shfl_down_sync(mask, dy, offset);
                dz += __shfl_down_sync(mask, dz, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 0]), dx);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 1]), dy);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 2]), dz);
            }
        }

        if (have_data_grad) {
            if (id0 >= 0 && id1 < sweep_samples) {
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
    if (id0 < 0 || id1 >= sweep_samples) {
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
    if (id0 < 0 || id1 >= sweep_samples) {
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
          int normalization) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idr >= Nr || idtheta >= Ntheta) {
        return;
    }

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    const float x = r * sqrtf(1.0f - theta*theta);
    const float y = r * theta;

    // Angular size of resolution cell at nadir.
    const float min_look_angle = sqrtf(2.0f * dr/pos[idbatch * nsweeps * 3 + (nsweeps/2) * 3 + 2]);

    float pixel = 0.0f;

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

        // Avoid nans due to numerical precision by clamping to valid range.
        const float look_angle = asinf(fmaxf(-pos_z / d, -1.0f));
        const float el_deg = look_angle - att[idbatch * nsweeps * 3 + 3 * i + 0];
        const float az_deg = atan2f(py, px) - att[idbatch * nsweeps * 3 + 3 * i + 2];
        // TODO: consider platform pitch

        const float el_idx = (el_deg - g_el0) / g_del;
        const float az_idx = (az_deg - g_az0) / g_daz;

        const int el_int = el_idx;
        const int az_int = az_idx;
        const float el_frac = el_idx - el_int;
        const float az_frac = az_idx - az_int;

        if (el_int < 0 || el_int+1 >= g_nel) {
            continue;
        }
        if (az_int < 0 || az_int+1 >= g_naz) {
            continue;
        }
        float g_i = interp2d<float>(g, g_naz, g_nel, az_int, az_frac, el_int, el_frac);
        float sinl = 1.0f;

        if (normalization == 1) {
            // sigma_0
            sinl = sqrtf(fmaxf(min_look_angle, 1.0f - (pos_z * pos_z) / (d * d)));
        } else if (normalization == 2) {
            // gamma_0
            sinl = sqrtf(fmaxf(min_look_angle, 1.0f - (pos_z * pos_z) / (d * d))) * d / pos_z;
        } else if (normalization == 3) {
            // point
            // Scale as d^4 instead of d^3 for area target.
            sinl = d;
        }
        // beta_0 otherwise

        float w = wa[idbatch * nsweeps + i];
        pixel += g_i * g_i * w * w / (d*d*d * sinl);
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = sqrtf(pixel);
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
        if (id0 >= 0 && id1 < sweep_samples) {
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

    unsigned mask = __ballot_sync(FULL_MASK, idx < Nx && idy < Ny);

    if (idx >= Nx || idy >= Ny) {
        return;
    }

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;

    complex64_t g = grad[idbatch * Nx * Ny + idx * Ny + idy];

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
        if (id0 >= 0 && id1 < sweep_samples) {
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
                dx += __shfl_down_sync(mask, dx, offset);
                dy += __shfl_down_sync(mask, dy, offset);
                dz += __shfl_down_sync(mask, dz, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 0]), dx);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 1]), dy);
                atomicAdd(&(pos_grad[idbatch * nsweeps * 3 + i * 3 + 2]), dz);
            }
        }

        if (have_data_grad) {
            if (id0 >= 0 && id1 < sweep_samples) {
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


__global__ void projection_cart_2d_kernel(
          const complex64_t* img,
          const float* dem,
          const float* pos,
          const float* vel,
          const float* att,
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
          const float *g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          bool use_rvp,
          int normalization) {
    const int idt = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = idt % Ny;
    const int idx = idt / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned mask = __ballot_sync(FULL_MASK, idt < Nx * Ny);

    if (idx >= Nx || idy >= Ny) {
        return;
    }

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;
    const float z = (dem != nullptr) ? dem[idx * Ny + idy] : 0.0f;
    const complex64_t w_img = img[idbatch * Nx * Ny + idx * Ny + idy];

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz = (z - pos_z);

        float d = sqrtf(px * px + py * py + pz * pz);
        float tau = 2.0f * (d + d0) / kC0;

        complex64_t w = w_img / (d * d);

        if (g != nullptr) {
            const float look_angle = asinf(fmaxf(pz / d, -1.0f));
            const float el_deg = look_angle - att[idbatch * nsweeps * 3 + 3 * i + 0];
            const float az_deg = atan2f(py, px) - att[idbatch * nsweeps * 3 + 3 * i + 2];

            const float el_idx = (el_deg - g_el0) / g_del;
            const float az_idx = (az_deg - g_az0) / g_daz;

            const int el_int = el_idx;
            const int az_int = az_idx;
            const float el_frac = el_idx - el_int;
            const float az_frac = az_idx - az_int;

            if (el_int < 0 || el_int+1 >= g_nel || az_int < 0 || az_int+1 >= g_naz) {
                w = 0.0f;
            } else {
                w *= interp2d<float>(g, g_naz, g_nel, az_int, az_frac, el_int, el_frac);
            }
        }

        if (normalization == 1) {
            // gamma_0
            // sin of look_angle.
            // Square root because this is amplitude.
            w *= sqrtf(-pz / d);
        }
        // sigma_0 otherwise

        float vel_proj;
        if (vel != nullptr) {
            const float vx = vel[idbatch * nsweeps * 3 + i * 3 + 0];
            const float vy = vel[idbatch * nsweeps * 3 + i * 3 + 1];
            const float vz = vel[idbatch * nsweeps * 3 + i * 3 + 2];
            vel_proj = (px * vx + py * vy + pz * vz) / d;
            vel_proj /= fs;
        }

        for (int j = 0; j < sweep_samples; j++) {
            if (vel != nullptr) {
                tau = 2.0f * (d + d0 + vel_proj * (j - sweep_samples/2)) / kC0;
            }
            float phase0 = -2.0f * (fc * tau);
            if (use_rvp) {
                phase0 += gamma * tau * tau;
            }
            const float freq = -2.0f * gamma * tau / fs;
            const float phase = freq * j + phase0;

            float ref_sin, ref_cos;
            sincospif(phase, &ref_sin, &ref_cos);
            complex64_t p = {ref_cos, ref_sin};
            const complex64_t s = w * p;

            float s_re = s.real();
            float s_im = s.imag();

            __syncwarp();
            for (int offset = 16; offset > 0; offset /= 2) {
                s_re += __shfl_down_sync(mask, s_re, offset);
                s_im += __shfl_down_sync(mask, s_im, offset);
            }

            if (threadIdx.x % 32 == 0) {
                float2 *x0 = (float2*)&data[idbatch * sweep_samples * nsweeps + i * sweep_samples + j];
                atomicAdd(&x0->x, s_re);
                atomicAdd(&x0->y, s_im);
            }
        }
    }
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

    bool antenna_pattern = g.defined() || att.defined();
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
	const float* dem_ptr = nullptr;
    if (dem.defined()) {
        at::Tensor dem_contig = dem.contiguous();
        dem_ptr = dem_contig.data_ptr<float>();
    }
	const float* vel_ptr = nullptr;
    if (vel.defined()) {
        at::Tensor vel_contig = vel.contiguous();
        vel_ptr = vel_contig.data_ptr<float>();
    }
    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        at::Tensor att_contig = att.contiguous();
        at::Tensor g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
    }

    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();
    c10::complex<float>* data_ptr = data.data_ptr<c10::complex<float>>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    projection_cart_2d_kernel
          <<<block_count, thread_per_block, 0, stream>>>(
                  (complex64_t*)img_ptr,
                  dem_ptr,
                  pos_ptr,
                  vel_ptr,
                  att_ptr,
                  (complex64_t*)data_ptr,
                  sweep_samples,
                  nsweeps,
                  fc,
                  fs,
                  gamma,
                  x0, dx,
                  y0, dy,
                  Nx, Ny,
                  d0,
                  g_ptr,
                  g_az0,
                  g_el0,
                  g_daz,
                  g_del,
                  g_naz,
                  g_nel,
                  use_rvp,
                  normalization);

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
          double alias_fmod) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined() || att.defined();
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

    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        at::Tensor att_contig = att.contiguous();
        at::Tensor g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
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

    // Use template specialization to eliminate antenna pattern branch
    #define LAUNCH_KERNEL(T, has_antenna) \
        backprojection_polar_2d_kernel<T, has_antenna> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(complex64_t, true);
        } else {
            LAUNCH_KERNEL(complex64_t, false);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(half2, true);
        } else {
            LAUNCH_KERNEL(half2, false);
        }
    }

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
          double alias_fmod) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
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
          double alias_fmod) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined() || att.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
    }

	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* pos_ptr = pos_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        at::Tensor att_contig = att.contiguous();
        at::Tensor g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
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

    // Use template specialization to eliminate antenna pattern branch
    #define LAUNCH_KERNEL(T, has_antenna) \
        backprojection_polar_2d_kernel<T, has_antenna, InterpMethod::LANCZOS> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, \
                      order)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(complex64_t, true);
        } else {
            LAUNCH_KERNEL(complex64_t, false);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(half2, true);
        } else {
            LAUNCH_KERNEL(half2, false);
        }
    }

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
          double alias_fmod) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

    bool antenna_pattern = g.defined() || att.defined();
    if (antenna_pattern) {
        TORCH_CHECK(g.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(g.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(att.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
    }

	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor g_contig = g.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* pos_ptr = pos_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

    float* att_ptr = nullptr;
    float* g_ptr = nullptr;
    if (antenna_pattern) {
        at::Tensor att_contig = att.contiguous();
        at::Tensor g_contig = g.contiguous();
        att_ptr = att_contig.data_ptr<float>();
        g_ptr = g_contig.data_ptr<float>();
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

    // Use template specialization to eliminate antenna pattern branch
    #define LAUNCH_KERNEL(T, has_antenna) \
        backprojection_polar_2d_kernel<T, has_antenna, InterpMethod::KNAB> \
              <<<block_count, thread_per_block, 0, stream>>>( \
                      (T*)data_ptr, pos_ptr, att_ptr, (complex64_t*)img_ptr, \
                      sweep_samples, nsweeps, \
                      phase_coef, phase_offset, delta_r, \
                      r0, dr, theta0, dtheta, Nr, Ntheta, \
                      d0, dealias, z0, dealias_coef, dealias_fmod, \
                      g_ptr, g_az0, g_el0, g_daz, g_del, g_naz, g_nel, \
                      order, v)

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(complex64_t, true);
        } else {
            LAUNCH_KERNEL(complex64_t, false);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (antenna_pattern) {
            LAUNCH_KERNEL(half2, true);
        } else {
            LAUNCH_KERNEL(half2, false);
        }
    }

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
          int64_t normalization) {
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
                  normalization);
	return img;
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


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cuda);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cuda);
  m.impl("backprojection_polar_2d_lanczos", &backprojection_polar_2d_lanczos_cuda);
  m.impl("backprojection_polar_2d_knab", &backprojection_polar_2d_knab_cuda);
  m.impl("backprojection_cart_2d", &backprojection_cart_2d_cuda);
  m.impl("backprojection_cart_2d_grad", &backprojection_cart_2d_grad_cuda);
  m.impl("gpga_backprojection_2d", &gpga_backprojection_2d_cuda);
  m.impl("gpga_backprojection_2d_lanczos", &gpga_backprojection_2d_lanczos_cuda);
  m.impl("backprojection_polar_2d_tx_power", &backprojection_polar_2d_tx_power_cuda);
  m.impl("projection_cart_2d", &projection_cart_2d_cuda);
}

}
