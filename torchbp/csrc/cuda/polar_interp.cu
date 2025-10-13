#include "util.h"

namespace torchbp {

__global__ void polar_interp_kernel_linear(const complex64_t *img, complex64_t
        *out, const float *dorigin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float r1,
        float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1,
        float z1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const float d = r1 + dr1 * idr;
    float t = theta1 + dtheta1 * idtheta;
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const float dorig0 = dorigin[idbatch * 3 + 0];
    const float dorig1 = dorigin[idbatch * 3 + 1];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    // TODO: Add dorig2
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float tp = arg / sqrtf(1.0f + arg*arg);

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        complex64_t v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        const float z0 = z1 + dorigin[idbatch * 3 + 2];
        const float dz = sqrtf(z1*z1 + d*d);
        const float rpz = sqrtf(z0*z0 + rp*rp);
        sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

__global__ void polar_interp_kernel_linear_grad(const complex64_t *img, const
        float *dorigin, float rotation, float ref_phase, float r0, float dr,
        float theta0, float dtheta, int Nr, int Ntheta, float r1, float dr1,
        float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, const complex64_t
        *grad, complex64_t *img_grad, float *dorigin_grad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    const complex64_t I = {0.0f, 1.0f};

    const float d = r1 + dr1 * idr;
    float t = theta1 + dtheta1 * idtheta;
    if (t > 1.0f) {
        t = 1.0f;
    }
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }

    unsigned mask = __ballot_sync(FULL_MASK, idx < Nr1 * Ntheta1 && t >= -1.0f && t <= 1.0f);

    if (idx >= Nr1 * Ntheta1) {
        return;
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const float dorig0 = dorigin[idbatch * 3 + 0];
    const float dorig1 = dorigin[idbatch * 3 + 1];
    const float dorig2 = dorigin[idbatch * 3 + 2];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    // TODO: Add dorig2
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float cosarg = sqrtf(1.0f + arg*arg);
    const float tp = arg / cosarg;

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;

    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    complex64_t v = {0.0f, 0.0f};
    complex64_t ref = {0.0f, 0.0f};

    const float z0 = z1 + dorig2;
    const float rpz = sqrtf(z0*z0 + rp*rp);
    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        const float dz = sqrtf(z1*z1 + d*d);
        sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    if (dorigin_grad != nullptr) {
        const complex64_t dref_drpz = I * kPI * ref_phase * ref;
        const complex64_t dv_drp = interp2d_gradx<complex64_t>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                dti_int, dti_frac) / dr;
        const complex64_t dv_dt = interp2d_grady<complex64_t>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                dti_int, dti_frac) / dtheta;
        const float drp_dorig0 = (cost*d + dorig0) / rp;
        const float drp_dorig1 = (sint*d + dorig1) / rp;
        const float drpz_dorig0 = (cost*d + dorig0) / rpz;
        const float drpz_dorig1 = (sint*d + dorig1) / rpz;
        const float drpz_dorig2 = (dorig2 + z1) / rpz;
        const float dt_darg = -arg*arg/(cosarg*cosarg*cosarg) + 1.0f / cosarg;
        const float darg_dorig0 = -(d*sint + dorig1) / ((dorig0 + d*cost)*(dorig0 + d*cost));
        const float darg_dorig1 = 1.0f / (cost*d + dorig0);

        const complex64_t g = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta];
        const complex64_t dout_dorig0 = ref * (dv_drp * drp_dorig0 + dv_dt * dt_darg * darg_dorig0) + v * dref_drpz * drpz_dorig0;
        const complex64_t dout_dorig1 = ref * (dv_drp * drp_dorig1 + dv_dt * dt_darg * darg_dorig1) + v * dref_drpz * drpz_dorig1;
        const complex64_t dout_dorig2 = v * dref_drpz * drpz_dorig2;
        float g_dorig0 = cuda::std::real(g * cuda::std::conj(dout_dorig0));
        float g_dorig1 = cuda::std::real(g * cuda::std::conj(dout_dorig1));
        float g_dorig2 = cuda::std::real(g * cuda::std::conj(dout_dorig2));

        for (int offset = 16; offset > 0; offset /= 2) {
            g_dorig0 += __shfl_down_sync(mask, g_dorig0, offset);
            g_dorig1 += __shfl_down_sync(mask, g_dorig1, offset);
            g_dorig2 += __shfl_down_sync(mask, g_dorig2, offset);
        }

        if (threadIdx.x % 32 == 0) {
            atomicAdd(&(dorigin_grad[idbatch * 3 + 0]), g_dorig0);
            atomicAdd(&(dorigin_grad[idbatch * 3 + 1]), g_dorig1);
            atomicAdd(&(dorigin_grad[idbatch * 3 + 2]), g_dorig2);
        }
    }

    if (img_grad != nullptr) {
        if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
            complex64_t g = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] * cuda::std::conj(ref);

            complex64_t g11 = g * (1.0f-dri_frac)*(1.0f-dti_frac);
            complex64_t g12 = g * (1.0f-dri_frac)*dti_frac;
            complex64_t g21 = g * dri_frac*(1.0f-dti_frac);
            complex64_t g22 = g * dri_frac*dti_frac;
            float2 *x11 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
            float2 *x12 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
            float2 *x21 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
            float2 *x22 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
            atomicAdd(&x11->x, g11.real());
            atomicAdd(&x11->y, g11.imag());
            atomicAdd(&x12->x, g12.real());
            atomicAdd(&x12->y, g12.imag());
            atomicAdd(&x21->x, g21.real());
            atomicAdd(&x21->y, g21.imag());
            atomicAdd(&x22->x, g22.real());
            atomicAdd(&x22->y, g22.imag());
        }
    }
}

__global__ void polar_interp_kernel_bicubic(const complex64_t *img,
        const complex64_t *img_gx, const complex64_t *img_gy,
        const complex64_t *img_gxy, complex64_t *out, const float *dorigin,
        float rotation, float ref_phase, float r0, float dr, float theta0,
        float dtheta, int Nr, int Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const float d = r1 + dr1 * idr;
    float t = theta1 + dtheta1 * idtheta;
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const float dorig0 = dorigin[idbatch * 3 + 0];
    const float dorig1 = dorigin[idbatch * 3 + 1];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    // TODO: Add dorig2
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float tp = arg / sqrtf(1.0f + arg*arg);

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        complex64_t v = bicubic_interp2d<complex64_t>(
                &img[idbatch * Nr * Ntheta],
                &img_gx[idbatch * Nr * Ntheta],
                &img_gy[idbatch * Nr * Ntheta],
                &img_gxy[idbatch * Nr * Ntheta],
                Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        const float z0 = z1 + dorigin[idbatch * 3 + 2];
        const float dz = sqrtf(z1*z1 + d*d);
        const float rpz = sqrtf(z0*z0 + rp*rp);
        sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

__global__ void polar_interp_kernel_lanczos(const complex64_t *img,
        complex64_t *out, const float *dorigin,
        float rotation, float ref_phase, float r0, float dr, float theta0,
        float dtheta, int Nr, int Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const float d = r1 + dr1 * idr;
    float t = theta1 + dtheta1 * idtheta;
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const float dorig0 = dorigin[idbatch * 3 + 0];
    const float dorig1 = dorigin[idbatch * 3 + 1];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    // TODO: Add dorig2
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float tp = arg / sqrtf(1.0f + arg*arg);

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    if (dri >= 0 && dri < Nr-1 && dti >= 0 && dti < Ntheta-1) {
        complex64_t v = lanczos_interp_2d<complex64_t, complex64_t>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri, dti, order);
        float ref_sin, ref_cos;
        const float z0 = z1 + dorigin[idbatch * 3 + 2];
        const float dz = sqrtf(z1*z1 + d*d);
        const float rpz = sqrtf(z0*z0 + rp*rp);
        sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

template<typename T>
__global__ void polar_to_cart_kernel_linear(const T *img, T
        *out, const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    const float orig0 = origin[idbatch * 3 + 0];
    const float orig1 = origin[idbatch * 3 + 1];
    const float orig2 = origin[idbatch * 3 + 2];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1));
    const float dz = sqrtf(d*d + orig2*orig2);
    float t = (y - orig1) / d; // Sin of angle
    float tc = (x - orig0) / d; // Cos of angle
    float rs = sinf(rotation);
    float rc = cosf(rotation);
    float cosa = t*rs  + tc*rc;
    if (rotation != 0.0f) {
        t = rc * t - rs * tc;
    }
    const float dri = (d - r0) / dr;
    const float dti = (t - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        T v = interp2d<T>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            float ref_sin, ref_cos;
            sincospif(ref_phase * dz, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            out[idbatch * Nx * Ny + idx*Ny + idy] = v * ref;
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = v;
        }
    } else {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx*Ny + idy] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = 0.0f;
        }
    }
}

__global__ void polar_to_cart_kernel_linear_grad(const complex64_t *img,
        const float *origin, float rotation, float ref_phase,
        float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta,
        float x0, float dx, float y0, float dy, int Nx, int Ny,
        const complex64_t *grad, complex64_t *img_grad, float *origin_grad) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    const float orig0 = origin[idbatch * 3 + 0];
    const float orig1 = origin[idbatch * 3 + 1];
    const float orig2 = origin[idbatch * 3 + 2];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1));
    const float dz = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1) + orig2*orig2);
    float t = (y - orig1) / d; // Sin of angle
    float tc = (x - orig0) / d; // Cos of angle
    float rs = sinf(rotation);
    float rc = cosf(rotation);
    float cosa = t*rs  + tc*rc;
    if (rotation != 0.0f) {
        t = rc * t - rs * tc;
    }

    unsigned mask = __ballot_sync(FULL_MASK, cosa >= 0 && id1 < Nr * Ntheta && t >= theta0 && t <= theta0 + dtheta * Ntheta);

    if (id1 >= Nx * Ny) {
        return;
    }

    const float dri = (d - r0) / dr;
    const float dti = (t - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        complex64_t v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        sincospif(ref_phase * dz, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};

        if (origin_grad != nullptr) {
            const complex64_t I = {0.0f, 1.0f};

            const complex64_t dref_dz = I * kPI * ref_phase * ref;
            const complex64_t dv_dd = interp2d_gradx<complex64_t>(
                    &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                    dti_int, dti_frac) / dr;
            const complex64_t dv_dt = interp2d_grady<complex64_t>(
                    &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                    dti_int, dti_frac) / dtheta;
            const float dt1_dorig0 = rc * ((x - orig0) * (y - orig1) / (d*d*d));
            const float dt1_dorig1 = rc * ((y - orig1) * (y - orig1) / (d*d*d) - 1.0f / d);
            const float dt2_dorig0 = -rs * ((x - orig0) * (x - orig0) / (d*d*d) - 1.0f / d);
            const float dt2_dorig1 = -rs * ((x - orig0) * (y - orig1) / (d*d*d));
            const float dt_dorig0 = dt1_dorig0 + dt2_dorig0;
            const float dt_dorig1 = dt1_dorig1 + dt2_dorig1;
            const float dd_dorig0 = (orig0 - x) / d;
            const float dd_dorig1 = (orig1 - y) / d;
            const complex64_t dv_dorig0 = dv_dd * dd_dorig0 + dv_dt * dt_dorig0;
            const complex64_t dv_dorig1 = dv_dd * dd_dorig1 + dv_dt * dt_dorig1;
            const float dz_dorig0 = (orig0 - x) / dz;
            const float dz_dorig1 = (orig1 - y) / dz;
            const float dz_dorig2 = orig2 / dz;
            const complex64_t dref_dorig0 = dref_dz * dz_dorig0;
            const complex64_t dref_dorig1 = dref_dz * dz_dorig1;
            const complex64_t dref_dorig2 = dref_dz * dz_dorig2;
            const complex64_t dout_dorig0 = dv_dorig0 * ref + v * dref_dorig0;
            const complex64_t dout_dorig1 = dv_dorig1 * ref + v * dref_dorig1;
            const complex64_t dout_dorig2 = v * dref_dorig2;

            const complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy];
            float g_origin0 = cuda::std::real(g * cuda::std::conj(dout_dorig0));
            float g_origin1 = cuda::std::real(g * cuda::std::conj(dout_dorig1));
            float g_origin2 = cuda::std::real(g * cuda::std::conj(dout_dorig2));

            for (int offset = 16; offset > 0; offset /= 2) {
                g_origin0 += __shfl_down_sync(mask, g_origin0, offset);
                g_origin1 += __shfl_down_sync(mask, g_origin1, offset);
                g_origin2 += __shfl_down_sync(mask, g_origin2, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(origin_grad[idbatch * 3 + 0]), g_origin0);
                atomicAdd(&(origin_grad[idbatch * 3 + 1]), g_origin1);
                atomicAdd(&(origin_grad[idbatch * 3 + 2]), g_origin2);
            }
        }

        if (img_grad != nullptr) {
            if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
                complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy] * cuda::std::conj(ref);

                complex64_t g11 = g * (1.0f-dri_frac)*(1.0f-dti_frac);
                complex64_t g12 = g * (1.0f-dri_frac)*dti_frac;
                complex64_t g21 = g * dri_frac*(1.0f-dti_frac);
                complex64_t g22 = g * dri_frac*dti_frac;
                float2 *x11 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
                float2 *x12 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
                float2 *x21 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
                float2 *x22 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
                atomicAdd(&x11->x, g11.real());
                atomicAdd(&x11->y, g11.imag());
                atomicAdd(&x12->x, g12.real());
                atomicAdd(&x12->y, g12.imag());
                atomicAdd(&x21->x, g21.real());
                atomicAdd(&x21->y, g21.imag());
                atomicAdd(&x22->x, g22.real());
                atomicAdd(&x22->y, g22.imag());
            }
        }
    }
}

at::Tensor polar_to_cart_linear_cuda(
          const at::Tensor &img,
          const at::Tensor &origin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double x0,
          double y0,
          double dx,
          double dy,
          int64_t Nx,
          int64_t Ny) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CUDA);
	at::Tensor origin_contig = origin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* origin_ptr = origin_contig.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

	if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        polar_to_cart_kernel_linear<complex64_t>
              <<<block_count, thread_per_block>>>(
                      (const complex64_t*)img_ptr,
                      (complex64_t*)out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny
                      );
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        polar_to_cart_kernel_linear<float>
              <<<block_count, thread_per_block>>>(
                      img_ptr,
                      out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny
                      );
    }
	return out;
}

std::vector<at::Tensor> polar_to_cart_linear_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &img,
          const at::Tensor &origin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double x0,
          double y0,
          double dx,
          double dy,
          int64_t Nx,
          int64_t Ny) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor origin_contig = origin.contiguous();
	const float* origin_ptr = origin_contig.data_ptr<float>();
	at::Tensor img_contig = img.contiguous();
	at::Tensor grad_contig = grad.contiguous();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();
    at::Tensor img_grad;
    c10::complex<float>* img_grad_ptr = nullptr;
    if (img.requires_grad()) {
        img_grad = torch::zeros_like(img);
        img_grad_ptr = img_grad.data_ptr<c10::complex<float>>();
    } else {
        img_grad = torch::Tensor();
    }

    at::Tensor origin_grad;
	float* origin_grad_ptr = nullptr;
    if (origin.requires_grad()) {
        origin_grad = torch::zeros_like(origin);
        origin_grad_ptr = origin_grad.data_ptr<float>();
    } else {
        origin_grad = torch::Tensor();
    }

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_to_cart_kernel_linear_grad
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  origin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr,
                  theta0,
                  dtheta,
                  Nr,
                  Ntheta,
                  x0,
                  dx,
                  y0,
                  dy,
                  Nx,
                  Ny,
                  (const complex64_t*)grad_ptr,
                  (complex64_t*)img_grad_ptr,
                  origin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(origin_grad);
	return ret;
}

template<typename T>
__global__ void polar_to_cart_kernel_bicubic(const T *img,
        const T *img_gx, const T *img_gy, const T *img_gxy,
        T *out, const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    const float orig0 = origin[idbatch * 3 + 0];
    const float orig1 = origin[idbatch * 3 + 1];
    const float orig2 = origin[idbatch * 3 + 2];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1));
    float t = (y - orig1) / d; // Sin of angle
    float tc = (x - orig0) / d; // Cos of angle
    float rs = sinf(rotation);
    float rc = cosf(rotation);
    float cosa = t*rs  + tc*rc;
    if (rotation != 0.0f) {
        t = rc * t - rs * tc;
    }

    const float dri = (d - r0) / dr;
    const float dti = (t - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        T v = bicubic_interp2d<T>(
                &img[idbatch * Nr * Ntheta],
                &img_gx[idbatch * Nr * Ntheta],
                &img_gy[idbatch * Nr * Ntheta],
                &img_gxy[idbatch * Nr * Ntheta],
                Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            float ref_sin, ref_cos;
            sincospif(ref_phase * sqrtf(d*d + orig2*orig2), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            out[idbatch * Nx * Ny + idx*Ny + idy] = v * ref;
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = v;
        }
    } else {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx*Ny + idy] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = 0.0f;
        }
    }
}

__global__ void polar_to_cart_kernel_bicubic_grad(const complex64_t *img,
        const complex64_t *img_gx, const complex64_t *img_gy, const complex64_t *img_gxy,
        const float *origin, float rotation, float ref_phase,
        float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta,
        float x0, float dx, float y0, float dy, int Nx, int Ny,
        const complex64_t *grad, complex64_t *img_grad, complex64_t *img_gx_grad,
        complex64_t *img_gy_grad, complex64_t *img_gxy_grad, float *origin_grad) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    const float orig0 = origin[idbatch * 3 + 0];
    const float orig1 = origin[idbatch * 3 + 1];
    const float orig2 = origin[idbatch * 3 + 2];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1));
    const float dz = sqrtf(d*d + orig2*orig2);
    float t = (y - orig1) / d; // Sin of angle
    float tc = (x - orig0) / d; // Cos of angle
    float rs = sinf(rotation);
    float rc = cosf(rotation);
    float cosa = t*rs  + tc*rc;
    if (rotation != 0.0f) {
        t = rc * t - rs * tc;
    }

    unsigned mask = __ballot_sync(FULL_MASK, cosa >= 0 && id1 < Nr * Ntheta && t >= theta0 && t <= theta0 + dtheta * Ntheta);

    if (id1 >= Nx * Ny) {
        return;
    }

    const float dri = (d - r0) / dr;
    const float dti = (t - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;
    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        complex64_t v = bicubic_interp2d<complex64_t>(
                &img[idbatch * Nr * Ntheta],
                &img_gx[idbatch * Nr * Ntheta],
                &img_gy[idbatch * Nr * Ntheta],
                &img_gxy[idbatch * Nr * Ntheta],
                Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        sincospif(ref_phase * dz, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};

        if (origin_grad != nullptr) {
            const complex64_t I = {0.0f, 1.0f};

            const complex64_t dref_dz = I * kPI * ref_phase * ref;
            const complex64_t dv_dd = bicubic_interp2d_gradx<complex64_t>(
                    &img[idbatch * Nr * Ntheta],
                    &img_gx[idbatch * Nr * Ntheta],
                    &img_gy[idbatch * Nr * Ntheta],
                    &img_gxy[idbatch * Nr * Ntheta],
                    Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac) / dr;
            const complex64_t dv_dt = bicubic_interp2d_grady<complex64_t>(
                    &img[idbatch * Nr * Ntheta],
                    &img_gx[idbatch * Nr * Ntheta],
                    &img_gy[idbatch * Nr * Ntheta],
                    &img_gxy[idbatch * Nr * Ntheta],
                    Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac) / dtheta;

            const float dt1_dorig0 = rc * ((x - orig0) * (y - orig1) / (d*d*d));
            const float dt1_dorig1 = rc * ((y - orig1) * (y - orig1) / (d*d*d) - 1.0f / d);
            const float dt2_dorig0 = -rs * ((x - orig0) * (x - orig0) / (d*d*d) - 1.0f / d);
            const float dt2_dorig1 = -rs * ((x - orig0) * (y - orig1) / (d*d*d));
            const float dt_dorig0 = dt1_dorig0 + dt2_dorig0;
            const float dt_dorig1 = dt1_dorig1 + dt2_dorig1;
            const float dd_dorig0 = (orig0 - x) / d;
            const float dd_dorig1 = (orig1 - y) / d;
            const complex64_t dv_dorig0 = dv_dd * dd_dorig0 + dv_dt * dt_dorig0;
            const complex64_t dv_dorig1 = dv_dd * dd_dorig1 + dv_dt * dt_dorig1;
            const float dz_dorig0 = (orig0 - x) / dz;
            const float dz_dorig1 = (orig1 - y) / dz;
            const float dz_dorig2 = orig2 / dz;
            const complex64_t dref_dorig0 = dref_dz * dz_dorig0;
            const complex64_t dref_dorig1 = dref_dz * dz_dorig1;
            const complex64_t dref_dorig2 = dref_dz * dz_dorig2;
            const complex64_t dout_dorig0 = dv_dorig0 * ref + v * dref_dorig0;
            const complex64_t dout_dorig1 = dv_dorig1 * ref + v * dref_dorig1;
            const complex64_t dout_dorig2 = v * dref_dorig2;

            const complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy];
            float g_origin0 = cuda::std::real(g * cuda::std::conj(dout_dorig0));
            float g_origin1 = cuda::std::real(g * cuda::std::conj(dout_dorig1));
            float g_origin2 = cuda::std::real(g * cuda::std::conj(dout_dorig2));

            for (int offset = 16; offset > 0; offset /= 2) {
                g_origin0 += __shfl_down_sync(mask, g_origin0, offset);
                g_origin1 += __shfl_down_sync(mask, g_origin1, offset);
                g_origin2 += __shfl_down_sync(mask, g_origin2, offset);
            }

            if (threadIdx.x % 32 == 0) {
                atomicAdd(&(origin_grad[idbatch * 3 + 0]), g_origin0);
                atomicAdd(&(origin_grad[idbatch * 3 + 1]), g_origin1);
                atomicAdd(&(origin_grad[idbatch * 3 + 2]), g_origin2);
            }
        }

        if (img_grad != nullptr) {
            if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
                complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy] * cuda::std::conj(ref);

                complex64_t img_grad_tmp[4];
                complex64_t img_gx_grad_tmp[4];
                complex64_t img_gy_grad_tmp[4];
                complex64_t img_gxy_grad_tmp[4];
                bicubic_interp2d_grad<complex64_t>(
                        (complex64_t*)&img_grad_tmp,
                        (complex64_t*)&img_gx_grad_tmp,
                        (complex64_t*)&img_gy_grad_tmp,
                        (complex64_t*)&img_gxy_grad_tmp,
                        dri_frac, dti_frac, g);
                float2 *g11 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
                float2 *g12 = (float2*)&img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
                float2 *g21 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
                float2 *g22 = (float2*)&img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
                atomicAdd(&g11->x, img_grad_tmp[0].real());
                atomicAdd(&g11->y, img_grad_tmp[0].imag());
                atomicAdd(&g12->x, img_grad_tmp[1].real());
                atomicAdd(&g12->y, img_grad_tmp[1].imag());
                atomicAdd(&g21->x, img_grad_tmp[2].real());
                atomicAdd(&g21->y, img_grad_tmp[2].imag());
                atomicAdd(&g22->x, img_grad_tmp[3].real());
                atomicAdd(&g22->y, img_grad_tmp[3].imag());
                float2 *gx11 = (float2*)&img_gx_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
                float2 *gx12 = (float2*)&img_gx_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
                float2 *gx21 = (float2*)&img_gx_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
                float2 *gx22 = (float2*)&img_gx_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
                atomicAdd(&gx11->x, img_gx_grad_tmp[0].real());
                atomicAdd(&gx11->y, img_gx_grad_tmp[0].imag());
                atomicAdd(&gx12->x, img_gx_grad_tmp[1].real());
                atomicAdd(&gx12->y, img_gx_grad_tmp[1].imag());
                atomicAdd(&gx21->x, img_gx_grad_tmp[2].real());
                atomicAdd(&gx21->y, img_gx_grad_tmp[2].imag());
                atomicAdd(&gx22->x, img_gx_grad_tmp[3].real());
                atomicAdd(&gx22->y, img_gx_grad_tmp[3].imag());
                float2 *gy11 = (float2*)&img_gy_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
                float2 *gy12 = (float2*)&img_gy_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
                float2 *gy21 = (float2*)&img_gy_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
                float2 *gy22 = (float2*)&img_gy_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
                atomicAdd(&gy11->x, img_gy_grad_tmp[0].real());
                atomicAdd(&gy11->y, img_gy_grad_tmp[0].imag());
                atomicAdd(&gy12->x, img_gy_grad_tmp[1].real());
                atomicAdd(&gy12->y, img_gy_grad_tmp[1].imag());
                atomicAdd(&gy21->x, img_gy_grad_tmp[2].real());
                atomicAdd(&gy21->y, img_gy_grad_tmp[2].imag());
                atomicAdd(&gy22->x, img_gy_grad_tmp[3].real());
                atomicAdd(&gy22->y, img_gy_grad_tmp[3].imag());
                float2 *gxy11 = (float2*)&img_gxy_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int];
                float2 *gxy12 = (float2*)&img_gxy_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1];
                float2 *gxy21 = (float2*)&img_gxy_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int];
                float2 *gxy22 = (float2*)&img_gxy_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1];
                atomicAdd(&gxy11->x, img_gxy_grad_tmp[0].real());
                atomicAdd(&gxy11->y, img_gxy_grad_tmp[0].imag());
                atomicAdd(&gxy12->x, img_gxy_grad_tmp[1].real());
                atomicAdd(&gxy12->y, img_gxy_grad_tmp[1].imag());
                atomicAdd(&gxy21->x, img_gxy_grad_tmp[2].real());
                atomicAdd(&gxy21->y, img_gxy_grad_tmp[2].imag());
                atomicAdd(&gxy22->x, img_gxy_grad_tmp[3].real());
                atomicAdd(&gxy22->y, img_gxy_grad_tmp[3].imag());
            }
        }
    }
}

at::Tensor polar_to_cart_bicubic_cuda(
          const at::Tensor &img,
          const at::Tensor &img_gx,
          const at::Tensor &img_gy,
          const at::Tensor &img_gxy,
          const at::Tensor &origin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double x0,
          double y0,
          double dx,
          double dy,
          int64_t Nx,
          int64_t Ny) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(img_gx.dtype() == img.dtype());
	TORCH_CHECK(img_gy.dtype() == img.dtype());
	TORCH_CHECK(img_gxy.dtype() == img.dtype());
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gx.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gy.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gxy.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CUDA);
	at::Tensor origin_contig = origin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor img_gx_contig = img_gx.contiguous();
	at::Tensor img_gy_contig = img_gy.contiguous();
	at::Tensor img_gxy_contig = img_gxy.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* origin_ptr = origin_contig.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* img_gx_ptr = img_gx_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* img_gy_ptr = img_gy_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* img_gxy_ptr = img_gxy_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        polar_to_cart_kernel_bicubic<complex64_t>
              <<<block_count, thread_per_block>>>(
                      (const complex64_t*)img_ptr,
                      (const complex64_t*)img_gx_ptr,
                      (const complex64_t*)img_gy_ptr,
                      (const complex64_t*)img_gxy_ptr,
                      (complex64_t*)out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny
                      );
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* img_gx_ptr = img_gx_contig.data_ptr<float>();
        float* img_gy_ptr = img_gy_contig.data_ptr<float>();
        float* img_gxy_ptr = img_gxy_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        polar_to_cart_kernel_bicubic<float>
              <<<block_count, thread_per_block>>>(
                      img_ptr,
                      img_gx_ptr,
                      img_gy_ptr,
                      img_gxy_ptr,
                      out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny
                      );
    }
	return out;
}

std::vector<at::Tensor> polar_to_cart_bicubic_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &img,
          const at::Tensor &img_gx,
          const at::Tensor &img_gy,
          const at::Tensor &img_gxy,
          const at::Tensor &origin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double x0,
          double y0,
          double dx,
          double dy,
          int64_t Nx,
          int64_t Ny) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gx.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gy.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gxy.dtype() == at::kComplexFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor origin_contig = origin.contiguous();
	const float* origin_ptr = origin_contig.data_ptr<float>();
	at::Tensor img_contig = img.contiguous();
	at::Tensor img_gx_contig = img_gx.contiguous();
	at::Tensor img_gy_contig = img_gy.contiguous();
	at::Tensor img_gxy_contig = img_gxy.contiguous();
	at::Tensor grad_contig = grad.contiguous();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gx_ptr = img_gx_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gy_ptr = img_gy_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gxy_ptr = img_gxy_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();
    at::Tensor img_grad;
    at::Tensor img_gx_grad;
    at::Tensor img_gy_grad;
    at::Tensor img_gxy_grad;
    c10::complex<float>* img_grad_ptr = nullptr;
    c10::complex<float>* img_gx_grad_ptr = nullptr;
    c10::complex<float>* img_gy_grad_ptr = nullptr;
    c10::complex<float>* img_gxy_grad_ptr = nullptr;
    if (img.requires_grad()) {
        img_grad = torch::zeros_like(img);
        img_gx_grad = torch::zeros_like(img);
        img_gy_grad = torch::zeros_like(img);
        img_gxy_grad = torch::zeros_like(img);
        img_grad_ptr = img_grad.data_ptr<c10::complex<float>>();
        img_gx_grad_ptr = img_gx_grad.data_ptr<c10::complex<float>>();
        img_gy_grad_ptr = img_gy_grad.data_ptr<c10::complex<float>>();
        img_gxy_grad_ptr = img_gxy_grad.data_ptr<c10::complex<float>>();
    } else {
        img_grad = torch::Tensor();
        img_gx_grad = torch::Tensor();
        img_gy_grad = torch::Tensor();
        img_gxy_grad = torch::Tensor();
    }

    at::Tensor origin_grad;
	float* origin_grad_ptr = nullptr;
    if (origin.requires_grad()) {
        origin_grad = torch::zeros_like(origin);
        origin_grad_ptr = origin_grad.data_ptr<float>();
    } else {
        origin_grad = torch::Tensor();
    }

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_to_cart_kernel_bicubic_grad
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  (const complex64_t*)img_gx_ptr,
                  (const complex64_t*)img_gy_ptr,
                  (const complex64_t*)img_gxy_ptr,
                  origin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr,
                  theta0,
                  dtheta,
                  Nr,
                  Ntheta,
                  x0,
                  dx,
                  y0,
                  dy,
                  Nx,
                  Ny,
                  (const complex64_t*)grad_ptr,
                  (complex64_t*)img_grad_ptr,
                  (complex64_t*)img_gx_grad_ptr,
                  (complex64_t*)img_gy_grad_ptr,
                  (complex64_t*)img_gxy_grad_ptr,
                  origin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(img_gx_grad);
    ret.push_back(img_gy_grad);
    ret.push_back(img_gxy_grad);
    ret.push_back(origin_grad);
	return ret;
}

template<typename T>
__global__ void polar_to_cart_kernel_lanczos(const T *img, T
        *out, const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, int order) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    const float orig0 = origin[idbatch * 3 + 0];
    const float orig1 = origin[idbatch * 3 + 1];
    const float orig2 = origin[idbatch * 3 + 2];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-orig0)*(x-orig0) + (y-orig1)*(y-orig1));
    const float dz = sqrtf(d*d + orig2*orig2);
    float t = (y - orig1) / d; // Sin of angle
    float tc = (x - orig0) / d; // Cos of angle
    float rs = sinf(rotation);
    float rc = cosf(rotation);
    float cosa = t*rs  + tc*rc;
    if (rotation != 0.0f) {
        t = rc * t - rs * tc;
    }
    const float dri = (d - r0) / dr;
    const float dti = (t - theta0) / dtheta;

    const int dri_int = dri;
    const int dti_int = dti;

    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        T v = lanczos_interp_2d<T, T>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri, dti, order);
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            float ref_sin, ref_cos;
            sincospif(ref_phase * dz, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            out[idbatch * Nx * Ny + idx*Ny + idy] = v * ref;
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = v;
        }
    } else {
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx*Ny + idy] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = 0.0f;
        }
    }
}


at::Tensor polar_to_cart_lanczos_cuda(
          const at::Tensor &img,
          const at::Tensor &origin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr,
          double theta0,
          double dtheta,
          int64_t Nr,
          int64_t Ntheta,
          double x0,
          double y0,
          double dx,
          double dy,
          int64_t Nx,
          int64_t Ny,
          int64_t order) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CUDA);
	at::Tensor origin_contig = origin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* origin_ptr = origin_contig.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

	if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        polar_to_cart_kernel_lanczos<complex64_t>
              <<<block_count, thread_per_block>>>(
                      (const complex64_t*)img_ptr,
                      (complex64_t*)out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny,
                      order
                      );
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        polar_to_cart_kernel_lanczos<float>
              <<<block_count, thread_per_block>>>(
                      img_ptr,
                      out_ptr,
                      origin_ptr,
                      rotation,
                      ref_phase,
                      r0,
                      dr,
                      theta0,
                      dtheta,
                      Nr,
                      Ntheta,
                      x0,
                      dx,
                      y0,
                      dy,
                      Nx,
                      Ny,
                      order
                      );
    }
	return out;
}

at::Tensor polar_interp_linear_cuda(
          const at::Tensor &img,
          const at::Tensor &dorigin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr0,
          double theta0,
          double dtheta0,
          int64_t Nr0,
          int64_t Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nr1, Ntheta1}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_interp_kernel_linear
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  (complex64_t*)out_ptr,
                  dorigin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr0,
                  theta0,
                  dtheta0,
                  Nr0,
                  Ntheta0,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1
                  );
	return out;
}

__global__ void ffbp_merge2_kernel_lanczos(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;
    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);

    complex64_t pixel{};

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        // TODO: Add dorig2
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = lanczos_interp_2d<complex64_t, complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorigin[id * 3 + 2];
            const float dz = sqrtf(z1*z1 + d*d);
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += v * ref;
        }
    }
    out[idr*Ntheta1 + idtheta] = pixel;
}

__global__ void ffbp_merge2_kernel_knab(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, float knab_v) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;
    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);

    complex64_t pixel{};
    const float knab_norm = knab_kernel_norm(order, knab_v);

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        // TODO: Add dorig2
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = knab_interp_2d<complex64_t, complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order, knab_v, knab_norm);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorigin[id * 3 + 2];
            const float dz = sqrtf(z1*z1 + d*d);
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospif(ref_phase * (rpz - dz), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += v * ref;
        }
    }
    out[idr*Ntheta1 + idtheta] = pixel;
}

std::vector<at::Tensor> polar_interp_linear_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &img,
          const at::Tensor &dorigin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr0,
          double theta0,
          double dtheta0,
          int64_t Nr0,
          int64_t Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
	at::Tensor img_contig = img.contiguous();
	at::Tensor grad_contig = grad.contiguous();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* grad_ptr = grad_contig.data_ptr<c10::complex<float>>();
    at::Tensor img_grad;
    c10::complex<float>* img_grad_ptr = nullptr;
    if (img.requires_grad()) {
        img_grad = torch::zeros_like(img);
        img_grad_ptr = img_grad.data_ptr<c10::complex<float>>();
    } else {
        img_grad = torch::Tensor();
    }

    at::Tensor dorigin_grad;
	float* dorigin_grad_ptr = nullptr;
    if (dorigin.requires_grad()) {
        dorigin_grad = torch::zeros_like(dorigin);
        dorigin_grad_ptr = dorigin_grad.data_ptr<float>();
    } else {
        dorigin_grad = torch::Tensor();
    }

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_interp_kernel_linear_grad
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  dorigin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr0,
                  theta0,
                  dtheta0,
                  Nr0,
                  Ntheta0,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1,
                  (complex64_t*)grad_ptr,
                  (complex64_t*)img_grad_ptr,
                  dorigin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(dorigin_grad);
	return ret;
}

at::Tensor polar_interp_bicubic_cuda(
          const at::Tensor &img,
          const at::Tensor &img_gx,
          const at::Tensor &img_gy,
          const at::Tensor &img_gxy,
          const at::Tensor &dorigin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr0,
          double theta0,
          double dtheta0,
          int64_t Nr0,
          int64_t Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gx.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gy.dtype() == at::kComplexFloat);
	TORCH_CHECK(img_gxy.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gx.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gy.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img_gxy.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor img_gx_contig = img_gx.contiguous();
	at::Tensor img_gy_contig = img_gy.contiguous();
	at::Tensor img_gxy_contig = img_gxy.contiguous();
	at::Tensor out = torch::empty({nbatch, Nr1, Ntheta1}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gx_ptr = img_gx_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gy_ptr = img_gy_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_gxy_ptr = img_gxy_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_interp_kernel_bicubic
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  (const complex64_t*)img_gx_ptr,
                  (const complex64_t*)img_gy_ptr,
                  (const complex64_t*)img_gxy_ptr,
                  (complex64_t*)out_ptr,
                  dorigin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr0,
                  theta0,
                  dtheta0,
                  Nr0,
                  Ntheta0,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1
                  );
	return out;
}

at::Tensor polar_interp_lanczos_cuda(
          const at::Tensor &img,
          const at::Tensor &dorigin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr0,
          double theta0,
          double dtheta0,
          int64_t Nr0,
          int64_t Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1,
          int64_t order) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nr1, Ntheta1}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch), 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_interp_kernel_lanczos
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  (complex64_t*)out_ptr,
                  dorigin_ptr,
                  rotation,
                  ref_phase,
                  r0,
                  dr0,
                  theta0,
                  dtheta0,
                  Nr0,
                  Ntheta0,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1,
                  order
                  );
	return out;
}

at::Tensor ffbp_merge2_lanczos_cuda(
          const at::Tensor &img0,
          const at::Tensor &img1,
          const at::Tensor &dorigin,
          double fc,
          const at::Tensor &r0,
          const at::Tensor &dr0,
          const at::Tensor &theta0,
          const at::Tensor &dtheta0,
          const at::Tensor &Nr0,
          const at::Tensor &Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1,
          int64_t order) {
	TORCH_CHECK(img0.dtype() == at::kComplexFloat);
	TORCH_CHECK(img1.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_CHECK(r0.dtype() == at::kFloat);
	TORCH_CHECK(dr0.dtype() == at::kFloat);
	TORCH_CHECK(theta0.dtype() == at::kFloat);
	TORCH_CHECK(dtheta0.dtype() == at::kFloat);
	TORCH_CHECK(Nr0.dtype() == at::kInt);
	TORCH_CHECK(Ntheta0.dtype() == at::kInt);
	TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(r0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dr0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(theta0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dtheta0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(Nr0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(Ntheta0.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img0_contig = img0.contiguous();
	at::Tensor img1_contig = img1.contiguous();
	at::Tensor r0_contig = r0.contiguous();
	at::Tensor dr0_contig = dr0.contiguous();
	at::Tensor theta0_contig = theta0.contiguous();
	at::Tensor dtheta0_contig = dtheta0.contiguous();
	at::Tensor Nr0_contig = Nr0.contiguous();
	at::Tensor Ntheta0_contig = Ntheta0.contiguous();
	at::Tensor out = torch::empty({Nr1, Ntheta1}, img0_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    c10::complex<float>* img0_ptr = img0_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img1_ptr = img1_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, 1, 1};

    const float ref_phase = 4.0f * fc / kC0;

    ffbp_merge2_kernel_lanczos
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img0_ptr,
                  (const complex64_t*)img1_ptr,
                  (complex64_t*)out_ptr,
                  dorigin_ptr,
                  ref_phase,
                  r0_ptr,
                  dr0_ptr,
                  theta0_ptr,
                  dtheta0_ptr,
                  Nr0_ptr,
                  Ntheta0_ptr,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1,
                  order
                  );
	return out;
}

at::Tensor ffbp_merge2_knab_cuda(
          const at::Tensor &img0,
          const at::Tensor &img1,
          const at::Tensor &dorigin,
          double fc,
          const at::Tensor &r0,
          const at::Tensor &dr0,
          const at::Tensor &theta0,
          const at::Tensor &dtheta0,
          const at::Tensor &Nr0,
          const at::Tensor &Ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t Nr1,
          int64_t Ntheta1,
          double z1,
          int64_t order,
          double oversample) {
	TORCH_CHECK(img0.dtype() == at::kComplexFloat);
	TORCH_CHECK(img1.dtype() == at::kComplexFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_CHECK(r0.dtype() == at::kFloat);
	TORCH_CHECK(dr0.dtype() == at::kFloat);
	TORCH_CHECK(theta0.dtype() == at::kFloat);
	TORCH_CHECK(dtheta0.dtype() == at::kFloat);
	TORCH_CHECK(Nr0.dtype() == at::kInt);
	TORCH_CHECK(Ntheta0.dtype() == at::kInt);
	TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(r0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dr0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(theta0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dtheta0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(Nr0.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(Ntheta0.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img0_contig = img0.contiguous();
	at::Tensor img1_contig = img1.contiguous();
	at::Tensor r0_contig = r0.contiguous();
	at::Tensor dr0_contig = dr0.contiguous();
	at::Tensor theta0_contig = theta0.contiguous();
	at::Tensor dtheta0_contig = dtheta0.contiguous();
	at::Tensor Nr0_contig = Nr0.contiguous();
	at::Tensor Ntheta0_contig = Ntheta0.contiguous();
	at::Tensor out = torch::empty({Nr1, Ntheta1}, img0_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    c10::complex<float>* img0_ptr = img0_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img1_ptr = img1_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr1 * Ntheta1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, 1, 1};

    const float ref_phase = 4.0f * fc / kC0;
    const float v = 1.0f - 1.0f / oversample;

    ffbp_merge2_kernel_knab
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img0_ptr,
                  (const complex64_t*)img1_ptr,
                  (complex64_t*)out_ptr,
                  dorigin_ptr,
                  ref_phase,
                  r0_ptr,
                  dr0_ptr,
                  theta0_ptr,
                  dtheta0_ptr,
                  Nr0_ptr,
                  Ntheta0_ptr,
                  r1,
                  dr1,
                  theta1,
                  dtheta1,
                  Nr1,
                  Ntheta1,
                  z1,
                  order,
                  v
                  );
	return out;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("polar_interp_linear", &polar_interp_linear_cuda);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cuda);
  m.impl("polar_interp_bicubic", &polar_interp_bicubic_cuda);
  m.impl("polar_interp_lanczos", &polar_interp_lanczos_cuda);
  m.impl("ffbp_merge2_lanczos", &ffbp_merge2_lanczos_cuda);
  m.impl("ffbp_merge2_knab", &ffbp_merge2_knab_cuda);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cuda);
  m.impl("polar_to_cart_linear_grad", &polar_to_cart_linear_grad_cuda);
  m.impl("polar_to_cart_bicubic", &polar_to_cart_bicubic_cuda);
  m.impl("polar_to_cart_bicubic_grad", &polar_to_cart_bicubic_grad_cuda);
  m.impl("polar_to_cart_lanczos", &polar_to_cart_lanczos_cuda);
}

}
