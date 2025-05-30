#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define LIBCUDACXX_ENABLE_HOST_NVFP16
// cuda::std::complex multiplication checks for nans which causes
// it to be extremely slow. This can be disabled by definition below on
// new version of the library, but it's not available on the old version
// that is shipped with cuda.
#ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#endif
#include <cuda/std/version>
#include <cuda/std/type_traits>
#if _LIBCUDACXX_CUDA_API_VERSION >= 2002000
#  include <cuda/std/complex>
#else
#  include "std_complex.h"
#endif

#define kPI 3.1415926535897932384626433f
#define kC0 299792458.0f
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace torchbp {

using complex64_t = cuda::std::complex<float>;
using complex32_t = cuda::std::complex<__half>;

template<class T>
static __device__ T interp1d_cubic(const T *y, int nx,
        int x_int, float x_frac) {
    T p1 = y[x_int];
    T p2 = y[x_int+1];
    if (x_int < 1 || x_int > nx-2) {
        return p1 * (1.0f - x_frac) + p2 * x_frac;
    }
    T p0 = y[x_int-1];
    T p3 = y[x_int+2];
    T yi = (-p0/2.0f + 3.0f*p1/2.0f -3.0f*p2/2.0f + p3/2.0f)*x_frac*x_frac*x_frac +
        (p0 - 5.0f*p1/2.0f + 2.0f*p2 - p3/2.0f)*x_frac*x_frac +
        (-p0/2.0f + p2/2.0f)*x_frac + p1;
    return yi;
}

template<class T>
static __device__ T interp1d_cubic_x_frac_grad(const T *y, int nx,
        int x_int, float x_frac) {
    T p1 = y[x_int];
    T p2 = y[x_int+1];
    if (x_int < 1 || x_int > nx-2) {
        return p2 - p1;
    }
    T p0 = y[x_int-1];
    T p3 = y[x_int+2];
    T yi = -(p0/2.0f) + p2/2.0f +
        2.0f*x_frac*(p0 - (5.0f*p1)/2.0f + 2.0f*p2 - p3/2.0f) +
        3.0f*x_frac*x_frac*(-(p0/2.0f) + (3.0f*p1)/2.0f - (3.0f*p2)/2.0f + p3/2.0f);
    return yi;
}

template<class T>
static __device__ T interp2d(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return img[x_int*ny + y_int]*(1.0f-x_frac)*(1.0f-y_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac)*y_frac +
           img[(x_int+1)*ny + y_int]*x_frac*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*x_frac*y_frac;
}

template<class T>
static __device__ float interp2d_abs(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return abs(img[x_int*ny + y_int])*(1.0f-x_frac)*(1.0f-y_frac) +
           abs(img[x_int*ny + y_int+1])*(1.0f-x_frac)*y_frac +
           abs(img[(x_int+1)*ny + y_int])*x_frac*(1.0f-y_frac) +
           abs(img[(x_int+1)*ny + y_int+1])*x_frac*y_frac;
}

template<class T>
static __device__ T bicubic_interp2d(const T *img, const T *gx, const T *gy,
        const T *gxy, int nx, int ny, int xi, float x,
        int yi, float y) {

    const T f[16] = {
        img[xi*ny + yi],
        img[xi*ny + yi + 1],
        gy[xi*ny + yi],
        gy[xi*ny + yi + 1],
        img[(xi+1)*ny + yi],
        img[(xi+1)*ny + yi + 1],
        gy[(xi+1)*ny + yi],
        gy[(xi+1)*ny + yi + 1],
        gx[xi*ny + yi],
        gx[xi*ny + yi + 1],
        gxy[xi*ny + yi],
        gxy[xi*ny + yi + 1],
        gx[(xi+1)*ny + yi],
        gx[(xi+1)*ny + yi + 1],
        gxy[(xi+1)*ny + yi],
        gxy[(xi+1)*ny + yi + 1]
    };

    const float x2 = x*x;
    const float x3 = x*x*x;
    const float y2 = y*y;
    const float y3 = y*y*y;
    const float xb[] = {1.0f - 3.0f*x2 + 2.0f*x3, 3.0f*x2-2.0f*x3, x-2.0f*x2+x3, -x2+x3};
    const float by[] = {1.0f - 3.0f*y2 + 2.0f*y3, 3.0f*y2-2.0f*y3, y-2.0f*y2+y3, -y2+y3};
    T v = 0.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            v += xb[i] * by[j] * f[i*4 + j];
        }
    }
    return v;
}

template<class T>
static __device__ void bicubic_interp2d_grad(T *img_grad, T *img_gx_grad,
        T *img_gy_grad, T *img_gxy_grad, float x, float y, T g) {
    const float x2 = x*x;
    const float x3 = x*x*x;
    const float y2 = y*y;
    const float y3 = y*y*y;
    const float xb[] = {1.0f - 3.0f*x2 + 2.0f*x3, 3.0f*x2-2.0f*x3, x-2.0f*x2+x3, -x2+x3};
    const float by[] = {1.0f - 3.0f*y2 + 2.0f*y3, 3.0f*y2-2.0f*y3, y-2.0f*y2+y3, -y2+y3};

    img_grad[0] = g * (xb[0] * by[0]);
    img_grad[1] = g * (xb[0] * by[1]);
    img_gy_grad[0] = g * (xb[0] * by[2]);
    img_gy_grad[1] = g * (xb[0] * by[3]);
    img_grad[2] = g * (xb[1] * by[0]);
    img_grad[3] = g * (xb[1] * by[1]);
    img_gy_grad[2] = g * (xb[1] * by[2]);
    img_gy_grad[3] = g * (xb[1] * by[3]);
    img_gx_grad[0] = g * (xb[2] * by[0]);
    img_gx_grad[1] = g * (xb[2] * by[1]);
    img_gxy_grad[0] = g * (xb[2] * by[2]);
    img_gxy_grad[1] = g * (xb[2] * by[3]);
    img_gx_grad[2] = g * (xb[3] * by[0]);
    img_gx_grad[3] = g * (xb[3] * by[1]);
    img_gxy_grad[2] = g * (xb[3] * by[2]);
    img_gxy_grad[3] = g * (xb[3] * by[3]);
}

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

template<typename T, bool bistatic>
__global__ void backprojection_polar_2d_kernel(
          const T* data,
          const float* pos,
          const float* vel,
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
          float ant_tx_dy) {
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

    complex64_t pixel = {0.0f, 0.0f};

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz2 = pos_z * pos_z;

        // Calculate distance to the pixel.
        float d;
        if constexpr (bistatic) {
            float tx_dx = sinf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;
            float tx_dy = cosf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;

            float drx = sqrtf(px * px + py * py + pz2);
            float dtx = sqrt((px - tx_dx) * (px - tx_dx) + (py - tx_dy) * (py - tx_dy) + pz2);
            d = drx + dtx - d0;
        } else {
            float drx = sqrtf(px * px + py * py + pz2);
            d = 2.0f * drx - d0;
        }

        float sx = delta_r * d;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (id0 < 0 || id1 >= sweep_samples) {
            continue;
        }
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
        sincospif(ref_phase * d, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        pixel += s * ref;
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = pixel;
}

template<typename T, bool have_pos_grad, bool have_data_grad>
__global__ void backprojection_polar_2d_grad_kernel(
          const T* data,
          const float* pos,
          const float* vel,
          const float* att,
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
          float ant_tx_dy,
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

        float tx_dx = sinf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;
        float tx_dy = cosf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;

        // Calculate distance to the pixel.

        float drx = sqrtf(px * px + py * py + pz2);
        float dtx = sqrt((px - tx_dx) * (px - tx_dx) + (py - tx_dy) * (py - tx_dy) + pz2);
        float d = drx + dtx - d0;

        float sx = delta_r * d;

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
            sincospif(ref_phase * d, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * (I * (kPI * ref_phase) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * cuda::std::conj(dout);

                // Take real part
                float gd = cuda::std::real(gdout);

                dx = -px / dtx - (px - tx_dx) / drx;
                dy = -py / dtx - (py - tx_dy) / drx;
                // Different from x,y because pos_z is handled differently.
                dz = pos_z / dtx + pos_z / drx;
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

template<typename T, bool bistatic>
__global__ void gpga_backprojection_2d_kernel(
          const float* target_pos,
          const T* data,
          const float* pos,
          const float* vel,
          const float* att,
          complex64_t* data_out,
          int sweep_samples,
          int nsweeps,
          float ref_phase,
          float delta_r,
          int Ntarget,
          float d0,
          float ant_tx_dy) {
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
    float d;
    if constexpr (bistatic) {
        float tx_dx = sinf(att[idsweep * 3 + 2]) * ant_tx_dy;
        float tx_dy = cosf(att[idsweep * 3 + 2]) * ant_tx_dy;

        float drx = sqrtf(px * px + py * py + pz*pz);
        float dtx = sqrt((px - tx_dx) * (px - tx_dx) + (py - tx_dy) * (py - tx_dy) + pz * pz);
        d = drx + dtx - d0;
    } else {
        float drx = sqrtf(px * px + py * py + pz * pz);
        d = 2.0f * drx - d0;
    }

    float sx = delta_r * d;

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
        sincospif(ref_phase * d, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        data_out[idtarget * nsweeps + idsweep] = s * ref;
    }
}

__global__ void cfar_2d_kernel(
          const float* img,
          float* detections,
          int N0,
          int N1,
          int Navg0,
          int Navg1,
          int Nguard0,
          int Nguard1,
          float threshold,
          int peaks_only) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = id0 % N1;
    const int idy = id0 / N1;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id0 >= N0 * N1) {
        return;
    }

    if (peaks_only) {
        float v = img[idbatch * N0 * N1 + idy * N1 + idx];
        float v00 = img[idbatch * N0 * N1 + idy * N1 + max(0, idx - 1)];
        float v01 = img[idbatch * N0 * N1 + idy * N1 + min(N1, idx + 1)];
        float v10 = img[idbatch * N0 * N1 + max(0, idy - 1) * N1 + idx];
        float v11 = img[idbatch * N0 * N1 + min(N0, idy + 1) * N1 + idx];
        if (v < v00 || v < v01 || v < v10 || v < v11) {
            detections[idbatch * N0 * N1 + idy * N1 + idx] = 0.0f;
            return;
        }
    }

    float avg = 0.0f;
    int Navg = 0;
    for (int i=-Navg0; i <= Navg0; i++) {
        int x = idx + i;
        if (x < 0 || x >= N1) continue;
        for (int j=-Navg1; j <= Navg1; j++) {
            if (i >= -Nguard0 && i <= Nguard0 && j >= -Nguard1 && j <= Nguard1) continue;
            int y = idy + j;
            if (y < 0 || y >= N0) continue;
            avg += img[idbatch * N0 * N1 + y * N1 + x];
            Navg += 1;
        }
    }
    avg /= Navg;
    float v = img[idbatch * N0 * N1 + idy * N1 + idx];
    if (v > avg * threshold) {
        detections[idbatch * N0 * N1 + idy * N1 + idx] = v / avg;
    } else {
        detections[idbatch * N0 * N1 + idy * N1 + idx] = 0.0f;
    }
}

__global__ void backprojection_polar_2d_tx_power_kernel(
          const float* wa,
          const float* pos,
          const float* att,
          const float* gtx,
          const float* grx,
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
          bool sin_look_angle) {
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

        float drx = sqrtf(px * px + py * py + pz2);
        // Bistatic term ignored in power calculations.
        float d = drx + drx;

        // Elevation angle.
        // Add roll to elevation.
        const float look_angle = atan2f(pos_z, d);
        const float el_deg = -look_angle - att[idbatch * nsweeps * 3 + 3 * i + 0];
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
        float gtx_i = interp2d<float>(gtx, g_naz, g_nel, az_int, az_frac, el_int, el_frac);
        float grx_i = interp2d<float>(grx, g_naz, g_nel, az_int, az_frac, el_int, el_frac);
        const float g2 = gtx_i * grx_i;
        float sinl = 1.0f;

        if (sin_look_angle) {
            sinl = pos_z / d;
            if (sinl < 0.0f) sinl = 0.0f;
            if (sinl > 1.0f) sinl = 1.0f;
        }

        pixel += sinl * g2 * wa[idbatch * nsweeps + i] / (d*d*d*d);
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = pixel;
}

__global__ void backprojection_cart_2d_kernel(
          const complex64_t* data,
          const float* pos,
          const float* vel,
          const float* att,
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
          float ant_tx_dy) {
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

        float tx_dx = sinf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;
        float tx_dy = cosf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;

        float target_angle = atan2(px, py);
        //if (fabsf(target_angle - att[i*3 + 2]) > beamwidth) {
        //    // Pixel outside of the beam.
        //    continue;
        //}
        // Calculate distance to the pixel.

        float drx = sqrtf(px * px + py * py + pz2);
        float dtx = sqrt((px - tx_dx) * (px - tx_dx) + (py - tx_dy) * (py - tx_dy) + pz2);
        float d = drx + dtx - d0;

        complex64_t s;
        float sx = delta_r * d;

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (id0 >= 0 && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospif(ref_phase * d, &ref_sin, &ref_cos);
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
          const float* vel,
          const float* att,
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
          float ant_tx_dy,
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

        float tx_dx = sinf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;
        float tx_dy = cosf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;

        float target_angle = atan2(px, py);
        //if (fabsf(target_angle - att[i*3 + 2]) > beamwidth) {
        //    // Pixel outside of the beam.
        //    continue;
        //}
        // Calculate distance to the pixel.

        float drx = sqrtf(px * px + py * py + pz2);
        float dtx = sqrt((px - tx_dx) * (px - tx_dx) + (py - tx_dy) * (py - tx_dy) + pz2);
        float d = drx + dtx - d0;

        float sx = delta_r * d;

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
            sincospif(ref_phase * d, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * ((I * kPI * ref_phase) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * cuda::std::conj(dout);

                float gd = cuda::std::real(gdout);

                dx = -px / dtx - (px - tx_dx) / drx;
                dy = -py / dtx - (py - tx_dy) / drx;
                // Different from x,y because pos_z is handled differently.
                dz = pos_z / dtx + pos_z / drx;
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
	at::Tensor res = torch::zeros({nbatch, 1}, options);
    float* res_ptr = res.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

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
	dim3 block_count = {block_x, nbatch};

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
	at::Tensor res = torch::zeros({nbatch}, options);
    float* res_ptr = res.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = data.numel() / nbatch;
	TORCH_INTERNAL_ASSERT(blocks * nbatch == data.numel());
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

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
	dim3 block_count = {block_x, nbatch};

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

at::Tensor backprojection_polar_2d_cuda(
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &vel,
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
          double ant_tx_dy) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(vel.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

    // delta_r, ref_phase, d0 are bistatic, 2x monostatic
	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (ant_tx_dy != 0.0f) {
            backprojection_polar_2d_kernel<complex64_t, true>
                  <<<block_count, thread_per_block>>>(
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0, ant_tx_dy);
        } else {
            backprojection_polar_2d_kernel<complex64_t, false>
                  <<<block_count, thread_per_block>>>(
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0, ant_tx_dy);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (ant_tx_dy != 0.0f) {
            backprojection_polar_2d_kernel<half2, true>
                  <<<block_count, thread_per_block>>>(
                          (half2*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0, ant_tx_dy);
        } else {
            backprojection_polar_2d_kernel<half2, false>
                  <<<block_count, thread_per_block>>>(
                          (half2*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0, ant_tx_dy);
        }
    }
	return img;
}

std::vector<at::Tensor> backprojection_polar_2d_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &vel,
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
          double ant_tx_dy) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(vel.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
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

    // delta_r, ref_phase, d0 are bistatic, 2x monostatic
	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch, 1};

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (have_pos_grad) {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<complex64_t, true, true>
                      <<<block_count, thread_per_block>>>(
                              (complex64_t*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                backprojection_polar_2d_grad_kernel<complex64_t, true, false>
                      <<<block_count, thread_per_block>>>(
                              (complex64_t*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );

            }
        } else {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<complex64_t, false, true>
                      <<<block_count, thread_per_block>>>(
                              (complex64_t*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
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
                      <<<block_count, thread_per_block>>>(
                              (half2*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );
            } else {
                backprojection_polar_2d_grad_kernel<half2, true, false>
                      <<<block_count, thread_per_block>>>(
                              (half2*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
                              (complex64_t*)grad_ptr,
                              pos_grad_ptr,
                              (complex64_t*)data_grad_ptr
                              );

            }
        } else {
            if (have_data_grad) {
                backprojection_polar_2d_grad_kernel<half2, false, true>
                      <<<block_count, thread_per_block>>>(
                              (half2*)data_ptr,
                              pos_ptr,
                              vel_ptr,
                              att_ptr,
                              sweep_samples,
                              nsweeps,
                              ref_phase,
                              delta_r,
                              r0, dr,
                              theta0, dtheta,
                              Nr, Ntheta,
                              d0, ant_tx_dy,
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

at::Tensor gpga_backprojection_2d_cuda(
          const at::Tensor &target_pos,
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &vel,
          const at::Tensor &att,
          int64_t sweep_samples,
          int64_t nsweeps,
          double fc,
          double r_res,
          int64_t Ntarget,
          double d0,
          double ant_tx_dy) {
	TORCH_CHECK(target_pos.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(vel.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat || data.dtype() == at::kComplexHalf);
	TORCH_INTERNAL_ASSERT(target_pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);

	at::Tensor target_pos_contig = target_pos.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(data.device());
	at::Tensor data_out = torch::zeros({Ntarget, nsweeps}, options);
	const float* target_pos_ptr = target_pos_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
    c10::complex<float>* data_out_ptr = data_out.data_ptr<c10::complex<float>>();

    // delta_r, ref_phase, d0 are bistatic, 2x monostatic
	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

	dim3 thread_per_block = {256};
	// Up-rounding division.
    int blocks = Ntarget * nsweeps;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x};

	if (data.dtype() == at::kComplexFloat) {
        const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
        if (ant_tx_dy != 0.0f) {
            gpga_backprojection_2d_kernel<complex64_t, true>
                  <<<block_count, thread_per_block>>>(
                          target_pos_ptr,
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)data_out_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          Ntarget,
                          d0, ant_tx_dy);
        } else {
            gpga_backprojection_2d_kernel<complex64_t, false>
                  <<<block_count, thread_per_block>>>(
                          target_pos_ptr,
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)data_out_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          Ntarget,
                          d0, ant_tx_dy);
        }
    } else if (data.dtype() == at::kComplexHalf) {
        const c10::complex<at::Half>* data_ptr = data_contig.data_ptr<c10::complex<at::Half>>();
        if (ant_tx_dy != 0.0f) {
            gpga_backprojection_2d_kernel<half2, true>
                  <<<block_count, thread_per_block>>>(
                          target_pos_ptr,
                          (half2*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)data_out_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          Ntarget,
                          d0, ant_tx_dy);
        } else {
            gpga_backprojection_2d_kernel<half2, false>
                  <<<block_count, thread_per_block>>>(
                          target_pos_ptr,
                          (half2*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          (complex64_t*)data_out_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          Ntarget,
                          d0, ant_tx_dy);
        }
    }
	return data_out;
}

at::Tensor backprojection_polar_2d_tx_power_cuda(
          const at::Tensor &wa,
          const at::Tensor &pos,
          const at::Tensor &att,
          const at::Tensor &gtx,
          const at::Tensor &grx,
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
          int64_t sin_look_angle) {
	TORCH_CHECK(wa.dtype() == at::kFloat);
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(gtx.dtype() == at::kFloat);
	TORCH_CHECK(grx.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(wa.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(gtx.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grx.device().type() == at::DeviceType::CUDA);

	at::Tensor wa_contig = wa.contiguous();
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor gtx_contig = gtx.contiguous();
	at::Tensor grx_contig = grx.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(wa.device());
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* wa_ptr = wa_contig.data_ptr<float>();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const float* gtx_ptr = gtx_contig.data_ptr<float>();
	const float* grx_ptr = grx_contig.data_ptr<float>();
	float* img_ptr = img.data_ptr<float>();

    // delta_r, d0 are bistatic, 2x monostatic
	const float delta_r = 0.5f / r_res;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nr * Ntheta;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

    backprojection_polar_2d_tx_power_kernel
          <<<block_count, thread_per_block>>>(
                  wa_ptr,
                  pos_ptr,
                  att_ptr,
                  gtx_ptr,
                  grx_ptr,
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
                  sin_look_angle);
	return img;
}

at::Tensor backprojection_cart_2d_cuda(
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &vel,
          const at::Tensor &att,
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
          double ant_tx_dy) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(vel.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor img = torch::zeros({nbatch, Nx, Ny}, data_contig.options());
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

	// Divide by 2 to get angle from the center.
	const float beamwidth_f = beamwidth / 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (Nx * Ny + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

	backprojection_cart_2d_kernel
          <<<block_count, thread_per_block>>>(
                  (complex64_t*)data_ptr,
                  pos_ptr,
                  vel_ptr,
                  att_ptr,
                  (complex64_t*)img_ptr,
                  sweep_samples,
                  nsweeps,
                  ref_phase,
                  delta_r,
                  x0, dx,
                  y0, dy,
                  Nx, Ny,
                  beamwidth_f, d0, ant_tx_dy);
	return img;
}

std::vector<at::Tensor> backprojection_cart_2d_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &data,
          const at::Tensor &pos,
          const at::Tensor &vel,
          const at::Tensor &att,
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
          double ant_tx_dy) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(vel.dtype() == at::kFloat);
	TORCH_CHECK(att.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
	at::Tensor grad_contig = grad.contiguous();
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
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

	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

	// Divide by 2 to get angle from the center.
	const float beamwidth_f = beamwidth / 2.0f;

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (Nx * Ny + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

    if (have_pos_grad) {
        if (have_data_grad) {
            backprojection_cart_2d_grad_kernel<true, true>
                  <<<block_count, thread_per_block>>>(
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          beamwidth_f, d0, ant_tx_dy,
                          (complex64_t*)grad_ptr,
                          pos_grad_ptr,
                          (complex64_t*)data_grad_ptr
                          );
        } else {
            backprojection_cart_2d_grad_kernel<true, false>
                  <<<block_count, thread_per_block>>>(
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          beamwidth_f, d0, ant_tx_dy,
                          (complex64_t*)grad_ptr,
                          pos_grad_ptr,
                          (complex64_t*)data_grad_ptr
                          );
        }
    } else {
        if (have_data_grad) {
            backprojection_cart_2d_grad_kernel<false, true>
                  <<<block_count, thread_per_block>>>(
                          (complex64_t*)data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          x0, dx,
                          y0, dy,
                          Nx, Ny,
                          beamwidth_f, d0, ant_tx_dy,
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

at::Tensor cfar_2d_cuda(
          const at::Tensor &img,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t Navg0,
          int64_t Navg1,
          int64_t Nguard0,
          int64_t Nguard1,
          double threshold,
          int64_t peaks_only) {
	TORCH_CHECK(img.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	at::Tensor img_contig = img.contiguous();
	const float* img_ptr = img_contig.data_ptr<float>();

    at::Tensor detections = torch::zeros_like(img);
	float* detections_ptr = detections.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
	unsigned int block_x = (N0 * N1 + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch};

    cfar_2d_kernel<<<block_count, thread_per_block>>>(
          img_ptr,
          detections_ptr,
          N0,
          N1,
          Navg0,
          Navg1,
          Nguard0,
          Nguard1,
          threshold,
          peaks_only);

	return detections;
}

__global__ void polar_interp_kernel_linear(const complex64_t *img, complex64_t
        *out, const float *dorigin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float r1,
        float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1) {
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
    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
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
        sincospif(ref_phase * (rp - d), &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

__global__ void polar_interp_kernel_linear_grad(const complex64_t *img, const
        float *dorigin, float rotation, float ref_phase, float r0, float dr,
        float theta0, float dtheta, int Nr, int Ntheta, float r1, float dr1,
        float theta1, float dtheta1, int Nr1, int Ntheta1, const complex64_t
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
    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float tp = arg / sqrtf(1.0f + arg*arg);

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const float dri_frac = dri - dri_int;

    const int dti_int = dti;
    const float dti_frac = dti - dti_int;

    complex64_t v = {0.0f, 0.0f};
    complex64_t ref = {0.0f, 0.0f};

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        sincospif(ref_phase * (rp - d), &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    if (dorigin_grad != nullptr) {
        complex64_t dout_drp = I * kPI * ref_phase * ref * v;
        float drp_dorigin0 = 0.50f * (2.0f*d*cost + 2.0f*dorig0) / rp;
        float drp_dorigin1 = 0.50f * (2.0f*d*sint + 2.0f*dorig1) / rp;

        complex64_t drp_conj = cuda::std::conj(dout_drp);
        complex64_t gdrp = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] * drp_conj;
        float gd = cuda::std::real(gdrp);
        drp_dorigin0 *= gd;
        drp_dorigin1 *= gd;

        for (int offset = 16; offset > 0; offset /= 2) {
            drp_dorigin0 += __shfl_down_sync(mask, drp_dorigin0, offset);
            drp_dorigin1 += __shfl_down_sync(mask, drp_dorigin1, offset);
        }

        if (threadIdx.x % 32 == 0) {
            atomicAdd(&(dorigin_grad[idbatch * 2 + 0]), drp_dorigin0);
            atomicAdd(&(dorigin_grad[idbatch * 2 + 1]), drp_dorigin1);
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

template<typename T>
__global__ void polar_to_cart_kernel_linear(const T *img, T
        *out, const float *dorigin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, int polar_interp) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-dorig0)*(x-dorig0) + (y-dorig1)*(y-dorig1));
    float t = (y - dorig1) / d; // Sin of angle
    float tc = (x - dorig0) / d; // Cos of angle
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
        // This is needed to avoid artifacts in amplitude when
        // range dimension spectrum is not band limited
        if constexpr (::cuda::std::is_same_v<T, complex64_t>) {
            if (polar_interp) {
                float absv = interp2d_abs<T>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
                v = absv * v / abs(v);
            }
            float ref_sin, ref_cos;
            sincospif(ref_phase * d, &ref_sin, &ref_cos);
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
        const float *dorigin, float rotation, float ref_phase,
        float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta,
        float x0, float dx, float y0, float dy, int Nx, int Ny,
        const complex64_t *grad, complex64_t *img_grad, float *dorigin_grad) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-dorig0)*(x-dorig0) + (y-dorig1)*(y-dorig1));
    float t = (y - dorig1) / d; // Sin of angle
    float tc = (x - dorig0) / d; // Cos of angle
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

    complex64_t v = {0.0f, 0.0f};
    complex64_t ref = {0.0f, 0.0f};
    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        //float absv = interp2d_abs<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        //v = absv * v / abs(v);
        float ref_sin, ref_cos;
        sincospif(ref_phase * d, &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    // FIXME: Doesn't consider absv (polar_interp)
    if (dorigin_grad != nullptr) {
        const complex64_t I = {0.0f, 1.0f};
        complex64_t dout_drp = I * kPI * ref_phase * ref * v;

        complex64_t drp_conj = cuda::std::conj(dout_drp);
        complex64_t gdrp = grad[idbatch * Nx * Ny + idx*Ny + idy] * drp_conj;
        float gd = cuda::std::real(gdrp);
        float dd_dorigin0 = gd * (dorig0 - x) / d;
        float dd_dorigin1 = gd * (dorig1 - y) / d;

        for (int offset = 16; offset > 0; offset /= 2) {
            dd_dorigin0 += __shfl_down_sync(mask, dd_dorigin0, offset);
            dd_dorigin1 += __shfl_down_sync(mask, dd_dorigin1, offset);
        }

        if (threadIdx.x % 32 == 0) {
            atomicAdd(&(dorigin_grad[idbatch * 2 + 0]), dd_dorigin0);
            atomicAdd(&(dorigin_grad[idbatch * 2 + 1]), dd_dorigin1);
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

at::Tensor polar_to_cart_linear_cuda(
          const at::Tensor &img,
          const at::Tensor &dorigin,
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
          int64_t polar_interp) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch, 1};

    const float ref_phase = 4.0f * fc / kC0;

	if (img.dtype() == at::kComplexFloat) {
        c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        polar_to_cart_kernel_linear<complex64_t>
              <<<block_count, thread_per_block>>>(
                      (const complex64_t*)img_ptr,
                      (complex64_t*)out_ptr,
                      dorigin_ptr,
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
                      polar_interp
                      );
    } else {
        float* img_ptr = img_contig.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        polar_to_cart_kernel_linear<float>
              <<<block_count, thread_per_block>>>(
                      img_ptr,
                      out_ptr,
                      dorigin_ptr,
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
                      polar_interp
                      );
    }
	return out;
}

std::vector<at::Tensor> polar_to_cart_linear_grad_cuda(
          const at::Tensor &grad,
          const at::Tensor &img,
          const at::Tensor &dorigin,
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
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch, 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_to_cart_kernel_linear_grad
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  dorigin_ptr,
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
                  dorigin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(dorigin_grad);
	return ret;
}

template<typename T>
__global__ void polar_to_cart_kernel_bicubic(const T *img,
        const T *img_gx, const T *img_gy, const T *img_gxy,
        T *out, const float *dorigin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, int polar_interp) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    if (id1 >= Nx * Ny) {
        return;
    }

    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-dorig0)*(x-dorig0) + (y-dorig1)*(y-dorig1));
    float t = (y - dorig1) / d; // Sin of angle
    float tc = (x - dorig0) / d; // Cos of angle
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
            if (polar_interp) {
                float absv = interp2d_abs<T>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
                v = absv * v / abs(v);
            }
            float ref_sin, ref_cos;
            sincospif(ref_phase * d, &ref_sin, &ref_cos);
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
        const float *dorigin, float rotation, float ref_phase,
        float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta,
        float x0, float dx, float y0, float dy, int Nx, int Ny,
        const complex64_t *grad, complex64_t *img_grad, complex64_t *img_gx_grad,
        complex64_t *img_gy_grad, complex64_t *img_gxy_grad, float *dorigin_grad) {
    const int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;

    const float dorig0 = dorigin[idbatch * 2 + 0];
    const float dorig1 = dorigin[idbatch * 2 + 1];
    const float x = x0 + dx * idx;
    const float y = y0 + dy * idy;
    const float d = sqrtf((x-dorig0)*(x-dorig0) + (y-dorig1)*(y-dorig1));
    float t = (y - dorig1) / d; // Sin of angle
    float tc = (x - dorig0) / d; // Cos of angle
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

    complex64_t v = {0.0f, 0.0f};
    complex64_t ref = {0.0f, 0.0f};
    if (cosa >= 0 && dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        complex64_t v = bicubic_interp2d<complex64_t>(
                &img[idbatch * Nr * Ntheta],
                &img_gx[idbatch * Nr * Ntheta],
                &img_gy[idbatch * Nr * Ntheta],
                &img_gxy[idbatch * Nr * Ntheta],
                Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        //float absv = interp2d_abs<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        //v = absv * v / abs(v);
        float ref_sin, ref_cos;
        sincospif(ref_phase * d, &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    // FIXME: Doesn't consider absv
    if (dorigin_grad != nullptr) {
        const complex64_t I = {0.0f, 1.0f};
        complex64_t dout_drp = I * kPI * ref_phase * ref * v;

        complex64_t drp_conj = cuda::std::conj(dout_drp);
        complex64_t gdrp = grad[idbatch * Nx * Ny + idx*Ny + idy] * drp_conj;
        float gd = cuda::std::real(gdrp);
        float dd_dorigin0 = gd * (dorig0 - x) / d;
        float dd_dorigin1 = gd * (dorig1 - y) / d;

        for (int offset = 16; offset > 0; offset /= 2) {
            dd_dorigin0 += __shfl_down_sync(mask, dd_dorigin0, offset);
            dd_dorigin1 += __shfl_down_sync(mask, dd_dorigin1, offset);
        }

        if (threadIdx.x % 32 == 0) {
            atomicAdd(&(dorigin_grad[idbatch * 2 + 0]), dd_dorigin0);
            atomicAdd(&(dorigin_grad[idbatch * 2 + 1]), dd_dorigin1);
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

at::Tensor polar_to_cart_bicubic_cuda(
          const at::Tensor &img,
          const at::Tensor &img_gx,
          const at::Tensor &img_gy,
          const at::Tensor &img_gxy,
          const at::Tensor &dorigin,
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
          int64_t polar_interp) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(img_gx.dtype() == img.dtype());
	TORCH_CHECK(img_gy.dtype() == img.dtype());
	TORCH_CHECK(img_gxy.dtype() == img.dtype());
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
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch, 1};

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
                      dorigin_ptr,
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
                      polar_interp
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
                      dorigin_ptr,
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
                      polar_interp
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
          const at::Tensor &dorigin,
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
	TORCH_CHECK(dorigin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CUDA);
	at::Tensor dorigin_contig = dorigin.contiguous();
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
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
    int blocks = Nx * Ny;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, nbatch, 1};

    const float ref_phase = 4.0f * fc / kC0;

    polar_to_cart_kernel_bicubic_grad
          <<<block_count, thread_per_block>>>(
                  (const complex64_t*)img_ptr,
                  (const complex64_t*)img_gx_ptr,
                  (const complex64_t*)img_gy_ptr,
                  (const complex64_t*)img_gxy_ptr,
                  dorigin_ptr,
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
                  dorigin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(img_gx_grad);
    ret.push_back(img_gy_grad);
    ret.push_back(img_gxy_grad);
    ret.push_back(dorigin_grad);
	return ret;
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
          int64_t Ntheta1) {
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
	dim3 block_count = {block_x, nbatch, 1};

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
                  Ntheta1
                  );
	return out;
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
          int64_t Ntheta1) {
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
	dim3 block_count = {block_x, nbatch, 1};

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
                  (complex64_t*)grad_ptr,
                  (complex64_t*)img_grad_ptr,
                  dorigin_grad_ptr
                  );
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(dorigin_grad);
	return ret;
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cuda);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cuda);
  m.impl("backprojection_cart_2d", &backprojection_cart_2d_cuda);
  m.impl("backprojection_cart_2d_grad", &backprojection_cart_2d_grad_cuda);
  m.impl("gpga_backprojection_2d", &gpga_backprojection_2d_cuda);
  m.impl("cfar_2d", &cfar_2d_cuda);
  m.impl("polar_interp_linear", &polar_interp_linear_cuda);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cuda);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cuda);
  m.impl("polar_to_cart_linear_grad", &polar_to_cart_linear_grad_cuda);
  m.impl("polar_to_cart_bicubic", &polar_to_cart_bicubic_cuda);
  m.impl("polar_to_cart_bicubic_grad", &polar_to_cart_bicubic_grad_cuda);
  m.impl("backprojection_polar_2d_tx_power", &backprojection_polar_2d_tx_power_cuda);
  m.impl("entropy", &entropy_cuda);
  m.impl("entropy_grad", &entropy_grad_cuda);
  m.impl("abs_sum", &abs_sum_cuda);
  m.impl("abs_sum_grad", &abs_sum_grad_cuda);
}

}
