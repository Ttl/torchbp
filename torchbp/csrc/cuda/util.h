#ifndef _TORCHBP_UTIL_CUH
#define _TORCHBP_UTIL_CUH
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
__device__ T interp1d_cubic(const T *y, int nx,
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
__device__ T interp1d_cubic_x_frac_grad(const T *y, int nx,
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
__device__ T interp2d(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return img[x_int*ny + y_int]*(1.0f-x_frac)*(1.0f-y_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac)*y_frac +
           img[(x_int+1)*ny + y_int]*x_frac*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*x_frac*y_frac;
}

template<class T>
__device__ T interp2d_gradx(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-y_frac) +
           -img[x_int*ny + y_int+1]*y_frac +
           img[(x_int+1)*ny + y_int]*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*y_frac;
}

template<class T>
__device__ T interp2d_grady(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-x_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac) +
           -img[(x_int+1)*ny + y_int]*x_frac +
           img[(x_int+1)*ny + y_int+1]*x_frac;
}

template<class T>
__device__ float interp2d_abs(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return abs(img[x_int*ny + y_int])*(1.0f-x_frac)*(1.0f-y_frac) +
           abs(img[x_int*ny + y_int+1])*(1.0f-x_frac)*y_frac +
           abs(img[(x_int+1)*ny + y_int])*x_frac*(1.0f-y_frac) +
           abs(img[(x_int+1)*ny + y_int+1])*x_frac*y_frac;
}

template<class T>
__device__ T bicubic_interp2d(const T *img, const T *gx, const T *gy,
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
__device__ T bicubic_interp2d_gradx(const T *img, const T *gx, const T *gy,
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
    const float y2 = y*y;
    const float y3 = y*y*y;
    const float xb[] = {-6.0f*x + 6.0f*x2, 6.0f*x-6.0f*x2, 1-4.0f*x+3.0f*x2, -2.0f*x+3.0f*x2};
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
__device__ T bicubic_interp2d_grady(const T *img, const T *gx, const T *gy,
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
    const float xb[] = {1.0f - 3.0f*x2 + 2.0f*x3, 3.0f*x2-2.0f*x3, x-2.0f*x2+x3, -x2+x3};
    const float by[] = {-6.0f*y + 6.0f*y2, 6.0f*y-6.0f*y2, 1-4.0f*y+3.0f*y2, -2.0f*y+3.0f*y2};
    T v = 0.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            v += xb[i] * by[j] * f[i*4 + j];
        }
    }
    return v;
}

template<class T>
__device__ void bicubic_interp2d_grad(T *img_grad, T *img_gx_grad,
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

inline __device__ float lanczos_kernel(float x, int a) {
    // Ensured by calling code
    //if (fabsf(x) >= a) {
    //    return 0.0f;
    //}
    if (x == 0.0f) {
        return 1.0f;
    }
    return sinpif(x) / (kPI * x) * sinpif(x/a) / (kPI * x / a);
}

template<class T, class T2>
__device__ T lanczos_interp_1d(const T2 *img, int n, float pos, int a) {
    int start = max(0, (int)ceilf(pos - a));
    int end = min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = lanczos_kernel(dx, a);
        T val;
        if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
            half2 val_h = ((half2*)img)[i];
            val = {__half2float(val_h.x), __half2float(val_h.y)};
        } else {
            val = img[i];
        }
        sum += w * val;
    }
    return sum;
}

template<class T, class T2>
__device__ T lanczos_interp_2d(const T2 *img, int nx, int ny, float x, float y, int a) {
    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = lanczos_kernel(dx, a);
        T row_val = lanczos_interp_1d<T, T2>(img + i * ny, ny, y, a);
        sum += wx * row_val;
    }
    return sum;
}

inline __device__ float knab_kernel_norm(int a, float v) {
    return 1.0f / coshf(kPI * v * a);
}

inline __device__ float knab_kernel(float x, int a, float v, float norm) {
    // This is needed due to rounding errors.
    if (fabsf(x) >= a) {
        return 0.0f;
    }
    if (x == 0.0f) {
        return 1.0f;
    }
    float xa = x / a;
    return (sinpif(x) / (kPI * x)) * coshf(kPI * v * a * sqrtf(1.0f - xa*xa)) * norm;
}

template<class T, class T2>
__device__ T knab_interp_1d(const T2 *img, int n, float pos, int a, float v, float norm) {
    int start = max(0, (int)ceilf(pos - a));
    int end = min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = knab_kernel(dx, a, v, norm);
        T val;
        if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
            half2 val_h = ((half2*)img)[i];
            val = {__half2float(val_h.x), __half2float(val_h.y)};
        } else {
            val = img[i];
        }
        sum += w * val;
    }
    return sum;
}

template<class T, class T2>
__device__ T knab_interp_2d(const T2 *img, int nx, int ny, float x, float y, int a, float v, float norm) {
    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel(dx, a, v, norm);
        T row_val = knab_interp_1d<T, T2>(img + i * ny, ny, y, a, v, norm);
        sum += wx * row_val;
    }
    return sum;
}

}
#endif
