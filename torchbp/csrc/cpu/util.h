#ifndef _TORCHBP_CPU_UTIL_H
#define _TORCHBP_CPU_UTIL_H
// Shared CPU helpers (defines, complex operators, bilinear / lanczos /
// knab / polynomial interpolation, sincospi). CPU analogue of cuda/util.h.
#include <ATen/Operators.h>
#include <ATen/ops/fft_ifft.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <map>
#include <mutex>
#include <tuple>
#include <omp.h>

namespace torchbp {

#define kPI 3.1415926535897932384626433f
#define kC0 299792458.0f

using complex64_t = c10::complex<float>;

inline c10::complex<double> operator * (const float &a, const c10::complex<double> &b){
    return c10::complex<double>(b.real() * (double)a, b.imag() * (double)a);
}

inline c10::complex<double> operator * (const c10::complex<double> &b, const float &a){
    return c10::complex<double>(b.real() * (double)a, b.imag() * (double)a);
}

template<class T>
static T interp2d(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return img[x_int*ny + y_int]*(1.0f-x_frac)*(1.0f-y_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac)*y_frac +
           img[(x_int+1)*ny + y_int]*x_frac*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*x_frac*y_frac;
}

template<class T>
static T interp2d_gradx(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-y_frac) +
           -img[x_int*ny + y_int+1]*y_frac +
           img[(x_int+1)*ny + y_int]*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*y_frac;
}

template<class T>
static T interp2d_grady(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-x_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac) +
           -img[(x_int+1)*ny + y_int]*x_frac +
           img[(x_int+1)*ny + y_int+1]*x_frac;
}

template <typename T>
static void sincospi(T x, T *sinx, T *cosx) {
    if constexpr (std::is_same_v<T, float>) {
        // Fast float (sin(pi*x), cos(pi*x)). Working in "pi units" makes range
        // reduction exact and cheap: the period in x is 2, so a single rintf
        // + subtract maps x to r in [-0.5, 0.5].
        // CPU analogue of CUDA's __sincosf.
        const float k = rintf(x); // nearest integer, ties to even
        const float r = x - k;    // r in [-0.5, 0.5]
        const float u = r * r;
        // sinpi(r) = r * Ps(r^2)
        float s = 0.0778885794857989f;
        s = s * u - 0.5983952894635156f;
        s = s * u + 2.550091904145188f;
        s = s * u - 5.167710704395367f;
        s = s * u + 3.1415926441706428f;
        s *= r;
        // cospi(r) = Pc(r^2)
        float c = 0.22049100664642599f;
        c = c * u - 1.3322375115287677f;
        c = c * u + 4.058461261992171f;
        c = c * u - 4.934794985432785f;
        c = c * u + 0.9999999672684953f;
        // Odd integer part of x flips the sign of both sinpi and cospi.
        // Branchless so the function stays vectorizable when inlined into
        // SIMD loops (and the branch is data-dependent, i.e. unpredictable).
        const float sgn = 1.0f - 2.0f * (float)(((int)k) & 1);
        *sinx = s * sgn;
        *cosx = c * sgn;
    } else {
        *sinx = sin(static_cast<T>(kPI) * x);
        *cosx = cos(static_cast<T>(kPI) * x);
    }
}

// Branchless float asin (cephes single-precision coefficients), a few ulp
// from libm asinf. No calls or branches, so it vectorizes when inlined into
// SIMD loops. libm asinf blocks vectorization. |x| > 1 returns NaN.
static inline float asinf_fast(float x) {
    const float a = fabsf(x);
    const bool big = a > 0.5f;
    const float z1 = 0.5f * (1.0f - a);
    const float z = big ? z1 : a * a;
    const float w = big ? sqrtf(z1) : a;
    float p = 4.2163199048e-2f;
    p = p * z + 2.4181311049e-2f;
    p = p * z + 4.5470025998e-2f;
    p = p * z + 7.4953002686e-2f;
    p = p * z + 1.6666752422e-1f;
    p = p * z * w + w;
    p = big ? 0.5f * kPI - 2.0f * p : p;
    return copysignf(p, x);
}

// Branchless float atan2 (cephes single-precision polynomial), a few ulp from
// libm atan2f. Vectorizable like asinf_fast. atan2f_fast(0, 0) is NaN where
// libm returns 0.
static inline float atan2f_fast(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y);
    // a = tan(angle folded to [0, pi/4]), in [0, 1]. Ternary min/max instead
    // of fminf/fmaxf: the libm calls have NaN semantics that don't map to
    // vmin/vmaxps, which blocks vectorization.
    const float mn = ax < ay ? ax : ay, mx = ax < ay ? ay : ax;
    const float a = mn / mx;
    // Cephes-style second reduction to |t| <= tan(pi/8)
    const bool red = a > 0.4142135623730950f;
    const float t = red ? (a - 1.0f) / (a + 1.0f) : a;
    const float z = t * t;
    float r = 8.05374449538e-2f;
    r = r * z - 1.38776856032e-1f;
    r = r * z + 1.99777106478e-1f;
    r = r * z - 3.33329491539e-1f;
    r = r * z * t + t;
    r = red ? r + 0.25f * kPI : r;
    // Undo the min/max fold and the quadrant fold
    r = ay > ax ? 0.5f * kPI - r : r;
    r = x < 0.0f ? kPI - r : r;
    return copysignf(r, y);
}

static inline float sinpi_f(float x) {
    float s, c;
    sincospi<float>(x, &s, &c);
    return s;
}

static inline float lanczos_kernel_cpu(float x, float a) {
    if (x == 0.0f) {
        return 1.0f;
    }
    return sinpi_f(x) / (kPI * x) * sinpi_f(x/a) / (kPI * x / a);
}

template<class T>
static T lanczos_interp_1d_cpu(const T *img, int n, float pos, int order) {
    float a = 0.5f * order;
    int start = std::max(0, (int)ceilf(pos - a));
    int end = std::min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = lanczos_kernel_cpu(dx, a);
        sum += img[i] * w;
    }
    return sum;
}

template<class T>
static T lanczos_interp_2d_cpu(const T *img, int nx, int ny, float x, float y, int order) {
    float a = 0.5f * order;
    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = lanczos_kernel_cpu(dx, a);
        T row_val = lanczos_interp_1d_cpu<T>(img + i * ny, ny, y, order);
        sum += row_val * wx;
    }
    return sum;
}

static inline float knab_kernel_norm_cpu(int order, float v) {
    float a = 0.5f * order;
    return expf(-2.0f*a*kPI*v);
}

static inline float knab_kernel_cpu(float x, float a, float v, float norm) {
    if (fabsf(x) >= a) {
        return 0.0f;
    }
    if (x == 0.0f) {
        return 1.0f;
    }
    float xa = x / a;
    float n = expf(kPI * a * v * (sqrtf(1.0f - xa*xa) - 1.0f));
    return (sinpi_f(x) / (kPI * x)) * (norm/(n*(norm + 1.0f)) + n/(norm + 1.0f));
}

template<class T>
static T knab_interp_1d_cpu(const T *img, int n, float pos, int order, float v, float norm) {
    float a = 0.5f * order;
    int start = std::max(0, (int)ceilf(pos - a));
    int end = std::min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = knab_kernel_cpu(dx, a, v, norm);
        sum += img[i] * w;
    }
    return sum;
}

template<class T>
static T knab_interp_2d_cpu(const T *img, int nx, int ny, float x, float y, int order, float v, float norm) {
    float a = 0.5f * order;
    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel_cpu(dx, a, v, norm);
        T row_val = knab_interp_1d_cpu<T>(img + i * ny, ny, y, order, v, norm);
        sum += row_val * wx;
    }
    return sum;
}

// Evaluates 1 + c1*x + c2*x^2 + ... + cn*x^n using Horner's method.
static inline float polyval_c0_one_cpu(const float *coefs, int n_coefs, float x) {
    float inner = coefs[n_coefs - 1];
    for (int i = n_coefs - 2; i >= 0; i--) {
        inner = inner * x + coefs[i];
    }
    return x * inner + 1.0f;
}

static inline float poly_interp_kernel_cpu(const float *coefs, int n_coefs, float x, float inv_a2) {
    float x2 = x * x;
    float t = x2 * inv_a2;
    return polyval_c0_one_cpu(coefs, n_coefs, t);
}

template<class T>
static T interp_2d_poly_cpu(const T *img, int nx, int ny, float x, float y,
        int order, const float *coefs, int n_coefs) {
    float a = 0.5f * order;
    float inv_a2 = 1.0f / (a * a);

    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    int start_y = std::max(0, (int)ceilf(y - a));
    int end_y = std::min(ny-1, (int)floorf(y + a));

    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - (float)i;
        float wx = poly_interp_kernel_cpu(coefs, n_coefs, dx, inv_a2);
        const T *row = img + i * ny;
        for (int j = start_y; j <= end_y; j++) {
            float dy = y - (float)j;
            float wy = poly_interp_kernel_cpu(coefs, n_coefs, dy, inv_a2);
            sum += row[j] * (wx * wy);
        }
    }
    return sum;
}

}
#endif
