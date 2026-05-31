#include <Python.h>
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

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace torchbp {

#define kPI 3.1415926535897932384626433f
#define kC0 299792458.0f

using complex64_t = c10::complex<float>;

c10::complex<double> operator * (const float &a, const c10::complex<double> &b){
    return c10::complex<double>(b.real() * (double)a, b.imag() * (double)a);
}

c10::complex<double> operator * (const c10::complex<double> &b, const float &a){
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
        if (((int)k) & 1) { s = -s; c = -c; }
        *sinx = s;
        *cosx = c;
    } else {
        *sinx = sin(static_cast<T>(kPI) * x);
        *cosx = cos(static_cast<T>(kPI) * x);
    }
}

template <typename T>
static void polar_interp_kernel_linear_cpu(const c10::complex<T> *img, c10::complex<T> *out, const T *dorigin, T rotation,
                  T ref_phase, T r0, T dr, T theta0, T dtheta, int Nr, int Ntheta,
                  T r1, T dr1, T theta1, T dtheta1, int Nr1, int Ntheta1, T z1, T alias_fmod, int idx, int idbatch) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const T d = r1 + dr1 * idr;
    T t = theta1 + dtheta1 * idtheta;
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const T dorig0 = dorigin[idbatch * 3 + 0];
    const T dorig1 = dorigin[idbatch * 3 + 1];
    const T sint = t;
    const T cost = sqrt(1.0f - t*t);
    const T rp = sqrt(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const T arg = (d*sint + dorig1) / (d*cost + dorig0);
    const T tp = arg / sqrt(1.0f + arg*arg);

    const T dri = (rp - r0) / dr;
    const T dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const T dri_frac = dri - dri_int;
    const int dti_int = dti;
    const T dti_frac = dti - dti_int;

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        c10::complex<T> v = interp2d<c10::complex<T>>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        T ref_sin, ref_cos;
        const T z0 = z1 + dorigin[idbatch * 3 + 2];
        const T dz = sqrt(z1*z1 + d*d);
        const T rpz = sqrt(z0*z0 + rp*rp);
        sincospi<T>(ref_phase * (rpz - dz)  - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
        c10::complex<T> ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

template <typename T>
static void polar_interp_kernel_linear_grad_cpu(const c10::complex<T> *img, const T *dorigin, T rotation,
                  T ref_phase, T r0, T dr, T theta0, T dtheta, int Nr, int Ntheta,
                  T r1, T dr1, T theta1, T dtheta1, int Nr1, int Ntheta1, T z1, T alias_fmod,
                  const c10::complex<T> *grad, c10::complex<T> *img_grad, T *dorigin_grad,
                  int idx, int idbatch) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;
    c10::complex<T> I = {0.0f, 1.0f};

    const T d = r1 + dr1 * idr;
    T t = theta1 + dtheta1 * idtheta;
    if (t > 1.0f) {
        t = 1.0f;
    }
    if (rotation != 0.0f) {
        t = sinf(asinf(t) - rotation);
    }

    if (idx >= Nr1 * Ntheta1) {
        return;
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const T dorig0 = dorigin[idbatch * 3 + 0];
    const T dorig1 = dorigin[idbatch * 3 + 1];
    const T dorig2 = dorigin[idbatch * 3 + 2];
    const T sint = t;
    const T cost = sqrt(1.0f - t*t);
    // TODO: Add dorig2
    const T rp = sqrt(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const T arg = (d*sint + dorig1) / (d*cost + dorig0);
    const T cosarg = sqrt(1.0f + arg*arg);
    const T tp = arg / cosarg;

    const T dri = (rp - r0) / dr;
    const T dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const T dri_frac = dri - dri_int;

    const int dti_int = dti;
    const T dti_frac = dti - dti_int;

    c10::complex<T> v = {0.0f, 0.0f};
    c10::complex<T> ref = {0.0f, 0.0f};

    const T z0 = z1 + dorig2;
    const T rpz = sqrt(z0*z0 + rp*rp);
    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        v = interp2d<c10::complex<T>>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        T ref_sin, ref_cos;
        const T dz = sqrt(z1*z1 + d*d);
        sincospi<T>(ref_phase * (rpz - dz) - alias_fmod * (dri - idr), &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    if (dorigin_grad != nullptr) {
        const c10::complex<T> dref_drpz = I * kPI * ref_phase * ref;
        const c10::complex<T> dref_ddri = -I * kPI * alias_fmod * ref;
        const c10::complex<T> dv_drp = interp2d_gradx<c10::complex<T>>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                dti_int, dti_frac) / dr;
        const c10::complex<T> dv_dt = interp2d_grady<c10::complex<T>>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac,
                dti_int, dti_frac) / dtheta;
        const T drp_dorig0 = (cost*d + dorig0) / rp;
        const T drp_dorig1 = (sint*d + dorig1) / rp;
        const T drpz_dorig0 = (cost*d + dorig0) / rpz;
        const T drpz_dorig1 = (sint*d + dorig1) / rpz;
        const T drpz_dorig2 = (dorig2 + z1) / rpz;
        const T dt_darg = -arg*arg/(cosarg*cosarg*cosarg) + 1.0f / cosarg;
        const T darg_dorig0 = -(d*sint + dorig1) / ((dorig0 + d*cost)*(dorig0 + d*cost));
        const T darg_dorig1 = 1.0f / (cost*d + dorig0);

        const c10::complex<T> g = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta];
        const c10::complex<T> dout_dorig0 = ref * (dv_drp * drp_dorig0 + dv_dt * dt_darg * darg_dorig0) + v * dref_drpz * drpz_dorig0 + v * dref_ddri * drp_dorig0 / dr;
        const c10::complex<T> dout_dorig1 = ref * (dv_drp * drp_dorig1 + dv_dt * dt_darg * darg_dorig1) + v * dref_drpz * drpz_dorig1 + v * dref_ddri * drp_dorig1 / dr;
        const c10::complex<T> dout_dorig2 = v * dref_drpz * drpz_dorig2;
        T g_dorig0 = std::real(g * std::conj(dout_dorig0));
        T g_dorig1 = std::real(g * std::conj(dout_dorig1));
        T g_dorig2 = std::real(g * std::conj(dout_dorig2));

#pragma omp atomic
        dorigin_grad[idbatch * 3 + 0] += g_dorig0;
#pragma omp atomic
        dorigin_grad[idbatch * 3 + 1] += g_dorig1;
#pragma omp atomic
        dorigin_grad[idbatch * 3 + 2] += g_dorig2;
    }

    if (img_grad != nullptr) {
        if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
            c10::complex<T> g = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] * std::conj(ref);

            c10::complex<T> g11 = g * (1.0f-dri_frac)*(1.0f-dti_frac);
            c10::complex<T> g12 = g * (1.0f-dri_frac)*dti_frac;
            c10::complex<T> g21 = g * dri_frac*(1.0f-dti_frac);
            c10::complex<T> g22 = g * dri_frac*dti_frac;
            // Slow
            #pragma omp critical
            {
                img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int] += g11;
                img_grad[idbatch * Nr * Ntheta + dri_int*Ntheta + dti_int + 1] += g12;
                img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int] += g21;
                img_grad[idbatch * Nr * Ntheta + (dri_int+1)*Ntheta + dti_int + 1] += g22;
            }
        }
    }
}

at::Tensor polar_interp_linear_cpu(
          const at::Tensor &img,
          const at::Tensor &dorigin,
          int64_t nbatch,
          double rotation,
          double fc,
          double r0,
          double dr0,
          double theta0,
          double dtheta0,
          int64_t nr0,
          int64_t ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t nr1,
          int64_t ntheta1,
          double z1,
          double alias_fmod) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kComplexDouble);
    TORCH_CHECK(dorigin.dtype() == at::kFloat || dorigin.dtype() == at::kDouble);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, nr1, ntheta1}, img_contig.options());
    at::Tensor dorigin_contig = dorigin.contiguous();

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

    if (img.dtype() == at::kComplexFloat) {
        TORCH_CHECK(dorigin.dtype() == at::kFloat);
        const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
        c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();
        c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
        const float ref_phase = 4.0f * fc / kC0;

#pragma omp parallel for collapse(2)
        for(int idbatch = 0; idbatch < nbatch; idbatch++) {
            for(int idx = 0; idx < nr1 * ntheta1; idx++) {
                polar_interp_kernel_linear_cpu<float>(img_ptr, out_ptr, dorigin_ptr, rotation,
                      ref_phase, r0, dr0, theta0, dtheta0, nr0, ntheta0,
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, z1, alias_fmod/kPI, idx, idbatch);
            }
        }
    } else {
        TORCH_CHECK(dorigin.dtype() == at::kDouble);
        const double* dorigin_ptr = dorigin_contig.data_ptr<double>();
        c10::complex<double>* img_ptr = img.data_ptr<c10::complex<double>>();
        c10::complex<double>* out_ptr = out.data_ptr<c10::complex<double>>();
        const double ref_phase = 4.0 * fc / kC0;

#pragma omp parallel for collapse(2)
        for(int idbatch = 0; idbatch < nbatch; idbatch++) {
            for(int idx = 0; idx < nr1 * ntheta1; idx++) {
                polar_interp_kernel_linear_cpu<double>(img_ptr, out_ptr, dorigin_ptr, rotation,
                      ref_phase, r0, dr0, theta0, dtheta0, nr0, ntheta0,
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, z1, alias_fmod/kPI, idx, idbatch);
            }
        }
    }
	return out;
}

std::vector<at::Tensor> polar_interp_linear_grad_cpu(
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
          int64_t nr0,
          int64_t ntheta0,
          double r1,
          double dr1,
          double theta1,
          double dtheta1,
          int64_t nr1,
          int64_t ntheta1,
          double z1,
          double alias_fmod) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kComplexDouble);
    TORCH_CHECK(dorigin.dtype() == at::kFloat || dorigin.dtype() == at::kDouble);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat || grad.dtype() == at::kComplexDouble);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor grad_contig = grad.contiguous();
    at::Tensor img_grad;
    at::Tensor dorigin_grad;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

    if (img.dtype() == at::kComplexFloat) {
        TORCH_CHECK(dorigin.dtype() == at::kFloat);
        TORCH_CHECK(grad.dtype() == at::kComplexFloat);
        const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
        c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();
        c10::complex<float>* grad_ptr = grad.data_ptr<c10::complex<float>>();
        c10::complex<float>* img_grad_ptr = nullptr;
        if (img.requires_grad()) {
            img_grad = torch::zeros_like(img);
            img_grad_ptr = img_grad.data_ptr<c10::complex<float>>();
        } else {
            img_grad = torch::Tensor();
        }

        float* dorigin_grad_ptr = nullptr;
        if (dorigin.requires_grad()) {
            dorigin_grad = torch::zeros_like(dorigin);
            dorigin_grad_ptr = dorigin_grad.data_ptr<float>();
        } else {
            dorigin_grad = torch::Tensor();
        }

        const float ref_phase = 4.0f * fc / kC0;

#pragma omp parallel for collapse(2)
        for(int idbatch = 0; idbatch < nbatch; idbatch++) {
            for(int idx = 0; idx < nr1 * ntheta1; idx++) {
                polar_interp_kernel_linear_grad_cpu<float>(img_ptr, dorigin_ptr, rotation,
                      ref_phase, r0, dr0, theta0, dtheta0, nr0, ntheta0,
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, z1, alias_fmod/kPI,
                      grad_ptr, img_grad_ptr, dorigin_grad_ptr,
                      idx, idbatch);
            }
        }
    } else {
        TORCH_CHECK(dorigin.dtype() == at::kDouble);
        TORCH_CHECK(grad.dtype() == at::kComplexDouble);
        const double* dorigin_ptr = dorigin_contig.data_ptr<double>();
        c10::complex<double>* img_ptr = img.data_ptr<c10::complex<double>>();
        c10::complex<double>* grad_ptr = grad.data_ptr<c10::complex<double>>();
        c10::complex<double>* img_grad_ptr = nullptr;
        if (img.requires_grad()) {
            img_grad = torch::zeros_like(img);
            img_grad_ptr = img_grad.data_ptr<c10::complex<double>>();
        } else {
            img_grad = torch::Tensor();
        }

        double* dorigin_grad_ptr = nullptr;
        if (dorigin.requires_grad()) {
            dorigin_grad = torch::zeros_like(dorigin);
            dorigin_grad_ptr = dorigin_grad.data_ptr<double>();
        } else {
            dorigin_grad = torch::Tensor();
        }

        const double ref_phase = 4.0 * fc / kC0;

#pragma omp parallel for collapse(2)
        for(int idbatch = 0; idbatch < nbatch; idbatch++) {
            for(int idx = 0; idx < nr1 * ntheta1; idx++) {
                polar_interp_kernel_linear_grad_cpu<double>(img_ptr, dorigin_ptr, rotation,
                      ref_phase, r0, dr0, theta0, dtheta0, nr0, ntheta0,
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, z1, alias_fmod/kPI,
                      grad_ptr, img_grad_ptr, dorigin_grad_ptr,
                      idx, idbatch);
            }
        }
    }

    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(dorigin_grad);
	return ret;
}

static void backprojection_polar_2d_kernel_cpu(
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
          bool dealias,
          float z0,
          const float *g,
          float g_az0,
          float g_el0,
          float g_daz,
          float g_del,
          int g_naz,
          int g_nel,
          float data_fmod,
          float alias_fmod,
          int idx,
          int idbatch) {
    const int idtheta = idx % Ntheta;
    const int idr = idx / Ntheta;
    if (idr >= Nr || idtheta >= Ntheta) {
        return;
    }

    const float r = r0 + idr * dr;
    const float theta = theta0 + idtheta * dtheta;
    const float x = r * sqrtf(1.0f - theta*theta);
    const float y = r * theta;

    complex64_t pixel{};
    float w_sum = 0.0f;

    for(int i = 0; i < nsweeps; i++) {
        // Sweep reference position.
        float pos_x = pos[idbatch * nsweeps * 3 + i * 3 + 0];
        float pos_y = pos[idbatch * nsweeps * 3 + i * 3 + 1];
        float pos_z = pos[idbatch * nsweeps * 3 + i * 3 + 2];
        float px = (x - pos_x);
        float py = (y - pos_y);
        float pz2 = pos_z * pos_z;

        // Calculate distance to the pixel.
        const float d = sqrtf(px * px + py * py + pz2);

        float sx = delta_r * (d + d0);

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (id0 < 0 || id1 >= sweep_samples) {
            continue;
        }
        complex64_t s0 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0];
        complex64_t s1 = data[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1];

        float interp_idx = sx - id0;
        complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

        float ref_sin, ref_cos;
        sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};

        if (g != nullptr) {
            const float look_angle = asinf(fmaxf(-pos_z / d, -1.0f));
            const float el_deg = look_angle - att[idbatch * nsweeps * 3 + 3 * i + 0];
            const float az_deg = atan2f(py, px) - att[idbatch * nsweeps * 3 + 3 * i + 2];

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
            const float w = interp2d<float>(g, g_naz, g_nel, az_int, az_frac, el_int, el_frac);

            pixel += w * s * ref;
            w_sum += w;
        } else {
            pixel += s * ref;
        }
    }
    if (g != nullptr) {
        if (w_sum > 0.0f) {
            pixel *= nsweeps / w_sum;
        }
    }
    if (dealias) {
        const float d = sqrtf(x*x + y*y + z0*z0);
        float ref_sin, ref_cos;
        sincospi(-ref_phase * d + alias_fmod * idr, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        pixel *= ref;
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = pixel;
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
          bool normalize) {
	TORCH_CHECK(pos.dtype() == at::kFloat);
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);

    bool antenna_pattern = g.defined() || att.defined();
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
    const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nr * Ntheta; idx++) {
            backprojection_polar_2d_kernel_cpu(
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
                          data_fmod/kPI,
                          alias_fmod/kPI,
                          idx, idbatch);
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
    const float x = r * sqrtf(1.0f - theta*theta);
    const float y = r * theta;

    complex64_t g = grad[idbatch * Nr * Ntheta + idr * Ntheta + idtheta];

    float arg_dealias = 0.0f;
    if (dealias) {
        const float d = sqrtf(x*x + y*y + z0*z0);
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
                if (!isfinite(dx)) dx = 0.0f;
                if (!isfinite(dy)) dy = 0.0f;
                if (!isfinite(dz)) dz = 0.0f;
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
            if (id0 >= 0 && id1 < sweep_samples) {
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
          bool normalize) {
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

static void backprojection_cart_2d_kernel_cpu(
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
          float data_fmod,
          int idt,
          int idbatch) {
    const int idy = idt % Ny;
    const int idx = idt / Ny;

    if (idx >= Nx || idy >= Ny) {
        return;
    }

    const float x = x0 + idx * dx;
    const float y = y0 + idy * dy;

    complex64_t pixel{};

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

        // Linear interpolation.
        int id0 = sx;
        int id1 = id0 + 1;
        if (id0 >= 0 && id1 < sweep_samples) {
            complex64_t s0 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id0];
            complex64_t s1 = data[idbatch * nsweeps * sweep_samples + i * sweep_samples + id1];

            float interp_idx = sx - id0;
            complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

            float ref_sin, ref_cos;
            sincospi(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += s * ref;
        }
    }
    img[idbatch * Nx * Ny + idx * Ny + idy] = pixel;
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

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idt = 0; idt < Nx * Ny; idt++) {
            backprojection_cart_2d_kernel_cpu(
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
                          beamwidth_f, d0,
                          data_fmod/kPI,
                          idt, idbatch);
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
        if (id0 >= 0 && id1 < sweep_samples) {
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
                if (!isfinite(gx)) gx = 0.0f;
                if (!isfinite(gy)) gy = 0.0f;
                if (!isfinite(gz)) gz = 0.0f;
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
            if (id0 >= 0 && id1 < sweep_samples) {
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

template<typename T>
static void polar_to_cart_kernel_linear_cpu(const T *img, T
        *out, const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, float alias_fmod,
        int id1, int idbatch) {
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;

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
        if constexpr (std::is_same_v<T, complex64_t>) {
            float ref_sin, ref_cos;
            sincospi(ref_phase * dz - alias_fmod * dri, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            out[idbatch * Nx * Ny + idx*Ny + idy] = v * ref;
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = v;
        }
    } else {
        if constexpr (std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx*Ny + idy] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = 0.0f;
        }
    }
}

at::Tensor polar_to_cart_linear_cpu(
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
          double alias_fmod) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CPU);
	at::Tensor origin_contig = origin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* origin_ptr = origin_contig.data_ptr<float>();

    const float ref_phase = 4.0f * fc / kC0;

    // Torch sets the OpenMP team size for this thread to 1 to avoid nested
    // parallelism inside its own intra-op pool, so a bare "omp parallel for"
    // would run single threaded. Re-enable threading explicitly. Use the full
    // logical processor count (incl. SMT/hyperthreads).
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int id1 = 0; id1 < Nx * Ny; id1++) {
            if (img.dtype() == at::kComplexFloat) {
                c10::complex<float>* img_ptr = img_contig.data_ptr<c10::complex<float>>();
                c10::complex<float>* out_ptr = out.data_ptr<c10::complex<float>>();
                polar_to_cart_kernel_linear_cpu<complex64_t>(
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
                              alias_fmod/kPI,
                              id1,
                              idbatch
                              );
            } else {
                float* img_ptr = img_contig.data_ptr<float>();
                float* out_ptr = out.data_ptr<float>();
                polar_to_cart_kernel_linear_cpu<float>(
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
                              alias_fmod/kPI,
                              id1,
                              idbatch
                              );
            }
        }
    }
	return out;
}

// ---------------------------------------------------------------------------
// CPU implementations of the lanczos / knab / polynomial interpolation kernels
// and the FFBP merge operators. These mirror the CUDA versions in
// cuda/polar_interp.cu and cuda/util.h so that ffbp() and polar_to_cart()
// (lanczos / antenna-pattern paths) work on the CPU as well.
// ---------------------------------------------------------------------------

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

// Applies the optional output-alias phase common to all merge kernels.
static inline void apply_merge_alias(complex64_t &pixel, int alias,
        float ref_phase, float alias_fmod, float dz, int idr) {
    if (!alias) {
        return;
    }
    float af = alias_fmod;
    float rp = ref_phase;
    if (alias == 2) {
        af = 0.0f;
    }
    if (alias == 3) {
        rp = 0.0f;
    }
    float ref_sin, ref_cos;
    sincospi<float>(rp * dz - af * idr, &ref_sin, &ref_cos);
    complex64_t ref = {ref_cos, ref_sin};
    pixel *= ref;
}

static void ffbp_merge2_kernel_lanczos_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, int alias, float alias_fmod,
        int idx) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;
    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    const float dz = sqrtf(z1*z1 + d*d);

    complex64_t pixel{};

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = lanczos_interp_2d_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorigin[id * 3 + 2];
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospi<float>(ref_phase * (rpz - dz) - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += v * ref;
        }
    }
    apply_merge_alias(pixel, alias, ref_phase, alias_fmod, dz, idr);
    out[idr*Ntheta1 + idtheta] = pixel;
}

static void ffbp_merge2_kernel_knab_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, float knab_v, int alias,
        float alias_fmod, int idx) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;
    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    const float dz = sqrtf(z1*z1 + d*d);

    complex64_t pixel{};
    const float knab_norm = knab_kernel_norm_cpu(order, knab_v);

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = knab_interp_2d_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order, knab_v, knab_norm);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorigin[id * 3 + 2];
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospi<float>(ref_phase * (rpz - dz) - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += v * ref;
        }
    }
    apply_merge_alias(pixel, alias, ref_phase, alias_fmod, dz, idr);
    out[idr*Ntheta1 + idtheta] = pixel;
}

static void ffbp_merge2_kernel_poly_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, const float *coefs, int n_coefs,
        int alias, float alias_fmod, int idx) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;
    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    const float dz = sqrtf(z1*z1 + d*d);

    complex64_t pixel{};

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = interp_2d_poly_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order, coefs, n_coefs);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorigin[id * 3 + 2];
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospi<float>(ref_phase * (rpz - dz) - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            pixel += v * ref;
        }
    }
    apply_merge_alias(pixel, alias, ref_phase, alias_fmod, dz, idr);
    out[idr*Ntheta1 + idtheta] = pixel;
}

at::Tensor ffbp_merge2_lanczos_cpu(
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
          int64_t alias,
          double alias_fmod) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_CHECK(dorigin.dtype() == at::kFloat);
    TORCH_CHECK(r0.dtype() == at::kFloat);
    TORCH_CHECK(Nr0.dtype() == at::kInt);
    TORCH_CHECK(Ntheta0.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
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
    const complex64_t* img0_ptr = img0_contig.data_ptr<complex64_t>();
    const complex64_t* img1_ptr = img1_contig.data_ptr<complex64_t>();
    complex64_t* out_ptr = out.data_ptr<complex64_t>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();

    const float ref_phase = 4.0f * fc / kC0;

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idx = 0; idx < Nr1 * Ntheta1; idx++) {
        ffbp_merge2_kernel_lanczos_cpu(img0_ptr, img1_ptr, out_ptr, dorigin_ptr,
                ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, alias, alias_fmod/kPI, idx);
    }
    return out;
}

at::Tensor ffbp_merge2_knab_cpu(
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
          double oversample,
          int64_t alias,
          double alias_fmod) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_CHECK(dorigin.dtype() == at::kFloat);
    TORCH_CHECK(r0.dtype() == at::kFloat);
    TORCH_CHECK(Nr0.dtype() == at::kInt);
    TORCH_CHECK(Ntheta0.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
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
    const complex64_t* img0_ptr = img0_contig.data_ptr<complex64_t>();
    const complex64_t* img1_ptr = img1_contig.data_ptr<complex64_t>();
    complex64_t* out_ptr = out.data_ptr<complex64_t>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();

    const float ref_phase = 4.0f * fc / kC0;
    const float v = 1.0f - 1.0f / oversample;

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idx = 0; idx < Nr1 * Ntheta1; idx++) {
        ffbp_merge2_kernel_knab_cpu(img0_ptr, img1_ptr, out_ptr, dorigin_ptr,
                ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, v, alias, alias_fmod/kPI, idx);
    }
    return out;
}

at::Tensor ffbp_merge2_poly_cpu(
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
          const at::Tensor &poly_coefs,
          int64_t alias,
          double alias_fmod) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_CHECK(dorigin.dtype() == at::kFloat);
    TORCH_CHECK(r0.dtype() == at::kFloat);
    TORCH_CHECK(Nr0.dtype() == at::kInt);
    TORCH_CHECK(Ntheta0.dtype() == at::kInt);
    TORCH_CHECK(poly_coefs.dtype() == at::kFloat);
    TORCH_CHECK(order >= 2 && order <= 8,
        "ffbp_merge2_poly requires order in [2, 8], got ", order);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
    at::Tensor dorigin_contig = dorigin.contiguous();
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor r0_contig = r0.contiguous();
    at::Tensor dr0_contig = dr0.contiguous();
    at::Tensor theta0_contig = theta0.contiguous();
    at::Tensor dtheta0_contig = dtheta0.contiguous();
    at::Tensor Nr0_contig = Nr0.contiguous();
    at::Tensor Ntheta0_contig = Ntheta0.contiguous();
    at::Tensor poly_coefs_contig = poly_coefs.contiguous();
    at::Tensor out = torch::empty({Nr1, Ntheta1}, img0_contig.options());
    const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    const complex64_t* img0_ptr = img0_contig.data_ptr<complex64_t>();
    const complex64_t* img1_ptr = img1_contig.data_ptr<complex64_t>();
    complex64_t* out_ptr = out.data_ptr<complex64_t>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();
    const float* coefs_ptr = poly_coefs_contig.data_ptr<float>();
    const int n_coefs = poly_coefs_contig.size(0);

    const float ref_phase = 4.0f * fc / kC0;

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idx = 0; idx < Nr1 * Ntheta1; idx++) {
        ffbp_merge2_kernel_poly_cpu(img0_ptr, img1_ptr, out_ptr, dorigin_ptr,
                ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, coefs_ptr, n_coefs,
                alias, alias_fmod/kPI, idx);
    }
    return out;
}

static void ffbp_merge2_kernel_poly_weighted_cpu(
        const complex64_t *img0, const complex64_t *img1,
        complex64_t *out,
        float *w1_out,
        float *w2_out,
        const float *dorigin,
        float ref_phase,
        const float *r0, const float *dr, const float *theta0, const float *dtheta,
        const int *Nr, const int *Ntheta,
        float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1,
        float z1, int order, const float *coefs, int n_coefs, int alias, float alias_fmod,
        const float *w1_map0, const float *w2_map0,
        float w_r0_0, float w_dr0, float w_theta0_0, float w_dtheta0,
        int w_nr0, int w_ntheta0,
        const float *w1_map1, const float *w2_map1,
        float w_r0_1, float w_dr1, float w_theta0_1, float w_dtheta1,
        int w_nr1, int w_ntheta1,
        int output_weight_decimation, int idx) {

    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    const float d = r1 + dr1 * idr;
    const float t = theta1 + dtheta1 * idtheta;

    const int dec = output_weight_decimation;
    const int out_ntheta_dec = (Ntheta1 + dec - 1) / dec;
    const bool should_write_weight = (w1_out != nullptr) &&
                                     (idr % dec == 0) && (idtheta % dec == 0);
    const int w_out_idx = should_write_weight ?
                          (idr / dec) * out_ntheta_dec + (idtheta / dec) : 0;

    if (t < -1.0f || t > 1.0f) {
        out[idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
        if (should_write_weight) {
            w1_out[w_out_idx] = 0.0f;
            w2_out[w_out_idx] = 0.0f;
        }
        return;
    }

    const float sint = t;
    const float cost = sqrtf(1.0f - t*t);
    const float dz = sqrtf(z1*z1 + d*d);

    complex64_t A_total{};
    float W1_total = 0.0f;
    float W2_total = 0.0f;

    const float *w1_maps[2] = {w1_map0, w1_map1};
    const float *w2_maps[2] = {w2_map0, w2_map1};
    const float w_r0[2] = {w_r0_0, w_r0_1};
    const float w_dr[2] = {w_dr0, w_dr1};
    const float w_theta0[2] = {w_theta0_0, w_theta0_1};
    const float w_dtheta[2] = {w_dtheta0, w_dtheta1};
    const int w_nr[2] = {w_nr0, w_nr1};
    const int w_ntheta[2] = {w_ntheta0, w_ntheta1};

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        const float dorig0 = dorigin[id * 3 + 0];
        const float dorig1 = dorigin[id * 3 + 1];
        const float dorig2 = dorigin[id * 3 + 2];

        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0[id]) / dr[id];
        const float dti = (tp - theta0[id]) / dtheta[id];

        if (dri >= 0 && dri < Nr[id]-1 && dti >= 0 && dti < Ntheta[id]-1) {
            complex64_t v = interp_2d_poly_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti, order, coefs, n_coefs);

            float ref_sin, ref_cos;
            const float z0 = z1 + dorig2;
            const float rpz = sqrtf(z0*z0 + rp*rp);
            sincospi<float>(ref_phase * (rpz - dz) - alias_fmod * (dri - idr), &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            float w1 = 0.0f;
            float w2 = 0.0f;
            if (w1_maps[id] != nullptr && w2_maps[id] != nullptr) {
                const float w_ri = (rp - w_r0[id]) / w_dr[id];
                const float w_ti = (tp - w_theta0[id]) / w_dtheta[id];

                const int w_ri_int = std::max(0, std::min((int)w_ri, w_nr[id] - 2));
                const int w_ti_int = std::max(0, std::min((int)w_ti, w_ntheta[id] - 2));
                const float w_ri_frac = std::max(0.0f, std::min(w_ri - (float)w_ri_int, 1.0f));
                const float w_ti_frac = std::max(0.0f, std::min(w_ti - (float)w_ti_int, 1.0f));

                w1 = interp2d<float>(w1_maps[id], w_nr[id], w_ntheta[id],
                                     w_ri_int, w_ri_frac, w_ti_int, w_ti_frac);
                w2 = interp2d<float>(w2_maps[id], w_nr[id], w_ntheta[id],
                                     w_ri_int, w_ri_frac, w_ti_int, w_ti_frac);
            }

            if (w1 > 0.0f && w2 > 0.0f) {
                A_total += v * ref;
                W1_total += w1;
                W2_total += w2;
            } else if (w1_maps[id] == nullptr) {
                A_total += v * ref;
            }
        }
    }

    complex64_t pixel = A_total;
    if (w1_out == nullptr && W2_total > 0.0f) {
        pixel = A_total * (W1_total / W2_total);
    }

    apply_merge_alias(pixel, alias, ref_phase, alias_fmod, dz, idr);
    out[idr*Ntheta1 + idtheta] = pixel;

    if (should_write_weight) {
        w1_out[w_out_idx] = W1_total;
        w2_out[w_out_idx] = W2_total;
    }
}

std::vector<at::Tensor> ffbp_merge2_poly_weighted_cpu(
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
          const at::Tensor &poly_coefs,
          int64_t alias,
          double alias_fmod,
          const at::Tensor &w1_map0,
          const at::Tensor &w2_map0,
          double w_r0_0, double w_dr0, double w_theta0_0, double w_dtheta0,
          int64_t w_nr0, int64_t w_ntheta0,
          const at::Tensor &w1_map1,
          const at::Tensor &w2_map1,
          double w_r0_1, double w_dr1, double w_theta0_1, double w_dtheta1,
          int64_t w_nr1, int64_t w_ntheta1,
          int64_t output_weight_map,
          int64_t output_weight_decimation) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_CHECK(dorigin.dtype() == at::kFloat);
    TORCH_CHECK(r0.dtype() == at::kFloat);
    TORCH_CHECK(Nr0.dtype() == at::kInt);
    TORCH_CHECK(Ntheta0.dtype() == at::kInt);
    TORCH_CHECK(poly_coefs.dtype() == at::kFloat);
    TORCH_CHECK(order >= 2 && order <= 8,
        "ffbp_merge2_poly_weighted requires order in [2, 8], got ", order);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);

    bool has_weight0 = w1_map0.defined() && w1_map0.numel() > 0 && w2_map0.defined() && w2_map0.numel() > 0;
    bool has_weight1 = w1_map1.defined() && w1_map1.numel() > 0 && w2_map1.defined() && w2_map1.numel() > 0;

    at::Tensor dorigin_contig = dorigin.contiguous();
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor r0_contig = r0.contiguous();
    at::Tensor dr0_contig = dr0.contiguous();
    at::Tensor theta0_contig = theta0.contiguous();
    at::Tensor dtheta0_contig = dtheta0.contiguous();
    at::Tensor Nr0_contig = Nr0.contiguous();
    at::Tensor Ntheta0_contig = Ntheta0.contiguous();
    at::Tensor poly_coefs_contig = poly_coefs.contiguous();
    at::Tensor out = torch::empty({Nr1, Ntheta1}, img0_contig.options());
    const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    const complex64_t* img0_ptr = img0_contig.data_ptr<complex64_t>();
    const complex64_t* img1_ptr = img1_contig.data_ptr<complex64_t>();
    complex64_t* out_ptr = out.data_ptr<complex64_t>();
    const float* r0_ptr = r0_contig.data_ptr<float>();
    const float* dr0_ptr = dr0_contig.data_ptr<float>();
    const float* theta0_ptr = theta0_contig.data_ptr<float>();
    const float* dtheta0_ptr = dtheta0_contig.data_ptr<float>();
    const int* Nr0_ptr = Nr0_contig.data_ptr<int>();
    const int* Ntheta0_ptr = Ntheta0_contig.data_ptr<int>();
    const float* coefs_ptr = poly_coefs_contig.data_ptr<float>();
    const int n_coefs = poly_coefs_contig.size(0);

    at::Tensor w1_map0_contig, w2_map0_contig, w1_map1_contig, w2_map1_contig;
    const float* w1_map0_ptr = nullptr;
    const float* w2_map0_ptr = nullptr;
    const float* w1_map1_ptr = nullptr;
    const float* w2_map1_ptr = nullptr;
    if (has_weight0) {
        TORCH_CHECK(w1_map0.dtype() == at::kFloat);
        TORCH_CHECK(w2_map0.dtype() == at::kFloat);
        w1_map0_contig = w1_map0.contiguous();
        w2_map0_contig = w2_map0.contiguous();
        w1_map0_ptr = w1_map0_contig.data_ptr<float>();
        w2_map0_ptr = w2_map0_contig.data_ptr<float>();
    }
    if (has_weight1) {
        TORCH_CHECK(w1_map1.dtype() == at::kFloat);
        TORCH_CHECK(w2_map1.dtype() == at::kFloat);
        w1_map1_contig = w1_map1.contiguous();
        w2_map1_contig = w2_map1.contiguous();
        w1_map1_ptr = w1_map1_contig.data_ptr<float>();
        w2_map1_ptr = w2_map1_contig.data_ptr<float>();
    }

    at::Tensor w1_out, w2_out;
    float* w1_out_ptr = nullptr;
    float* w2_out_ptr = nullptr;
    int dec = output_weight_decimation > 0 ? output_weight_decimation : 1;
    if (output_weight_map) {
        int64_t out_nr_dec = (Nr1 + dec - 1) / dec;
        int64_t out_ntheta_dec = (Ntheta1 + dec - 1) / dec;
        w1_out = torch::empty({out_nr_dec, out_ntheta_dec},
                torch::TensorOptions().dtype(at::kFloat).device(img0.device()));
        w2_out = torch::empty({out_nr_dec, out_ntheta_dec},
                torch::TensorOptions().dtype(at::kFloat).device(img0.device()));
        w1_out_ptr = w1_out.data_ptr<float>();
        w2_out_ptr = w2_out.data_ptr<float>();
    }

    const float ref_phase = 4.0f * fc / kC0;

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
    for (int idx = 0; idx < Nr1 * Ntheta1; idx++) {
        ffbp_merge2_kernel_poly_weighted_cpu(
                img0_ptr, img1_ptr, out_ptr, w1_out_ptr, w2_out_ptr, dorigin_ptr, ref_phase,
                r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, coefs_ptr, n_coefs,
                alias, alias_fmod/kPI,
                w1_map0_ptr, w2_map0_ptr, w_r0_0, w_dr0, w_theta0_0, w_dtheta0, w_nr0, w_ntheta0,
                w1_map1_ptr, w2_map1_ptr, w_r0_1, w_dr1, w_theta0_1, w_dtheta1, w_nr1, w_ntheta1,
                dec, idx);
    }

    std::vector<at::Tensor> ret;
    ret.push_back(out);
    ret.push_back(w1_out);
    ret.push_back(w2_out);
    return ret;
}

template<typename T>
static void polar_to_cart_kernel_lanczos_cpu(const T *img, T
        *out, const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, float alias_fmod, int order,
        int id1, int idbatch) {
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;

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
        T v = lanczos_interp_2d_cpu<T>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri, dti, order);
        if constexpr (std::is_same_v<T, complex64_t>) {
            float ref_sin, ref_cos;
            sincospi<float>(ref_phase * dz - alias_fmod * dri, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};
            out[idbatch * Nx * Ny + idx*Ny + idy] = v * ref;
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = v;
        }
    } else {
        if constexpr (std::is_same_v<T, complex64_t>) {
            out[idbatch * Nx * Ny + idx*Ny + idy] = {0.0f, 0.0f};
        } else {
            out[idbatch * Nx * Ny + idx*Ny + idy] = 0.0f;
        }
    }
}

at::Tensor polar_to_cart_lanczos_cpu(
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
          double alias_fmod,
          int64_t order) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
    TORCH_CHECK(origin.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CPU);
    at::Tensor origin_contig = origin.contiguous();
    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
    const float* origin_ptr = origin_contig.data_ptr<float>();

    const float ref_phase = 4.0f * fc / kC0;

    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int id1 = 0; id1 < Nx * Ny; id1++) {
            if (img.dtype() == at::kComplexFloat) {
                const complex64_t* img_ptr = img_contig.data_ptr<complex64_t>();
                complex64_t* out_ptr = out.data_ptr<complex64_t>();
                polar_to_cart_kernel_lanczos_cpu<complex64_t>(
                        img_ptr, out_ptr, origin_ptr, rotation, ref_phase,
                        r0, dr, theta0, dtheta, Nr, Ntheta,
                        x0, dx, y0, dy, Nx, Ny, alias_fmod/kPI, order, id1, idbatch);
            } else {
                const float* img_ptr = img_contig.data_ptr<float>();
                float* out_ptr = out.data_ptr<float>();
                polar_to_cart_kernel_lanczos_cpu<float>(
                        img_ptr, out_ptr, origin_ptr, rotation, ref_phase,
                        r0, dr, theta0, dtheta, Nr, Ntheta,
                        x0, dx, y0, dy, Nx, Ny, alias_fmod/kPI, order, id1, idbatch);
            }
        }
    }
    return out;
}

static void compute_illumination_kernel_cpu(
          const float* pos,
          const float* att,
          const float* g,
          float* w1_out,
          float* w2_out,
          int nsweeps,
          float r0, float dr, float theta0, float dtheta, int nr, int ntheta,
          float g_el0, float g_del, float g_az0, float g_daz, int g_nel, int g_naz,
          int decimation, int idx) {

    const int out_ntheta = (ntheta + decimation - 1) / decimation;
    const int out_nr = (nr + decimation - 1) / decimation;

    if (idx >= out_nr * out_ntheta) return;

    const int out_idtheta = idx % out_ntheta;
    const int out_idr = idx / out_ntheta;

    const int full_idr = out_idr * decimation;
    const int full_idtheta = out_idtheta * decimation;

    const float r = r0 + dr * full_idr;
    const float t = theta0 + dtheta * full_idtheta;  // t is sin(theta)
    const float cost = sqrtf(1.0f - t*t);
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
        const float pz = -pos_z;
        const float d = sqrtf(px*px + py*py + pz*pz);

        const float look_angle = asinf(std::max(-1.0f, std::min(1.0f, -pos_z / d)));

        float att_el = 0.0f;
        float att_az = 0.0f;
        if (att != nullptr) {
            att_el = att[i * 3 + 0];
            att_az = att[i * 3 + 2];
        }

        const float el = look_angle - att_el;
        const float az = atan2f(py, px) - att_az;

        const float el_idx = (el - g_el0) / g_del;
        const float az_idx = (az - g_az0) / g_daz;

        if (el_idx >= 0 && el_idx < g_nel - 1 && az_idx >= 0 && az_idx < g_naz - 1) {
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
    for (int idx = 0; idx < out_nr * out_ntheta; idx++) {
        compute_illumination_kernel_cpu(
                pos_ptr, att_ptr, g_ptr, w1_out_ptr, w2_out_ptr, nsweeps,
                r0, dr, theta0, dtheta, nr, ntheta,
                g_el0, g_del, g_az0, g_daz, g_nel, g_naz, dec, idx);
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

    for (int p = 0; p < N; p++) {
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
            if (el_int < 0 || el_int+1 >= g_nel || az_int < 0 || az_int+1 >= g_naz)
                norm = 0.0f;
            else
                norm *= interp2d<float>(g, g_nel, g_naz,
                                        el_int, el_idx - el_int,
                                        az_int, az_idx - az_int);
        }
        if (normalization == 1) norm *= sqrtf(-dpz / d);

        if (norm == 0.0f) continue;

        const complex64_t w_img = img[p];
        const float wr = w_img.real() * norm;
        const float wi = w_img.imag() * norm;

        const float tau  = 2.0f * (d + d0) / kC0;
        const float nu_p = gamma * tau / fs;

        // exp(+j 2pi u_p k/M_ext) = exp(-j 2pi nu_p k) for integer k.
        const float up = M_ext_f - nu_p * M_ext_f;

        // Carrier phase + centering term exp(-j pi nu_p M).
        float phase = -2.0f * fc * tau;
        if (use_rvp) phase += gamma * tau * tau;
        phase -= nu_p * M_ext_f * 0.5f;
        float sn, cs;
        sincospi(phase, &sn, &cs);
        const float ar = wr * cs - wi * sn;
        const float ai = wr * sn + wi * cs;

        const int   j0  = (int)rintf(up);
        const float frc = up - (float)j0;

        for (int delta = -(half_W - 1); delta <= half_W; ++delta) {
            float dist    = fabsf(frc - (float)delta);
            int   lut_idx = (int)(dist * lut_scale);
            if (lut_idx > N_LUT - 1) lut_idx = N_LUT - 1;
            float kb = kb_lut[lut_idx];

            int gidx = ((j0 + delta) % M_ext + M_ext) % M_ext;
            grid_row[gidx] += complex64_t(ar * kb, ai * kb);
        }
    }
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
            if (el_int < 0 || el_int+1 >= g_nel || az_int < 0 || az_int+1 >= g_naz)
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

// ---------------------------------------------------------------------------
// CPU implementations of the element-wise interp combine, coherence, resample
// and GPGA operators. These mirror the CUDA versions in cuda/util.cu,
// cuda/coherence.cu, cuda/resample.cu and cuda/backproj.cu so that the
// polarimetry / interferometry / autofocus paths run on the CPU as well.
// ---------------------------------------------------------------------------

// Element-wise a (*|/) interp2d(b). Output grid [Na0, Na1] is mapped onto the
// input grid [Nb0, Nb1] with bilinear interpolation, matching the CUDA kernel
// in cuda/util.cu bit-for-bit.
template<typename T, typename T2>
static void interp2d_combine_kernel_cpu(const T* a, const T2* b, T* out,
        int Na0, int Na1, int Nb0, int Nb1, bool is_div, int idx, int idbatch) {
    const int id0 = idx / Na1;
    const int id1 = idx % Na1;

    if (id0 >= Na0 || id1 >= Na1) return;

    // Map from output grid [0, Na0-1] x [0, Na1-1] to input grid [0, Nb0-1] x [0, Nb1-1]
    float b0_float = (float)id0 * (Nb0 - 1) / (Na0 - 1);
    float b1_float = (float)id1 * (Nb1 - 1) / (Na1 - 1);

    int b0_int = (int)floorf(b0_float);
    int b1_int = (int)floorf(b1_float);
    float b0_frac = b0_float - b0_int;
    float b1_frac = b1_float - b1_int;

    // Clamp to valid range
    b0_int = std::min(b0_int, Nb0 - 2);
    b1_int = std::min(b1_int, Nb1 - 2);
    b0_int = std::max(b0_int, 0);
    b1_int = std::max(b1_int, 0);

    const T2 v = interp2d<T2>(&b[idbatch * Nb1 * Nb0], Nb0, Nb1, b0_int, b0_frac, b1_int, b1_frac);
    const int oidx = idbatch * Na1 * Na0 + id0 * Na1 + id1;
    if (is_div) {
        out[oidx] = a[oidx] / static_cast<T>(v);
    } else {
        out[oidx] = a[oidx] * static_cast<T>(v);
    }
}

static at::Tensor interp2d_combine_cpu(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1,
          bool is_div) {
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor out = torch::zeros({nbatch, Na0, Na1}, a_contig.options());

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    const bool a_complex = a.scalar_type() == at::ScalarType::ComplexFloat;
    const bool b_complex = b.scalar_type() == at::ScalarType::ComplexFloat;

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int idx = 0; idx < Na0 * Na1; idx++) {
            if (a_complex && b_complex) {
                interp2d_combine_kernel_cpu<complex64_t, complex64_t>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        (const complex64_t*)b_contig.data_ptr<c10::complex<float>>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (a_complex && !b_complex) {
                interp2d_combine_kernel_cpu<complex64_t, float>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        b_contig.data_ptr<float>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (!a_complex && !b_complex) {
                interp2d_combine_kernel_cpu<float, float>(
                        a_contig.data_ptr<float>(),
                        b_contig.data_ptr<float>(),
                        out.data_ptr<float>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else {
                AT_ERROR("Unsupported dtype combination for interp2d combine");
            }
        }
    }
    return out;
}

at::Tensor div_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/true);
}

at::Tensor mul_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/false);
}

// "Quick and dirty" power coherence over a moving window. Mirrors
// power_coherence_2d_kernel in cuda/coherence.cu.
static void power_coherence_2d_kernel_cpu(
          const complex64_t* img0, const complex64_t* img1, float* out,
          int N0, int N1, int w0, int w1, bool corr_output, int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    float corr = 0.0f;
    float p0 = 0.0f;
    float p1 = 0.0f;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            float v0_abs2 = v0.real() * v0.real() + v0.imag() * v0.imag();
            float v1_abs2 = v1.real() * v1.real() + v1.imag() * v1.imag();
            p0 += v0_abs2 * v0_abs2;
            p1 += v1_abs2 * v1_abs2;
            corr += v0_abs2 * v1_abs2;
        }
    }
    float denom = sqrtf(p0 * p1);
    float v = denom > 0.0f ? corr / denom : 0.0f;
    if (corr_output) {
        v = v > 0.5f ? sqrtf(2.0f * v - 1.0f) : 0.0f;
    }
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}

at::Tensor power_coherence_2d_cpu(
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1,
          int64_t corr_output) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
    at::Tensor out = torch::empty({nbatch, N0, N1}, options);

    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();
    float* out_ptr = out.data_ptr<float>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            power_coherence_2d_kernel_cpu(
                    img0_ptr, img1_ptr, out_ptr,
                    N0, N1, w0, w1, (bool)corr_output, id0, idbatch);
        }
    }
    return out;
}

// Generic 2D image resampling. Each output pixel (ir, iaz) reads from input at
// (ir + shift_r, iaz + shift_az). Mirrors resample_2d_*_kernel in
// cuda/resample.cu. interp_2d is a callable (img_row_base, x, y) -> T.
template<typename T, typename Interp>
static void resample_2d_kernel_cpu(
        const T* img, const float* shift_r, const float* shift_az, T* out,
        int Nr, int Naz, int order, Interp interp_2d, int id1, int idbatch) {
    const int iaz = id1 % Naz;
    const int ir = id1 / Naz;

    if (id1 >= Nr * Naz) return;

    const int shift_idx = ir * Naz + iaz;
    const float src_r = ir + shift_r[shift_idx];
    const float src_az = iaz + shift_az[shift_idx];

    const float a = 0.5f * order;
    if (src_r >= -a && src_r < Nr + a && src_az >= -a && src_az < Naz + a) {
        out[idbatch * Nr * Naz + shift_idx] =
            interp_2d(&img[idbatch * Nr * Naz], src_r, src_az);
    } else {
        out[idbatch * Nr * Naz + shift_idx] = T{};
    }
}

template<typename T, typename Interp>
static at::Tensor resample_2d_cpu_impl(
          const at::Tensor &img,
          const at::Tensor &shift_r,
          const at::Tensor &shift_az,
          int64_t nbatch, int64_t Nr, int64_t Naz, int64_t order,
          Interp interp_2d) {
    at::Tensor img_contig = img.contiguous();
    at::Tensor shift_r_contig = shift_r.contiguous();
    at::Tensor shift_az_contig = shift_az.contiguous();
    at::Tensor out = torch::empty({nbatch, Nr, Naz}, img_contig.options());

    const float* shift_r_ptr = shift_r_contig.data_ptr<float>();
    const float* shift_az_ptr = shift_az_contig.data_ptr<float>();
    const T* img_ptr = (const T*)img_contig.template data_ptr<T>();
    T* out_ptr = (T*)out.template data_ptr<T>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id1 = 0; id1 < Nr * Naz; id1++) {
            resample_2d_kernel_cpu<T>(
                    img_ptr, shift_r_ptr, shift_az_ptr, out_ptr,
                    Nr, Naz, order, interp_2d, id1, idbatch);
        }
    }
    return out;
}

at::Tensor resample_2d_lanczos_cpu(
          const at::Tensor &img,
          const at::Tensor &shift_r,
          const at::Tensor &shift_az,
          int64_t nbatch, int64_t Nr, int64_t Naz, int64_t order) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_2d_lanczos: img must be complex64 or float32");
    TORCH_CHECK(shift_r.dtype() == at::kFloat, "resample_2d_lanczos: shift_r must be float32");
    TORCH_CHECK(shift_az.dtype() == at::kFloat, "resample_2d_lanczos: shift_az must be float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(shift_r.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(shift_az.device().type() == at::DeviceType::CPU);

    if (img.dtype() == at::kComplexFloat) {
        auto interp = [Nr, Naz, order](const complex64_t* base, float x, float y) {
            return lanczos_interp_2d_cpu<complex64_t>(base, Nr, Naz, x, y, order);
        };
        return resample_2d_cpu_impl<complex64_t>(img, shift_r, shift_az, nbatch, Nr, Naz, order, interp);
    } else {
        auto interp = [Nr, Naz, order](const float* base, float x, float y) {
            return lanczos_interp_2d_cpu<float>(base, Nr, Naz, x, y, order);
        };
        return resample_2d_cpu_impl<float>(img, shift_r, shift_az, nbatch, Nr, Naz, order, interp);
    }
}

at::Tensor resample_2d_knab_cpu(
          const at::Tensor &img,
          const at::Tensor &shift_r,
          const at::Tensor &shift_az,
          int64_t nbatch, int64_t Nr, int64_t Naz, int64_t order,
          double oversample) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_2d_knab: img must be complex64 or float32");
    TORCH_CHECK(shift_r.dtype() == at::kFloat, "resample_2d_knab: shift_r must be float32");
    TORCH_CHECK(shift_az.dtype() == at::kFloat, "resample_2d_knab: shift_az must be float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(shift_r.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(shift_az.device().type() == at::DeviceType::CPU);

    // Knab window parameter: v = 1 - 1/oversample
    const float v = 1.0f - 1.0f / static_cast<float>(oversample);
    const float norm = knab_kernel_norm_cpu(order, v);

    if (img.dtype() == at::kComplexFloat) {
        auto interp = [Nr, Naz, order, v, norm](const complex64_t* base, float x, float y) {
            return knab_interp_2d_cpu<complex64_t>(base, Nr, Naz, x, y, order, v, norm);
        };
        return resample_2d_cpu_impl<complex64_t>(img, shift_r, shift_az, nbatch, Nr, Naz, order, interp);
    } else {
        auto interp = [Nr, Naz, order, v, norm](const float* base, float x, float y) {
            return knab_interp_2d_cpu<float>(base, Nr, Naz, x, y, order, v, norm);
        };
        return resample_2d_cpu_impl<float>(img, shift_r, shift_az, nbatch, Nr, Naz, order, interp);
    }
}

// Generalized phase gradient autofocus: demodulated phase of each target for
// every sweep. Mirrors gpga_backprojection_2d_kernel in cuda/backproj.cu
// (complex64 data, linear range interpolation).
static void gpga_backprojection_2d_kernel_cpu(
          const float* target_pos, const complex64_t* data, const float* pos,
          complex64_t* data_out, int sweep_samples, int nsweeps,
          float ref_phase, float delta_r, int Ntarget, float d0, float data_fmod,
          int idtarget, int idsweep) {
    if (idtarget >= Ntarget || idsweep >= nsweeps) return;

    const float x = target_pos[idtarget * 3 + 0];
    const float y = target_pos[idtarget * 3 + 1];
    const float z = target_pos[idtarget * 3 + 2];

    float pos_x = pos[idsweep * 3 + 0];
    float pos_y = pos[idsweep * 3 + 1];
    float pos_z = pos[idsweep * 3 + 2];
    float px = (x - pos_x);
    float py = (y - pos_y);
    float pz = (z - pos_z);

    const float d = sqrtf(px * px + py * py + pz * pz);

    float sx = delta_r * (d + d0);

    int id0 = sx;
    int id1 = id0 + 1;
    if (id0 < 0 || id1 >= sweep_samples) {
        data_out[idtarget * nsweeps + idsweep] = {0.0f, 0.0f};
    } else {
        complex64_t s0 = data[idsweep * sweep_samples + id0];
        complex64_t s1 = data[idsweep * sweep_samples + id1];
        float interp_idx = sx - id0;
        complex64_t s = (1.0f - interp_idx) * s0 + interp_idx * s1;

        float ref_sin, ref_cos;
        sincospi<float>(ref_phase * d - data_fmod * sx, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        data_out[idtarget * nsweeps + idsweep] = s * ref;
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

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idtarget = 0; idtarget < Ntarget; idtarget++) {
        for (int idsweep = 0; idsweep < nsweeps; idsweep++) {
            gpga_backprojection_2d_kernel_cpu(
                    target_pos_ptr, data_ptr, pos_ptr, data_out_ptr,
                    sweep_samples, nsweeps, ref_phase, delta_r, Ntarget, d0,
                    data_fmod / kPI, idtarget, idsweep);
        }
    }
    return data_out;
}

// Interferometric coherence over a moving window. Mirrors coherence_2d_kernel
// in cuda/coherence.cu.
static void coherence_2d_kernel_cpu(
          const complex64_t* img0, const complex64_t* img1, float* out,
          int N0, int N1, int w0, int w1, int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * std::conj(v1);
            Navg += 1;
        }
    }
    corr /= static_cast<float>(Navg);
    p0 /= static_cast<float>(Navg);
    p1 /= static_cast<float>(Navg);
    float denom = sqrtf(p0 * p1);
    float v = denom > 0.0f ? std::abs(corr) / denom : 0.0f;
    out[idbatch * N0 * N1 + idy * N1 + idx] = v;
}

at::Tensor coherence_2d_cpu(
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1) {
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();

    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img0.device());
    at::Tensor out = torch::empty({nbatch, N0, N1}, options);

    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();
    float* out_ptr = out.data_ptr<float>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            coherence_2d_kernel_cpu(
                    img0_ptr, img1_ptr, out_ptr, N0, N1, w0, w1, id0, idbatch);
        }
    }
    return out;
}

// Gradient of coherence_2d. Scatters the per-output-pixel gradient back to
// every input pixel in the window. Mirrors coherence_2d_grad_kernel in
// cuda/coherence.cu (CUDA atomicAdd -> "#pragma omp atomic" here).
static void coherence_2d_grad_kernel_cpu(
          const float* grad, const complex64_t* img0, const complex64_t* img1,
          int N0, int N1, int w0, int w1,
          complex64_t* img0_grad, complex64_t* img1_grad,
          int id0_, int idbatch) {
    const int idx = id0_ % N1;
    const int idy = id0_ / N1;

    if (id0_ >= N0 * N1) return;

    complex64_t corr{};
    float p0 = 0.0f;
    float p1 = 0.0f;
    int Navg = 0;
    int start_i = std::max(-w0, -idx);
    int end_i = std::min(w0, N1 - 1 - idx);
    int start_j = std::max(-w1, -idy);
    int end_j = std::min(w1, N0 - 1 - idy);
    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];
            p0 += v0.real() * v0.real() + v0.imag() * v0.imag();
            p1 += v1.real() * v1.real() + v1.imag() * v1.imag();
            corr += v0 * std::conj(v1);
            Navg += 1;
        }
    }
    corr /= static_cast<float>(Navg);
    p0 /= static_cast<float>(Navg);
    p1 /= static_cast<float>(Navg);
    float p0p1 = p0 * p1;
    if (p0p1 <= 0.0f) {
        return;
    }
    float dout2 = (corr.real() * corr.real() + corr.imag() * corr.imag()) / p0p1;
    float dout = sqrtf(dout2);

    float dout_ddout2 = dout > 0.0f
        ? grad[idbatch * N0 * N1 + idy * N1 + idx] / dout
        : 0.0f;
    complex64_t dout_dc = dout_ddout2 * corr / p0p1 / static_cast<float>(Navg);
    float dout_dnorm1 = dout_ddout2 * (-dout2 / p0) / static_cast<float>(Navg);
    float dout_dnorm2 = dout_ddout2 * (-dout2 / p1) / static_cast<float>(Navg);

    for (int y = idy + start_j; y <= idy + end_j; y++) {
        for (int x = idx + start_i; x <= idx + end_i; x++) {
            complex64_t v0 = img0[idbatch * N0 * N1 + y * N1 + x];
            complex64_t v1 = img1[idbatch * N0 * N1 + y * N1 + x];

            if (img0_grad != nullptr) {
                complex64_t gx1 = dout_dc * v1 + dout_dnorm1 * v0;
                float* g0 = reinterpret_cast<float*>(&img0_grad[idbatch * N0 * N1 + y * N1 + x]);
                #pragma omp atomic
                g0[0] += gx1.real();
                #pragma omp atomic
                g0[1] += gx1.imag();
            }
            if (img1_grad != nullptr) {
                complex64_t gx2 = std::conj(dout_dc) * v0 + dout_dnorm2 * v1;
                float* g1 = reinterpret_cast<float*>(&img1_grad[idbatch * N0 * N1 + y * N1 + x]);
                #pragma omp atomic
                g1[0] += gx2.real();
                #pragma omp atomic
                g1[1] += gx2.imag();
            }
        }
    }
}

std::vector<at::Tensor> coherence_2d_grad_cpu(
          const at::Tensor &grad,
          const at::Tensor &img0,
          const at::Tensor &img1,
          int64_t nbatch,
          int64_t N0,
          int64_t N1,
          int64_t w0,
          int64_t w1) {
    TORCH_CHECK(grad.dtype() == at::kFloat);
    TORCH_CHECK(img0.dtype() == at::kComplexFloat);
    TORCH_CHECK(img1.dtype() == at::kComplexFloat);
    TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(img1.device().type() == at::DeviceType::CPU);
    at::Tensor img0_contig = img0.contiguous();
    at::Tensor img1_contig = img1.contiguous();
    at::Tensor grad_contig = grad.contiguous();

    const float* grad_ptr = grad_contig.data_ptr<float>();
    const complex64_t* img0_ptr = (const complex64_t*)img0_contig.data_ptr<c10::complex<float>>();
    const complex64_t* img1_ptr = (const complex64_t*)img1_contig.data_ptr<c10::complex<float>>();

    at::Tensor img0_grad;
    at::Tensor img1_grad;
    complex64_t* img0_grad_ptr = nullptr;
    complex64_t* img1_grad_ptr = nullptr;

    if (img0.requires_grad()) {
        img0_grad = torch::zeros_like(img0);
        img0_grad_ptr = (complex64_t*)img0_grad.data_ptr<c10::complex<float>>();
    } else {
        img0_grad = torch::Tensor();
    }
    if (img1.requires_grad()) {
        img1_grad = torch::zeros_like(img1);
        img1_grad_ptr = (complex64_t*)img1_grad.data_ptr<c10::complex<float>>();
    } else {
        img1_grad = torch::Tensor();
    }

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id0 = 0; id0 < N0 * N1; id0++) {
            coherence_2d_grad_kernel_cpu(
                    grad_ptr, img0_ptr, img1_ptr, N0, N1, w0, w1,
                    img0_grad_ptr, img1_grad_ptr, id0, idbatch);
        }
    }

    std::vector<at::Tensor> ret;
    ret.push_back(img0_grad);
    ret.push_back(img1_grad);
    return ret;
}

// Lee speckle filter. Mirrors lee_filter_kernel in cuda/speckle_filter.cu.
// The window statistics use Welford's online mean/variance and the same
// boundary conventions as the CUDA kernel (note the strict-less-than upper
// bounds and that nan pixels still advance the count).
template<typename T>
static void lee_filter_kernel_cpu(const T* img, float* out, int Nx, int Ny,
        int wx, int wy, float cu, int id1, int idbatch) {
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;

    if (id1 >= Nx * Ny) return;

    int count = 0;
    float mean = 0.0f;
    float m2 = 0.0f;
    for (int i = std::max(idx - wx, 0); i < std::min(idx + wx, Nx - 1); i++) {
        for (int j = std::max(idy - wy, 0); j < std::min(idy + wy, Ny - 1); j++) {
            count++;
            T val = img[idbatch * Nx * Ny + i * Ny + j];
            float v;
            if constexpr (std::is_same_v<T, complex64_t>) {
                if (std::isnan(val.real())) {
                    continue;
                }
                v = std::abs(val);
            } else {
                if (std::isnan(val)) {
                    continue;
                }
                v = val;
            }
            float delta = v - mean;
            mean += delta / count;
            float delta2 = v - mean;
            m2 += delta * delta2;
        }
    }
    const int oidx = idbatch * Nx * Ny + idx * Ny + idy;
    if (count == 0) {
        T c = img[oidx];
        if constexpr (std::is_same_v<T, complex64_t>) {
            out[oidx] = std::abs(c);
        } else {
            out[oidx] = c;
        }
    } else {
        float var = m2 / count;
        float ci = sqrtf(var) / mean;
        float w;
        if (ci < cu) {
            w = 0.0f;
        } else {
            w = 1.0f - (cu * cu) / (ci * ci);
        }
        T val = img[oidx];
        if constexpr (std::is_same_v<T, complex64_t>) {
            out[oidx] = mean + w * (std::abs(val) - mean);
        } else {
            out[oidx] = mean + w * (val - mean);
        }
    }
}

at::Tensor lee_filter_cpu(
          const at::Tensor &img,
          int64_t nbatch,
          int64_t Nx,
          int64_t Ny,
          int64_t wx,
          int64_t wy,
          double cu) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    at::Tensor img_contig = img.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kFloat)
        .layout(torch::kStrided)
        .device(img.device());
    at::Tensor out = torch::empty({nbatch, Nx, Ny}, options);
    float* out_ptr = out.data_ptr<float>();

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    const bool is_complex = img.dtype() == at::kComplexFloat;

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id1 = 0; id1 < Nx * Ny; id1++) {
            if (is_complex) {
                lee_filter_kernel_cpu<complex64_t>(
                        (const complex64_t*)img_contig.data_ptr<c10::complex<float>>(),
                        out_ptr, Nx, Ny, wx, wy, cu, id1, idbatch);
            } else {
                lee_filter_kernel_cpu<float>(
                        img_contig.data_ptr<float>(),
                        out_ptr, Nx, Ny, wx, wy, cu, id1, idbatch);
            }
        }
    }
    return out;
}

// Defines the operators
TORCH_LIBRARY(torchbp, m) {
  m.def("backprojection_polar_2d(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize) -> Tensor");
  m.def("backprojection_polar_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize) -> Tensor[]");
  m.def("backprojection_polar_2d_lanczos(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize) -> Tensor");
  m.def("backprojection_polar_2d_knab(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, float oversample, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize) -> Tensor");
  m.def("backprojection_cart_2d(Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor");
  m.def("backprojection_cart_2d_grad(Tensor grad, Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor[]");
  m.def("gpga_backprojection_2d(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, float data_fmod) -> Tensor");
  m.def("gpga_backprojection_2d_lanczos(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, int order, float data_fmod) -> Tensor");
  m.def("cfar_2d(Tensor img, int nbatch, int N0, int N1, int Navg0, int Navg1, int Nguard0, int Nguard1, float threshold, int peaks_only) -> Tensor");
  m.def("polar_interp_linear(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor");
  m.def("polar_interp_linear_grad(Tensor grad, Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor[]");
  m.def("polar_interp_lanczos(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float alias_fmod) -> Tensor");
  m.def("ffbp_merge2_lanczos(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, int alias, float alias_fmod) -> Tensor");
  m.def("ffbp_merge2_knab(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float oversample, int alias, float alias_fmod) -> Tensor");
  m.def("ffbp_merge2_poly(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod) -> Tensor");
  m.def("ffbp_merge2_poly_weighted(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod, Tensor w1_map0, Tensor w2_map0, float w_r0_0, float w_dr0, float w_theta0_0, float w_dtheta0, int w_nr0, int w_ntheta0, Tensor w1_map1, Tensor w2_map1, float w_r0_1, float w_dr1, float w_theta0_1, float w_dtheta1, int w_nr1, int w_ntheta1, int output_weight_map, int output_weight_decimation) -> Tensor[]");
  m.def("polar_to_cart_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor");
  m.def("polar_to_cart_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor[]");
  m.def("polar_to_cart_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod, int order) -> Tensor");
  m.def("backprojection_polar_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization) -> Tensor");
  m.def("backprojection_polar_2d_tx_power_slant(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, float altitude) -> Tensor");
  m.def("compute_illumination(Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, float r0, float dr, float theta0, float dtheta, int nr, int ntheta, int decimation) -> Tensor[]");
  m.def("entropy(Tensor data, Tensor norm, int nbatch) -> Tensor");
  m.def("entropy_grad(Tensor data, Tensor norm, Tensor grad, int nbatch) -> Tensor[]");
  m.def("abs_sum(Tensor data, int nbatch) -> Tensor");
  m.def("abs_sum_grad(Tensor data, Tensor grad, int nbatch) -> Tensor");
  m.def("lee_filter(Tensor img, int nbatch, int Nx, int Ny, int wx, int wy, float cu) -> Tensor");
  m.def("coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor");
  m.def("coherence_2d_grad(Tensor grad, Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor[]");
  m.def("power_coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1, int corr_output) -> Tensor");
  m.def("projection_cart_2d(Tensor img, Tensor dem, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float fs, float gamma, float x0, float dx, float y0, float dy, int Nx, int Ny, float d0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int use_rvp, int normalization) -> Tensor");
  m.def("projection_cart_2d_nufft(Tensor img, Tensor dem, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float fs, float gamma, float x0, float dx, float y0, float dy, int Nx, int Ny, float d0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int use_rvp, int normalization) -> Tensor");
  m.def("subpixel_correlation(Tensor im_m, Tensor im_s, Tensor mean_m, Tensor mean_s, int nbatch, int N0, int N1) -> Tensor[]");
  m.def("div_2d_interp_linear(Tensor a, Tensor b, int nbatch, int Na0, int Na1, int Nb0, int Nb1) -> Tensor");
  m.def("mul_2d_interp_linear(Tensor a, Tensor b, int nbatch, int Na0, int Na1, int Nb0, int Nb1) -> Tensor");
  m.def("resample_2d_lanczos(Tensor img, Tensor shift_r, Tensor shift_az, int nbatch, int Nr, int Naz, int order) -> Tensor");
  m.def("resample_2d_knab(Tensor img, Tensor shift_r, Tensor shift_az, int nbatch, int Nr, int Naz, int order, float oversample) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cpu);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cpu);
  m.impl("backprojection_cart_2d", &backprojection_cart_2d_cpu);
  m.impl("backprojection_cart_2d_grad", &backprojection_cart_2d_grad_cpu);
  m.impl("projection_cart_2d", &projection_cart_2d_cpu);
  m.impl("projection_cart_2d_nufft", &projection_cart_2d_nufft_cpu);
  m.impl("polar_interp_linear", &polar_interp_linear_cpu);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cpu);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cpu);
  m.impl("polar_to_cart_lanczos", &polar_to_cart_lanczos_cpu);
  m.impl("ffbp_merge2_lanczos", &ffbp_merge2_lanczos_cpu);
  m.impl("ffbp_merge2_knab", &ffbp_merge2_knab_cpu);
  m.impl("ffbp_merge2_poly", &ffbp_merge2_poly_cpu);
  m.impl("ffbp_merge2_poly_weighted", &ffbp_merge2_poly_weighted_cpu);
  m.impl("compute_illumination", &compute_illumination_cpu);
  m.impl("div_2d_interp_linear", &div_2d_interp_linear_cpu);
  m.impl("mul_2d_interp_linear", &mul_2d_interp_linear_cpu);
  m.impl("power_coherence_2d", &power_coherence_2d_cpu);
  m.impl("resample_2d_lanczos", &resample_2d_lanczos_cpu);
  m.impl("resample_2d_knab", &resample_2d_knab_cpu);
  m.impl("gpga_backprojection_2d", &gpga_backprojection_2d_cpu);
  m.impl("coherence_2d", &coherence_2d_cpu);
  m.impl("coherence_2d_grad", &coherence_2d_grad_cpu);
  m.impl("lee_filter", &lee_filter_cpu);
}

}
