#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
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
    *sinx = sin(static_cast<T>(kPI) * x);
    *cosx = cos(static_cast<T>(kPI) * x);
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
            w_sum += w*w;
        } else {
            pixel += s * ref;
        }
    }
    if (g != nullptr) {
        if (w_sum > 0.0f) {
            pixel *= nsweeps / sqrtf(w_sum);
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
          double alias_fmod) {
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
          double alias_fmod) {
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

// Defines the operators
TORCH_LIBRARY(torchbp, m) {
  m.def("backprojection_polar_2d(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod) -> Tensor");
  m.def("backprojection_polar_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod) -> Tensor[]");
  m.def("backprojection_polar_2d_lanczos(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod) -> Tensor");
  m.def("backprojection_polar_2d_knab(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, float oversample, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod) -> Tensor");
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
  m.def("polar_to_cart_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor");
  m.def("polar_to_cart_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor[]");
  m.def("polar_to_cart_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod, int order) -> Tensor");
  m.def("backprojection_polar_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization) -> Tensor");
  m.def("entropy(Tensor data, Tensor norm, int nbatch) -> Tensor");
  m.def("entropy_grad(Tensor data, Tensor norm, Tensor grad, int nbatch) -> Tensor[]");
  m.def("abs_sum(Tensor data, int nbatch) -> Tensor");
  m.def("abs_sum_grad(Tensor data, Tensor grad, int nbatch) -> Tensor");
  m.def("lee_filter(Tensor img, int nbatch, int Nx, int Ny, int wx, int wy, float cu) -> Tensor");
  m.def("coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor");
  m.def("coherence_2d_grad(Tensor grad, Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor[]");
  m.def("power_coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1, int corr_output) -> Tensor");
  m.def("projection_cart_2d(Tensor img, Tensor dem, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float fs, float gamma, float x0, float dx, float y0, float dy, int Nx, int Ny, float d0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int use_rvp, int normalization) -> Tensor");
  m.def("subpixel_correlation(Tensor im_m, Tensor im_s, Tensor mean_m, Tensor mean_s, int nbatch, int N0, int N1) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cpu);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cpu);
  m.impl("polar_interp_linear", &polar_interp_linear_cpu);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cpu);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cpu);
}

}
