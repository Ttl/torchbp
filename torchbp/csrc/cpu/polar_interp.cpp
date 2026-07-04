#include "util.h"

// CPU polar interpolation, polar-to-cart and FFBP merge ops. Mirrors cuda/polar_interp.cu.
namespace torchbp {

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

static void polar_interp_kernel_lanczos_cpu(const complex64_t *img, complex64_t *out, const float *dorigin,
                  float rotation, float ref_phase, float r0, float dr, float theta0, float dtheta,
                  int Nr, int Ntheta, float r1, float dr1, float theta1, float dtheta1,
                  int Nr1, int Ntheta1, float z1, int order, float alias_fmod, int idx, int idbatch) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

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
    const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const float arg = (d*sint + dorig1) / (d*cost + dorig0);
    const float tp = arg / sqrtf(1.0f + arg*arg);

    const float dri = (rp - r0) / dr;
    const float dti = (tp - theta0) / dtheta;

    if (dri >= 0 && dri < Nr-1 && dti >= 0 && dti < Ntheta-1) {
        complex64_t v = lanczos_interp_2d_cpu<complex64_t>(
                &img[idbatch * Nr * Ntheta], Nr, Ntheta, dri, dti, order);
        float ref_sin, ref_cos;
        const float z0 = z1 + dorigin[idbatch * 3 + 2];
        const float dz = sqrtf(z1*z1 + d*d);
        const float rpz = sqrtf(z0*z0 + rp*rp);
        sincospi<float>(ref_phase * (rpz - dz) - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
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

at::Tensor polar_interp_lanczos_cpu(
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
          int64_t order,
          double alias_fmod) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat);
    TORCH_CHECK(dorigin.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, nr1, ntheta1}, img_contig.options());
    at::Tensor dorigin_contig = dorigin.contiguous();

    // See polar_interp_linear_cpu for why the thread count is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    const float* dorigin_ptr = dorigin_contig.data_ptr<float>();
    complex64_t* img_ptr = img_contig.data_ptr<complex64_t>();
    complex64_t* out_ptr = out.data_ptr<complex64_t>();
    const float ref_phase = 4.0f * fc / kC0;

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < nr1 * ntheta1; idx++) {
            polar_interp_kernel_lanczos_cpu(img_ptr, out_ptr, dorigin_ptr, rotation,
                  ref_phase, r0, dr0, theta0, dtheta0, nr0, ntheta0,
                  r1, dr1, theta1, dtheta1, nr1, ntheta1, z1, order, alias_fmod/kPI, idx, idbatch);
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

// Per-pixel math matches polar_to_cart_kernel_linear_grad in
// cuda/polar_interp.cu.
static void polar_to_cart_kernel_linear_grad_cpu(const complex64_t *img,
        const float *origin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, float alias_fmod,
        const complex64_t *grad, complex64_t *img_grad, float *origin_grad,
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
    float cosa = t*rs + tc*rc;
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
        complex64_t v = interp2d<complex64_t>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        float ref_sin, ref_cos;
        sincospi<float>(ref_phase * dz - alias_fmod * dri, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};

        if (origin_grad != nullptr) {
            const complex64_t I = {0.0f, 1.0f};

            const complex64_t dref_dz = I * kPI * ref_phase * ref;
            const complex64_t dref_ddri = -I * kPI * alias_fmod * ref;
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
            const complex64_t dref_dorig0 = dref_dz * dz_dorig0 + dref_ddri * dd_dorig0 / dr;
            const complex64_t dref_dorig1 = dref_dz * dz_dorig1 + dref_ddri * dd_dorig1 / dr;
            const complex64_t dref_dorig2 = dref_dz * dz_dorig2;
            const complex64_t dout_dorig0 = dv_dorig0 * ref + v * dref_dorig0;
            const complex64_t dout_dorig1 = dv_dorig1 * ref + v * dref_dorig1;
            const complex64_t dout_dorig2 = v * dref_dorig2;

            const complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy];
            const float g_origin0 = std::real(g * std::conj(dout_dorig0));
            const float g_origin1 = std::real(g * std::conj(dout_dorig1));
            const float g_origin2 = std::real(g * std::conj(dout_dorig2));

#pragma omp atomic
            origin_grad[idbatch * 3 + 0] += g_origin0;
#pragma omp atomic
            origin_grad[idbatch * 3 + 1] += g_origin1;
#pragma omp atomic
            origin_grad[idbatch * 3 + 2] += g_origin2;
        }

        if (img_grad != nullptr) {
            complex64_t g = grad[idbatch * Nx * Ny + idx*Ny + idy] * std::conj(ref);

            complex64_t g11 = g * (1.0f-dri_frac)*(1.0f-dti_frac);
            complex64_t g12 = g * (1.0f-dri_frac)*dti_frac;
            complex64_t g21 = g * dri_frac*(1.0f-dti_frac);
            complex64_t g22 = g * dri_frac*dti_frac;
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

std::vector<at::Tensor> polar_to_cart_linear_grad_cpu(
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
          int64_t Ny,
          double alias_fmod) {
	TORCH_CHECK(img.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_CHECK(origin.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(origin.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
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

    const float ref_phase = 4.0f * fc / kC0;

    // See polar_interp_linear_cpu for why the thread count is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int id1 = 0; id1 < Nx * Ny; id1++) {
            polar_to_cart_kernel_linear_grad_cpu(
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
                          alias_fmod/kPI,
                          (const complex64_t*)grad_ptr,
                          (complex64_t*)img_grad_ptr,
                          origin_grad_ptr,
                          id1,
                          idbatch);
        }
    }
    std::vector<at::Tensor> ret;
    ret.push_back(img_grad);
    ret.push_back(origin_grad);
	return ret;
}

// FFBP merge kernels: one theta chunk of one output range row per call.
//
// Structured for SIMD like backprojection_polar_2d_row_cpu: for each of the
// two input images a vectorizable geometry + phase pass writes per-pixel
// input-grid coordinates and the reference phasor to small buffers, then a
// scalar pass does the data-dependent interpolation gather and accumulates.
// Per-pixel math matches the ffbp_merge2 kernels in cuda/polar_interp.cu.
constexpr int MERGE_CHUNK = 256;

// Geometry + phase pass for one input image. dri_buf < 0 marks pixels that
// fall outside the input grid (or outside |theta| <= 1).
static void ffbp_merge2_geom_pass_cpu(float d, float dz, float theta1,
        float dtheta1, int tb, float dorig0, float dorig1, float dorig2,
        float z1, float ref_phase, float r0, float dr, float theta0,
        float dtheta, int Nr, int Ntheta, float alias_fmod, int idr,
        int nchunk, float *dri_buf, float *dti_buf, float *rp_buf,
        float *tp_buf, float *cs_buf, float *sn_buf) {
    const float z0 = z1 + dorig2;
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        const float t = theta1 + dtheta1 * (tb + q);
        const bool tok = (t >= -1.0f) & (t <= 1.0f);
        // Zero theta on the out-of-range lanes so sqrtf sees a non-negative
        // argument and no NaNs enter the pipeline.
        const float sint = tok ? t : 0.0f;
        const float cost = sqrtf(1.0f - sint*sint);
        const float rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
        const float arg = (d*sint + dorig1) / (d*cost + dorig0);
        const float tp = arg / sqrtf(1.0f + arg*arg);

        const float dri = (rp - r0) / dr;
        const float dti = (tp - theta0) / dtheta;

        const bool ok = tok & (dri >= 0.0f) & (dri < Nr-1) &
                        (dti >= 0.0f) & (dti < Ntheta-1);

        const float rpz = sqrtf(z0*z0 + rp*rp);
        float ref_sin, ref_cos;
        sincospi<float>(ref_phase * (rpz - dz) - alias_fmod*(dri - idr), &ref_sin, &ref_cos);
        cs_buf[q] = ref_cos;
        sn_buf[q] = ref_sin;
        rp_buf[q] = rp;
        tp_buf[q] = tp;
        dri_buf[q] = ok ? dri : -1.0f;
        dti_buf[q] = dti;
    }
}

// Interpolation window and weight pass for the polynomial merge kernels:
// computes the same clamped window and weights as interp_2d_poly in
// cuda/util.h, but for a whole chunk of pixels at once so that the Horner
// recurrences vectorize. Weights are stored [tap][pixel]. Lanes with
// dri_buf < 0 hold garbage; the scalar pass skips them.
template<int N_COEFS, int MAX_ORDER>
static void ffbp_merge2_poly_weights_pass_cpu(const float *coefs, int order,
        int Nr, int Ntheta, int nchunk,
        const float *dri_buf, const float *dti_buf,
        int *ir_buf, int *it_buf, int *nrc_buf, int *ntc_buf,
        float wr_buf[][MERGE_CHUNK], float wt_buf[][MERGE_CHUNK]) {
    const float a = 0.5f * order;
    const float inv_a2 = 1.0f / (a * a);
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        const float x = dri_buf[q];
        const float y = dti_buf[q];
        const int start_x = std::max(0, (int)ceilf(x - a));
        const int end_x = std::min(Nr-1, (int)floorf(x + a));
        const int start_y = std::max(0, (int)ceilf(y - a));
        const int end_y = std::min(Ntheta-1, (int)floorf(y + a));
        ir_buf[q] = start_x;
        it_buf[q] = start_y;
        nrc_buf[q] = std::min(end_x - start_x + 1, MAX_ORDER);
        ntc_buf[q] = std::min(end_y - start_y + 1, MAX_ORDER);
        for (int j = 0; j < MAX_ORDER; j++) {
            wr_buf[j][q] = poly_interp_kernel_cpu<N_COEFS>(coefs, x - (float)(start_x + j), inv_a2);
            wt_buf[j][q] = poly_interp_kernel_cpu<N_COEFS>(coefs, y - (float)(start_y + j), inv_a2);
        }
    }
}

// Scalar per-pixel interpolation tap loop of the polynomial merge kernels:
// same taps and accumulation order as interp_2d_poly in cuda/util.h with the
// window and weights read from the buffers filled by the weight pass.
static inline complex64_t ffbp_merge2_poly_taps_cpu(const complex64_t *img,
        int Ntheta, int q,
        const int *ir_buf, const int *it_buf, const int *nrc_buf, const int *ntc_buf,
        const float wr_buf[][MERGE_CHUNK], const float wt_buf[][MERGE_CHUNK]) {
    // Accumulate the real and imaginary parts in plain floats: gcc optimizes
    // this much better than the c10::complex operator chain.
    float sr = 0.0f, si = 0.0f;
    for (int i = 0; i < nrc_buf[q]; i++) {
        const float *row = (const float*)(img + (size_t)(ir_buf[q] + i) * Ntheta + it_buf[q]);
        const float wxi = wr_buf[i][q];
        for (int j = 0; j < ntc_buf[q]; j++) {
            const float w = wxi * wt_buf[j][q];
            sr += row[2*j] * w;
            si += row[2*j+1] * w;
        }
    }
    return {sr, si};
}

// Applies the optional output-alias phase common to all merge kernels. The
// phasor depends only on the range row, so one chunk shares a single phasor.
static void apply_merge_alias(float *accr, float *acci, int nchunk, int alias,
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
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        const float pr = accr[q] * ref_cos - acci[q] * ref_sin;
        const float pi = accr[q] * ref_sin + acci[q] * ref_cos;
        accr[q] = pr;
        acci[q] = pi;
    }
}

static void ffbp_merge2_kernel_lanczos_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, int alias, float alias_fmod,
        int idr, int tb) {
    float dri_buf[MERGE_CHUNK], dti_buf[MERGE_CHUNK];
    float rp_buf[MERGE_CHUNK], tp_buf[MERGE_CHUNK];
    float cs_buf[MERGE_CHUNK], sn_buf[MERGE_CHUNK];
    float accr[MERGE_CHUNK], acci[MERGE_CHUNK];

    const int nchunk = std::min(MERGE_CHUNK, Ntheta1 - tb);
    const float d = r1 + dr1 * idr;
    const float dz = sqrtf(z1*z1 + d*d);

#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        accr[q] = 0.0f;
        acci[q] = 0.0f;
    }

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        ffbp_merge2_geom_pass_cpu(d, dz, theta1, dtheta1, tb,
                dorigin[id * 3 + 0], dorigin[id * 3 + 1], dorigin[id * 3 + 2],
                z1, ref_phase, r0[id], dr[id], theta0[id], dtheta[id],
                Nr[id], Ntheta[id], alias_fmod, idr, nchunk,
                dri_buf, dti_buf, rp_buf, tp_buf, cs_buf, sn_buf);
        // Scalar pass: data-dependent interpolation gather.
        for (int q = 0; q < nchunk; q++) {
            const float dri = dri_buf[q];
            if (dri < 0.0f) {
                continue;
            }
            complex64_t v = lanczos_interp_2d_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti_buf[q], order);
            accr[q] += v.real() * cs_buf[q] - v.imag() * sn_buf[q];
            acci[q] += v.real() * sn_buf[q] + v.imag() * cs_buf[q];
        }
    }
    apply_merge_alias(accr, acci, nchunk, alias, ref_phase, alias_fmod, dz, idr);
    complex64_t *out_row = out + (size_t)idr * Ntheta1 + tb;
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        out_row[q] = complex64_t(accr[q], acci[q]);
    }
}

static void ffbp_merge2_kernel_knab_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, float knab_v, int alias,
        float alias_fmod, int idr, int tb) {
    float dri_buf[MERGE_CHUNK], dti_buf[MERGE_CHUNK];
    float rp_buf[MERGE_CHUNK], tp_buf[MERGE_CHUNK];
    float cs_buf[MERGE_CHUNK], sn_buf[MERGE_CHUNK];
    float accr[MERGE_CHUNK], acci[MERGE_CHUNK];

    const int nchunk = std::min(MERGE_CHUNK, Ntheta1 - tb);
    const float d = r1 + dr1 * idr;
    const float dz = sqrtf(z1*z1 + d*d);
    const float knab_norm = knab_kernel_norm_cpu(order, knab_v);

#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        accr[q] = 0.0f;
        acci[q] = 0.0f;
    }

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        ffbp_merge2_geom_pass_cpu(d, dz, theta1, dtheta1, tb,
                dorigin[id * 3 + 0], dorigin[id * 3 + 1], dorigin[id * 3 + 2],
                z1, ref_phase, r0[id], dr[id], theta0[id], dtheta[id],
                Nr[id], Ntheta[id], alias_fmod, idr, nchunk,
                dri_buf, dti_buf, rp_buf, tp_buf, cs_buf, sn_buf);
        // Scalar pass: data-dependent interpolation gather.
        for (int q = 0; q < nchunk; q++) {
            const float dri = dri_buf[q];
            if (dri < 0.0f) {
                continue;
            }
            complex64_t v = knab_interp_2d_cpu<complex64_t>(
                    img, Nr[id], Ntheta[id], dri, dti_buf[q], order, knab_v, knab_norm);
            accr[q] += v.real() * cs_buf[q] - v.imag() * sn_buf[q];
            acci[q] += v.real() * sn_buf[q] + v.imag() * cs_buf[q];
        }
    }
    apply_merge_alias(accr, acci, nchunk, alias, ref_phase, alias_fmod, dz, idr);
    complex64_t *out_row = out + (size_t)idr * Ntheta1 + tb;
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        out_row[q] = complex64_t(accr[q], acci[q]);
    }
}

template<int N_COEFS>
static void ffbp_merge2_kernel_poly_cpu(const complex64_t *img0, const complex64_t *img1,
        complex64_t *out, const float *dorigin,
        float ref_phase, const float *r0, const float *dr, const float *theta0,
        const float *dtheta, const int *Nr, const int *Ntheta, float r1, float dr1, float theta1,
        float dtheta1, int Nr1, int Ntheta1, float z1, int order, const float *coefs,
        int alias, float alias_fmod, int idr, int tb) {
    constexpr int MAX_ORDER = 8;
    float dri_buf[MERGE_CHUNK], dti_buf[MERGE_CHUNK];
    float rp_buf[MERGE_CHUNK], tp_buf[MERGE_CHUNK];
    float cs_buf[MERGE_CHUNK], sn_buf[MERGE_CHUNK];
    float accr[MERGE_CHUNK], acci[MERGE_CHUNK];
    float wr_buf[MAX_ORDER][MERGE_CHUNK], wt_buf[MAX_ORDER][MERGE_CHUNK];
    int ir_buf[MERGE_CHUNK], it_buf[MERGE_CHUNK];
    int nrc_buf[MERGE_CHUNK], ntc_buf[MERGE_CHUNK];

    const int nchunk = std::min(MERGE_CHUNK, Ntheta1 - tb);
    const float d = r1 + dr1 * idr;
    const float dz = sqrtf(z1*z1 + d*d);

#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        accr[q] = 0.0f;
        acci[q] = 0.0f;
    }

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        ffbp_merge2_geom_pass_cpu(d, dz, theta1, dtheta1, tb,
                dorigin[id * 3 + 0], dorigin[id * 3 + 1], dorigin[id * 3 + 2],
                z1, ref_phase, r0[id], dr[id], theta0[id], dtheta[id],
                Nr[id], Ntheta[id], alias_fmod, idr, nchunk,
                dri_buf, dti_buf, rp_buf, tp_buf, cs_buf, sn_buf);
        ffbp_merge2_poly_weights_pass_cpu<N_COEFS, MAX_ORDER>(coefs, order,
                Nr[id], Ntheta[id], nchunk, dri_buf, dti_buf,
                ir_buf, it_buf, nrc_buf, ntc_buf, wr_buf, wt_buf);
        // Scalar pass: data-dependent interpolation gather.
        for (int q = 0; q < nchunk; q++) {
            if (dri_buf[q] < 0.0f) {
                continue;
            }
            complex64_t v = ffbp_merge2_poly_taps_cpu(img, Ntheta[id], q,
                    ir_buf, it_buf, nrc_buf, ntc_buf, wr_buf, wt_buf);
            accr[q] += v.real() * cs_buf[q] - v.imag() * sn_buf[q];
            acci[q] += v.real() * sn_buf[q] + v.imag() * cs_buf[q];
        }
    }
    apply_merge_alias(accr, acci, nchunk, alias, ref_phase, alias_fmod, dz, idr);
    complex64_t *out_row = out + (size_t)idr * Ntheta1 + tb;
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        out_row[q] = complex64_t(accr[q], acci[q]);
    }
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

    const int ntchunks = (Ntheta1 + MERGE_CHUNK - 1) / MERGE_CHUNK;
#pragma omp parallel for collapse(2)
    for (int idr = 0; idr < Nr1; idr++) {
        for (int tc = 0; tc < ntchunks; tc++) {
            ffbp_merge2_kernel_lanczos_cpu(img0_ptr, img1_ptr, out_ptr, dorigin_ptr,
                    ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                    r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, alias, alias_fmod/kPI,
                    idr, tc * MERGE_CHUNK);
        }
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

    const int ntchunks = (Ntheta1 + MERGE_CHUNK - 1) / MERGE_CHUNK;
#pragma omp parallel for collapse(2)
    for (int idr = 0; idr < Nr1; idr++) {
        for (int tc = 0; tc < ntchunks; tc++) {
            ffbp_merge2_kernel_knab_cpu(img0_ptr, img1_ptr, out_ptr, dorigin_ptr,
                    ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr,
                    r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, v, alias, alias_fmod/kPI,
                    idr, tc * MERGE_CHUNK);
        }
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

    const int ntchunks = (Ntheta1 + MERGE_CHUNK - 1) / MERGE_CHUNK;

    // Dispatch to template-specialized kernel based on n_coefs. This enables
    // full compile-time unrolling of the polynomial evaluation. Mirrors the
    // dispatch in ffbp_merge2_poly_cuda.
    #define LAUNCH_KERNEL(N) \
        _Pragma("omp parallel for collapse(2)") \
        for (int idr = 0; idr < Nr1; idr++) { \
            for (int tc = 0; tc < ntchunks; tc++) { \
                ffbp_merge2_kernel_poly_cpu<N>(img0_ptr, img1_ptr, out_ptr, dorigin_ptr, \
                        ref_phase, r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr, \
                        r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, coefs_ptr, \
                        alias, alias_fmod/kPI, idr, tc * MERGE_CHUNK); \
            } \
        }

    switch (n_coefs) {
        case 4: LAUNCH_KERNEL(4); break;
        case 5: LAUNCH_KERNEL(5); break;
        case 6: LAUNCH_KERNEL(6); break;
        case 7: LAUNCH_KERNEL(7); break;
        case 8: LAUNCH_KERNEL(8); break;
        case 9: LAUNCH_KERNEL(9); break;
        case 10: LAUNCH_KERNEL(10); break;
        case 11: LAUNCH_KERNEL(11); break;
        case 12: LAUNCH_KERNEL(12); break;
        case 13: LAUNCH_KERNEL(13); break;
        case 14: LAUNCH_KERNEL(14); break;
        default:
            TORCH_CHECK(false, "ffbp_merge2_poly: n_coefs must be 4-14, got ", n_coefs);
    }
    #undef LAUNCH_KERNEL
    return out;
}

template<int N_COEFS>
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
        float z1, int order, const float *coefs, int alias, float alias_fmod,
        const float *w1_map0, const float *w2_map0,
        float w_r0_0, float w_dr0, float w_theta0_0, float w_dtheta0,
        int w_nr0, int w_ntheta0,
        const float *w1_map1, const float *w2_map1,
        float w_r0_1, float w_dr1, float w_theta0_1, float w_dtheta1,
        int w_nr1, int w_ntheta1,
        int output_weight_decimation, int idr, int tb) {
    constexpr int MAX_ORDER = 8;
    float dri_buf[MERGE_CHUNK], dti_buf[MERGE_CHUNK];
    float rp_buf[MERGE_CHUNK], tp_buf[MERGE_CHUNK];
    float cs_buf[MERGE_CHUNK], sn_buf[MERGE_CHUNK];
    float accr[MERGE_CHUNK], acci[MERGE_CHUNK];
    float accw1[MERGE_CHUNK], accw2[MERGE_CHUNK];
    float wr_buf[MAX_ORDER][MERGE_CHUNK], wt_buf[MAX_ORDER][MERGE_CHUNK];
    int ir_buf[MERGE_CHUNK], it_buf[MERGE_CHUNK];
    int nrc_buf[MERGE_CHUNK], ntc_buf[MERGE_CHUNK];

    const int nchunk = std::min(MERGE_CHUNK, Ntheta1 - tb);
    const float d = r1 + dr1 * idr;
    const float dz = sqrtf(z1*z1 + d*d);

    const float *w1_maps[2] = {w1_map0, w1_map1};
    const float *w2_maps[2] = {w2_map0, w2_map1};
    const float w_r0[2] = {w_r0_0, w_r0_1};
    const float w_dr[2] = {w_dr0, w_dr1};
    const float w_theta0[2] = {w_theta0_0, w_theta0_1};
    const float w_dtheta[2] = {w_dtheta0, w_dtheta1};
    const int w_nr[2] = {w_nr0, w_nr1};
    const int w_ntheta[2] = {w_ntheta0, w_ntheta1};

#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        accr[q] = 0.0f;
        acci[q] = 0.0f;
        accw1[q] = 0.0f;
        accw2[q] = 0.0f;
    }

    for (int id=0; id < 2; id++) {
        const complex64_t *img = id == 0 ? img0 : img1;
        ffbp_merge2_geom_pass_cpu(d, dz, theta1, dtheta1, tb,
                dorigin[id * 3 + 0], dorigin[id * 3 + 1], dorigin[id * 3 + 2],
                z1, ref_phase, r0[id], dr[id], theta0[id], dtheta[id],
                Nr[id], Ntheta[id], alias_fmod, idr, nchunk,
                dri_buf, dti_buf, rp_buf, tp_buf, cs_buf, sn_buf);
        ffbp_merge2_poly_weights_pass_cpu<N_COEFS, MAX_ORDER>(coefs, order,
                Nr[id], Ntheta[id], nchunk, dri_buf, dti_buf,
                ir_buf, it_buf, nrc_buf, ntc_buf, wr_buf, wt_buf);
        // Scalar pass: data-dependent interpolation and weight map gathers.
        for (int q = 0; q < nchunk; q++) {
            if (dri_buf[q] < 0.0f) {
                continue;
            }
            complex64_t v = ffbp_merge2_poly_taps_cpu(img, Ntheta[id], q,
                    ir_buf, it_buf, nrc_buf, ntc_buf, wr_buf, wt_buf);

            const float vr = v.real() * cs_buf[q] - v.imag() * sn_buf[q];
            const float vi = v.real() * sn_buf[q] + v.imag() * cs_buf[q];

            float w1 = 0.0f;
            float w2 = 0.0f;
            if (w1_maps[id] != nullptr && w2_maps[id] != nullptr) {
                const float w_ri = (rp_buf[q] - w_r0[id]) / w_dr[id];
                const float w_ti = (tp_buf[q] - w_theta0[id]) / w_dtheta[id];

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
                accr[q] += vr;
                acci[q] += vi;
                accw1[q] += w1;
                accw2[q] += w2;
            } else if (w1_maps[id] == nullptr) {
                accr[q] += vr;
                acci[q] += vi;
            }
        }
    }

    // Normalize only at final output (when no weight maps are emitted).
    if (w1_out == nullptr) {
#pragma omp simd
        for (int q = 0; q < nchunk; q++) {
            const float scale = accw2[q] > 0.0f ? accw1[q] / accw2[q] : 1.0f;
            accr[q] *= scale;
            acci[q] *= scale;
        }
    }

    apply_merge_alias(accr, acci, nchunk, alias, ref_phase, alias_fmod, dz, idr);
    complex64_t *out_row = out + (size_t)idr * Ntheta1 + tb;
#pragma omp simd
    for (int q = 0; q < nchunk; q++) {
        out_row[q] = complex64_t(accr[q], acci[q]);
    }

    const int dec = output_weight_decimation;
    if (w1_out != nullptr && idr % dec == 0) {
        const int out_ntheta_dec = (Ntheta1 + dec - 1) / dec;
        float *w1_row = w1_out + (size_t)(idr / dec) * out_ntheta_dec;
        float *w2_row = w2_out + (size_t)(idr / dec) * out_ntheta_dec;
        for (int q = 0; q < nchunk; q++) {
            const int idtheta = tb + q;
            if (idtheta % dec == 0) {
                w1_row[idtheta / dec] = accw1[q];
                w2_row[idtheta / dec] = accw2[q];
            }
        }
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

    const int ntchunks = (Ntheta1 + MERGE_CHUNK - 1) / MERGE_CHUNK;

    // Dispatch to template-specialized kernel based on n_coefs. This enables
    // full compile-time unrolling of the polynomial evaluation. Mirrors the
    // dispatch in ffbp_merge2_poly_weighted_cuda.
    #define LAUNCH_KERNEL(N) \
        _Pragma("omp parallel for collapse(2)") \
        for (int idr = 0; idr < Nr1; idr++) { \
            for (int tc = 0; tc < ntchunks; tc++) { \
                ffbp_merge2_kernel_poly_weighted_cpu<N>( \
                        img0_ptr, img1_ptr, out_ptr, w1_out_ptr, w2_out_ptr, dorigin_ptr, ref_phase, \
                        r0_ptr, dr0_ptr, theta0_ptr, dtheta0_ptr, Nr0_ptr, Ntheta0_ptr, \
                        r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1, order, coefs_ptr, \
                        alias, alias_fmod/kPI, \
                        w1_map0_ptr, w2_map0_ptr, w_r0_0, w_dr0, w_theta0_0, w_dtheta0, w_nr0, w_ntheta0, \
                        w1_map1_ptr, w2_map1_ptr, w_r0_1, w_dr1, w_theta0_1, w_dtheta1, w_nr1, w_ntheta1, \
                        dec, idr, tc * MERGE_CHUNK); \
            } \
        }

    switch (n_coefs) {
        case 4: LAUNCH_KERNEL(4); break;
        case 5: LAUNCH_KERNEL(5); break;
        case 6: LAUNCH_KERNEL(6); break;
        case 7: LAUNCH_KERNEL(7); break;
        case 8: LAUNCH_KERNEL(8); break;
        case 9: LAUNCH_KERNEL(9); break;
        case 10: LAUNCH_KERNEL(10); break;
        case 11: LAUNCH_KERNEL(11); break;
        case 12: LAUNCH_KERNEL(12); break;
        case 13: LAUNCH_KERNEL(13); break;
        case 14: LAUNCH_KERNEL(14); break;
        default:
            TORCH_CHECK(false, "ffbp_merge2_poly_weighted: n_coefs must be 4-14, got ", n_coefs);
    }
    #undef LAUNCH_KERNEL

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

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("polar_interp_linear", &polar_interp_linear_cpu);
  m.impl("polar_interp_lanczos", &polar_interp_lanczos_cpu);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cpu);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cpu);
  m.impl("polar_to_cart_linear_grad", &polar_to_cart_linear_grad_cpu);
  m.impl("polar_to_cart_lanczos", &polar_to_cart_lanczos_cpu);
  m.impl("ffbp_merge2_lanczos", &ffbp_merge2_lanczos_cpu);
  m.impl("ffbp_merge2_knab", &ffbp_merge2_knab_cpu);
  m.impl("ffbp_merge2_poly", &ffbp_merge2_poly_cpu);
  m.impl("ffbp_merge2_poly_weighted", &ffbp_merge2_poly_weighted_cpu);
}

}
