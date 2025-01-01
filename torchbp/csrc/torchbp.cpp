#include <omp.h>
#include <torch/extension.h>

namespace torchbp {

#define kPI 3.1415926535897932384626433f
#define kC0 299792458.0f

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

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
static float interp2d_abs(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return abs(img[x_int*ny + y_int])*(1.0f-x_frac)*(1.0f-y_frac) +
           abs(img[x_int*ny + y_int+1])*(1.0f-x_frac)*y_frac +
           abs(img[(x_int+1)*ny + y_int])*x_frac*(1.0f-y_frac) +
           abs(img[(x_int+1)*ny + y_int+1])*x_frac*y_frac;
}

template <typename T>
static void sincospi(T x, T *sinx, T *cosx) {
    *sinx = sin(static_cast<T>(kPI) * x);
    *cosx = cos(static_cast<T>(kPI) * x);
}

template <typename T>
void polar_interp_kernel_linear_cpu(const c10::complex<T> *img, c10::complex<T> *out, const T *dorigin, T rotation,
                  T ref_phase, T r0, T dr, T theta0, T dtheta, int Nr, int Ntheta,
                  T r1, T dr1, T theta1, T dtheta1, int Nr1, int Ntheta1, int idx, int idbatch) {
    const int idtheta = idx % Ntheta1;
    const int idr = idx / Ntheta1;

    if (idx >= Nr1 * Ntheta1) {
        return;
    }

    const T d = r1 + dr1 * idr;
    T t = theta1 + dtheta1 * idtheta;
    if (rotation != 0.0f) {
        t = sinf(asinf(t) + rotation);
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const T dorig0 = dorigin[idbatch * 2 + 0];
    const T dorig1 = dorigin[idbatch * 2 + 1];
    const T sint = t;
    const T cost = sqrtf(1.0f - t*t);
    const T rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const T arg = (d*sint + dorig1) / (d*cost + dorig0);
    const T tp = arg / sqrtf(1.0f + arg*arg);

    const T dri = (rp - r0) / dr;
    const T dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const T dri_frac = dri - dri_int;
    const int dti_int = dti;
    const T dti_frac = dti - dti_int;

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        c10::complex<T> v = interp2d<c10::complex<T>>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        T ref_sin, ref_cos;
        sincospi<T>(ref_phase * (rp - d), &ref_sin, &ref_cos);
        c10::complex<T> ref = {ref_cos, ref_sin};
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = v * ref;
    } else {
        out[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] = {0.0f, 0.0f};
    }
}

template <typename T>
void polar_interp_kernel_linear_grad_cpu(const c10::complex<T> *img, const T *dorigin, T rotation,
                  T ref_phase, T r0, T dr, T theta0, T dtheta, int Nr, int Ntheta,
                  T r1, T dr1, T theta1, T dtheta1, int Nr1, int Ntheta1,
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
        t = sinf(asinf(t) + rotation);
    }

    if (idx >= Nr1 * Ntheta1) {
        return;
    }
    if (t < -1.0f || t > 1.0f) {
        return;
    }
    const T dorig0 = dorigin[idbatch * 2 + 0];
    const T dorig1 = dorigin[idbatch * 2 + 1];
    const T sint = t;
    const T cost = sqrtf(1.0f - t*t);
    const T rp = sqrtf(d*d + dorig0*dorig0 + dorig1*dorig1 + 2*d*(dorig0*cost + dorig1*sint));
    const T arg = (d*sint + dorig1) / (d*cost + dorig0);
    const T tp = arg / sqrtf(1.0f + arg*arg);

    const T dri = (rp - r0) / dr;
    const T dti = (tp - theta0) / dtheta;

    const int dri_int = dri;
    const T dri_frac = dri - dri_int;

    const int dti_int = dti;
    const T dti_frac = dti - dti_int;

    c10::complex<T> v = {0.0f, 0.0f};
    c10::complex<T> ref = {0.0f, 0.0f};

    if (dri_int >= 0 && dri_int < Nr-1 && dti_int >= 0 && dti_int < Ntheta-1) {
        v = interp2d<c10::complex<T>>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
        T ref_sin, ref_cos;
        sincospi<T>(ref_phase * (rp - d), &ref_sin, &ref_cos);
        ref = {ref_cos, ref_sin};
    }

    if (dorigin_grad != nullptr) {
        c10::complex<T> dout_drp = I * static_cast<T>(kPI) * ref_phase * ref * v;
        T drp_dorigin0 = 0.50f * (2.0f*d*cost + 2.0f*dorig0) / rp;
        T drp_dorigin1 = 0.50f * (2.0f*d*sint + 2.0f*dorig1) / rp;

        c10::complex<T> drp_conj = std::conj(dout_drp);
        c10::complex<T> gdrp = grad[idbatch * Nr1 * Ntheta1 + idr*Ntheta1 + idtheta] * drp_conj;
        T gd = std::real(gdrp);
        drp_dorigin0 *= gd;
        drp_dorigin1 *= gd;

#pragma omp atomic
        dorigin_grad[idbatch * 2 + 0] += drp_dorigin0;
#pragma omp atomic
        dorigin_grad[idbatch * 2 + 1] += drp_dorigin1;
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
          int64_t ntheta1) {
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
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, idx, idbatch);
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
                      r1, dr1, theta1, dtheta1, nr1, ntheta1, idx, idbatch);
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
          int64_t ntheta1) {
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
                      r1, dr1, theta1, dtheta1, nr1, ntheta1,
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
                      r1, dr1, theta1, dtheta1, nr1, ntheta1,
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

void backprojection_polar_2d_kernel_cpu(
          const complex64_t* data,
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
          float ant_tx_dy,
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

        // Calculate distance to the pixel.

        float dtx = sqrtf(px * px + py * py + pz2);
        float drx = sqrt((px + tx_dx) * (px + tx_dx) + (py + tx_dy) * (py + tx_dy) + pz2);
        float d = drx + dtx - d0;

        float sx = delta_r * d;

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
        sincospi(ref_phase * d, &ref_sin, &ref_cos);
        complex64_t ref = {ref_cos, ref_sin};
        pixel += s * ref;
    }
    img[idbatch * Nr * Ntheta + idr * Ntheta + idtheta] = pixel;
}

at::Tensor backprojection_polar_2d_cpu(
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);

	at::Tensor pos_contig = pos.contiguous();
	at::Tensor vel_contig = vel.contiguous();
	at::Tensor att_contig = att.contiguous();
	at::Tensor data_contig = data.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(torch::kComplexFloat)
        .layout(torch::kStrided)
        .device(torch::kCPU, 1);
	at::Tensor img = torch::zeros({nbatch, Nr, Ntheta}, options);
	const float* pos_ptr = pos_contig.data_ptr<float>();
	const float* vel_ptr = vel_contig.data_ptr<float>();
	const float* att_ptr = att_contig.data_ptr<float>();
	const c10::complex<float>* data_ptr = data_contig.data_ptr<c10::complex<float>>();
    c10::complex<float>* img_ptr = img.data_ptr<c10::complex<float>>();

    // delta_r, ref_phase, d0 are bistatic, 2x monostatic
	const float delta_r = 0.5f / r_res;
    const float ref_phase = 2.0f * fc / kC0;
    d0 *= 2.0f;

#pragma omp parallel for collapse(2)
    for(int idx = 0; idx < Nr * Ntheta; idx++) {
        for(int idbatch = 0; idbatch < nbatch; idbatch++) {
            backprojection_polar_2d_kernel_cpu(
                          data_ptr,
                          pos_ptr,
                          vel_ptr,
                          att_ptr,
                          img_ptr,
                          sweep_samples,
                          nsweeps,
                          ref_phase,
                          delta_r,
                          r0, dr,
                          theta0, dtheta,
                          Nr, Ntheta,
                          d0, ant_tx_dy,
                          idx, idbatch);
        }
    }
	return img;
}

void backprojection_polar_2d_grad_kernel_cpu(
          const complex64_t* data,
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

        float tx_dx = sinf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;
        float tx_dy = cosf(att[idbatch * nsweeps * 3 + i * 3 + 2]) * ant_tx_dy;

        // Calculate distance to the pixel.

        float dtx = sqrtf(px * px + py * py + pz2);
        float drx = sqrt((px + tx_dx) * (px + tx_dx) + (py + tx_dy) * (py + tx_dy) + pz2);
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
            sincospi(ref_phase * d, &ref_sin, &ref_cos);
            complex64_t ref = {ref_cos, ref_sin};

            if (have_pos_grad) {
                complex64_t dout = ref * ((I * kPI * ref_phase) * s + (s1 - s0) * delta_r);
                complex64_t gdout = g * std::conj(dout);

                // Take real part
                float gd = std::real(gdout);

                dx = -px / dtx - (px + tx_dx) / drx;
                dy = -py / dtx - (py + tx_dy) / drx;
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
            // Slow
            #pragma omp critical
            {
                data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id0] += ds0;
                data_grad[idbatch * sweep_samples * nsweeps + i * sweep_samples + id1] += ds1;
            }
            }
        }
    }
}

std::vector<at::Tensor> backprojection_polar_2d_grad_cpu(
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
	TORCH_CHECK(data.dtype() == at::kComplexFloat);
	TORCH_CHECK(grad.dtype() == at::kComplexFloat);
	TORCH_INTERNAL_ASSERT(pos.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(vel.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(att.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(data.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(grad.device().type() == at::DeviceType::CPU);
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

#pragma omp parallel for collapse(2)
    for(int idbatch = 0; idbatch < nbatch; idbatch++) {
        for(int idx = 0; idx < Nr * Ntheta; idx++) {
            backprojection_polar_2d_grad_kernel_cpu(
                          data_ptr,
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
void polar_to_cart_kernel_linear_cpu(const T *img, T
        *out, const float *dorigin, float rotation, float ref_phase, float r0,
        float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0,
        float dx, float y0, float dy, int Nx, int Ny, int polar_interp,
        int id1, int idbatch) {
    const int idy = id1 % Ny;
    const int idx = id1 / Ny;

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
        if constexpr (std::is_same_v<T, complex64_t>) {
            // This is needed to avoid artifacts in amplitude when
            // range dimension spectrum is not band limited
            if (polar_interp) {
                float absv = interp2d_abs<T>(&img[idbatch * Nr * Ntheta], Nr, Ntheta, dri_int, dri_frac, dti_int, dti_frac);
                v = absv * v / abs(v);
            }
            float ref_sin, ref_cos;
            sincospi(ref_phase * d, &ref_sin, &ref_cos);
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
	TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(dorigin.device().type() == at::DeviceType::CPU);
	at::Tensor dorigin_contig = dorigin.contiguous();
	at::Tensor img_contig = img.contiguous();
	at::Tensor out = torch::empty({nbatch, Nx, Ny}, img_contig.options());
	const float* dorigin_ptr = dorigin_contig.data_ptr<float>();

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
                              polar_interp,
                              id1,
                              idbatch
                              );
            } else {
                float* img_ptr = img_contig.data_ptr<float>();
                float* out_ptr = out.data_ptr<float>();
                polar_to_cart_kernel_linear_cpu<float>(
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
                              polar_interp,
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
  m.def("backprojection_polar_2d(Tensor data, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, float ant_tx_dy) -> Tensor");
  m.def("backprojection_polar_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, float ant_tx_dy) -> Tensor[]");
  m.def("backprojection_cart_2d(Tensor data, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float ant_tx_dy) -> Tensor");
  m.def("backprojection_cart_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float ant_tx_dy) -> Tensor[]");
  m.def("polar_interp_linear(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1) -> Tensor");
  m.def("polar_interp_linear_grad(Tensor grad, Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1) -> Tensor[]");
  m.def("polar_to_cart_linear(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, int polar_interp) -> Tensor");
  m.def("polar_to_cart_linear_grad(Tensor grad, Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny) -> Tensor[]");
  m.def("polar_to_cart_bicubic(Tensor img, Tensor img_gx, Tensor img_gy, Tensor img_gxy, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, int polar_interp) -> Tensor");
  m.def("polar_to_cart_bicubic_grad(Tensor grad, Tensor img, Tensor img_gx, Tensor img_gy, Tensor img_gxy, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny) -> Tensor[]");
  m.def("backprojection_polar_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor gtx, Tensor grx, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta) -> Tensor");
  m.def("entropy(Tensor data, Tensor norm, int nbatch) -> Tensor");
  m.def("entropy_grad(Tensor data, Tensor norm, Tensor grad, int nbatch) -> Tensor[]");
  m.def("abs_sum(Tensor data, int nbatch) -> Tensor");
  m.def("abs_sum_grad(Tensor data, Tensor grad, int nbatch) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("backprojection_polar_2d", &backprojection_polar_2d_cpu);
  m.impl("backprojection_polar_2d_grad", &backprojection_polar_2d_grad_cpu);
  m.impl("polar_interp_linear", &polar_interp_linear_cpu);
  m.impl("polar_interp_linear_grad", &polar_interp_linear_grad_cpu);
  m.impl("polar_to_cart_linear", &polar_to_cart_linear_cpu);
}

}
