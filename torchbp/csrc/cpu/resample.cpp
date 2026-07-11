#include "util.h"

// CPU image resampling ops. Mirrors cuda/resample.cu.
namespace torchbp {

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

// Generic 1D signal rate change. Resamples N input samples to M output samples
// along the last axis; everything else is folded into nbatch. Output element
// id1 reads the input at continuous position id1 * (N / M) with a windowed-sinc
// kernel, lowpassed to the output rate when M < N (decimation). Mirrors
// resample_1d_*_kernel in cuda/resample.cu. interp_1d is a callable
// (row_base, src) -> T.
template<typename T, typename Interp>
static void resample_1d_kernel_cpu(
        const T* img, T* out,
        int N, int M, float step, Interp interp_1d, int id1, int idbatch) {
    if (id1 >= M) return;

    const float src = id1 * step;
    out[idbatch * M + id1] = interp_1d(&img[idbatch * N], src);
}

template<typename T, typename Interp>
static at::Tensor resample_1d_cpu_impl(
          const at::Tensor &img,
          int64_t nbatch, int64_t N, int64_t M,
          Interp interp_1d) {
    at::Tensor img_contig = img.contiguous();
    at::Tensor out = torch::empty({nbatch, M}, img_contig.options());

    const T* img_ptr = (const T*)img_contig.template data_ptr<T>();
    T* out_ptr = (T*)out.template data_ptr<T>();

    const float step = (float)N / (float)M;

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int id1 = 0; id1 < M; id1++) {
            resample_1d_kernel_cpu<T>(
                    img_ptr, out_ptr, N, M, step, interp_1d, id1, idbatch);
        }
    }
    return out;
}

at::Tensor resample_1d_lanczos_cpu(
          const at::Tensor &img,
          int64_t nbatch, int64_t N, int64_t M, int64_t order) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_1d_lanczos: img must be complex64 or float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);

    // Anti-alias cutoff: input Nyquist when up/equal-rate, output Nyquist when
    // decimating.
    const float cutoff = M >= N ? 1.0f : (float)M / (float)N;

    if (img.dtype() == at::kComplexFloat) {
        auto interp = [N, order, cutoff](const complex64_t* base, float src) {
            return lanczos_resample_1d_cpu<complex64_t>(base, N, src, order, cutoff);
        };
        return resample_1d_cpu_impl<complex64_t>(img, nbatch, N, M, interp);
    } else {
        auto interp = [N, order, cutoff](const float* base, float src) {
            return lanczos_resample_1d_cpu<float>(base, N, src, order, cutoff);
        };
        return resample_1d_cpu_impl<float>(img, nbatch, N, M, interp);
    }
}

at::Tensor resample_1d_knab_cpu(
          const at::Tensor &img,
          int64_t nbatch, int64_t N, int64_t M, int64_t order,
          double oversample) {
    TORCH_CHECK(img.dtype() == at::kComplexFloat || img.dtype() == at::kFloat,
                "resample_1d_knab: img must be complex64 or float32");
    TORCH_INTERNAL_ASSERT(img.device().type() == at::DeviceType::CPU);

    // Knab window parameter: v = 1 - 1/oversample
    const float v = 1.0f - 1.0f / static_cast<float>(oversample);
    const float norm = knab_kernel_norm_cpu(order, v);
    const float cutoff = M >= N ? 1.0f : (float)M / (float)N;

    if (img.dtype() == at::kComplexFloat) {
        auto interp = [N, order, v, norm, cutoff](const complex64_t* base, float src) {
            return knab_resample_1d_cpu<complex64_t>(base, N, src, order, v, norm, cutoff);
        };
        return resample_1d_cpu_impl<complex64_t>(img, nbatch, N, M, interp);
    } else {
        auto interp = [N, order, v, norm, cutoff](const float* base, float src) {
            return knab_resample_1d_cpu<float>(base, N, src, order, v, norm, cutoff);
        };
        return resample_1d_cpu_impl<float>(img, nbatch, N, M, interp);
    }
}

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("resample_2d_lanczos", &resample_2d_lanczos_cpu);
  m.impl("resample_2d_knab", &resample_2d_knab_cpu);
  m.impl("resample_1d_lanczos", &resample_1d_lanczos_cpu);
  m.impl("resample_1d_knab", &resample_1d_knab_cpu);
}

}
