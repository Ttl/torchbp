#include "util.h"

// CPU speckle filter ops. Mirrors cuda/speckle_filter.cu.
namespace torchbp {


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

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("lee_filter", &lee_filter_cpu);
}

}
