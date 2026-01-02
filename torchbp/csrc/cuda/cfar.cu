#include "util.h"

namespace torchbp {

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
        float v01 = img[idbatch * N0 * N1 + idy * N1 + min(N1-1, idx + 1)];
        float v10 = img[idbatch * N0 * N1 + max(0, idy - 1) * N1 + idx];
        float v11 = img[idbatch * N0 * N1 + min(N0-1, idy + 1) * N1 + idx];
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
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cfar_2d_kernel<<<block_count, thread_per_block, 0, stream>>>(
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


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("cfar_2d", &cfar_2d_cuda);
}

}
