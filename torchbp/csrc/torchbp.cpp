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

// Defines the operators
TORCH_LIBRARY(torchbp, m) {
  m.def("backprojection_polar_2d(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor");
  m.def("backprojection_polar_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor[]");
  m.def("backprojection_polar_2d_lanczos(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor");
  m.def("backprojection_polar_2d_knab(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, float oversample, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor");
  m.def("backprojection_cart_2d(Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor");
  m.def("backprojection_cart_2d_grad(Tensor grad, Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor[]");
  m.def("polar_range_dealias(Tensor img, Tensor dem, int nbatch, int Nr, int Ntheta, float fc, float r0, float dr, float theta0, float dtheta, float ox, float oy, float oz, float alias_fmod) -> Tensor");
  m.def("gpga_backprojection_2d(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, float data_fmod) -> Tensor");
  m.def("gpga_backprojection_2d_lanczos(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, int order, float data_fmod) -> Tensor");
  m.def("blocksvd_alpha(Tensor img, Tensor data, Tensor pos, Tensor blocks, int sweep_samples, int nsweeps, int nblocks, int Ntheta, float fc, float r_res, float r0, float dr, float theta0, float dtheta, float d0, float data_fmod) -> Tensor");
  m.def("cfar_2d(Tensor img, int nbatch, int N0, int N1, int Navg0, int Navg1, int Nguard0, int Nguard1, float threshold, int peaks_only) -> Tensor");
  m.def("polar_interp_linear(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor");
  m.def("polar_interp_linear_grad(Tensor grad, Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor[]");
  m.def("polar_interp_lanczos(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float alias_fmod) -> Tensor");
  m.def("ffbp_merge2_lanczos(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, int alias, float alias_fmod, Tensor dem) -> Tensor");
  m.def("ffbp_merge2_knab(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float oversample, int alias, float alias_fmod, Tensor dem) -> Tensor");
  m.def("ffbp_merge2_poly(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod, Tensor dem) -> Tensor");
  m.def("ffbp_merge2_poly_weighted(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod, Tensor w1_map0, Tensor w2_map0, float w_r0_0, float w_dr0, float w_theta0_0, float w_dtheta0, int w_nr0, int w_ntheta0, Tensor w1_map1, Tensor w2_map1, float w_r0_1, float w_dr1, float w_theta0_1, float w_dtheta1, int w_nr1, int w_ntheta1, int output_weight_map, int output_weight_decimation, Tensor dem) -> Tensor[]");
  m.def("cfbp_merge2(Tensor img0, Tensor img1, Tensor w0, Tensor idx0, Tensor w1, Tensor idx1, int nbatch, int Nx, int Ny0, int Ny1, int Nyout, int order0, int order1, float dx, float dy, float ox0, float oy0, float z0, float ox1, float oy1, float z1, float oxp, float oyp, float zp, float ref_phase) -> Tensor");
  m.def("polar_to_cart_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor");
  m.def("polar_to_cart_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor[]");
  m.def("polar_to_cart_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod, int order) -> Tensor");
  m.def("cart_to_polar_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod) -> Tensor");
  m.def("cart_to_polar_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod) -> Tensor[]");
  m.def("cart_to_polar_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod, int order) -> Tensor");
  m.def("backprojection_polar_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, int azimuth_resolution) -> Tensor");
  m.def("backprojection_polar_2d_tx_power_slant(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, int azimuth_resolution, float altitude) -> Tensor");
  m.def("backprojection_cart_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, int normalization, int azimuth_resolution) -> Tensor");
  m.def("backprojection_polar_2d_tx_power_accum(Tensor wa, Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, float dr_ref, float h_ref, float altitude, int theta_psi) -> Tensor");
  m.def("backprojection_cart_2d_tx_power_accum(Tensor wa, Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float x0, float dx, float y0, float dy, int Nx, int Ny, int normalization, float dx_ref, float h_ref) -> Tensor");
  m.def("cart_tx_power_merge2(Tensor acc0, Tensor acc1, float x0_0, float dx_0, float y0_0, float dy_0, int Nx_0, int Ny_0, float x0_1, float dx_1, float y0_1, float dy_1, int Nx_1, int Ny_1, float x1, float dx1, float y1, float dy1, int Nx1, int Ny1) -> Tensor");
  m.def("ffbp_tx_power_merge2(Tensor acc0, Tensor acc1, Tensor dorigin, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float altitude, int in_psi, int out_psi) -> Tensor");
  m.def("compute_illumination(Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, float r0, float dr, float theta0, float dtheta, int nr, int ntheta, int decimation, Tensor dem) -> Tensor[]");
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
  m.def("resample_1d_lanczos(Tensor img, int nbatch, int N, int M, int order) -> Tensor");
  m.def("resample_1d_knab(Tensor img, int nbatch, int N, int M, int order, float oversample) -> Tensor");
  m.def("afbp_fuse(Tensor S, Tensor nua, Tensor xs, Tensor inv_kr, Tensor x_half, Tensor band, float x_c, float x_taper, int kmax) -> Tensor");
}

}
