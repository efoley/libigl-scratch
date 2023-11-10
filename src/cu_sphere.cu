#include "cuda.hpp"

#include <Eigen/Core>
#include <curand_kernel.h>

namespace edf {

template <typename T> T div_up(T a, T b) { return (a + b - 1) / b; }

__global__ void setup_curand(curandState *state, int num_states,
                             unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_states) {
    curand_init(seed, idx, 0, &state[idx]);
  }
}

/**
 * Generate random points on the unit sphere.
 *
 * @param state curand states
 * @param points_phi_theta output containing points in spherical coordinates
 (phi, theta)
 *    in radians on the unit sphere; coordinates are stored in column-major
 order
 * @param points_xyz output containing points (x, y, z) on the unit sphere;
 coordinates
 *    are stored in column-major order
 * @param num_points number of points on the sphere
 * @param num_points_per_thread number of points to generate per thread

*/
__global__ void random_unit_sphere_points(curandState *state,
                                          float *points_phi_theta,
                                          float *points_xyz,
                                          const int num_points,
                                          const int num_points_per_thread) {
  const auto thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

  auto s = &state[thread_idx];

  const auto idx0 = thread_idx * num_points_per_thread;
  for (int i = 0; i < num_points_per_thread; i++) {
    const auto idx = idx0 + i;

    if (idx >= num_points) {
      break;
    }

    // Generate random azimuth angle phi
    float phi = curand_uniform(s) * 2.0f * M_PI;
    // Generate random inclination angle theta
    float theta = acosf(2.0f * curand_uniform(s) - 1.0f);

    // column-major indexing
    const size_t phi_idx = idx;
    const size_t theta_idx = num_points + idx;

    const size_t x_idx = 0 + idx;
    const size_t y_idx = num_points + idx;
    const size_t z_idx = 2 * num_points + idx;

    points_phi_theta[phi_idx] = phi;
    points_phi_theta[theta_idx] = theta;

    // spherical (r=1, theta, phi) to Cartesian (x, y, z)
    points_xyz[x_idx] = sinf(theta) * cosf(phi); // x
    points_xyz[y_idx] = sinf(theta) * sinf(phi); // y
    points_xyz[z_idx] = cosf(theta);             // z
  }
}

Eigen::MatrixXf random_sphere_points_cu(int num_points, unsigned long seed,
                                        int num_points_per_thread,
                                        int block_size) {
  const int num_threads = div_up(num_points, num_points_per_thread);

  const int num_blocks = div_up(num_threads, block_size);

  // we'll have one curand state per thread
  curandState *d_states;
  CUDA_CALL(cudaMalloc(&d_states, num_threads * sizeof(curandState)));

  // Setup the curand states
  setup_curand<<<num_blocks, block_size>>>(d_states, num_threads, seed);
  CUDA_CALL(cudaPeekAtLastError());

  // allocate memory for the points
  float *d_points_xyz;
  CUDA_CALL(cudaMalloc(&d_points_xyz, num_points * 3 * sizeof(float)));

  float *d_points_phi_theta;
  CUDA_CALL(cudaMalloc(&d_points_phi_theta, num_points * 2 * sizeof(float)));

  // Generate the random points on the sphere
  random_unit_sphere_points<<<num_blocks, block_size>>>(
      d_states, d_points_phi_theta, d_points_xyz, num_points,
      num_points_per_thread);
  CUDA_CALL(cudaPeekAtLastError());

  // cudaDeviceSynchronize();

  // make an eigen matrix to hold the points
  Eigen::MatrixXf points_xyz(num_points, 3);

  // copy from the device into the matrix
  CUDA_CALL(cudaMemcpy(points_xyz.data(), d_points_xyz, num_points * 3 * sizeof(float),
                       cudaMemcpyDeviceToHost));

  // Free the memory
  CUDA_CALL(cudaFree(d_states));
  CUDA_CALL(cudaFree(d_points_xyz));
  CUDA_CALL(cudaFree(d_points_phi_theta));

  return points_xyz;
}

} // namespace edf