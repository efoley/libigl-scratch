#include "sphere.hpp"

#include <random>

#include "util.hpp"

namespace edf {

Eigen::MatrixXf random_sphere_points_cpu(int num_points, unsigned long seed) {
  std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with seed
  std::uniform_real_distribution<float> dist_phi(0.0f, 2.0f * M_PI);
  std::uniform_real_distribution<float> dist_cos_theta(-1.0f, 1.0f);

  Eigen::MatrixXf points(num_points, 3);

  for (int idx = 0; idx < num_points; ++idx) {
    // Generate random azimuth angle phi
    float phi = dist_phi(gen);
    // Generate random inclination angle theta
    float theta = acosf(dist_cos_theta(gen));

    // Convert spherical coordinates (r=1, theta, phi) to Cartesian coordinates
    // (x, y, z)
    points(idx, 0) = std::sin(theta) * std::cos(phi); // x
    points(idx, 1) = std::sin(theta) * std::sin(phi); // y
    points(idx, 2) = std::cos(theta);                 // z
  }

  return points;
}
} // namespace edf