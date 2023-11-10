#pragma once
#include <Eigen/Core>

namespace edf {

Eigen::MatrixXf random_sphere_points_cu(int num_points, unsigned long seed,
                                        int num_points_per_thread,
                                        int block_size);

} // namespace edf