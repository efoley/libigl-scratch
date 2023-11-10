#pragma once

#include "cgal.hpp"
#include <igl/polygons_to_triangles.h>

namespace edf {

Eigen::MatrixXf random_sphere_points(int num_points, unsigned long seed);

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> random_tri_sphere(int num_points,
                                                              uint32_t seed);

} // namespace edf
