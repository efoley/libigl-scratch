#pragma once

#include "cgal.hpp"
#include <igl/polygons_to_triangles.h>

namespace edf {

Eigen::MatrixXf random_sphere_points(int num_points, unsigned long seed);

std::pair<std::vector<Point_3>, Mesh>
random_tri_sphere(int numPoints, int numRelaxations, uint32_t seed);

} // namespace edf
