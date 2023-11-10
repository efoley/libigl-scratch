#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>

#include <Eigen/Core>

namespace edf {

using Point_3 = CGAL::Exact_predicates_inexact_constructions_kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cgal_to_igl_mesh(const Mesh& mesh);

} // namespace edf