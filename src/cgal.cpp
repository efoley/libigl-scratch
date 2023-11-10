#include "cgal.hpp"
#include "util.hpp"

namespace edf {

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cgal_to_igl_mesh(const Mesh &mesh) {
  Eigen::MatrixXd V(mesh.number_of_vertices(), 3);
  Eigen::MatrixXi F(mesh.number_of_faces(), 3);

  enumerate(mesh.vertices(), [&](auto i, auto v) {
    const auto p = mesh.point(v);
    V.row(i) << p.x(), p.y(), p.z();
  });

  enumerate(mesh.faces(), [&](auto i, auto f) {
    int j = 0;
    for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
      F(i, j++) = v.idx();
    }
    DEBUG_CHECK(j == 3);
  });

  return std::make_pair(V, F);
}

} // namespace edf