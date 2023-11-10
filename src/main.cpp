#include <igl/opengl/glfw/Viewer.h>

#include "sphere.hpp"

int main(int argc, char *argv[]) {
  // Inline mesh of a cube
  const Eigen::MatrixXd V = (Eigen::MatrixXd(8, 3) << 0.0, 0.0, 0.0, 0.0, 0.0,
                             1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                             1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
                                .finished();
  const Eigen::MatrixXi F =
      (Eigen::MatrixXi(12, 3) << 0, 6, 4, 0, 2, 6, 0, 3, 2, 0, 1, 3, 2, 7, 6, 2,
       3, 7, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 1, 5, 7, 1, 7, 3)
          .finished();

  // const auto &points =
  //     2 * Eigen::MatrixXd{edf::random_sphere_points(1000, 42).cast<double>()};

  // // print some points
  // for (int i = 0; i < points.rows(); i += 100) {
  //   std::cout << points.row(i) << std::endl;
  // }

  // igl::opengl::glfw::Viewer viewer;
  // viewer.data().set_points(points, Eigen::RowVector3d(0, 1, 0)); // Red points
  // viewer.launch();

  // const auto &[points, mesh] = edf::random_tri_sphere(100000, 0, 0);

  // const auto& [mv, mf] = edf::cgal_to_igl_mesh(mesh);

  // // Plot the mesh
  // igl::opengl::glfw::Viewer viewer;
  // viewer.data().set_mesh(mv, mf);
  // viewer.data().set_face_based(true);
  // viewer.launch();

  return EXIT_SUCCESS;
}
