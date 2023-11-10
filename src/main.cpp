#include <igl/opengl/glfw/Viewer.h>

#include "sphere.hpp"

const bool SHOW_POINTS = false;

int main(int argc, char *argv[]) {
  const auto &[points, mesh_faces] = edf::random_tri_sphere(100000, 42);


  igl::opengl::glfw::Viewer viewer;

  if (SHOW_POINTS) {
    // add points  
    auto points_out = Eigen::MatrixXd{1.001 * points};
    viewer.data().set_points(points_out, Eigen::RowVector3d(1, 0, 0));
    viewer.data().point_size = 0.25;
  }

  // add mesh
  viewer.data().set_mesh(points, mesh_faces);
  viewer.data().set_face_based(true);
  viewer.launch();

  return EXIT_SUCCESS;
}
