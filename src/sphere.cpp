#include "sphere.hpp"
#include "cu_sphere.hpp"
#include "legendre.hpp"
#include "util.hpp"

#include "CGAL/convex_hull_3.h"
#include <CGAL/point_generators_3.h>

static constexpr bool USE_CUDA = true;

namespace edf {

using Vector_3 = CGAL::Exact_predicates_inexact_constructions_kernel::Vector_3;

Point_3 geo_centroid(Mesh::Face_index face, const Mesh &mesh) {
  double sx = 0, sy = 0, sz = 0;
  int n = 0;

  for (auto v : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
    const auto p = mesh.point(v);

    sx += p.x();
    sy += p.y();
    sz += p.z();
    n++;
  }

  sx /= n;
  sy /= n;
  sz /= n;

  // normalize so point is on sphere
  const auto rr = sqrt(sx * sx + sy * sy + sz * sz);
  sx /= rr;
  sy /= rr;
  sz /= rr;

  return Point_3{sx, sy, sz};
}

Point_3 normalize(Point_3 p) {
  const auto v = p - CGAL::ORIGIN;

  return CGAL::ORIGIN + v / sqrt(v.squared_length());
}

// Note that the orientation of the output mesh will be the opposite of the
// input mesh. This works if the input mesh comes from CGAL's convex_hull_3,
// which orients faces inwards.
Mesh voronoi_from_delaunay(const Mesh &in) {
  Mesh out;

  std::vector<Mesh::Vertex_index> circumcenters;

  for (auto tri : in.faces()) {
    assert(boost::size(CGAL::vertices_around_face(in.halfedge(tri), in)) == 3);

    // compute circumcenter
    std::array<Point_3, 3> points;
    int i = 0;
    for (auto v : CGAL::vertices_around_face(in.halfedge(tri), in)) {
      points.at(i++) = in.point(v);
    }
    auto cp = normalize(CGAL::circumcenter(points[0], points[1], points[2]));

    // add the circumcenter as a vertex in our dual mesh
    auto v = out.add_vertex(cp);

    // ensure that face indices are packed to range [0, # faces)
    CHECK(tri.idx() == circumcenters.size());

    circumcenters.push_back(v);
  }

  std::vector<Point_3> sites;
  for (auto v : in.vertices()) {
    // TODO EDF could build one edge at a time to avoid alloc
    std::vector<Mesh::Vertex_index> vertices;
    for (auto face : CGAL::faces_around_target(in.halfedge(v), in)) {
      vertices.push_back(circumcenters.at(face.idx()));
    }

    const auto f = out.add_face(vertices);

    CHECK(f != decltype(out)::null_face());

    sites.push_back(in.point(v));

    CHECK(f.idx() == sites.size() - 1);
  }

  auto [site, _] =
      out.add_property_map<Mesh::Face_index, Point_3>("f:site", {});
  for (auto f : out.faces()) {
    site[f] = sites.at(f.idx());
  }

  return out;
}

Mesh geo_voronoi(const std::vector<Point_3> &pointsIn) {
  // convex hull of points on sphere will be Delaunay triangulation
  Mesh hull;
  CGAL::convex_hull_3(pointsIn.begin(), pointsIn.end(), hull);

  // see if all faces are triangles
  for (auto face : hull.faces()) {
    auto nv =
        boost::size(CGAL::vertices_around_face(hull.halfedge(face), hull));
    CHECK(nv == 3);
  }

  Mesh vor = voronoi_from_delaunay(hull);

  return vor;
}

std::vector<Point_3> relax(const std::vector<Point_3> &pointsIn) {
  std::vector<Point_3> pointsOut;

  Mesh vor = geo_voronoi(pointsIn);

  for (auto face : vor.faces()) {
    pointsOut.push_back(geo_centroid(face, vor));
  }

  return pointsOut;
}

Eigen::MatrixXf random_sphere_points_cpu(int num_points, unsigned long seed);

Eigen::MatrixXf random_sphere_points(int num_points, unsigned long seed) {
  if constexpr (USE_CUDA) {
    return random_sphere_points_cu(num_points, seed, 128, 64);
  } else {
    return random_sphere_points_cpu(num_points, seed);
  }
}


std::pair<std::vector<Point_3>, Mesh>
random_tri_sphere(int numPoints, int numRelaxations, uint32_t seed) {
  // auto rand = CGAL::Random(seed);
  // auto g = CGAL::Random_points_on_sphere_3<Point_3>(1., rand);

  auto m = random_sphere_points(numPoints, seed);

#if 0
  for (int i = 0; i < numPoints; i += 100) {
    std::cout << "m(" << i << ", 0) = " << m(i, 0) << "; m(" << i
              << ", 1) = " << m(i, 1) << "; m(" << i << ", 2) = " << m(i, 2)
              << std::endl;
    // output norm of point
    std::cout << "\tnorm = " << sqrt(m(i, 0) * m(i, 0) + m(i, 1) * m(i, 1) +
                                   m(i, 2) * m(i, 2))
              << std::endl;
  }
#endif

  // iterate over rows of matrix and make cgal points
  std::vector<Point_3> points;
  for (int i = 0; i < numPoints; i++) {
    points.push_back(Point_3{m(i, 0), m(i, 1), m(i, 2)});
  }

  // // generate random points on the sphere
  // std::vector<Point_3> points;
  // for (int i = 0; i < numPoints; i++) {
  //   points.push_back(*g);
  //   g++;
  // }

  // for (int i = 0; i < numRelaxations; i++) {
  //   points = relax(points);
  // }

  Mesh hull;
  CGAL::convex_hull_3(points.begin(), points.end(), hull);

  for (auto face : hull.faces()) {
    auto h = hull.halfedge(face);

    // check face orientation
    auto p1 = hull.point(hull.target(h));
    auto p2 = hull.point(hull.target(hull.next(h)));
    auto p3 = hull.point(hull.target(hull.next(hull.next(h))));
    CHECK(CGAL::orientation(p1, p2, p3, Point_3{0., 0., 0.}) == CGAL::NEGATIVE);

    auto nv =
        boost::size(CGAL::vertices_around_face(hull.halfedge(face), hull));
    CHECK(nv == 3);
  }

  return {points, hull};
}

} // namespace edf