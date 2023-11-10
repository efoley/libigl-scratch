cmake_minimum_required(VERSION 3.22)
project(example CXX CUDA)

enable_language(CUDA)

# option(LIBIGL_USE_STATIC_LIBRARY "Use libIGL as static library" ON)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# CLI11
include(cli11)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

# Other modules you could enable
# igl_include(embree)
# igl_include(imgui)
# igl_include(opengl)
# igl_include(stb)
# igl_include(predicates)
# igl_include(xml)
igl_include(copyleft cgal)

# igl_include(copyleft comiso)
igl_include(copyleft core)

# igl_include(copyleft cork)
# igl_include(copyleft tetgen)
# igl_include(restricted matlab)
# igl_include(restricted mosek)
# igl_include(restricted triangle)
find_package(CUDA REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# find_package(GLEW REQUIRED)

# cu_sphere
add_library(cu_sphere
  src/cu_sphere.cu
)
set_target_properties(cu_sphere PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cu_sphere PUBLIC cxx_std_20)
set_target_properties(cu_sphere PROPERTIES CUDA_ARCHITECTURES "70")
target_link_libraries(cu_sphere PUBLIC Eigen3::Eigen)

# cu_main
add_executable(cu_main
  src/cu_main.cpp
)
set_target_properties(cu_main PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cu_main PUBLIC cxx_std_20)
set_target_properties(cu_main PROPERTIES CUDA_ARCHITECTURES "70")
target_link_libraries(cu_main cu_sphere Eigen3::Eigen CLI11::CLI11)

# main
add_executable(main
  src/main.cpp
  src/sphere.cpp
  src/cgal.cpp
  src/sphere_rand_points.cpp
)
target_compile_features(main PUBLIC cxx_std_20)

target_link_libraries(main PRIVATE cu_sphere)

# Link igl (and the glfw module) to your project
target_link_libraries(main PUBLIC
  igl::glfw

  # # Other modules you could link to
  # igl::embree
  # igl::imgui
  # igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  igl_copyleft::cgal

  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
)
