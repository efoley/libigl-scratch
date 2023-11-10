#include <CLI/CLI.hpp>
#include <cuda_runtime.h>
#include <optional>

#include "cu_sphere.hpp"

using namespace edf;

struct Params {
  int num_points = 1000000;
  std::optional<unsigned long> seed;
  int skip_reps = 5;
  int num_reps = 30;
  int num_points_per_thread = 128;
  int block_size = 64;
};

Params cliParams(int argc, char *argv[]) {
  CLI::App app;

  Params params;

  app.add_option("num_points", params.num_points, "Number of points");
  app.add_option("num_reps", params.num_reps, "Number of repetitions");
  app.add_option("-s,--seed", params.seed, "Random seed");
  app.add_option("-p,--num_points_per_thread", params.num_points_per_thread,
                 "Number of points per thread");
  app.add_option("-b,--block_size", params.block_size, "Block size");

  //   std::string output_dir = "/tmp";
  //   app.add_option("-d,--output_dir",
  //   output_dir)->check(CLI::ExistingDirectory);
  try {
    app.parse((argc), (argv));
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return params;
}

int main(int argc, char *argv[]) {

  // force CUDA to initialize
  cudaFree(0);

  const auto params = cliParams(argc, argv);

  // time the kernel on each rep
  std::vector<double> durations;

  for (int rep = 0; rep < params.num_reps; rep++) {
    // get start time
    const auto start = std::chrono::high_resolution_clock::now();

    auto seed = params.seed ? *params.seed : time(NULL);

    const auto &points = edf::random_sphere_points_cu(
        params.num_points, seed, params.num_points_per_thread,
        params.block_size);

    // get end time
    const auto end = std::chrono::high_resolution_clock::now();

    if (rep < params.skip_reps) {
      continue;
    }

    // compute duration
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // convert to double us
    const auto duration_us = duration.count() / 1e3;
    durations.push_back(duration_us);
  }

  // compute mean and standard deviation
  double mean = 0.;
  for (const auto &d : durations) {
    mean += d;
  }
  mean /= durations.size();

  double std = 0.;
  for (const auto &d : durations) {
    std += (d - mean) * (d - mean);
  }
  std /= durations.size();
  std = sqrt(std);

  // print labeled number of points, points per thread, and block size
  std::cout << "Num Points: " << params.num_points << std::endl;
  std::cout << "Num Points Per Thread: " << params.num_points_per_thread
            << std::endl;
  std::cout << "Block Size: " << params.block_size << std::endl;
  std::cout << "Mean Duration (µs): " << mean << std::endl;
  std::cout << "Std Duration (µs): " << std << std::endl;

  return EXIT_SUCCESS;
}