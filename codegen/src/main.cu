#include <cub/cub.cuh>
#include "graph.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "driver.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<graph> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 4\n";
    exit(1);
  }
  Graph g(argv[1]);
  int device_id = 0;
  int chunk_size = 1024;
  if (argc > 3) device_id = atoi(argv[2]);
  if (argc > 4) chunk_size = atoi(argv[3]);
  g.print_meta_data();
 
  int num_patterns = 1;
  std::vector<uint64_t> total(num_patterns, 0);
  PatternSolver(g, total);
  for (int i = 0; i < num_patterns; i++)
    std::cout << "pattern " << i << ": " << total[i] << "\n";
  return 0;
}

