#include "graph.h"
#include "pattern.hh"

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<graph> <k> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 1\n";
    exit(1);
  }
  Graph g(argv[1]);
  int k = atoi(argv[2]);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 3) n_devices = atoi(argv[3]);
  if (argc > 4) chunk_size = atoi(argv[4]);
  std::cout << "SELECTION: " << k << "\n";
  g.print_meta_data();
 
  int num_patterns = 1;
  std::cout << "num_patterns: " << num_patterns << "\n";
  std::vector<uint64_t> total(num_patterns, 0);
  PatternSolver(g, k, total, n_devices, chunk_size);
  for (int i = 0; i < num_patterns; i++)
    std::cout << "pattern " << i << ": " << total[i] << "\n";
  return 0;
}

