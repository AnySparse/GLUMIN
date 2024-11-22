
#include "graph.h"

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<graph> <k> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 4\n";
    exit(1);
  }
  std::cout << "k-clique listing with undirected graphs\n";
  if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  Graph g(argv[1], USE_DAG); // use DAG
  int k = atoi(argv[2]);
  int n_devices = 8;
  int chunk_size = 1024;
  int select_device = 3;
  int choose = 0;
  if (argc > 3) select_device = atoi(argv[3]);
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);
  g.print_meta_data();
 
  uint64_t total = 0;
  CliqueSolver(g, k, total, select_device, n_devices);
  std::cout << "num_" << k << "-cliques = " << total << "\n";
  return 0;
}
