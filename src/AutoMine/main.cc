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
  std::string pattern_name = argv[2];
  std::regex pattern("^P(\\d+)$");
  std::smatch match;
  int k;
  if (std::regex_match(pattern_name, match, pattern)) {
    k = std::stoi(match[1].str());
    std::cout << "Pattern P" << k << std::endl;
  } else {
    std::cerr << "Invalid input format. Expected format: Px, where x is an integer." << std::endl;
    return 1;
  }
  std::cout << "P" << k << "(only for undirected graphs)\n";
  std::string use_lut;
  if (argc > 3) use_lut = argv[3];
  if (use_lut != "lut") k = k + 1;

  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);
  g.print_meta_data();
 
  int num_patterns = 1;
  std::cout << "num_patterns: " << num_patterns << "\n";
  std::vector<uint64_t> total(num_patterns, 0);
  PatternSolver(g, k, total, n_devices, chunk_size);
  for (int i = 0; i < num_patterns; i++)
    std::cout << "pattern " << i << ": " << total[i] << "\n";
  return 0;
}

