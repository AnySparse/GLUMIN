#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "codegen_LUT.cuh"
#include "codegen_utils.cuh"
#define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "GM_LUT.cuh"
#include "GM_build_LUT.cuh"
#include "GM_LUT_deep.cuh"
#include "BS_vertex.cuh"
#include "BS_edge.cuh"

#define BLK_SZ BLOCK_SIZE
#include "clique4_warp_edge.cuh"
#include "clique5_warp_edge.cuh"
#include "clique6_warp_edge.cuh"
#include "clique7_warp_edge.cuh"

// #define THREAD_PARALLEL

__global__ void clear(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int) {
  assert(k >= 1);
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
            << " GB, graph_mem = " << mem_graph/1024/1024 << " MB\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";
  // CUDA_SAFE_CALL(cudaSetDevice(CUDA_SELECT_DEVICE));
  GraphGPU gg(g);
  gg.init_edgelist(g);

  size_t npatterns = 3;
  AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
  AccType *d_counts;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_counts);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists;
  size_t n_bitmaps;

  n_lists = 5;
  n_bitmaps = 1;

  vidType switch_lut = 1;
  switch_lut = Select_func(nv, ne, md);

  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  if (k == 1){
    max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  }
  else {
    max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  } 
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

  nblocks = std::min(6*max_blocks, nblocks);

  nblocks = 640;
  std::cout << "CUDA " << k << "-pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  size_t bitmap_size = nblocks * per_block_bitmap_size;
  std::cout << "lut rows size: " << bitmap_size/(1024*1024) << " MB\n";
  bitmapType *frontier_bitmap; // each thread has lut rows to store midresult of lut compute
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));

  LUTManager<> lut_manager(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT, true); 

  // split vertex tasks
  std::vector<vidType> vid_warp, vid_block, vid_global;

  for (int vid = 0; vid < nv; ++vid) {
    auto degree = g.get_degree(vid);
    if (degree <= WARP_LIMIT) {
        vid_warp.push_back(vid);
    } else if (degree <= BLOCK_LIMIT) {
        vid_block.push_back(vid);
    } else {
        vid_global.push_back(vid);
    }
  }
  vidType vid_warp_size = vid_warp.size();
  vidType vid_block_size = vid_block.size();
  vidType vid_global_size = vid_global.size();
  std::cout << "warp_task: " << vid_warp_size << " block_task: " << vid_block_size << " global_task: " << vid_global_size << "\n";
  vidType *d_vid_warp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_warp, vid_warp_size * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_vid_warp, vid_warp.data(), vid_warp_size * sizeof(vidType), cudaMemcpyHostToDevice));

  vidType *d_vid_block;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_block, vid_block_size * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_vid_block, vid_block.data(), vid_block_size * sizeof(vidType), cudaMemcpyHostToDevice));

  Timer t;
  t.Start();
  // G2Miner
  if (k == 1){
    std::cout << "P1 Run G2Miner\n";
    P1_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 2){
    std::cout << "P2 Run G2Miner\n";
    P2_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 3){
    std::cout << "P3 Run G2Miner\n";
    P3_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 6){
    std::cout << "P6 Run G2Miner\n";
    P6_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 7){
    std::cout << "P7 Run G2Miner\n";
    P7_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 7){
    std::cout << "P7 Run G2Miner\n";
    P7_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 8){
    std::cout << "P8 Run G2Miner\n";
    P8_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 9){
    std::cout << "P9 Run G2Miner\n";
    P9_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 10){
    std::cout << "P10 Run G2Miner\n";
    P10_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 11){
    std::cout << "P11 Run G2Miner\n";
    P11_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 12){
    std::cout << "P12 Run G2Miner\n";
    P12_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 13){
    std::cout << "P13 Run G2Miner\n";
    P13_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 14){
    std::cout << "P14 Run G2Miner\n";
    P14_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 15){
    std::cout << "P15 Run G2Miner\n";
    P15_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 16){
    std::cout << "P16 Run G2Miner\n";
    P16_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 17){
    std::cout << "P17 Run G2Miner\n";
    P17_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }              
  else if (k == 18){
    std::cout << "P18 Run G2Miner\n";
    P18_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 19){
    std::cout << "P19 Run G2Miner\n";
    P19_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 20){
    std::cout << "P20 Run G2Miner\n";
    P20_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 21){
    std::cout << "P21 Run G2Miner\n";
    P21_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else if (k == 22){
    std::cout << "P22 Run G2Miner\n";
    P22_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
  }
  else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < npatterns; i ++) accum[i] = h_counts[i];
  t.Stop();


  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_counts));
}

void CliqueSolver(Graph &g, int k, uint64_t &total, int, int) {
  assert(k > 3);
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  GraphGPU gg(g);
  gg.init_edgelist(g);
  size_t nwarps = WARPS_PER_BLOCK;
  size_t nthreads = BLK_SZ;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;

  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(clique4_warp_edge, nthreads, 0);

  if (k==5) max_blocks_per_SM = maximum_residency(clique5_warp_edge, nthreads, 0);  
  if (k==6) max_blocks_per_SM = maximum_residency(clique6_warp_edge, nthreads, 0);
  if (k==7) max_blocks_per_SM = maximum_residency(clique7_warp_edge, nthreads, 0);
  std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

  nblocks = std::min(16*max_blocks, nblocks);  
  
  std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());


  Timer t;
  t.Start();
  if (k == 4) {
    std::cout << "P4 Run G2Miner\n";
    clique4_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else if (k == 5) {
    std::cout << "P5 Run G2Miner\n";
    clique5_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else if (k == 6) {
    std::cout << "P23 Run G2Miner\n";
    clique6_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else if (k == 7) {
    std::cout << "P24 Run G2Miner\n";
    clique7_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
  } else {
    std::cout << "Not supported right now\n";
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}