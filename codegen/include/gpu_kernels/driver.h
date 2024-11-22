typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "common.h"
#include "LUT.cuh"

#include "codegen_utils.cuh"
#include "generated_inc.cuh"

__global__ void clear(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void PatternSolver(Graph &g, std::vector<uint64_t> &accum) {
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto max_degree = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);

  std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
            << " GB, graph_mem = " << mem_graph/1024/1024/1024 << " GB\n";

  assert(memsize >= mem_graph && "Out of Memory.");

  GraphGPU gg(g);
  gg.init_edgelist(g);
  //std::cout << gg.out_colidx() << " " << gg.get_dst_ptr(0) << std::endl;

  size_t npatterns = 1;

  AccType *h_accumulators = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_accumulators[i] = 0;
  AccType *d_accumulators;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_accumulators, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_accumulators);
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists = __N_LISTS__;
  size_t n_bitmaps = __N_BITMAPS__;

  size_t per_block_vlist_size = nwarps * n_lists * size_t(max_degree) * sizeof(vidType);
  size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(max_degree) + 32-1)/32) * sizeof(vidType);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize*0.9 - mem_graph) / (per_block_vlist_size + per_block_bitmap_size);
  if (nb < nblocks) nblocks = nb;

  // size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  // nblocks = BLOCK_GROUP;
  nblocks = 640;
  // nblocks = 1;
  // nblocks = std::min(6*max_blocks, nblocks);
  
  std::cout << "CUDA " << nblocks << " CTAs, " << nthreads << " threads/CTA\n";

  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "Intermediate result list size: " << list_size/(1024*1024) << " MB\n";
  size_t bitmap_size = nblocks * per_block_bitmap_size;
  std::cout << "Intermediate result bitmap size: " << bitmap_size/(1024*1024) << " MB\n";

  vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));
  bitmapType* frontier_bitmap;
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));

  LUTManager<> LUTs(/*LUT_num=*/nblocks * nwarps, /*nrows=*/2048, /*ncols=*/2048, /*use_gpu=*/true);
  size_t lut_size = nblocks * nwarps * 2038 * 2048 / 32;
  std::cout << "LUT size: " << lut_size / (1024*1024) << " MB\n";

  printf("Total number of warp: %d\n", nblocks * nwarps);

  Timer t;
  t.Start();

  generated_kernel<<<nblocks, nthreads>>>(nv, gg, frontier_list, frontier_bitmap, max_degree, d_accumulators, LUTs);
  //star3_warp_edge_<<<nblocks, nthreads>>>(ne, gg, frontier_list, max_degree, d_accumulators);
  //star3_warp_vertex_subgraph_small<<<nblocks, nthreads>>>(nv, gg, frontier_list, max_degree, d_accumulators, sub_graph);
  cudaDeviceSynchronize();
  CUDA_SAFE_CALL(cudaMemcpy(h_accumulators, d_accumulators, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < npatterns; i ++) accum[i] = h_accumulators[i];
  t.Stop();

  std::cout << "runtime = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_accumulators));
}
