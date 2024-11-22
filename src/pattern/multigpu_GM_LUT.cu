#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "codegen_LUT.cuh"
#include "codegen_utils.cuh"
#include "scheduler.h"
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "GM_LUT.cuh"
#include "GM_LUT_edge.cuh"
#include "GM_build_LUT.cuh"
#include <thread>

__global__ void clear(AccType *counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  counts[i] = 0;
}

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int n_gpus, int chunk_size) {
  assert(k >= 1);
  int ndevices = 0;
  eidType nnz = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  size_t memsize = print_device_info(0);
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  nnz = g.init_edgelist();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  // std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
  //           << " GB, graph_mem = " << mem_graph/1024/1024/1024 << " GB\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;

  // split the edgelist onto multiple gpus
  Timer t;
  double split_time = 0;
  t.Start();
  eidType n_tasks_per_gpu = eidType(nnz-1) / eidType(ndevices) + 1;
  std::vector<vidType*> src_ptrs, dst_ptrs;
  Scheduler scheduler;
  std::vector<vidType*> vert_ptrs;
  auto num_tasks = scheduler.round_robin(ndevices, g, src_ptrs, dst_ptrs, chunk_size);
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "GPU[" << i << "] is assigned " << num_tasks[i] << " tasks\n";
  split_time = t.Seconds();
  std::cout << "Time on splitting edgelist to GPUs: " << split_time << " sec\n";

  std::vector<GraphGPU> d_graphs(ndevices);
  std::vector<std::thread> threads2;
  double graph_copy_time = 0, edgelist_copy_time = 0;
  
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads2.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(g, i, ndevices);
    }));
  }
  for (auto &thread: threads2) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  graph_copy_time = t.Seconds();
  std::cout << "Time on copying graph to GPUs: " << graph_copy_time << " sec\n";

  std::vector<std::thread> thread1;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    thread1.push_back(std::thread([&,i]() {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
    }));
  }
  for (auto &thread: thread1) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  edgelist_copy_time = t.Seconds();
  // end copy graphGPU, edgelist  
 
  size_t nwarps = WARPS_PER_BLOCK;
  size_t n_lists, n_bitmaps;
  if (k == 1) {
    n_lists = __N_LISTS1;
    n_bitmaps = __N_BITMAPS1;
  }
  else if (k == 2) {
    n_lists = __N_LISTS2;
    n_bitmaps = __N_BITMAPS2;
  }
  else {
    n_lists = 0;
  }

  size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
  size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

  size_t n_lutrows;
  if (k == 6) {
    n_lutrows = 2;
  } 
  else {
    n_lutrows = 0;
  }

  size_t per_block_lutrows_size = BLOCK_SIZE * n_lutrows * size_t((md - 1) / BITMAP_WIDTH + 1) * sizeof(bitmapType);

  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM;
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

  nblocks = std::min(6*max_blocks, nblocks);

  nblocks = 640;
  std::cout << "CUDA " << k << "-pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  size_t bitmap_size = nblocks * per_block_bitmap_size;
  std::cout << "lut rows size: " << bitmap_size/(1024*1024) << " MB\n";

  size_t npatterns = 1;

  std::vector<AccType> h_counts(ndevices * npatterns, 0);
  std::vector<AccType *> d_counts(ndevices);
  std::vector<vidType *> frontier_list(ndevices);
  std::vector<bitmapType *> frontier_bitmap(ndevices);
  std::vector<LUTManager<>> lut_manager(ndevices);

  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_counts[i], sizeof(AccType) * npatterns));
    clear<<<1, npatterns>>>(d_counts[i]);
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list[i], list_size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap[i], bitmap_size));
    lut_manager[i].init(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT, true);
  }

  std::vector<std::thread> threads;
  std::vector<Timer> subt(ndevices);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());


  vidType base_size = nv / ndevices;
  vidType remainder = nv % ndevices;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(d_counts[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));

    vidType begin = 0, end = nv;
    if (i < remainder) {
      begin = i * (base_size + 1);
      end = begin + base_size + 1;
    } else {
      begin = remainder * (base_size + 1) + (i - remainder) * base_size;
      end = begin + base_size;
    }
    if (k == 1) {
      std::cout << "GM LUT device " << i <<"\n";
      GM_LUT_warp<<<nblocks, nthreads>>>(begin, end, d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i]);
      lut_manager[i].recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
      GM_LUT_block<<<nblocks, nthreads>>>(begin, end, d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i]);
      lut_manager[i].recreate(1, md, md, true);
      nblocks = BLOCK_GROUP;
      for (vidType idx = begin; idx < end; idx++) {
        if (g.get_degree(idx) > BLOCK_LIMIT) {
          lut_manager[i].update_para(1, g.get_degree(idx), g.get_degree(idx), true);
          GM_build_LUT<<<nblocks, nthreads>>>(begin, end, d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i], idx);
          GM_LUT_global<<<nblocks, nthreads>>>(begin, end, d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i], idx);
        }
      }
    }
    else if (k == 2) {
      std::cout << "Edge-centric GM LUT device " << i <<"\n";
      GM_LUT_warp_edge<<<nblocks, nthreads>>>(0, num_tasks[i], d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i]);
      lut_manager[i].recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
      GM_LUT_block_edge<<<nblocks, nthreads>>>(0, num_tasks[i], d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i]);
      lut_manager[i].recreate(BLOCK_GROUP, md, md, true);
      nblocks = BLOCK_GROUP;
      GM_LUT_block_large_edge<<<nblocks, nthreads>>>(0, num_tasks[i], d_graphs[i], frontier_list[i], frontier_bitmap[i], md, d_counts[i], lut_manager[i]);
    }
    else {
      std::cout << "Not supported right now\n";
    }
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i*npatterns], d_counts[i], sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < npatterns; i++) {
    for (int j = 0; j < ndevices; j++) {
      accum[i] += h_counts[j * npatterns + i];
    }
  }  
  t.Stop();

  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
}

