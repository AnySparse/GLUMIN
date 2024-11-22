// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique4_warp_vertex_subgraph(vidType nv, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total, BinaryEncode<> sub_graph) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  AccType counter = 0;
  size_t warpMapHead = warp_id * max_deg * ((max_deg - 1) / 32 + 1);
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  for (auto v0 = warp_id; v0 < nv; v0 += num_warps) {
    auto v0_ptr = g.N(v0);
    auto v0_size = g.get_degree(v0);

    for (vidType i = 0; i < v0_size; i++) {
      auto search = g.N(v0_ptr[i]);
      vidType search_size = g.get_degree(v0_ptr[i]);
      for (int j = thread_lane; j < v0_size; j += WARP_SIZE) {
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, v0_ptr[j], search_size);
        __syncwarp(active);
        // set binary_encode
        sub_graph.warp_cover(warpMapHead, i, j, flag);
      }
    }
    __syncwarp();

    for (vidType v1 = thread_lane; v1 < v0_size; v1 += WARP_SIZE) {
      for (vidType v2 = 0; v2 < v0_size; v2 ++) {
        if (sub_graph.get(warpMapHead, v1, v2)) {
          counter += sub_graph.intersect_num_thread(warpMapHead, (v0_size - 1) / 32 + 1,v1, v2);         
        }
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

