// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique5_warp_edge_subgraph(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total, BinaryEncode<> sub_graph) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType *vlist  = & vlists[int64_t(warp_id)*int64_t(max_deg)];
  AccType counter = 0;
  size_t warpMapHead = warp_id * max_deg * ((max_deg - 1) / 32 + 1);
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  for(eidType eid = warp_id ; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.get_degree(v0);
    vidType v1_size = g.get_degree(v1);
    vidType count1 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane] = count1;
    __syncwarp();
    // build binary_encode
    for (vidType i = 0; i < list_size[warp_lane]; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.get_degree(vlist[i]);
      for (int j = thread_lane; j < list_size[warp_lane]; j += WARP_SIZE) {
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        __syncwarp(active);
        // set binary_encode
        sub_graph.warp_cover(warpMapHead, i, j, flag);
      }
    }
    __syncwarp();

    for (vidType v1 = thread_lane; v1 < list_size[warp_lane]; v1 += WARP_SIZE) {
      for (vidType v2 = 0; v2 < list_size[warp_lane]; v2 ++) {
        if (sub_graph.get(warpMapHead, v1, v2)) {
          counter += sub_graph.intersect_num_thread(warpMapHead, (list_size[warp_lane] - 1) / 32 + 1,v1, v2);
        }
      }
    }
    // __syncwarp();

    // sub_graph.warp_clear(warpMapHead);
  }

  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}


