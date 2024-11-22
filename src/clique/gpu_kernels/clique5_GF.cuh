// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique5_GF(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total, BinaryEncode<> sub_graph, int *allocator) {
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
  __shared__ bitmapType sub_graph_shared[WARPS_PER_BLOCK][BITMAP_WIDTH];
  
  for(int eid = warp_id ; eid < ne; ) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.get_degree(v0);
    vidType v1_size = g.get_degree(v1);
    vidType count1 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane] = count1;
    __syncwarp();
    // build binary_encode

    if (list_size[warp_lane] <= 32) {
      for (vidType i = 0; i < list_size[warp_lane]; i++) {
        auto search = g.N(vlist[i]);
        vidType search_size = g.get_degree(vlist[i]);
        for (int j = thread_lane; j < list_size[warp_lane]; j += WARP_SIZE) {
          unsigned active = __activemask();
          bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
          __syncwarp(active);
          // set binary_encode
          unsigned mask = __ballot_sync(active, flag);
          if (thread_lane == 0) sub_graph_shared[warp_lane][i] = mask;
        }
      }
      int countElement = (list_size[warp_lane] - 1) / BITMAP_WIDTH + 1;

      for (vidType v1 = 0; v1 < list_size[warp_lane]; v1++) {
        for (vidType v2 = thread_lane; v2 < list_size[warp_lane]; v2 += WARP_SIZE) {
          if (sub_graph_shared[warp_lane][v1] & (1 << v2)){
            counter += __popc(sub_graph_shared[warp_lane][v1] & sub_graph_shared[warp_lane][v2]);
          }
        }
      }
    }

    else {
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
      int countElement = (list_size[warp_lane] - 1) / BITMAP_WIDTH + 1;
  
      for (vidType v1 = 0; v1 < list_size[warp_lane]; v1++) {
        for (vidType v2 = 0; v2 < list_size[warp_lane]; v2++) {
          if (sub_graph.get(warpMapHead, v1, v2)){
            counter += sub_graph.intersect_num_warp(warpMapHead, (list_size[warp_lane] - 1) / 32 + 1,v1, v2);
          }
        }
      }
    }
    // __syncwarp();

    // sub_graph.warp_clear(warpMapHead);

    if (thread_lane == 0) {
      eid = atomicAdd(allocator, 1);
    }
    eid = __shfl_sync(0xffffffff, eid, 0);
  }

  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}
