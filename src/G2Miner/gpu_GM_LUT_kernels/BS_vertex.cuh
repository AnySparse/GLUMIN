#define __N_LISTS4 2
#define __N_BITMAPS4 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
BS_vertex(vidType nv, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 2];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  AccType count = 0;
  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(warp_id);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = nv;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 2;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = warp_id; v0_idx < candidate_v0.size(); v0_idx += num_warps){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    for(vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx ++){
      auto v1 = candidate_v1[v1_idx];
      __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
      auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
      for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
        auto v2 = candidate_v2[v2_idx];
        __intersect(meta, __get_vlist_from_heap(g, meta, /*slot_id=*/0), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/1);
        auto candidate_v3 = __get_vlist_from_heap(g, meta, /*slot_id=*/1);
        for(vidType v3_idx = 0; v3_idx < candidate_v3.size(); v3_idx ++){
          auto v3 = candidate_v3[v3_idx];
          count += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/1), __get_vlist_from_graph(g, meta, /*vid=*/v3), /*upper_bound=*/-1);
        }
      }
    }
  }
  // END OF CODEGEN

  atomicAdd(counter, count);
}
