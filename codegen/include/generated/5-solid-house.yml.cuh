#define __N_LISTS 4
#define __N_BITMAPS 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
generated_kernel(vidType nv, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 4];
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
  meta.capacity = 4;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = warp_id; v0_idx < candidate_v0.size(); v0_idx += num_warps){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    __build_LUT(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    for (vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx += 1) {
      auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
      __build_index_from_vmap(g, meta, __get_vmap_from_lut_vid_limit(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/v1), /*slot_id=*/3);
      __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v1), __get_vlist_from_graph(g, meta, /*vid=*/v0), /*upper_bound=*/-1, /*output_slot=*/1);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/3);
      for(vidType v2_idx_idx = 0; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx ++){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        auto v2 = __build_vid_from_vidx(g, meta, v2_idx);
        __intersect(meta, __get_vlist_from_heap(g, meta, /*slot_id=*/1), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/2);
        auto candidate_v3 = __get_vlist_from_heap(g, meta, /*slot_id=*/2);
        for(vidType v3_idx = 0; v3_idx < candidate_v3.size(); v3_idx ++){
          auto v3 = candidate_v3[v3_idx];
          count += __intersect_num(__get_vlist_from_heap(g, meta, /*slot_id=*/2), __get_vlist_from_graph(g, meta, /*vid=*/v3), /*upper_bound=*/v3);
        }
      }
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}
