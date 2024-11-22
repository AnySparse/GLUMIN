#define __N_LISTS1 5
#define __N_BITMAPS1 2

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_warp(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 5];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 2];
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
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 5;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = begin + warp_id; v0_idx < candidate_v0.size(); v0_idx += num_warps){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    if (candidate_v1.size() > WARP_LIMIT) continue; 
    __build_LUT(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    for (vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx += 1) {
      auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
      __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v1), __get_vlist_from_graph(g, meta, /*vid=*/v0), /*upper_bound=*/-1, /*output_slot=*/0);
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/4);      
      auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
      for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
        auto v2 = candidate_v2[v2_idx];
        __intersect(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v1, /*output_slot=*/2);
        __difference(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/3);
        __build_bitmap_from_vmap(g, meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/3), /*bitmap_id=*/0);
        auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
        for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
          auto v3_idx = candidate_v3_idx[v3_idx_idx];
          count += __difference_num(__get_vmap_from_heap(g, meta, /*bitmap_id=*/0, /*slot_id=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/-1);
        }
      }
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_block(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 5];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 2];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int block_id    = blockIdx.x;
  int num_blocks  = gridDim.x;
  AccType count = 0;
  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(block_id);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 5;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = begin + block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){ /*add begin end, block_id!!!*/
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    if (candidate_v1.size() <= WARP_LIMIT || candidate_v1.size() > BLOCK_LIMIT) continue; /*add condition!!!*/
    __build_LUT_block(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));; /*use block!!!*/
    __syncthreads(); /*syncthread!!!*/
    for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) {
      auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
      __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v1), __get_vlist_from_graph(g, meta, /*vid=*/v0), /*upper_bound=*/-1, /*output_slot=*/0);
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/4);
      auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
      for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
        auto v2 = candidate_v2[v2_idx];
        __intersect(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v1, /*output_slot=*/2);
        __difference(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/3);
        __build_bitmap_from_vmap(g, meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/3), /*bitmap_id=*/0);
        auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
        for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
          auto v3_idx = candidate_v3_idx[v3_idx_idx];
          count += __difference_num(__get_vmap_from_heap(g, meta, /*bitmap_id=*/0, /*slot_id=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/-1);
        }
      }
    }
    __syncthreads();
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_global(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs,
                  vidType task_id){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 5];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 2];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  AccType count = 0;
  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(0);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 5;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto v0 = task_id;
  auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
  __set_LUT_para(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0)); /*no build!!!*/
  for (vidType v1_idx = warp_id; v1_idx < candidate_v1.size(); v1_idx += num_warps) {
    auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
    __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v1), __get_vlist_from_graph(g, meta, /*vid=*/v0), /*upper_bound=*/-1, /*output_slot=*/0);
    __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/4);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
      auto v2 = candidate_v2[v2_idx];
      __intersect(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v1, /*output_slot=*/2);
      __difference(meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/4), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/3);
      __build_bitmap_from_vmap(g, meta, __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/3), /*bitmap_id=*/0);
      auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
      for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
        auto v3_idx = candidate_v3_idx[v3_idx_idx];
        count += __difference_num(__get_vmap_from_heap(g, meta, /*bitmap_id=*/0, /*slot_id=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/-1);
      }
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}