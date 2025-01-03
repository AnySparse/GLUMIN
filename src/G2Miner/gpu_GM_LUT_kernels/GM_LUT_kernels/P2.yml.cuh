#define __N_LISTS1 3
#define __N_BITMAPS1 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_LUT_warp(vidType begin, vidType end, 
                  vidType *vid_list,
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
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
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 3;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = begin + warp_id; v0_idx < candidate_v0.size(); v0_idx += num_warps){
    // auto v0 = candidate_v0[v0_idx];
    auto v0 = vid_list[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    __build_LUT(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    for (vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx += 1) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), /*slot_id=*/1);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
      for(vidType v2_idx_idx = thread_lane; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += WARP_SIZE){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        count += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
      }
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_LUT_block(vidType begin, vidType end, 
                  vidType *vid_list,
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id    = blockIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
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
  meta.capacity = 3;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = begin + block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){
    auto v0 = vid_list[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    __build_LUT_block(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    __syncthreads();
    for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), /*slot_id=*/1);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
      for(vidType v2_idx_idx = thread_lane; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += WARP_SIZE){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        count += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
      }
    }
    __syncthreads();
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_LUT_global(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs,
                  vidType task_id){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
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
  meta.lut = LUTs.getEmptyLUT(0);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 3;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto v0 = task_id;
  auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
  __set_LUT_para(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));
  for (vidType v1_idx = warp_id; v1_idx < candidate_v1.size(); v1_idx += num_warps) {
    __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), /*slot_id=*/1);
    auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
    for(vidType v2_idx_idx = thread_lane; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += WARP_SIZE){
      auto v2_idx = candidate_v2_idx[v2_idx_idx];
      count += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}