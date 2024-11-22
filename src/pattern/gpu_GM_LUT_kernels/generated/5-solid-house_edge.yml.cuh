#define __N_LISTS1 4
#define __N_BITMAPS1 2

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_warp_edge(eidType begin, eidType ne, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 4];
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
  // meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 4;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  for (eidType e01_idx = begin + warp_id; e01_idx < ne; e01_idx += num_warps){
    auto v0 = g.get_src(e01_idx);
    auto v0_size = g.getOutDegree(v0);
    if (v0_size > WARP_LIMIT) continue;
    auto v1 = g.get_dst(e01_idx);
    __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    __build_LUT(g, meta, candidate_v2);
    for (vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx++) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/2);
      auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
      for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
        auto v3_idx = candidate_v3_idx[v3_idx_idx];
        count += __intersect_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/true, /*upper_bound=*/-1), /*upper_bound=*/v3_idx);
      }    
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_block_edge(eidType begin, eidType ne, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 4];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 2];
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
  // meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 4;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  for (eidType e01_idx = begin + block_id; e01_idx < ne; e01_idx += num_blocks){
    auto v0 = g.get_src(e01_idx);
    auto v0_size = g.getOutDegree(v0);
    if (v0_size <= WARP_LIMIT || v0_size > BLOCK_LIMIT) continue;
    auto v1 = g.get_dst(e01_idx);
    __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0); 
    __build_LUT_block(g, meta, candidate_v2);
    __syncthreads();
    for (vidType v2_idx = warp_lane; v2_idx < candidate_v2.size(); v2_idx += WARPS_PER_BLOCK) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/2);
      auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
      for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
        auto v3_idx = candidate_v3_idx[v3_idx_idx];
        count += __intersect_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/true, /*upper_bound=*/-1), /*upper_bound=*/v3_idx);
      }
    }
    __syncthreads();
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_LUT_block_large_edge(eidType begin, eidType ne, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 4];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 2];
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
  // meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 4;
  meta.bitmap_capacity = 2;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  for (eidType e01_idx = begin + block_id; e01_idx < ne; e01_idx += num_blocks){
    auto v0 = g.get_src(e01_idx);
    auto v0_size = g.getOutDegree(v0);
    if (v0_size <= BLOCK_LIMIT) continue;
    auto v1 = g.get_dst(e01_idx);
    __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    __build_LUT_block(g, meta, candidate_v2);
    __syncthreads();
    for (vidType v2_idx = warp_lane; v2_idx < candidate_v2.size(); v2_idx += WARPS_PER_BLOCK) {
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*slot_id=*/2);
      auto candidate_v3_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/2);
      for(vidType v3_idx_idx = thread_lane; v3_idx_idx < candidate_v3_idx.size(); v3_idx_idx += WARP_SIZE){
        auto v3_idx = candidate_v3_idx[v3_idx_idx];
        count += __intersect_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v3_idx, /*connected=*/true, /*upper_bound=*/-1), /*upper_bound=*/v3_idx);
      }
    }
    __syncthreads();
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}

