__global__ void P7_GF_LUT_warp(vidType begin, eidType end, GraphGPU g,
  vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
  AccType *INDEX1, Roaring_LUTManager<> LUTs) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  vidType bitmap_base_size = (max_deg + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
  bitmapType *bitmap_buffer = &bitmaps[int64_t(warp_id) * int64_t(bitmap_base_size) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P7_count = 0;
  AccType correct_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(warp_id);
  Roaring_Bitmap1D<> vlist0, vlist1, vlist2;
  bool vlist0_opt, vlist1_opt, vlist2_opt;
  __syncthreads();
  for (eidType eid = warp_id; eid < end;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    if (v0_size > WARP_LIMIT) {
      if (thread_lane == 0) eid = atomicAdd(INDEX1, 1);
      __syncwarp();
      eid = __shfl_sync(0xffffffff, eid, 0);
      continue;
    }
    auto cnt = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
      vlist[max_deg * 3 + i] = 0;
    }
    __syncwarp();

    for (int i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
        vidType key = vlist[j]; // each thread picks a vertex as the key
        // store BS result
        bool bs_res = binary_search(g.N(v2), key, v2_size);
        if (j < i && !bs_res)
          atomicAdd(&vlist[max_deg * 3 + j], 1);
        // warp_set bitmap  
        bool flag = (j!=i) && bs_res;
        lut.warp_set(i, j, flag);
      }
    }
    __syncwarp();
    for (vidType v2_idx = 0; v2_idx < list_size[warp_lane][0]; v2_idx++) {
      auto dif_cnt = lut.difference_set(v2_idx, v2_idx, vlist + max_deg, vlist1, vlist1_opt);
      if(thread_lane == 0) {
        P7_count += (dif_cnt * vlist[max_deg * 3 + v2_idx]);
      }
      __syncwarp();
      auto int_cnt = lut.intersect_set(list_size[warp_lane][0], v2_idx, vlist + max_deg * 2, vlist2, vlist2_opt);
      if (thread_lane == 0) {
        list_size[warp_lane][1] = dif_cnt;
        list_size[warp_lane][2] = int_cnt;
      }
      __syncwarp();
      for (vidType i = thread_lane; i < list_size[warp_lane][2]; i += WARP_SIZE) {
        vidType v3_idx = vlist[max_deg * 2 + i];
        correct_count += lut.difference_num_thread_lower(vlist1_opt, v3_idx, v2_idx, vlist1, v3_idx);
      }
    }

    if (thread_lane == 0) eid = atomicAdd(INDEX1, 1);
    __syncwarp();
    eid = __shfl_sync(0xffffffff, eid, 0);
  }
  atomicAdd(&counters[0], P7_count);
  atomicAdd(&counters[1], correct_count);
}

__global__ void P7_GF_LUT_block(vidType begin, eidType end, GraphGPU g,
  vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
  AccType *INDEX1, Roaring_LUTManager<> LUTs) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  int block_id  = blockIdx.x;
  int num_blocks = gridDim.x;
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  vidType *block_vlist = &vlists[int64_t(num_warps) * int64_t(max_deg) * 4 + int64_t(block_id) * int64_t(max_deg)];
  vidType bitmap_base_size = (max_deg + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
  bitmapType *bitmap_buffer = &bitmaps[int64_t(warp_id) * int64_t(bitmap_base_size) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P7_count = 0;
  AccType correct_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(block_id);
  Roaring_Bitmap1D<> vlist0, vlist1, vlist2;
  bool vlist0_opt, vlist1_opt, vlist2_opt;
  __shared__ vidType shared_eid;
  __syncthreads();
  for (eidType eid = block_id; eid < end;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    if (v0_size <= WARP_LIMIT) {
      if (threadIdx.x == 0) shared_eid = atomicAdd(INDEX1, 1);
      __syncthreads();
      eid = shared_eid;
      continue;
    }
    auto cnt = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType i = threadIdx.x; i < list_size[warp_lane][0]; i += BLOCK_SIZE) {
      block_vlist[i] = 0;
    }
    __syncthreads();

    for (int i = warp_lane; i < list_size[warp_lane][0]; i += WARPS_PER_BLOCK) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
        vidType key = vlist[j]; // each thread picks a vertex as the key
        // store BS result
        bool bs_res = binary_search(g.N(v2), key, v2_size);
        if (j < i && !bs_res)
          atomicAdd(&block_vlist[j], 1);
        // warp_set bitmap  
        bool flag = (j!=i) && bs_res;
        lut.warp_set(i, j, flag);
      }
    }
    __syncthreads();
    for (vidType v2_idx = warp_lane; v2_idx < list_size[warp_lane][0]; v2_idx += WARPS_PER_BLOCK) {
      auto dif_cnt = lut.difference_set(v2_idx, v2_idx, vlist + max_deg, vlist1, vlist1_opt);
      if(thread_lane == 0) {
        P7_count += (dif_cnt * block_vlist[v2_idx]);
      }
      __syncwarp();
      auto int_cnt = lut.intersect_set(list_size[warp_lane][0], v2_idx, vlist + max_deg * 2, vlist2, vlist2_opt);
      if (thread_lane == 0) {
        list_size[warp_lane][1] = dif_cnt;
        list_size[warp_lane][2] = int_cnt;
      }
      __syncwarp();
      for (vidType i = thread_lane; i < list_size[warp_lane][2]; i += WARP_SIZE) {
        vidType v3_idx = vlist[max_deg * 2 + i];
        correct_count += lut.difference_num_thread_lower(vlist1_opt, v3_idx, v2_idx, vlist1, v3_idx);
      }
    }

    if (threadIdx.x == 0) shared_eid = atomicAdd(INDEX1, 1);
    __syncthreads();
    eid = shared_eid;
  }
  atomicAdd(&counters[0], P7_count);
  atomicAdd(&counters[1], correct_count);
}

