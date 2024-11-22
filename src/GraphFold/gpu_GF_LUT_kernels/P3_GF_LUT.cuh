__global__ void P3_GF_LUT_warp(vidType begin, vidType end, GraphGPU g,
  vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
  AccType *INDEX1, Roaring_LUTManager<> LUTs) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  vidType bitmap_base_size = (max_deg + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
  bitmapType *bitmap_buffer = &bitmaps[int64_t(warp_id) * int64_t(bitmap_base_size) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType correct_count = 0;
  AccType P3_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(warp_id);
  Roaring_Bitmap1D<> vlist0, vlist1, vlist2;
  bool vlist0_opt, vlist1_opt;
  __syncthreads();
  for (vidType vid = begin + warp_id; vid < end;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    if (v0_size > WARP_LIMIT) {
      if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
      __syncwarp();
      vid = __shfl_sync(0xffffffff, vid, 0);
      continue;
    }
    lut.build_LUT(g.N(v0), v0_size, g);
    for (vidType v1_idx = 0; v1_idx < v0_size; v1_idx++) {
      vidType v1 = g.N(v0)[v1_idx];
      vidType v1_size = g.getOutDegree(v1);
      auto cnt = lut.difference_set(v0_size, v1_idx, vlist, vlist0, vlist0_opt);
      if (thread_lane == 0) list_size[warp_lane][0] = cnt;
      __syncwarp();
      // if (thread_lane == 0) P3_count += list_size[warp_lane][0];
      for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
        vlist[max_deg * 4 + i] = 0;
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2_idx = vlist[i];
        for (auto j = thread_lane; j < i; j += WARP_SIZE) {
          vidType key = vlist[j];
          if (!lut.get(v2_idx, key)) {
            atomicAdd(&vlist[max_deg * 4 + j], 1);
          }
        }
      }
      __syncwarp();
      // for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
      //   vidType v2_idx = vlist[i];
      //   vidType tmp_cnt = lut.difference_num_thread(vlist0_opt, v2_idx, vlist0, v2_idx);
      //   P3_count += (tmp_cnt * vlist[max_deg * 4 + i]);
      // }
      // __syncwarp();


      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2_idx = vlist[i];
        vlist1.init(bitmap_buffer, v0_size);
        cnt = lut.difference_set(vlist0_opt, v2_idx, vlist0, v2_idx, vlist + max_deg, vlist1);
        if (thread_lane == 0) list_size[warp_lane][1] = cnt;
        __syncwarp();
        if (thread_lane == 0) P3_count += (cnt * vlist[max_deg * 4 + i]);
        vlist2.init(bitmap_buffer + bitmap_base_size, v0_size);
        cnt = lut.intersect_set(vlist0_opt, v2_idx, vlist0, v2_idx, vlist + max_deg * 2, vlist2);
        if (thread_lane == 0) list_size[warp_lane][2] = cnt;
        __syncwarp();
        for (vidType ii = thread_lane; ii < list_size[warp_lane][2]; ii += WARP_SIZE) {
          vidType v3_idx = vlist[max_deg * 2 + ii];
          correct_count += lut.difference_num_thread_lower(true, v3_idx, v2_idx, vlist1, v3_idx);
        }
      }
    }
    if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
    __syncwarp();
    vid = __shfl_sync(0xffffffff, vid, 0);
  }
  atomicAdd(&counters[0], P3_count);
  atomicAdd(&counters[1], correct_count);
}

__global__ void P3_GF_LUT_block(vidType begin, vidType end, GraphGPU g,
  vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
  AccType *INDEX1, Roaring_LUTManager<> LUTs) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  int block_id  = blockIdx.x;
  int num_blocks = gridDim.x;
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  vidType bitmap_base_size = (max_deg + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
  bitmapType *bitmap_buffer = &bitmaps[int64_t(warp_id) * int64_t(bitmap_base_size) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  __shared__ vidType shared_vid;
  AccType correct_count = 0;
  AccType P3_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(block_id);
  Roaring_Bitmap1D<> vlist0, vlist1, vlist2;
  bool vlist0_opt, vlist1_opt;
  __syncthreads();
  for (vidType vid = begin + block_id; vid < end;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    if (v0_size <= WARP_LIMIT) {
      if (threadIdx.x == 0) shared_vid = atomicAdd(INDEX1, 1);
      __syncthreads();
      vid = shared_vid;
      continue;
    }
    lut.build_roaring_LUT_block(g.N(v0), v0_size, g);
    __syncthreads();
    for (vidType v1_idx = warp_lane; v1_idx < v0_size; v1_idx += WARPS_PER_BLOCK) {
      vidType v1 = g.N(v0)[v1_idx];
      vidType v1_size = g.getOutDegree(v1);
      auto cnt = lut.difference_set(v0_size, v1_idx, vlist, vlist0, vlist0_opt);
      if (thread_lane == 0) list_size[warp_lane][0] = cnt;
      __syncwarp();
      // if (thread_lane == 0) P3_count += list_size[warp_lane][0];
      for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
        vlist[max_deg * 4 + i] = 0;
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2_idx = vlist[i];
        for (auto j = thread_lane; j < i; j += WARP_SIZE) {
          vidType key = vlist[j];
          if (!lut.get(v2_idx, key)) {
            atomicAdd(&vlist[max_deg * 4 + j], 1);
          }
        }
      }
      __syncwarp();

      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2_idx = vlist[i];
        vlist1.init(bitmap_buffer, v0_size);
        cnt = lut.difference_set(vlist0_opt, v2_idx, vlist0, v2_idx, vlist + max_deg, vlist1);
        if (thread_lane == 0) list_size[warp_lane][1] = cnt;
        __syncwarp();
        if (thread_lane == 0) P3_count += (cnt * vlist[max_deg * 4 + i]);
        vlist2.init(bitmap_buffer + bitmap_base_size, v0_size);
        cnt = lut.intersect_set(vlist0_opt, v2_idx, vlist0, v2_idx, vlist + max_deg * 2, vlist2);
        if (thread_lane == 0) list_size[warp_lane][2] = cnt;
        __syncwarp();
        for (vidType ii = thread_lane; ii < list_size[warp_lane][2]; ii += WARP_SIZE) {
          vidType v3_idx = vlist[max_deg * 2 + ii];
          correct_count += lut.difference_num_thread_lower(true, v3_idx, v2_idx, vlist1, v3_idx);
        }
      }
    }
    if (threadIdx.x == 0) shared_vid = atomicAdd(INDEX1, 1);
    __syncthreads();
    vid = shared_vid;
  }
  atomicAdd(&counters[0], P3_count);
  atomicAdd(&counters[1], correct_count);
}