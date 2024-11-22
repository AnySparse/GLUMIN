__global__ void P2_GF_LUT_warp(vidType begin, vidType end, GraphGPU g,
  vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
  AccType *INDEX1, Roaring_LUTManager<> LUTs) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType correct_count = 0;
  AccType P2_count = 0;
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
      if (thread_lane == 0)
        list_size[warp_lane][0] = cnt;
      __syncwarp();
      cnt = difference_set(g.N(v1), v1_size, v0_ptr, v0_size, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
        vlist[max_deg * 4 + i] = 0;
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
        vidType v4 = vlist[max_deg + i];
        vidType v4_size = g.getOutDegree(v4);
        for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
          vidType key = v0_ptr[vlist[j]];
          if (!binary_search(g.N(v4), key, v4_size)) {
            atomicAdd(&vlist[max_deg * 4 + j], 1);
          }
        }
      }
      __syncwarp();
      for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE) {
        vidType v2_idx = vlist[i];
        vidType tmp_cnt = lut.difference_num_thread(vlist0_opt, v2_idx, vlist0, v2_idx);
        P2_count += (tmp_cnt * vlist[max_deg * 4 + i]);
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
        vidType v4 = vlist[max_deg + i];
        vidType v4_size = g.getOutDegree(v4);
        vlist2.init(vlist + max_deg * 3, v0_size);
        cnt = lut.difference_set(vlist, v0_ptr, list_size[warp_lane][0], g.N(v4), v4_size, &vlist[max_deg * 2], vlist2);
        if (thread_lane == 0) list_size[warp_lane][2] = cnt;
        __syncwarp();
        cnt = lut.intersect_set(vlist, v0_ptr, list_size[warp_lane][0], g.N(v4), v4_size, &vlist[max_deg * 3]);
        if (thread_lane == 0) list_size[warp_lane][3] = cnt;
        __syncwarp();
        for (vidType ii = thread_lane; ii < list_size[warp_lane][3]; ii += WARP_SIZE) {
          vidType v2_idx = vlist[max_deg * 3 + ii];
          correct_count += lut.difference_num_thread_lower(true, v2_idx, v0_size, vlist2, v2_idx);
        }
      }
    }

    if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
    __syncwarp();
    vid = __shfl_sync(0xffffffff, vid, 0);
  }
  atomicAdd(&counters[0], P2_count);
  atomicAdd(&counters[1], correct_count);
}