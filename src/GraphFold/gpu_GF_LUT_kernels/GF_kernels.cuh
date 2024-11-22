__global__ void P1_count_correction(eidType ne, GraphGPU g,
                                    vidType *vlists, vidType max_deg, AccType *counters,
                                    AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  AccType count = 0;
  __syncthreads();
  for (eidType eid = warp_id; eid < ne;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    if (v1 == v0) {
      if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
      __syncwarp();
      eid = __shfl_sync(0xffffffff, eid, 0);
      continue;
    }
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto dif_cnt = difference_set(g.N(v0), v0_size, g.N(v1),
                                  v1_size, v1, vlist);
    auto int_cnt = intersect(g.N(v0), v0_size, g.N(v1),
                             v1_size, v1, &vlist[max_deg]); // y0y1
    if (thread_lane == 0) {
      list_size[warp_lane][0] = dif_cnt;
      list_size[warp_lane][1] = int_cnt;
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
      vidType v2 = vlist[max_deg + i];
      vidType v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
        auto key = vlist[j];
        vidType key_size = g.getOutDegree(key);
        if (key > v2 && !binary_search(g.N(key), v2, key_size))
          count += 1;
      }
    }
    __syncwarp();
    if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
    __syncwarp();
    eid = __shfl_sync(0xffffffff, eid, 0);
  }
  atomicAdd(&counters[1], count);
}

__global__ void P1_frequency_count(vidType nv, GraphGPU g,
                                   vidType *vlists, vidType max_deg, AccType *counters,
                                   AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  AccType count = 0;
  AccType star3_count = 0;
  __syncthreads();
  for (vidType vid = warp_id; vid < nv;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    for (vidType i = thread_lane; i < v0_size; i += WARP_SIZE) {
      vlist[max_deg + i] = 0;
    }
    __syncwarp();
    // build lut merge
    for (int j = 0; j < v0_size; j++) {
      vidType v1 = g.N(v0)[j];
      vidType v1_size = g.getOutDegree(v1);
      for (auto i = thread_lane; i < v0_size; i += WARP_SIZE) {
        vidType key = g.N(v0)[i]; // each thread picks a vertex as the key
        int is_smaller = key < v1 ? 1 : 0;
        if (is_smaller && !binary_search(g.N(v1), key, v1_size))
          atomicAdd(&vlist[max_deg + i], 1);
      }
    }
    __syncwarp();
    for (vidType v2_idx = 0; v2_idx < v0_size; v2_idx++) {
      vidType v2 = g.N(v0)[v2_idx];
      vidType v2_size = g.getOutDegree(v2);
      vidType tmp_cnt = difference_num(g.N(v0), v0_size,
                                   g.N(v2), v2_size, v2);
      vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        star3_count += (warp_cnt * vlist[max_deg + v2_idx]);
      __syncwarp();
    }
    if (thread_lane == 0) vid = atomicAdd(INDEX, 1);
    __syncwarp();
    vid = __shfl_sync(0xffffffff, vid, 0);
  }
  atomicAdd(&counters[0], star3_count);
}

__global__ void P1_fused_matching(vidType nv, vidType ne, GraphGPU g,
                                   vidType *vlists, vidType max_deg, AccType *counters,
                                   AccType *INDEX1, AccType *INDEX2) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  AccType count = 0;
  AccType star3_count = 0;
  __syncthreads();
  for (vidType vid = warp_id; vid < nv;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    for (vidType i = thread_lane; i < v0_size; i += WARP_SIZE) {
      vlist[max_deg + i] = 0;
    }
    __syncwarp();
    // build lut merge
    for (int j = 0; j < v0_size; j++) {
      vidType v1 = g.N(v0)[j];
      vidType v1_size = g.getOutDegree(v1);
      for (auto i = thread_lane; i < v0_size; i += WARP_SIZE) {
        vidType key = g.N(v0)[i]; // each thread picks a vertex as the key
        int is_smaller = key < v1 ? 1 : 0;
        if (is_smaller && !binary_search(g.N(v1), key, v1_size))
          atomicAdd(&vlist[max_deg + i], 1);
      }
    }
    __syncwarp();
    for (vidType v2_idx = 0; v2_idx < v0_size; v2_idx++) {
      vidType v2 = g.N(v0)[v2_idx];
      vidType v2_size = g.getOutDegree(v2);
      vidType tmp_cnt = difference_num(g.N(v0), v0_size,
                                   g.N(v2), v2_size, v2);
      vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        star3_count += (warp_cnt * vlist[max_deg + v2_idx]);
      __syncwarp();
      auto dif_cnt = difference_set(g.N(v0), v0_size, g.N(v2),
                                  v2_size, v2, vlist);
      auto int_cnt = intersect(g.N(v0), v0_size, g.N(v2),
                             v2_size, v2, &vlist[max_deg]); // y0y1
      if (thread_lane == 0) {
        list_size[warp_lane][0] = dif_cnt;
        list_size[warp_lane][1] = int_cnt;
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
        vidType v3 = vlist[max_deg + i];
        vidType v3_size = g.getOutDegree(v3);
        for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
          auto key = vlist[j];
          vidType key_size = g.getOutDegree(key);
          if (key > v3 && !binary_search(g.N(key), v3, key_size))
            count += 1;
        }
      }
    }
    if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
    __syncwarp();
    vid = __shfl_sync(0xffffffff, vid, 0);
  }
  atomicAdd(&counters[0], star3_count);
  atomicAdd(&counters[1], count);
}

__global__ void P2_fused_matching(eidType ne, GraphGPU g,
                                  vidType *vlists, vidType max_deg, AccType *counters,
                                  AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P2_count = 0;
  AccType correct_count = 0;
  AccType calculate_count = 0;
  __syncthreads();
  for (eidType eid = warp_id; eid < ne;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.N(v0);
    auto v1_ptr = g.N(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]);
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
        vidType key = vlist[j];
        if (!binary_search(g.N(v4), key, v4_size)) {
          atomicAdd(&vlist[max_deg * 4 + j], 1);
        }
      }
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      vidType tmp_cnt = difference_num(vlist, list_size[warp_lane][0],
                                   g.N(v2), v2_size, v2);
      vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P2_count += (warp_cnt * vlist[max_deg * 4 + i]);
      __syncwarp();
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
      vidType v4 = vlist[max_deg + i];
      vidType v4_size = g.getOutDegree(v4);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.N(v4),
                           v4_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.N(v4),
                      v4_size, &vlist[max_deg * 3]);
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();
      for (vidType ii = 0; ii < list_size[warp_lane][3]; ii++) {
        vidType v2 = vlist[max_deg * 3 + ii];
        vidType v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][2]; j += WARP_SIZE) {
          auto key = vlist[max_deg * 2 + j];
          vidType key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.N(key), v2, key_size))
            correct_count += 1;
        }
      }
    }
    if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
    __syncwarp();
    eid = __shfl_sync(0xffffffff, eid, 0);
  }
  atomicAdd(&counters[0], P2_count);
  atomicAdd(&counters[1], correct_count);
}

__global__ void P3_fused_matching(eidType ne, GraphGPU g,
                                  vidType *vlists, vidType max_deg, AccType *counters,
                                  AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P3_count = 0;
  AccType correct_count = 0;
  __syncthreads();
  for (vidType eid = warp_id; eid < ne;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.N(v0);
    auto v1_ptr = g.N(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    // if (thread_lane == 0) P3_count += list_size[warp_lane][0];
    for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += 32) {
      vlist[max_deg * 4 + i] = 0;
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < i; j += WARP_SIZE) {
        vidType key = vlist[j];
        if (!binary_search(g.N(v2), key, v2_size)) {
          atomicAdd(&vlist[max_deg * 4 + j], 1);
        }
      }
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      vidType tmp_cnt = difference_num(vlist, list_size[warp_lane][0],
                                    g.N(v2), v2_size, v2);
      vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P3_count += (warp_cnt * vlist[max_deg * 4 + i]);
      __syncwarp();
    }
    __syncwarp();

    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.N(v2),
                            v2_size, v2, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.N(v2),
                      v2_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      for (vidType ii = 0; ii < list_size[warp_lane][2]; ii++) {
        vidType v2 = vlist[max_deg * 2 + ii];
        vidType v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][1];
              j += WARP_SIZE) {
          auto key = vlist[max_deg * 1 + j];
          vidType key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.N(key), v2, key_size))
            correct_count += 1;
        }
      }
    }
    if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
    __syncwarp();
    eid = __shfl_sync(0xffffffff, eid, 0);
  }

  atomicAdd(&counters[0], P3_count);
  atomicAdd(&counters[1], correct_count);
}



__global__ void P3_fused_matching_vertex(eidType nv, GraphGPU g,
  vidType *vlists, vidType max_deg, AccType *counters,
  AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
  threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P3_count = 0;
  AccType correct_count = 0;
  __syncthreads();
  for (vidType vid = warp_id; vid < nv;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    auto v0_ptr = g.N(v0);
    for (vidType v1_idx = 0; v1_idx < v0_size; v1_idx++){
      vidType v1 = v0_ptr[v1_idx];
      vidType v1_size = g.getOutDegree(v1);
      auto v1_ptr = g.N(v1);
      auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
      if (thread_lane == 0)
        list_size[warp_lane][0] = cnt;
        __syncwarp();
        // if (thread_lane == 0) P3_count += list_size[warp_lane][0];
        for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += 32) {
          vlist[max_deg * 4 + i] = 0;
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2 = vlist[i];
        vidType v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < i; j += WARP_SIZE) {
          vidType key = vlist[j];
          if (!binary_search(g.N(v2), key, v2_size)) {
            atomicAdd(&vlist[max_deg * 4 + j], 1);
          }
        }
      }
      __syncwarp();
      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2 = vlist[i];
        vidType v2_size = g.getOutDegree(v2);
        vidType tmp_cnt = difference_num(vlist, list_size[warp_lane][0],
            g.N(v2), v2_size, v2);
        vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
        __syncwarp();
        if (thread_lane == 0)
          P3_count += (warp_cnt * vlist[max_deg * 4 + i]);
        __syncwarp();
      }
      __syncwarp();

      for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
        vidType v2 = vlist[i];
        vidType v2_size = g.getOutDegree(v2);
        cnt = difference_set(vlist, list_size[warp_lane][0], g.N(v2),
        v2_size, v2, &vlist[max_deg]);
        if (thread_lane == 0)
          list_size[warp_lane][1] = cnt;
        __syncwarp();
        cnt = intersect(vlist, list_size[warp_lane][0], g.N(v2),
        v2_size, &vlist[max_deg * 2]);
        if (thread_lane == 0)
          list_size[warp_lane][2] = cnt;
        __syncwarp();
        for (vidType ii = 0; ii < list_size[warp_lane][2]; ii++) {
          vidType v2 = vlist[max_deg * 2 + ii];
          vidType v2_size = g.getOutDegree(v2);
          for (auto j = thread_lane; j < list_size[warp_lane][1]; j += WARP_SIZE) {
            auto key = vlist[max_deg * 1 + j];
            vidType key_size = g.getOutDegree(key);
            if (key > v2 && !binary_search(g.N(key), v2, key_size))
              correct_count += 1;
          }
        }
      }
    }
    if (thread_lane == 0) vid = atomicAdd(INDEX, 1);
    __syncwarp();
    vid = __shfl_sync(0xffffffff, vid, 0);
  }

  atomicAdd(&counters[0], P3_count);
  atomicAdd(&counters[1], correct_count);
}

__global__ void P7_fused_matching(eidType ne, GraphGPU g,
                                  vidType *vlists, vidType max_deg, AccType *counters,
                                  AccType *INDEX) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  __shared__ vidType list_size[WARPS_PER_BLOCK][4];
  AccType P7_count = 0;
  AccType correct_count = 0;
  __syncthreads();
  for (eidType eid = warp_id; eid < ne;) {
    vidType v0 = g.get_src(eid);
    vidType v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto cnt = intersect(g.N(v0), v0_size, g.N(v1), v1_size,
                         vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    for (vidType i = thread_lane; i < list_size[warp_lane][0]; i += 32) {
      vlist[max_deg * 3 + i] = 0;
    }
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < i; j += WARP_SIZE) {
        vidType key = vlist[j]; // each thread picks a vertex as the key
        if (!binary_search(g.N(v2), key, v2_size)) {
          atomicAdd(&vlist[max_deg * 3 + j], 1);
        }
      }
    }
    __syncwarp();

    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      vidType tmp_cnt = difference_num(vlist, list_size[warp_lane][0],
                                   g.N(v2), v2_size, v2);
      vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P7_count += (warp_cnt * vlist[max_deg * 3 + i]);
      __syncwarp();
    }

    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane][0]; i++) {
      vidType v2 = vlist[i];
      vidType v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.N(v2),
                           v2_size, v2, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.N(v2),
                      v2_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();

      for (vidType ii = 0; ii < list_size[warp_lane][2]; ii++) {
        vidType v2 = vlist[max_deg * 2 + ii];
        vidType v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][1];
             j += WARP_SIZE) {
          auto key = vlist[max_deg + j];
          vidType key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.N(key), v2, key_size))
            correct_count += 1;
        }
      }
    }
    if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
    __syncwarp();
    eid = __shfl_sync(0xffffffff, eid, 0);
  }

  atomicAdd(&counters[0], P7_count);
  atomicAdd(&counters[1], correct_count);
}