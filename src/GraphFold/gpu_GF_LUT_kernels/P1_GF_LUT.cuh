__global__ void P1_GF_LUT_warp(vidType begin, vidType end, GraphGPU g,
    vidType *vlists, bitmapType* bitmaps, vidType max_deg, AccType *counters,
    AccType *INDEX1, Roaring_LUTManager<> LUTs) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;                   // global warp index
    int thread_lane =
        threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
    int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
    vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 3];
    __shared__ vidType list_size[WARPS_PER_BLOCK][2];
    AccType count = 0;
    AccType star3_count = 0;
    Roaring_LUT<> lut;
    lut = LUTs.getEmptyLUT(warp_id);
    Roaring_Bitmap1D<> vlist0, vlist1;
    bool vlist0_opt, vlist1_opt;
    __syncthreads();
    for (vidType vid = begin + warp_id; vid < end;) {
      vidType v0 = vid;
      vidType v0_size = g.getOutDegree(v0);
      if (v0_size > WARP_LIMIT) {
        if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
        __syncwarp();
        vid = __shfl_sync(0xffffffff, vid, 0);
        continue;
      }
      for (vidType i = thread_lane; i < v0_size; i += WARP_SIZE) {
        vlist[max_deg * 2 + i] = 0;
      }
      __syncwarp();
      // build lut merge
      for (int i = 0; i < v0_size; i++) {
        vidType v1 = g.N(v0)[i];
        vidType v1_size = g.getOutDegree(v1);
        for (auto j = thread_lane; j < v0_size; j += WARP_SIZE) {
          vidType key = g.N(v0)[j]; // each thread picks a vertex as the key
          int is_smaller = key < v1 ? 1 : 0;
          // store BS result
          bool bs_res = binary_search(g.N(v1), key, v1_size);
          if (is_smaller && !bs_res)
            atomicAdd(&vlist[max_deg * 2 + j], 1);
          // warp_set bitmap  
          bool flag = (j!=i) && bs_res;
          lut.warp_set(i, j, flag);
        }
      }
      __syncwarp();
      for (vidType v2_idx = 0; v2_idx < v0_size; v2_idx++) {
        vidType v2 = g.N(v0)[v2_idx];
        vidType v2_size = g.getOutDegree(v2);
        auto dif_cnt = lut.difference_set(v2_idx, v2_idx, vlist, vlist0, vlist0_opt);
        if(thread_lane == 0) {
          star3_count += (dif_cnt * vlist[max_deg * 2 + v2_idx]);
        }
        __syncwarp();
        auto int_cnt = lut.intersect_set(v2_idx, v2_idx, vlist + max_deg, vlist1, vlist1_opt);
        if (thread_lane == 0) {
          list_size[warp_lane][0] = dif_cnt;
          list_size[warp_lane][1] = int_cnt;
        }
        __syncwarp();
        // how to bitmap?
        for (vidType i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE) {
        // for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
          vidType v3_idx = vlist[max_deg + i];
          vidType v3 = g.N(v0)[v3_idx];
          count += lut.difference_num_thread_lower(vlist0_opt, v3_idx, v2_idx, vlist0, v3_idx);
          // for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
          //   auto key_idx = vlist[j];
          //   auto key = g.N(v0)[key_idx];
          //   if (key_idx > v3_idx && !lut.get(v3_idx, key_idx)) count += 1;
          // }
        }
      }
      if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
      __syncwarp();
      vid = __shfl_sync(0xffffffff, vid, 0);
    }
    atomicAdd(&counters[0], star3_count);
    atomicAdd(&counters[1], count);
  }

__global__ void P1_GF_LUT_block(vidType begin, vidType end, GraphGPU g,
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
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  vidType *block_vlist = &vlists[int64_t(num_warps) * int64_t(max_deg) * 2 + int64_t(block_id) * int64_t(max_deg)];
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  AccType count = 0;
  AccType star3_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(block_id);
  Roaring_Bitmap1D<> vlist0, vlist1;
  bool vlist0_opt, vlist1_opt;
  __shared__ vidType shared_vid;
  __syncthreads();
  for (vidType vid = begin + block_id; vid < end;) {
    vidType v0 = vid;
    vidType v0_size = g.getOutDegree(v0);
    if (v0_size <= WARP_LIMIT || v0_size > BLOCK_LIMIT) {
      if (threadIdx.x == 0) shared_vid = atomicAdd(INDEX1, 1);
      __syncthreads();
      vid = shared_vid;
      continue;
    }
    for (vidType i = threadIdx.x; i < v0_size; i += BLOCK_SIZE) {
      block_vlist[i] = 0;
    }
    __syncthreads();
    // build lut merge
    for (int i = warp_lane; i < v0_size; i += WARPS_PER_BLOCK) {
      vidType v1 = g.N(v0)[i];
      vidType v1_size = g.getOutDegree(v1);
      for (auto j = thread_lane; j < v0_size; j += WARP_SIZE) {
        vidType key = g.N(v0)[j]; // each thread picks a vertex as the key
        int is_smaller = key < v1 ? 1 : 0;
        // store BS result
        bool bs_res = binary_search(g.N(v1), key, v1_size);
        if (is_smaller && !bs_res)
          atomicAdd(&block_vlist[j], 1);
        // warp_set bitmap  
        bool flag = (j!=i) && bs_res;
        lut.warp_set(i, j, flag);
      }
    }
    __syncthreads();
    for (vidType v2_idx = warp_lane; v2_idx < v0_size; v2_idx += WARPS_PER_BLOCK) {
      vidType v2 = g.N(v0)[v2_idx];
      vidType v2_size = g.getOutDegree(v2);
      auto dif_cnt = lut.difference_set(v2_idx, v2_idx, vlist, vlist0, vlist0_opt);
      if(thread_lane == 0) {
        star3_count += (dif_cnt * block_vlist[v2_idx]);
      }
      __syncwarp();
      auto int_cnt = lut.intersect_set(v2_idx, v2_idx, vlist + max_deg, vlist1, vlist1_opt);
      if (thread_lane == 0) {
        list_size[warp_lane][0] = dif_cnt;
        list_size[warp_lane][1] = int_cnt;
      }
      __syncwarp();
      // how to bitmap?
      for (vidType i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE) {
        vidType v3_idx = vlist[max_deg + i];
        vidType v3 = g.N(v0)[v3_idx];
        count += lut.difference_num_thread_lower(vlist0_opt, v3_idx, v2_idx, vlist0, v3_idx);
      }
    }
    if (threadIdx.x == 0) shared_vid = atomicAdd(INDEX1, 1);
    __syncthreads();
    vid = shared_vid;
  }
  atomicAdd(&counters[0], star3_count);
  atomicAdd(&counters[1], count);
}

__global__ void clear_counterlist(GraphGPU g, vidType *vlists, vidType max_deg, vidType task_id) {
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int num_threads = num_warps * WARP_SIZE;
  int block_id  = blockIdx.x;
  vidType *block_vlist = &vlists[int64_t(num_warps) * int64_t(max_deg) * 2];
  vidType v0 = task_id;
  vidType v0_size = g.getOutDegree(v0);
  for (vidType i = thread_id; i < v0_size; i += num_threads) {
    block_vlist[i] = 0;
  }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GF_build_LUT(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  Roaring_LUTManager<> LUTs,
                  vidType task_id){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 2];
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int num_threads = num_warps * WARP_SIZE;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int block_id    = blockIdx.x;
  AccType count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(0);
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  vidType *block_vlist = &vlists[int64_t(num_warps) * int64_t(max_deg) * 2];

  __syncwarp();
  vidType v0 = task_id;
  vidType v0_size = g.getOutDegree(v0);

  for (int i = warp_id; i < v0_size; i += num_warps) {
    vidType v1 = g.N(v0)[i];
    vidType v1_size = g.getOutDegree(v1);
    for (auto j = thread_lane; j < v0_size; j += WARP_SIZE) {
      vidType key = g.N(v0)[j]; // each thread picks a vertex as the key
      int is_smaller = key < v1 ? 1 : 0;
      // store BS result
      bool bs_res = binary_search(g.N(v1), key, v1_size);
      if (is_smaller && !bs_res)
        atomicAdd(&block_vlist[j], 1);
      // warp_set bitmap  
      bool flag = (j!=i) && bs_res;
      lut.warp_set(i, j, flag);
    }
  }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P1_GF_LUT_global(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counters,
                  Roaring_LUTManager<> LUTs,
                  vidType task_id){
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int num_threads = num_warps * WARP_SIZE;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int block_id    = blockIdx.x;
  __shared__ vidType list_size[WARPS_PER_BLOCK][2];
  AccType count = 0;
  AccType star3_count = 0;
  Roaring_LUT<> lut;
  lut = LUTs.getEmptyLUT(0);
  Roaring_Bitmap1D<> vlist0, vlist1;
  bool vlist0_opt, vlist1_opt;
  vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  vidType *block_vlist = &vlists[int64_t(num_warps) * int64_t(max_deg) * 2];

  __syncwarp();
  vidType v0 = task_id;
  vidType v0_size = g.getOutDegree(v0);
  for (vidType v2_idx = warp_id; v2_idx < v0_size; v2_idx += num_warps) {
    vidType v2 = g.N(v0)[v2_idx];
    vidType v2_size = g.getOutDegree(v2);
    auto dif_cnt = lut.difference_set(v2_idx, v2_idx, vlist, vlist0, vlist0_opt);
    if(thread_lane == 0) {
      star3_count += (dif_cnt * block_vlist[v2_idx]);
    }
    __syncwarp();
    auto int_cnt = lut.intersect_set(v2_idx, v2_idx, vlist + max_deg, vlist1, vlist1_opt);
    if (thread_lane == 0) {
      list_size[warp_lane][0] = dif_cnt;
      list_size[warp_lane][1] = int_cnt;
    }
    __syncwarp();
    for (vidType i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE) {
      vidType v3_idx = vlist[max_deg + i];
      vidType v3 = g.N(v0)[v3_idx];
      count += lut.difference_num_thread_lower(vlist0_opt, v3_idx, v2_idx, vlist0, v3_idx);
    }
  }
  atomicAdd(&counters[0], star3_count);
  atomicAdd(&counters[1], count);
}
