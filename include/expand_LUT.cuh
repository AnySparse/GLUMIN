// Look up lut
#pragma once

#include "cutil_subset.h"
#include "common.h"
#include "expand_bitmap.cuh"

template <typename T = bitmapType, int W = BITMAP_WIDTH>
struct Roaring_LUT {
private:
  vidType* vlist_;
  vidType size_;
  vidType chunk_size_;
  Roaring_Bitmap2D<T, W> bitmap_;

public:
  __device__ size_t get_num_bytes(){
    return bitmap_.get_num_bytes(size_);
  }

  __device__ void init(T* heap_, uint32_t row_size){
    bitmap_.init(heap_, row_size);
  }

  __device__ void init(T* heap_, typeType* types_head_, uint32_t row_size){
    bitmap_.init(heap_, types_head_, row_size);
  }  

  __device__ void warp_set(uint32_t x, uint32_t y, bool flag) {
    bitmap_.warp_set(x, y, flag);
  }

  __device__ bool get(int x, int y) {
    return bitmap_.get(x, y);
  }

  __device__ Roaring_Bitmap1D<T,W> row(int x) {
    Roaring_Bitmap1D<T, W> ret;
#ifdef ROARING
    ret.init(bitmap_.row_head(x), bitmap_.types_row_head(x), size_);
#else
    ret.init(bitmap_.row_head(x), size_);
#endif
    return ret;
  }

  __device__ void build_LUT(vidType* sub_list, vidType list_size, GraphGPU g) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    vlist_ = sub_list;
    size_ = list_size;
    for (vidType i = 0; i < size_; i++) {
      auto search = g.N(vlist_[i]);
      vidType search_size = g.getOutDegree(vlist_[i]);
      for (int j = thread_lane; j < list_size; j += WARP_SIZE) {
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vlist_[j], search_size);
        __syncwarp(active);
        bitmap_.warp_set(i, j, flag);
      }
    }
  }

  __device__ vidType build_roaring_LUT(vidType* sub_list, vidType list_size, GraphGPU g) {
    vidType warp_lane = threadIdx.x / WARP_SIZE;
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    vlist_ = sub_list;
    size_ = list_size;
    chunk_size_ = (size_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    vidType count = 0;
#ifdef ROARING
    __shared__ vidType cnt_shared[WARPS_PER_BLOCK];
    __shared__ T chunk_buffer[WARPS_PER_BLOCK][PAD_PER_CHUNK];
    bitmap_.warp_clear_chunk_type(thread_lane);
    for (vidType i = 0; i < size_; i++) {
      auto search = g.N(vlist_[i]);
      vidType search_size = g.getOutDegree(vlist_[i]);
      for (int chunk_id = 0; chunk_id < chunk_size_; chunk_id++) {

        uint32_t pad_size, array_limit;
        if (chunk_id == chunk_size_ - 1) {
          pad_size = ((size_ - chunk_id * CHUNK_WIDTH) + W - 1) / W;
          array_limit = (size_ - (chunk_id * CHUNK_WIDTH)) / ARRAY_ID_WIDTH;
        }
        else {
          pad_size = PAD_PER_CHUNK;
          array_limit = ARRAY_LIMIT;
        }

        int cnt = 0;
        for (vidType idx = thread_lane; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size_; idx += WARP_SIZE) {
          int j = chunk_id * CHUNK_WIDTH + idx;
          unsigned active = __activemask();
          bool flag = (j!=i) && binary_search(search, vlist_[j], search_size);
          __syncwarp(active);
          // cnt += bitmap_.warp_set(i, j, flag);
          // set to buffer
          T mask = __ballot_sync(active, flag);
          uint32_t element = idx / W;
          if (thread_lane == 0) chunk_buffer[warp_lane][element] = mask;
          cnt += __popc(mask);
        }
        if (thread_lane == 0){
          cnt_shared[warp_lane] = cnt;
        }
        __syncwarp();

        // construct an array chunk
        if (cnt_shared[warp_lane] < array_limit) {
          arrayType* chunk_head = bitmap_.chunk_head(i, chunk_id);
          if (thread_lane == 0) {
            chunk_head[0] = cnt_shared[warp_lane];
            // count += (cnt_shared[warp_lane] + 1) * sizeof(arrayType);
            bitmap_.set_chunk_type(i, chunk_id, 1);
          }
          __syncwarp();
          vidType set_idx = 1;
          if (thread_lane == 0){
            for (vidType idx = 0; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size_; idx++) {
              arrayType chunk_lane = (arrayType)idx;
              if (chunk_buffer[warp_lane][idx / W] & (1 << (idx % W))) {
                chunk_head[set_idx] = chunk_lane;
                set_idx++;
              }
            }
          }
        }
        // else constuct bitmap chunk
        else {
          // count += pad_size * sizeof(bitmapType);
          for (vidType pad_id = thread_lane; pad_id < pad_size; pad_id += WARP_SIZE) {
            bitmap_.set_pad(i, chunk_id * PAD_PER_CHUNK + pad_id, chunk_buffer[warp_lane][pad_id]);
          }
        }
        __syncwarp();

      }
    }
#else
    for (vidType i = 0; i < size_; i++) {
      auto search = g.N(vlist_[i]);
      vidType search_size = g.getOutDegree(vlist_[i]);
      for (int j = thread_lane; j < size_; j += WARP_SIZE) {
        int cnt = 0;
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vlist_[j], search_size);
        __syncwarp(active);
        cnt += bitmap_.warp_set(i, j, flag);
      }
    }
#endif
    return count;
  }  

  __device__ vidType build_roaring_LUT_block(vidType* sub_list, vidType list_size, GraphGPU g) {
    vidType warp_lane = threadIdx.x / WARP_SIZE;
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    vlist_ = sub_list;
    size_ = list_size;
    vidType count = 0;
    for (vidType i = warp_lane; i < size_; i += WARPS_PER_BLOCK) {
      auto search = g.N(vlist_[i]);
      vidType search_size = g.getOutDegree(vlist_[i]);
      for (int j = thread_lane; j < size_; j += WARP_SIZE) {
        int cnt = 0;
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vlist_[j], search_size);
        __syncwarp(active);
        cnt += bitmap_.warp_set(i, j, flag);
      }
    }
    return count;
  }

  __device__ vidType build_roaring_LUT_global(vidType* sub_list, vidType list_size, GraphGPU g) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int num_warps = WARPS_PER_BLOCK * gridDim.x;
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    vlist_ = sub_list;
    size_ = list_size;
    vidType count = 0;
    for (vidType i = warp_id; i < size_; i += num_warps) {
      auto search = g.N(vlist_[i]);
      vidType search_size = g.getOutDegree(vlist_[i]);
      for (int j = thread_lane; j < size_; j += WARP_SIZE) {
        int cnt = 0;
        unsigned active = __activemask();
        bool flag = (j!=i) && binary_search(search, vlist_[j], search_size);
        __syncwarp(active);
        cnt += bitmap_.warp_set(i, j, flag);
      }
    }
    return count;
  }

  /*******************************************Warp parallel set operation******************************************************/
  __device__ vidType difference_set(vidType upper, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 0 index
    lutrow_result = row(row_id);
    opt_result = false;
    return 0;
  }


  __device__ vidType difference_set(vidType upper, vidType row_id, vidType* index_list, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 0 index
    lutrow_result = row(row_id);
    opt_result = false;
    return lutrow_result.to_non_nbr_index(index_list, upper);
  }

  __device__ vidType intersect_set(vidType upper, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 1 index
    lutrow_result = row(row_id);
    opt_result = true;
    return 0;
  }

  __device__ vidType intersect_set(vidType upper, vidType row_id, vidType* index_list, Roaring_Bitmap1D<T,W>& lutrow_result, bool& opt_result) {
    // get 1 indexd
    lutrow_result = row(row_id);
    opt_result = true;
    return lutrow_result.to_nbr_index(index_list, upper);
  }

  __device__ vidType intersect_set(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b, vidType* c) {
    return intersect_set_bs_cache(a, source, size_a, b, size_b, c);
  }


  // In: lut - vlist BS
  // Out: lut index, lutrow
  __device__ vidType difference_set(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b, vidType* c, Roaring_Bitmap1D<T,W>& lutrow_result) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_lane = threadIdx.x / WARP_SIZE;
    __shared__ vidType list_size[WARPS_PER_BLOCK];
    vidType cnt = difference_set_bs_cache(a, source, size_a, b, size_b, c);
    if (thread_lane == 0) list_size[warp_lane] = cnt;
    __syncwarp();
    lutrow_result.build_from_index(c, list_size[warp_lane], lutrow_result);
    return list_size[warp_lane];
  }

  // In: lut - vlist BS
  // Out: lut index
  __device__ vidType difference_set(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b, vidType upper, vidType* c) {
    return difference_set_bs_cache(a, source, size_a, b, size_b, upper, c);
  }

  // In: lut - vlist BS
  // Out: lut index
  __device__ vidType difference_set(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b, vidType* c) {
    return difference_set_bs_cache(a, source, size_a, b, size_b, c);
  }

  // In: lut - vlist BS
  // Out: num
  __device__ vidType difference_num(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b, vidType upper) {
    return difference_num_bs_cache(a, source, size_a, b, size_b, upper);
  }

  // In: lut - vlist BS
  // Out: num
  __device__ vidType difference_num(vidType* a, vidType* source, vidType size_a, vidType* b, vidType size_b) {
    return difference_num_bs_cache(a, source, size_a, b, size_b);
  }

  __device__ vidType intersect_set(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id,
                                    vidType* lutrow_index, Roaring_Bitmap1D<T, W>& lutrow_result) {
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and
      lutrow_input.AND(thread_lane, v2, upper, lutrow_result);
    }
    else{
      //not and
      lutrow_input.NOT_AND(thread_lane, v2, upper, lutrow_result);
    }
    return lutrow_result.to_nbr_index(lutrow_index, upper);
  }

  // opt true: lut inut:1, use a_and_nb
  // opt false: lut input 0, use not_or
  // In: lutrow-lutrow_head
  // Out: lut index, lutrow
  __device__ vidType difference_set(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id,
                                    vidType* lutrow_index, Roaring_Bitmap1D<T, W>& lutrow_result) {
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and not
      lutrow_input.AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    else{
      //not or
      lutrow_input.NOT_AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    return lutrow_result.to_nbr_index(lutrow_index, upper);
  }

  // opt true: lut inut:1, use a_and_nb
  // opt false: lut input 0, use not_or
  // In: lutrow-lutrow_head
  // Out: lutrow
  __device__ vidType difference_set(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result) {
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and not
      lutrow_input.AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    else{
      //not or
      lutrow_input.NOT_AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    return 0;
  }


  __device__ vidType difference_set_endlut(bool opt, vidType upper, vidType * source, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id,
                                           vidType* vlist_result, Roaring_Bitmap1D<T, W>& lutrow_result) {
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and not
      lutrow_input.AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    else{
      //not or
      lutrow_input.NOT_AND_NOT(thread_lane, v2, upper, lutrow_result);
    }
    return lutrow_result.to_nbr_list(source, vlist_result, upper);
  }

  __device__ vidType difference_num(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id) {
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if (opt){
      return lutrow_input.count_AND_NOT(thread_lane, v2, upper);
    }
    else {
      return lutrow_input.count_NOT_AND_NOT(thread_lane, v2, upper);
    }
    return 0;
  }

  // include judge list

  __device__ vidType difference_num(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, Roaring_Bitmap1D<T, W> judge_list, vidType row_id) {;
    vidType thread_lane = threadIdx.x & (WARP_SIZE - 1);
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(judge_list.bit(row_id)) {
      if(opt) {
        return lutrow_input.count_AND_NOT(thread_lane, v2, upper);
      }
      else {
        return lutrow_input.count_NOT_AND_NOT(thread_lane, v2, upper);
      }
    }
    return 0;
  }

  // /*******************************************Thread parallel set operation******************************************************/

  __device__ vidType difference_set_thread(vidType upper, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 0 index
    lutrow_result = row(row_id);
    opt_result = false;
    return 0;
  }

  __device__ vidType difference_set_thread(vidType upper, vidType row_id, vidType* index_list, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 0 index
    lutrow_result = row(row_id);
    opt_result = false;
    return lutrow_result.to_non_nbr_index_thread(index_list, upper);
  }

  __device__ vidType intersect_set_thread(vidType upper, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 1 index
    lutrow_result = row(row_id);
    opt_result = true;
    return 0;
  }

  __device__ vidType intersect_set_thread(vidType upper, vidType row_id, vidType* index_list, Roaring_Bitmap1D<T, W>& lutrow_result, bool& opt_result) {
    // get 1 index
    lutrow_result = row(row_id);
    opt_result = true;
    return lutrow_result.to_nbr_index_thread(index_list, upper);
  }

  // opt true: lut inut:1, use a_and_nb
  // opt false: lut input 0, use not_or
  // In: lutrow-lutrow_head
  // Out: lut index, lutrow
  __device__ vidType difference_set_thread(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id, 
                                          vidType* lutrow_index, Roaring_Bitmap1D<T, W>& lutrow_result) {
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and not
      lutrow_input.AND_NOT_thread(v2, upper, lutrow_result);
    }
    else{
      //not or
      lutrow_input.NOT_AND_NOT_thread(v2, upper, lutrow_result);
    }
    return lutrow_result.to_nbr_index_thread(lutrow_index, upper);
  }

  // opt true: lut inut:1, use a_and_nb
  // opt false: lut input 0, use not_or
  // In: lutrow-lutrow_head
  // Out: lutrow
  __device__ vidType difference_set_thread(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id, Roaring_Bitmap1D<T, W>& lutrow_result) {
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(opt) {
      //and not
      lutrow_input.AND_NOT_thread(v2, upper, lutrow_result);
    }
    else{
      //not or
      lutrow_input.NOT_AND_NOT_thread(v2, upper, lutrow_result);
    }
    return 0;
  }

  __device__ vidType intersect_num_thread(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id) {
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if (opt){
      return lutrow_input.count_AND_thread(v2, upper);
    }
    else {
      return lutrow_input.count_NOT_AND_thread(v2, upper);
    }
    return 0;
  }

  __device__ vidType difference_num_thread(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id) {
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if (opt){
      return lutrow_input.count_AND_NOT_thread(v2, upper);
    }
    else {
      return lutrow_input.count_NOT_AND_NOT_thread(v2, upper);
    }
    return 0;
  }

  __device__ vidType difference_num_thread_lower(bool opt, vidType lower, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, vidType row_id) {
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if (opt){
      return lutrow_input.count_AND_NOT_thread_lower(v2, lower, upper);
    }
    else {
      return lutrow_input.count_NOT_AND_NOT_thread_lower(v2, lower, upper);
    }
    return 0;
  }

  // include judge list

  __device__ vidType difference_num_thread(bool opt, vidType upper, Roaring_Bitmap1D<T, W> lutrow_input, Roaring_Bitmap1D<T, W> judge_list, vidType row_id) {;
    Roaring_Bitmap1D<T, W> v2 = row(row_id);
    if(judge_list.bit(row_id)) {
      if(opt) {
        return lutrow_input.count_AND_NOT_thread(v2, upper);
      }
      else {
        return lutrow_input.count_NOT_AND_NOT_thread(v2, upper);
      }
    }
    return 0;
  }
};

template <typename T = bitmapType, int W = BITMAP_WIDTH>
class Roaring_LUTManager{
private:
  T* heap_head_;
  typeType* types_head_;
  uint32_t LUT_num_;
  uint32_t nrow_;
  uint32_t ncol_;
  uint32_t padded_rowsize_;
  int64_t LUT_size_;
  uint32_t chunk_rowsize_;
  uint32_t types_pad_rowsize_;
  uint32_t types_pad_allsize_;

public:
  Roaring_LUTManager(){}
  Roaring_LUTManager(uint32_t LUT_num, uint32_t nrow, uint32_t ncol):
    LUT_num_(LUT_num), nrow_(nrow), ncol_(ncol) {
      padded_rowsize_ = (ncol_ + W - 1) / W;
      LUT_size_ = padded_rowsize_ * nrow_;
      chunk_rowsize_ = (ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
      types_pad_rowsize_ = (chunk_rowsize_ + TYPE_WIDTH - 1) / TYPE_WIDTH;
      types_pad_allsize_ = types_pad_rowsize_ * ncol;
      size_t totalMemSize = LUT_size_ * LUT_num_ * sizeof(T);
      CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
      CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
      std::cout << LUT_num << " LUT of " << nrow_ << " rows, " << ncol_ << " cols,"<<
        " total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
#ifdef ROARING
      size_t totalMemSize_types = types_pad_allsize_ * LUT_num_ * sizeof(typeType);
      CUDA_SAFE_CALL(cudaMalloc(&types_head_, totalMemSize_types));
      CUDA_SAFE_CALL(cudaMemset(types_head_, 0, totalMemSize_types));
      std::cout << LUT_num << " typelist of " << nrow_ << " rows, " << chunk_rowsize_ << " chunks,"<<
      " total memory size: " << float(totalMemSize_types)/(1024*1024) << " MB\n";
#endif
    }

  void init(uint32_t LUT_num, uint32_t nrow, uint32_t ncol) {
    LUT_num_ = LUT_num;
    nrow_ = nrow;
    ncol_ = ncol;
    padded_rowsize_ = (ncol_ + W - 1) / W;
    LUT_size_ = padded_rowsize_ * nrow_;
    chunk_rowsize_ = (ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    types_pad_rowsize_ = (chunk_rowsize_ + TYPE_WIDTH - 1) / TYPE_WIDTH;
    types_pad_allsize_ = types_pad_rowsize_ * ncol;
    size_t totalMemSize = LUT_size_ * LUT_num_ * sizeof(T);
    CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
    CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
    std::cout << LUT_num << " LUT of " << nrow_ << " rows, " << ncol_ << " cols,"<<
      " total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
#ifdef ROARING
    size_t totalMemSize_types = types_pad_allsize_ * LUT_num_ * sizeof(typeType);
    CUDA_SAFE_CALL(cudaMalloc(&types_head_, totalMemSize_types));
    CUDA_SAFE_CALL(cudaMemset(types_head_, 0, totalMemSize_types));
    std::cout << LUT_num << " typelist of " << nrow_ << " rows, " << chunk_rowsize_ << " chunks,"<<
    " total memory size: " << float(totalMemSize_types)/(1024*1024) << " MB\n";
#endif
  }    

  void recreate(uint32_t LUT_num, uint32_t nrow, uint32_t ncol){
    LUT_num_ = LUT_num;
    nrow_ = nrow;
    ncol_ = ncol;
    CUDA_SAFE_CALL(cudaFree(heap_head_));
#ifdef ROARING
    CUDA_SAFE_CALL(cudaFree(types_head_));
#endif
    padded_rowsize_ = (ncol_ + W - 1) / W;
    LUT_size_ = padded_rowsize_ * nrow_;
    chunk_rowsize_ = (ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    types_pad_rowsize_ = (chunk_rowsize_ + TYPE_WIDTH - 1) / TYPE_WIDTH;
    types_pad_allsize_ = types_pad_rowsize_ * ncol;
    size_t totalMemSize = LUT_size_ * LUT_num_ * sizeof(T);
    CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
    CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
    std::cout << LUT_num << " LUT of " << nrow_ << " rows, " << ncol_ << " cols,"<<
      " total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
#ifdef ROARING
    size_t totalMemSize_types = types_pad_allsize_ * LUT_num_ * sizeof(typeType);
    CUDA_SAFE_CALL(cudaMalloc(&types_head_, totalMemSize_types));
    CUDA_SAFE_CALL(cudaMemset(types_head_, 0, totalMemSize_types));
    std::cout << LUT_num << " typelist of " << nrow_ << " rows, " << chunk_rowsize_ << " chunks,"<<
    " total memory size: " << float(totalMemSize_types)/(1024*1024) << " MB\n";
#endif
  }

  void update_para(uint32_t LUT_num, uint32_t nrow, uint32_t ncol){
    LUT_num_ = LUT_num;
    nrow_ = nrow;
    ncol_ = ncol;
    padded_rowsize_ = (ncol_ + W - 1) / W;
    LUT_size_ = padded_rowsize_ * nrow_;
    chunk_rowsize_ = (ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    types_pad_rowsize_ = (chunk_rowsize_ + TYPE_WIDTH - 1) / TYPE_WIDTH;
    types_pad_allsize_ = types_pad_rowsize_ * ncol;
  }

  __device__ Roaring_LUT<T,W> getEmptyLUT(int lut_id) {
    Roaring_LUT<T,W> lut;
    assert(lut_id < LUT_num_);
    lut.init(heap_head_ + lut_id * LUT_size_, types_head_ + lut_id * types_pad_allsize_, ncol_);
    return lut;
  }
};
