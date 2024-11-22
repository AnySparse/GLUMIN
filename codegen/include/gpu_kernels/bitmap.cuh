#pragma once

#include "cutil_subset.h"
#include "common.h"

template<typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct Bitmap1DView {
  T* ptr_ = NULL;
  uint32_t size_;

  __device__ Bitmap1DView(){}
  __device__ Bitmap1DView(T* ptr, uint32_t size): ptr_(ptr), size_(size) {}

  __device__ uint32_t get_numElement(uint32_t size){
    return (size + W-1)/W; // how many BitmapType
  }

  __device__ void init(T* ptr, uint32_t size) {
    ptr_ = ptr;
    size_ = size;
  }

  __device__ uint32_t size() {
    return size_;
  }

  __device__  T load(int i) {
    return ptr_[i];
  }

  __device__ void set(int i, T word) {
    ptr_[i] = word;
  }

  __device__ bool bit(int i) {
    int idx = i / W;
    int bit_loc = i % W;
    uint32_t mask = (1u << bit_loc);
    return ptr_[idx] & mask;
  }

  __device__ void warp_clear(uint32_t lane) {
    uint32_t padded_size_ = (size_ + W - 1) / W;
    for (auto i = lane; i < padded_size_; i += WARP_SIZE) {
      ptr_[i] = 0;
    }
    __syncwarp();
  }

  __device__ void AND(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    auto thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      output.set(i, load(i) & ope.load(i));
    }
    if (remain) {
      output.set(countElement, load(countElement) & ope.load(countElement));
    }
    output.size_ = limit;
    if(thread_lane==0) *result_size = limit;
    __syncwarp();
  }

  __device__ void NOT_AND(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    auto thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      output.set(i, (~load(i)) & ope.load(i));
    }
    if (remain) {
      output.set(countElement, (~load(countElement)) & ope.load(countElement));
    }
    output.size_ = limit;
    if(thread_lane==0) *result_size = limit;
    __syncwarp();
  }

  __device__ void AND_NOT(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    auto thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      output.set(i, load(i) & (~ope.load(i)));
    }
    if (remain) {
      output.set(countElement, load(countElement) & (~ope.load(countElement)));
    }
    output.size_ = limit;
    if(thread_lane==0) *result_size = limit;
    __syncwarp();
  }

  __device__ void NOT_AND_NOT(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    auto thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit : valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      output.set(i, (~load(i)) & (~ope.load(i)));
    }
    if (remain) {
      output.set(countElement, (~load(countElement)) & (~ope.load(countElement)));
    }
    output.size_ = limit;
    if(thread_lane==0) *result_size = limit;
    __syncwarp();
  }

  // convert bitmap to raw vertex list.
  __device__ vidType _to_vlist(bool connected, vidType* raw_vlist, vidType * result, vidType* result_size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int limit = size_;
    int remain = limit % WARP_SIZE;
    if (thread_lane == 0) *result_size = 0;
    __syncwarp();
    for (int key = thread_lane; key < limit - remain; key += WARP_SIZE) {
      bool found = (connected == bit(key) ? 1 : 0);
      unsigned mask = __ballot_sync(FULL_MASK, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[*result_size + idx1 - 1] = raw_vlist[key];
      if (thread_lane == 0) *result_size += __popc(mask);
    }
    // N.B. This is not clear why CUDA compiler can not generate correct active_mask
    if (thread_lane < remain) {
      int key = thread_lane + limit - remain;
      bool found = (connected == bit(key) ? 1 : 0);
      unsigned activemask = __activemask();
      unsigned mask = __ballot_sync(activemask, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[*result_size + idx1 - 1] = raw_vlist[key];
      if (thread_lane == 0) *result_size += __popc(mask);
    }
    __syncwarp();
    return *result_size;
  }

  // convert bitmap to index list.
  __device__ vidType _to_index(bool connected, vidType* result, vidType* result_size) {
    int limit = size_;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int remain = limit % WARP_SIZE;
    if (thread_lane == 0) *result_size = 0;
    __syncwarp();
    for (int key = thread_lane; key < limit - remain; key += WARP_SIZE) {
      bool found = (connected == bit(key) ? 1 : 0);
      unsigned mask = __ballot_sync(FULL_MASK, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[*result_size + idx1 - 1] = key;
      if (thread_lane == 0) *result_size += __popc(mask);
    }
    // N.B. This is not clear why CUDA compiler can not generate correct active_mask
    if(thread_lane < remain) {
      int key = thread_lane + limit - remain;
      bool found = (connected == bit(key) ? 1 : 0);
      unsigned activemask = __activemask();
      unsigned mask = __ballot_sync(activemask, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[*result_size + idx1 - 1] = key;
      if (thread_lane == 0) *result_size += __popc(mask);
    }
    __syncwarp();
    return *result_size;
  }

  // for debug
  __device__ vidType _to_index_thread(bool connected, vidType *result, vidType* result_size) {
    *result_size = 0;
    for(int key = 0; key < size_; key++) {
      if(connected == bit(key)) {
        result[*result_size] = key;
        (*result_size)++;
      }
    }
    return *result_size;
  }

  __device__ vidType count_warp(bool connected, int limit) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    limit = limit < 0 ? size_ : (limit < size_ ? limit : size_);
    int remain = limit & (WARP_SIZE-1);
    int num = 0;
    for (int key = thread_lane; key < limit - remain; key += WARP_SIZE) {
      unsigned activemask = __activemask();
      bool found = (connected == bit(key) ? 1 : 0);
      //unsigned activemask = 0xffffffff;
      unsigned mask = __ballot_sync(activemask, found);
      if (thread_lane == 0) num += __popc(mask);
    }
    if (thread_lane < remain) {
      unsigned activemask = __activemask();
      int key = (limit - remain) + thread_lane;
      bool found = (connected == bit(key) ? 1 : 0);
      unsigned mask = __ballot_sync(activemask, found);
      if (thread_lane == 0) num += __popc(mask);
    }
    __syncwarp();
    return num;
  }

  __device__ vidType count_thread(bool connected, int limit) {
    limit = limit < 0 ? size_ : (limit < size_ ? limit : size_);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i ++) {
      auto element = load(i);
      if(!connected) element = ~element;
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement);
      if(!connected) element = ~element;
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_AND_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = load(i) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_AND_NOT_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = load(i) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_NOT_AND_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = (~load(i)) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_NOT_AND_NOT_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = (~load(i)) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }
};

template<typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct Bitmap2DView{
  T* ptr_;
  uint32_t nrow_; // how many bits in a row.
  uint32_t ncol_; // how many bits in a col.
  uint32_t padded_rowsize_; // number of T in a row.
  uint32_t capacity_; // how many T used to store data.

  //__device__ size_t get_bytes_num(size_t size) {
  //  return sizeof(T) * ((size+W-1) / W) * size;
  //}

  __device__ void init(T* ptr, uint32_t size) {
    ptr_ = ptr;
    nrow_ = size;
    ncol_ = size;
    padded_rowsize_ = (ncol_+W-1) / W;
    capacity_ = padded_rowsize_ * nrow_;
  }

  __device__ size_t get_valid_bits_num() { return size_t(nrow_) * ncol_; }

  __device__ T* row(int x) {
    return ptr_ + x * padded_rowsize_;
  }

  // set all bits in bitmap to 0.
  __device__ void clear() {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    for (auto i = thread_lane; i < capacity_; i += WARP_SIZE) {
      ptr_[i] = 0;
    }
    __syncwarp();
  }

  __device__ void build(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    for (vidType i = 0; i < size; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }

  __device__ T warp_load(uint32_t x, uint32_t warp_id) {
    return ptr_[x * padded_rowsize_ + warp_id];
  }

  // set target bits in bitmap[x][y/W].
  // not concurrent-safe, single (warp) writer.
  __device__ void warp_set(uint32_t x, uint32_t y, bool flag) {
    uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1);
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t activemask = __activemask();
    uint32_t mask = __ballot_sync(activemask, flag);
    if (thread_lane == 0) ptr_[x * padded_rowsize_ + element] = mask;
  }

  // set bitmap[x][y] to 1.
  // not concurrent-safe, single writer.
  __device__ void set1(int x, int y) {
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t bit_sft = y % W;
    T mask = 1 << bit_sft;
    ptr_[x * padded_rowsize_ + element] |= mask;
  }

  __device__ bool get(int x, int y) {
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t bit_sft = y % W;
    T mask = 1 << bit_sft;
    return ptr_[x * padded_rowsize_ + element] & mask;
  }
};
