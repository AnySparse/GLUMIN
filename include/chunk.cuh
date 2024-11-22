#pragma once

#include "cutil_subset.h"
#include "common.h"

template <typename T, int W>
__device__ struct ArrayChunk;


template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct BitmapChunk {
  T* head_;
  uint32_t size_;
  uint32_t pad_size_;

  __device__ BitmapChunk(){}
  __device__ void init(T* head, uint32_t size) {
    head_ = head;
    size_ = size;
    pad_size_ = (size_ + W - 1) / W; 
  }

  __device__ T load(int i) {
    return reinterpret_cast<T>(head_[i]);
  }

  __device__ void set(int i, T word) {
    head_[i] = reinterpret_cast<T>(word);
    return;
  }

  __device__ bool bit(int i) {
    int element = i / W;
    int bit_get = i % W;
    T mask = 1 << bit_get;
    return head_[element] & mask;
  }

  __device__ void warp_clear(uint32_t thread_lane) {
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      head_[i] = 0;
    }  
  }

  __device__ void thread_clear() {
    for (auto i = 0; i < pad_size_; i++) {
      head_[i] = 0;
    }
  }

  __device__ void set1(int i) {
    int element = i / W;
    int bit_get = i % W;
    T mask = 1 << bit_get;
    head_[element] |= mask;
    return;
  }

  __device__ void atomic_set1(int i) {
    int element = i / W;
    int bit_get = i % W;
    T mask = 1 << bit_get;
    atomicOr(&head_[element], mask);
    return;
  }

  __device__ void AND(uint32_t thread_lane, BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      output.set(i, load(i) & ope.load(i));
    }
    if (thread_lane == 0 && remain) {
      auto element = load(countElement) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
    __syncwarp();
  }  

  __device__ void AND_NOT(uint32_t thread_lane, BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      output.set(i, load(i) & (~ope.load(i)));
    }
    if (thread_lane == 0 && remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
    __syncwarp();
  }

   __device__ void NOT_AND(uint32_t thread_lane, BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      output.set(i, (~load(i)) & ope.load(i));
    }
    if (thread_lane == 0 && remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
    __syncwarp();
  }

   __device__ void NOT_AND_NOT(uint32_t thread_lane, BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      output.set(i, (~load(i)) & (~ope.load(i)));
    }
    if (thread_lane == 0 && remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
    __syncwarp();
  }

  __device__ vidType count_NOT_AND_NOT(uint32_t thread_lane, BitmapChunk<T,W> ope, int limit) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    vidType num = 0;
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      num += __popc((~load(i)) & (~ope.load(i)));
    }
    if (thread_lane == 0 && remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
    // __syncwarp();
    return num;
  }

  __device__ void build_from_index(vidType* index_list, vidType list_size, BitmapChunk<T, W> output) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    output.warp_clear(thread_lane);
    __syncwarp();
    for (int idx = thread_lane; idx < list_size; idx += WARP_SIZE) {
      auto setId = index_list[idx];
      output.atomic_set1(setId);      
    }
    __syncwarp();
  }

  __device__ vidType _to_index(bool connected, vidType *result, vidType chunk_lane, int limit) {
    int warp_lane = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    __shared__ vidType result_size[WARPS_PER_BLOCK];
    if (thread_lane == 0) result_size[warp_lane] = 0;
    __syncwarp();
    vidType size_group = limit / WARP_SIZE;
    for (int key = 0; key < size_group; key++) {
      bool found = 0;
      vidType keyId = (key * WARP_SIZE) + thread_lane;
      if (connected == bit(keyId)) found = 1;
      unsigned mask = __ballot_sync(0xffffffff, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[result_size[warp_lane] + idx1 - 1] = (chunk_lane * CHUNK_WIDTH) + keyId;
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    __syncwarp();
    if (thread_lane < limit % WARP_SIZE) {
      bool found = 0;
      vidType keyId = size_group * WARP_SIZE + thread_lane;
      if (connected == bit(keyId)) found = 1;
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[result_size[warp_lane] + idx1 - 1] = (chunk_lane * CHUNK_WIDTH) + keyId;
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    __syncwarp();
    return result_size[warp_lane];    
  }

  __device__ vidType _to_index_thread(bool connected, vidType *result, vidType chunk_lane, int limit) {
    vidType result_size = 0;
    for(int key = 0; key < limit; key++) {
      if(connected == bit(key)) {
        result[result_size] = (chunk_lane * CHUNK_WIDTH) + key;
        result_size++;
      }
    }
    return result_size;
  }

  __device__ vidType _to_vlist(bool connected, vidType *source, vidType *result, vidType chunk_lane, int limit) {
    int warp_lane = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    __shared__ vidType result_size[WARPS_PER_BLOCK];
    if (thread_lane == 0) result_size[warp_lane] = 0;
    __syncwarp();
    vidType size_group = limit / WARP_SIZE;
    for (int key = 0; key < size_group; key++) {
      bool found = 0;
      vidType keyId = (key * WARP_SIZE) + thread_lane;
      if (connected == bit(keyId)) found = 1;
      unsigned mask = __ballot_sync(0xffffffff, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[result_size[warp_lane] + idx1 - 1] = source[(chunk_lane * CHUNK_WIDTH) + keyId];
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    __syncwarp();
    if (thread_lane < limit % WARP_SIZE) {
      bool found = 0;
      vidType keyId = size_group * WARP_SIZE + thread_lane;
      if (connected == bit(keyId)) found = 1;
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, found);
      auto idx1 = __popc(mask << (WARP_SIZE - thread_lane - 1));
      if (found) result[result_size[warp_lane] + idx1 - 1] = source[(chunk_lane * CHUNK_WIDTH) + keyId];
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    return result_size[warp_lane];    
  }

  __device__ void AND_thread(BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, load(i) & ope.load(i));
    }
    if (remain) {
      auto element = load(countElement) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
  } 

  __device__ void AND_NOT_thread(BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, load(i) & (~ope.load(i)));
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
  }

  __device__ void NOT_AND_thread(BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, (~load(i)) & ope.load(i));
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
  }

  __device__ void NOT_AND_NOT_thread(BitmapChunk<T,W> ope, int limit, BitmapChunk<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, (~load(i)) & (~ope.load(i)));
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
  }

  __device__ vidType count_AND_thread(BitmapChunk<> ope, int limit) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i++) {
      auto element = load(i) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_AND_NOT_thread(BitmapChunk<> ope, int limit) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i++) {
      auto element = load(i) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_NOT_AND_thread(BitmapChunk<> ope, int limit) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i ++) {
      auto element = (~load(i)) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }  

  __device__ vidType count_NOT_AND_NOT_thread(BitmapChunk<> ope, int limit) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i++) {
      auto element = (~load(i)) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
    return num;
  }  

  __device__ vidType count_NOT_AND_NOT_thread(ArrayChunk<arrayType, W> ope, int limit) {
    bitmapType res[PAD_PER_CHUNK];
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i++) {
      res[i] = (~load(i));
    }
    if (remain) {
      res[countElement] = (~load(countElement));
    }
    for (auto i = 0; i < ope.get_word(0); i++) {
      vidType set_id = ope.get_word(i + 1);
      if (set_id >= limit) break;
      bitmapType mask = 1 << (set_id % BITMAP_WIDTH);
      res[set_id / BITMAP_WIDTH] &= ~mask;
    }
    for (auto i = 0; i < countElement; i++) {
      num += __popc(res[i]);
    }
    if (remain) {
      num += __popc(res[countElement] & ((1U << remain) - 1));
    }
    return num;
  }  
};

template <typename T = arrayType, int W = BITMAP_WIDTH>
__device__ struct ArrayChunk {
  T* head_;
  uint32_t size_;

  __device__ ArrayChunk(){}
  __device__ void init(T* head, uint32_t size) {
    head_ = head;
    size_ = size;
  }

  __device__ vidType get_word(int i) {
    return (vidType)head_[i];
  }

  __device__ vidType _to_index(bool connected, vidType *result, vidType chunk_lane, int limit) {
    // only one thread working
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    __shared__ bool present[WARPS_PER_BLOCK][CHUNK_WIDTH];
    for (vidType i = thread_lane; i < CHUNK_WIDTH; i += WARP_SIZE) {
      present[warp_lane][i] = false;
    }
    __syncwarp();
    // bool present[CHUNK_WIDTH] = {false};
    vidType result_size = 0;
    if (thread_lane == 0) {
      for(vidType idx = 0; idx < head_[0]; idx++) {
        vidType keyId = head_[idx + 1];
        // printf("keyId %d\n", keyId);
        if (keyId >= limit) break;
        present[warp_lane][keyId] = true;
      }
      for(vidType i = 0; i < limit; i++) {
        if(connected == present[warp_lane][i]) {
          result[result_size] = chunk_lane * CHUNK_WIDTH + i;
          result_size++;
        }
      }
    }
    __syncwarp();
    result_size = SHFL(result_size, 0);
    // result_size = limit - head_[0];
    return result_size;
  }

  __device__ vidType count_NOT_AND_NOT_thread(BitmapChunk<bitmapType, W> ope, int limit) {
    bitmapType res[PAD_PER_CHUNK];
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i++) {
      res[i] = (~ope.load(i));
    }
    if (remain) {
      res[countElement] = (~ope.load(countElement));
    }
    for (auto i = 0; i < get_word(0); i++) {
      vidType set_id = get_word(i + 1);
      if (set_id >= limit) break;
      bitmapType mask = 1 << (set_id % BITMAP_WIDTH);
      res[set_id / BITMAP_WIDTH] &= ~mask;
    }
    for (auto i = 0; i < countElement; i++) {
      num += __popc(res[i]);
    }
    if (remain) {
      num += __popc(res[countElement] & ((1U << remain) - 1));
    }
    return num;
  }

  __device__ vidType count_NOT_AND_NOT_thread(ArrayChunk<T, W> ope, int limit) {
    bitmapType res[PAD_PER_CHUNK] = {0};
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < get_word(0); i++) {
      vidType set_id = get_word(i + 1);
      if (set_id >= limit) break;
      bitmapType mask = 1 << (set_id % BITMAP_WIDTH);
      res[set_id / BITMAP_WIDTH] |= mask;
    }
    for (auto i = 0; i < ope.get_word(0); i++) {
      vidType set_id = ope.get_word(i + 1);
      if (set_id >= limit) break;
      bitmapType mask = 1 << (set_id % BITMAP_WIDTH);
      res[set_id / BITMAP_WIDTH] |= mask;
    }
    for (auto i = 0; i < countElement; i++) {
      num += __popc(~res[i]);
    }
    if (remain) {
      num += __popc(~res[countElement] & ((1U << remain) - 1));
    }
    return num;
  }
};

// template<typenme T = bitmapType, int W = BITMAP_WIDTH>
// __device__ struct ChunkManager {}