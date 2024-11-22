#pragma once

#include "cutil_subset.h"
#include "common.h"
#include "chunk.cuh"

template<typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct Roaring_Bitmap1D {
  T* head_;
  typeType* types_; // chunk type list
  uint32_t size_; // bit num in a bitmap1D
  uint32_t chunk_size_; // chunk num in a Bitmap1D
  uint32_t pad_size_;

  __device__ Roaring_Bitmap1D() {}
  __device__ Roaring_Bitmap1D(T* head, uint32_t size) {
    head_ = head;
    size_ = size;
    chunk_size_ = (size_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
  }

  __device__ void init(T* head, uint32_t size) {
    head_ = head;
    size_ = size;
    chunk_size_ = (size_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    pad_size_ = (size_ + W - 1) / W;
  }

  __device__ void init(T* head, typeType* types, uint32_t size) {
    head_ = head;
    types_ = types;
    size_ = size;
    chunk_size_ = (size_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    pad_size_ = (size_ + W - 1) / W;
  }

  // intput pad id, get pad_head
  __device__ T load(int i) {
    return reinterpret_cast<T>(head_[i]);
  }

  // input chunk id, output chunk head
  __device__ T* chunk_head(int i) {
    return head_ + i * PAD_PER_CHUNK;
  }

  __device__ arrayType* array_chunk_head(int i) {
    return (arrayType*)&head_[i * PAD_PER_CHUNK];
  }

  // input chunk id, output chunk
  __device__ BitmapChunk<T, W> load_bitmapchunk(int i) {
    BitmapChunk<T, W> ret;
    ret.init(chunk_head(i), CHUNK_WIDTH);
    return ret;
  }

  __device__ ArrayChunk<arrayType, W> load_arraychunk(int i) {
    ArrayChunk<arrayType, W> ret;
    ret.init(array_chunk_head(i), CHUNK_WIDTH);
    return ret;
  }

  __device__ void set(int i, T word) {
    head_[i] = word;
  }

  __device__ bool bit(int i) {
#ifdef ROARING
    int element = i / CHUNK_WIDTH;
    int bit_get = i % CHUNK_WIDTH;
    return load_bitmapchunk(element).bit(bit_get);   
#else
    int idx = i / W;
    int bit_loc = i % W;
    uint32_t mask = (1u << bit_loc);
    return head_[idx] & mask;
#endif
  }

  __device__ void set1(int i) {
    int element = i / CHUNK_WIDTH;
    int bit_set = i % CHUNK_WIDTH;
    load_bitmapchunk(element).set1(bit_set);
    return;
  }

  __device__ void atomic_set1(int i) {
    int element = i / CHUNK_WIDTH;
    int bit_set = i % CHUNK_WIDTH;
    load_bitmapchunk(element).atomic_set1(bit_set);
    return;  
  }

  __device__ bool get_chunk_type(int chunk_id) {
    uint32_t element = chunk_id / TYPE_WIDTH;
    uint32_t bit_get = chunk_id % TYPE_WIDTH;
    typeType mask = 1 << bit_get;
    return types_[element] & mask;
  }

  __device__ void set_chunk_type(int chunk_id, bool value) {
    uint32_t element = chunk_id / TYPE_WIDTH;
    uint32_t bit_set = chunk_id % TYPE_WIDTH;
    typeType mask = 1 << bit_set;
    if (value) {
      head_[element] |= mask;
    }
    else {
      head_[element] &= ~mask;
    }    
  }

  __device__ void warp_clear(uint32_t thread_lane) {
#ifdef ROARING   
    for (auto chunk_id = 0; chunk_id < chunk_size_; chunk_id++) {
      load_bitmapchunk(chunk_id).warp_clear(thread_lane);
    }
#else
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      head_[i] = 0;
    }
#endif
  }

  __device__ void thread_clear() {
#ifdef ROARING 
    for (auto chunk_id = 0; chunk_id < chunk_size_; chunk_id++) {
      load_bitmapchunk(chunk_id).thread_clear();
    }
#else
    for (auto i = 0; i < pad_size_; i++) {
      head_[i] = 0;
    }
#endif  
  }

  __device__ void AND(uint32_t thread_lane, Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).AND(thread_lane, ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }
#else
    for (auto i = thread_lane; i < countElement; i += WARP_SIZE) {
      output.set(i, load(i) & ope.load(i));
    }
    if (thread_lane == 0 && remain) {
      output.set(countElement, load(countElement) & ope.load(countElement) & ((1U << remain) - 1));
    }
    __syncwarp();
#endif      
  }  

  __device__ void AND_NOT(uint32_t thread_lane, Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).AND_NOT(thread_lane, ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }
#else
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      output.set(i, (load(i)) & ~ope.load(i));
    }
    __syncwarp();
#endif
  }

  __device__ void NOT_AND(uint32_t thread_lane, Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).NOT_AND(thread_lane, ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    } 
#else
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      output.set(i, (~load(i)) & ope.load(i));
    }
    __syncwarp();
#endif
  }

  __device__ void NOT_AND_NOT(uint32_t thread_lane, Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).NOT_AND_NOT(thread_lane, ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }  
#else
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      output.set(i, (~load(i)) & (~ope.load(i)));
    }
    __syncwarp();
#endif
  }

  __device__ vidType count_NOT_AND_NOT(uint32_t thread_lane, Roaring_Bitmap1D<T, W> ope, int limit) {
    vidType num = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      if (get_chunk_type(chunk_id)) {
        if (ope.get_chunk_type(chunk_id)) {
          // array @ array
          num += load_arraychunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_arraychunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
        else {
          // array @ bitmap
          num += load_arraychunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
      }
      else {
        if (ope.get_chunk_type(chunk_id)) {
          // bitmap @ array
          num += load_bitmapchunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_arraychunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
        else {
          // bitmap @ bitmap
          num += load_bitmapchunk(chunk_id).count_NOT_AND_NOT(thread_lane, ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
      }
      limit -= CHUNK_WIDTH;
    }
#else
    for (auto i = thread_lane; i < pad_size_; i += WARP_SIZE) {
      num += __popc((~load(i)) & (~ope.load(i)));
    }
#endif
    return num;
  }

  __device__ void build_from_index(vidType* index_list, vidType list_size, Roaring_Bitmap1D<T, W> output) {
#ifdef ROARING
#else
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    output.warp_clear(thread_lane);
    __syncwarp();
    for (int idx = thread_lane; idx < list_size; idx += WARP_SIZE) {
      auto set_id = index_list[idx];
      output.atomic_set1(set_id);
    }
#endif  
  }

  __device__ vidType _to_index(bool connected, vidType *result, int limit) {
#ifdef ROARING
    vidType res = 0;
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      if (get_chunk_type(chunk_id)) {
        // Array chunk
        res += load_arraychunk(chunk_id)._to_index(connected, result + res, chunk_id, min(CHUNK_WIDTH, limit));
      }
      else{
        // Bitmap chunk
        res += load_bitmapchunk(chunk_id)._to_index(connected, result + res, chunk_id, min(CHUNK_WIDTH, limit));
      }
      limit -= CHUNK_WIDTH;
    }
    return res;
#else
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
      if (found) result[result_size[warp_lane] + idx1 - 1] = keyId;
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
      if (found) result[result_size[warp_lane] + idx1 - 1] = keyId;
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    return result_size[warp_lane];  
  #endif
  }

  __device__ vidType to_nbr_index(vidType* result, int limit) {
    return _to_index(true, result, limit);
  }

  __device__ vidType to_non_nbr_index(vidType* result, int limit) {
    return _to_index(false, result, limit);
  }

  __device__ vidType _to_index_thread(bool connected, vidType *result, int limit) {
    vidType res = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      res += load_bitmapchunk(chunk_id)._to_index_thread(connected, result + res, chunk_id, min(CHUNK_WIDTH, limit));
      limit -= CHUNK_WIDTH;
    }
#else
    for(int key = 0; key < limit; key++) {
      if(connected == bit(key)) {
        result[res] = key;
        res++;
      }
    }
#endif
    return res;  
  }
  
  __device__ vidType to_nbr_index_thread(vidType* result, int limit) {
    return _to_index_thread(true, result, limit);
  }

  __device__ vidType to_non_nbr_index_thread(vidType* result, int limit) {
    return _to_index_thread(false, result, limit);
  }

  __device__ vidType _to_vlist(bool connected, vidType *source, vidType *result, int limit) { 
#ifdef ROARING
  vidType res = 0;
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      res += load_bitmapchunk(chunk_id)._to_vlist(connected, source, result + res, chunk_id, min(CHUNK_WIDTH, limit));
      limit -= CHUNK_WIDTH;
    }
  return res;  
#else
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
      if (found) result[result_size[warp_lane] + idx1 - 1] = source[keyId];
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
      if (found) result[result_size[warp_lane] + idx1 - 1] = source[keyId];
      if (thread_lane == 0) result_size[warp_lane] += __popc(mask);
    }
    return result_size[warp_lane];    
#endif
  }

  __device__ vidType to_nbr_list(vidType* source, vidType* result, int limit) {
    return _to_vlist(true, source, result, limit);
  }

  __device__ vidType to_non_nbr_list(vidType* source, vidType* result, int limit) {
    return _to_vlist(false, source, result, limit);
  }

  __device__ void AND_thread(Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).AND_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }
#else   
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, load(i) & ope.load(i));
    }
    if (remain) {
      auto element = load(countElement) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
#endif
  } 

  __device__ void AND_NOT_thread(Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    } 
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, load(i) & (~ope.load(i)));
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
#endif
  }

  __device__ void NOT_AND_thread(Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).NOT_AND_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }   
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, (~load(i)) & ope.load(i));
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      output.set(countElement, element & ((1U << remain) - 1));
    }
#endif
  }

  __device__ void NOT_AND_NOT_thread(Roaring_Bitmap1D<T,W> ope, int limit, Roaring_Bitmap1D<T,W> output) {
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      load_bitmapchunk(chunk_id).NOT_AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit), output.load_bitmapchunk(chunk_id));
      limit -= CHUNK_WIDTH;
    }    
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      output.set(i, (~load(i)) & (~ope.load(i)));
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      output.set(countElement, element & ((1U << remain) - 1));
    }
#endif
  }

  __device__ vidType count_AND_thread(Roaring_Bitmap1D<> ope, int limit) {
    vidType num = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      num += load_bitmapchunk(chunk_id).count_AND_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
      limit -= CHUNK_WIDTH;
    }
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      auto element = load(i) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
#endif
    return num;  
  }

  __device__ vidType count_AND_NOT_thread(Roaring_Bitmap1D<> ope, int limit) {
    vidType num = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      num += load_bitmapchunk(chunk_id).count_AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
      limit -= CHUNK_WIDTH;
    }
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      auto element = load(i) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
#endif
    return num;  
  }

  __device__ vidType count_NOT_AND_thread(Roaring_Bitmap1D<> ope, int limit) {
    vidType num = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      num += load_bitmapchunk(chunk_id).count_NOT_AND_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
      limit -= CHUNK_WIDTH;
    }
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i ++) {
      auto element = (~load(i)) & ope.load(i);
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      num += __popc(element & ((1U << remain) - 1));
    }
#endif
    return num;  
  }  

  __device__ vidType count_NOT_AND_NOT_thread(Roaring_Bitmap1D<> ope, int limit) {
    vidType num = 0;
#ifdef ROARING
    vidType chunk_limit = (limit + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    for (int chunk_id = 0; chunk_id < chunk_limit; chunk_id++) {
      if (get_chunk_type(chunk_id)) {
        if (ope.get_chunk_type(chunk_id)) {
          // array @ array
          num += load_arraychunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_arraychunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
        else {
          // array @ bitmap
          num += load_arraychunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
      }
      else {
        if (ope.get_chunk_type(chunk_id)) {
          // bitmap @ array
          num += load_bitmapchunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_arraychunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
        else {
          // bitmap @ bitmap
          num += load_bitmapchunk(chunk_id).count_NOT_AND_NOT_thread(ope.load_bitmapchunk(chunk_id), min(CHUNK_WIDTH, limit));
        }
      }
      limit -= CHUNK_WIDTH;
    }
#else
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    for (auto i = 0; i < countElement; i++) {
      auto element = (~load(i)) & (~ope.load(i));
      num += __popc(element);
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      num += __popc(element & ((1U << remain) - 1));
    }
#endif
    return num;
  }

  __device__ vidType count_AND_NOT_thread_lower(Roaring_Bitmap1D<> ope, int lower, int upper) {
    vidType num = 0;
    int start_index = (lower + 1) / BITMAP_WIDTH;
    int start_offset = (lower + 1) % BITMAP_WIDTH;
    int countElement = upper / BITMAP_WIDTH;
    int remain = upper % BITMAP_WIDTH;
    int countCheck = size_ / BITMAP_WIDTH;
    for (int i = lower + 1; i < upper; i++) {
      if (bit(i) && !ope.bit(i)) num++;
    }
    return num;
  }

  __device__ vidType count_NOT_AND_NOT_thread_lower(Roaring_Bitmap1D<> ope, int lower, int upper) {
    vidType num = 0;
    int start_index = (lower + 1) / BITMAP_WIDTH;
    int start_offset = (lower + 1) % BITMAP_WIDTH;
    int countElement = upper / BITMAP_WIDTH;
    int remain = upper % BITMAP_WIDTH;
    vidType check_num = 0;
    // for (int i = lower + 1; i < upper; i++) {
    //   if (!bit(i) && !ope.bit(i)) check_num++;
    // }
    if (lower >= upper) return 0;
    for (int i = start_index; i <= countElement; ++i) {
      uint32_t mask = 0xFFFFFFFF;

      if (i == start_index) {
          mask &= 0xFFFFFFFF << start_offset;
      }
      if (i == countElement) {
          mask &= ((1U << remain) - 1);
      }
      num += __popc((~load(i)) & (~ope.load(i)) & mask);
    }
    return num;
  }

};

template<typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct Roaring_Bitmap2D {
  T* head_;
  typeType* types_;
  uint32_t nrow_;
  uint32_t ncol_;
  uint32_t size_;
  uint32_t pad_rowsize_; // number of T in a row
  uint32_t pad_allsize_; // number of T in a Bitmap2D
  uint32_t chunk_rowsize_;
  uint32_t types_pad_rowsize_; // number of typeType in a row
  uint32_t types_pad_allsize_;

  __device__ Roaring_Bitmap2D(){}
  __device__ Roaring_Bitmap2D(T* head, uint32_t size) {
    head_ = head;
    nrow_ = size;
    ncol_ = size;
    pad_rowsize_ = (ncol_ + W - 1) / W;
    pad_allsize_ = pad_rowsize_ * nrow_;
  }

  __device__ void init(T* head, uint32_t size) {
    head_ = head;
    nrow_ = size;
    ncol_ = size;
    pad_rowsize_ = (ncol_ + W - 1) / W;
    pad_allsize_ = pad_rowsize_ * nrow_;
  }

  __device__ void init(T* head, typeType* types, uint32_t size) {
    head_ = head;
    types_ = types;
    nrow_ = size;
    ncol_ = size;
    pad_rowsize_ = (ncol_ + W - 1) / W;
    pad_allsize_ = pad_rowsize_ * nrow_;
    chunk_rowsize_ = (ncol_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    types_pad_rowsize_ = (chunk_rowsize_ + TYPE_WIDTH - 1) / TYPE_WIDTH;
    types_pad_allsize_ = types_pad_rowsize_ * nrow_;
  }

  __device__ size_t get_bytes_num(size_t size) {
    return sizeof(T) * ((size + W - 1) / W) * size;
  }

  __device__ size_t get_valid_bits_num() { return size_t(nrow_) * ncol_; }

  __device__ T* row_head(int x) {
    return head_ + x * pad_rowsize_;
  }

#ifdef ROARING
  __device__ arrayType* chunk_head(int x, int chunk_id) {
    return (arrayType*)&head_[x * pad_rowsize_ + chunk_id * PAD_PER_CHUNK];
  }

  // get each roaring bitmap1D's type list head
  __device__ typeType* types_row_head(int x) {
    return types_ + x * types_pad_rowsize_;    
  }

  __device__ void set_chunk_type(int x, int y, bool value) {
    typeType* v = types_row_head(x);
    uint32_t element = y / TYPE_WIDTH;
    uint32_t bit_set = y % TYPE_WIDTH;
    typeType mask = 1 << bit_set;
    if (value) {
      v[element] |= mask;
    }
    else {
      v[element] &= ~mask;
    }
  }

  __device__ void warp_clear_chunk_type(uint32_t thread_lane) {
    for (int i = thread_lane; i < types_pad_allsize_; i += WARP_SIZE) {
      types_[i] = 0;
    }
    __syncwarp();
  }
#endif

  // __device__ Roaring_Bitmap1D

  __device__ void warp_clear(uint32_t thread_lane) {
    for (int i = thread_lane; i < pad_allsize_; i += WARP_SIZE) {
      head_[i] = 0;
    }
    __syncwarp();
    return;
  }

  __device__ void set1(int x, int y) {
    uint32_t element = y / W;
    uint32_t bit_set = y % W;
    T mask = 1 << bit_set;
    head_[x * pad_rowsize_ + element] |= mask;
    return;
  }

  __device__ bool get(int x, int y) {
    uint32_t element = y / W;
    uint32_t bit_set = y % W;
    T mask = 1 << bit_set;
    return head_[x * pad_rowsize_ + element] & mask;
  }

  __device__ T warp_load(uint32_t x, uint32_t warp_id) {
    return head_[x * pad_rowsize_ + warp_id];
  }

  __device__ vidType warp_set(uint32_t x, uint32_t y, bool flag) {
    uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1);
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t active = __activemask();
    uint32_t mask = __ballot_sync(active, flag);
    if (thread_lane == 0) head_[x * pad_rowsize_ + element] = mask;
    return __popc(mask);
  }

  __device__ void set_pad(int x, int y, T word) {
    head_[x * pad_rowsize_ + y] = word;
  }

};
