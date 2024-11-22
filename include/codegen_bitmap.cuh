#pragma once

#include "cutil_subset.h"
#include "common.h"

template<typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ struct Bitmap1DView {
  T* ptr_ = NULL;
  uint32_t size_;
#ifdef ROARING
  typeType* types_ptr_;
#endif
#ifdef LOAD_SRAM
  T* sram_ptr_ = NULL;
#endif

  __device__ Bitmap1DView(){}
  __device__ Bitmap1DView(T* ptr, uint32_t size): ptr_(ptr), size_(size) {}
#ifdef ROARING
  __device__ Bitmap1DView(T* ptr, typeType* types_ptr, uint32_t size): ptr_(ptr),types_ptr_(types_ptr), size_(size) {}
#endif
#ifdef LOAD_SRAM
  __device__ Bitmap1DView(T* ptr, T* sram_ptr, uint32_t size): ptr_(ptr),sram_ptr_(sram_ptr), size_(size) {}
#endif

  __device__ uint32_t get_numElement(uint32_t size){
    return (size + W-1)/W; // how many BitmapType
  }

  __device__ void init(T* ptr, uint32_t size) {
    ptr_ = ptr;
    size_ = size;
  }

#ifdef ROARING
  __device__ void init(T* ptr, typeType* types_ptr, uint32_t size) {
    ptr_ = ptr;
    types_ptr_ = types_ptr;
    size_ = size;
  }
#endif
#ifdef LOAD_SRAM 
  __device__ void init(T* ptr, T* sram_ptr, uint32_t size) {
    ptr_ = ptr;
    sram_ptr_ = sram_ptr;
    size_ = size;
}
#endif
  __device__ uint32_t size() {
    return size_;
  }

#ifdef ROARING
  __device__  T load(int i) {
    int chunk_id = i / PAD_PER_CHUNK;
    typeType chunk_type = types_ptr_[chunk_id];
    int chunk_lane = i % PAD_PER_CHUNK;
    int pad_start = chunk_lane * BITMAP_WIDTH;
    int pad_end = min((chunk_lane + 1) * BITMAP_WIDTH, CHUNK_WIDTH);
    bitmapType ret = 0;
    if (chunk_type == 0) return 0;
    else if (chunk_type == (CHUNK_WIDTH - 1)) return FULL_MASK;
    else if (chunk_type == ARRAY_LIMIT) return ptr_[i];
    // Array type
    else {
      arrayType* checker = get_chunk_head(chunk_id);
      for (int iter = 0; iter < chunk_type; iter++) {
        int set_id = checker[iter];
        if (set_id >= pad_start && set_id < pad_end) {
          ret |= (1 << (set_id % BITMAP_WIDTH));
        }
      }
      return ret;
    };
  }

  __device__ bool bit(int i) {
    int chunk_id = i / CHUNK_WIDTH;
    typeType chunk_type = types_ptr_[chunk_id];
    int chunk_lane = i % CHUNK_WIDTH;
    if (chunk_type == 0) return 0;
    else if (chunk_type == (CHUNK_WIDTH - 1)) return 1;
    else if (chunk_type == ARRAY_LIMIT) {
      int idx = i / W;
      int bit_loc = i % W;
      uint32_t mask = (1u << bit_loc);
      return ptr_[idx] & mask; 
    }
    // Array chunk, check chunk_lane is in array id?
    else {
      arrayType* checker = get_chunk_head(chunk_id);
      for (int iter = 0; iter < chunk_type; iter++) {
        if (checker[iter] == chunk_lane) return 1;
      }
      return 0;
    } 
    return 0;
  }

  __device__ arrayType* get_chunk_head(int chunk_id) {
    return (arrayType*)&ptr_[chunk_id * PAD_PER_CHUNK];
  }

#else
  __device__  T load(int i) {
    return ptr_[i];
  }

  __device__ bool bit(int i) {
    int idx = i / W;
    int bit_loc = i % W;
    uint32_t mask = (1u << bit_loc);
    return ptr_[idx] & mask;
  }
#endif

  __device__ void set(int i, T word) {
    ptr_[i] = word;
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

  __device__ vidType count_AND(Bitmap1DView<>& ope, int limit) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = thread_lane; i < countElement ; i += WARP_SIZE) {
      auto element = load(i) & ope.load(i);
      num += __popc(element);
    }
    if (thread_lane == 0 && remain) {
      auto element = load(countElement) & ope.load(countElement);
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

  __device__ vidType count_AND_NOT(Bitmap1DView<>& ope, int limit) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = thread_lane; i < countElement ; i += WARP_SIZE) {
      auto element = load(i) & (~ope.load(i));
      num += __popc(element);
    }
    if (thread_lane == 0 && remain) {
      auto element = load(countElement) & (~ope.load(countElement));
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

  __device__ vidType count_NOT_AND(Bitmap1DView<>& ope, int limit) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = thread_lane; i < countElement ; i += WARP_SIZE) {
      auto element = (~load(i)) & ope.load(i);
      num += __popc(element);
    }
    if (thread_lane == 0 && remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
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

  __device__ vidType count_NOT_AND_NOT(Bitmap1DView<>& ope, int limit) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = thread_lane; i < countElement ; i += WARP_SIZE) {
      auto element = (~load(i)) & (~ope.load(i));
      num += __popc(element);
    }
    if (thread_lane == 0 && remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
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
#ifdef ROARING
  typeType* types_;
  uint32_t chunk_padded_rowsize_;
  uint32_t types_padded_rowsize_;
  uint32_t types_capacity_;
#endif
#ifdef LOAD_SRAM
  T* sram_ptr_ = NULL;
#endif

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

#ifdef ROARING
  __device__ typeType* row_types(int x) {
    return types_ + x * types_padded_rowsize_;
  }
#endif

#ifdef LOAD_SRAM
  __device__ T* row(int x) {
    vidType SRAM_row_num = (WARP_LIMIT / 32) * LOAD_RATE / padded_rowsize_;
    if (x < SRAM_row_num) return sram_ptr_ + x * padded_rowsize_;
    else return ptr_ + x * padded_rowsize_;
  }
#else
  __device__ T* row(int x) {
    return ptr_ + x * padded_rowsize_;
  }
#endif

  __device__ void set_pad(int x, int y, T word) {
    ptr_[x * padded_rowsize_ + y] = word;
  }

  // set all bits in bitmap to 0.
  __device__ void clear() {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    for (auto i = thread_lane; i < capacity_; i += WARP_SIZE) {
      ptr_[i] = 0;
    }
    __syncwarp();
  }

#ifdef ROARING
  __device__ void init(T* ptr, typeType* types, uint32_t size) {
    ptr_ = ptr;
    types_ = types;
    nrow_ = size;
    ncol_ = size;
    padded_rowsize_ = (ncol_+W-1) / W;
    capacity_ = padded_rowsize_ * nrow_;
    chunk_padded_rowsize_ = (ncol_ + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    types_padded_rowsize_ = chunk_padded_rowsize_;
    types_capacity_ = types_padded_rowsize_ * nrow_;
  }

  __device__ void warp_clear_chunk_type(uint32_t thread_lane) {
    for (int i = thread_lane; i < types_capacity_; i += WARP_SIZE) {
      types_[i] = 0;
    }
    __syncwarp();
  }
  __device__ arrayType* get_chunk_head(int x, int chunk_id) {
    return (arrayType*)&ptr_[x * padded_rowsize_ + chunk_id * PAD_PER_CHUNK];
  }

  __device__ void set_chunk_type(int x, int y, typeType value) {
    types_[x * types_padded_rowsize_ + y] = value;
  }

  __device__ typeType get_chunk_type(int x, int y) {
    return types_[x * types_padded_rowsize_ + y];
  }

  __device__ void build(GraphGPU& g, vidType* vlist, vidType size) {
    int warp_lane = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    __shared__ vidType cnt_shared[WARPS_PER_BLOCK];
    __shared__ T chunk_buffer[WARPS_PER_BLOCK][PAD_PER_CHUNK];
    warp_clear_chunk_type(thread_lane);
    for (vidType i = 0; i < size; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int chunk_id = 0; chunk_id < chunk_padded_rowsize_; chunk_id++) {
        uint32_t pad_size, array_limit;
        if (chunk_id == chunk_padded_rowsize_ - 1) {
          pad_size = ((size - chunk_id * CHUNK_WIDTH) + W - 1) / W;
          array_limit = min(ARRAY_LIMIT,(size - (chunk_id * CHUNK_WIDTH)) / ARRAY_ID_WIDTH);
        }
        else {
          pad_size = PAD_PER_CHUNK;
          array_limit = ARRAY_LIMIT;
        }
        int cnt = 0;
        for (vidType idx = thread_lane; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx += WARP_SIZE) {
          int j = chunk_id * CHUNK_WIDTH + idx;
          unsigned active = __activemask();
          bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
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

        // all bit 0
        if (cnt_shared[warp_lane] == 0) {
          set_chunk_type(i, chunk_id, 0);
        }
        // all bit 11
        else if (cnt_shared[warp_lane] == CHUNK_WIDTH){
          set_chunk_type(i, chunk_id, CHUNK_WIDTH - 1);
        }
        // array chunk (Set limit 0, Not used Now)
        else if (cnt_shared[warp_lane] < array_limit) {
          arrayType* chunk_head = get_chunk_head(i, chunk_id);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, cnt_shared[warp_lane]);
          }
          __syncwarp();
          vidType set_idx = 0;
          if (thread_lane == 0){
            for (vidType idx = 0; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx++) {
              arrayType chunk_lane = (arrayType)idx;
              if (chunk_buffer[warp_lane][idx / W] & (1 << (idx % W))) {
                chunk_head[set_idx] = chunk_lane;
                set_idx++;
              }
            }
          }
        }
        // bitmap chunk
        else {
          // count += pad_size * sizeof(bitmapType);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, ARRAY_LIMIT);
          }
          for (vidType pad_id = thread_lane; pad_id < pad_size; pad_id += WARP_SIZE) {
            set_pad(i, chunk_id * PAD_PER_CHUNK + pad_id, chunk_buffer[warp_lane][pad_id]);
          }
        }
      }
    }
  }

  __device__ void build_block(GraphGPU& g, vidType* vlist, vidType size) {
    int warp_lane = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    __shared__ vidType cnt_shared[WARPS_PER_BLOCK];
    __shared__ T chunk_buffer[WARPS_PER_BLOCK][PAD_PER_CHUNK];
    warp_clear_chunk_type(thread_lane);
    for (vidType i = warp_lane; i < size; i += WARPS_PER_BLOCK) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int chunk_id = 0; chunk_id < chunk_padded_rowsize_; chunk_id++) {
        uint32_t pad_size, array_limit;
        if (chunk_id == chunk_padded_rowsize_ - 1) {
          pad_size = ((size - chunk_id * CHUNK_WIDTH) + W - 1) / W;
          array_limit = min(ARRAY_LIMIT,(size - (chunk_id * CHUNK_WIDTH)) / ARRAY_ID_WIDTH);
        }
        else {
          pad_size = PAD_PER_CHUNK;
          array_limit = ARRAY_LIMIT;
        }
        int cnt = 0;
        for (vidType idx = thread_lane; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx += WARP_SIZE) {
          int j = chunk_id * CHUNK_WIDTH + idx;
          unsigned active = __activemask();
          bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
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

        // all bit 0
        if (cnt_shared[warp_lane] == 0) {
          set_chunk_type(i, chunk_id, 0);
        }
        // all bit 11
        else if (cnt_shared[warp_lane] == CHUNK_WIDTH){
          set_chunk_type(i, chunk_id, CHUNK_WIDTH - 1);
        }
        // array chunk (Set limit 0, Not used Now)
        else if (cnt_shared[warp_lane] < array_limit) {
          arrayType* chunk_head = get_chunk_head(i, chunk_id);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, cnt_shared[warp_lane]);
          }
          __syncwarp();
          vidType set_idx = 0;
          if (thread_lane == 0){
            for (vidType idx = 0; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx++) {
              arrayType chunk_lane = (arrayType)idx;
              if (chunk_buffer[warp_lane][idx / W] & (1 << (idx % W))) {
                chunk_head[set_idx] = chunk_lane;
                set_idx++;
              }
            }
          }
        }
        // bitmap chunk
        else {
          // count += pad_size * sizeof(bitmapType);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, ARRAY_LIMIT);
          }
          for (vidType pad_id = thread_lane; pad_id < pad_size; pad_id += WARP_SIZE) {
            set_pad(i, chunk_id * PAD_PER_CHUNK + pad_id, chunk_buffer[warp_lane][pad_id]);
          }
        }
      }
    }
  }

  __device__ void build_global(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;    
    int warp_id = thread_id / WARP_SIZE;
    int num_warps = WARPS_PER_BLOCK * gridDim.x;
    int warp_lane = threadIdx.x / WARP_SIZE;
    __shared__ vidType cnt_shared[WARPS_PER_BLOCK];
    __shared__ T chunk_buffer[WARPS_PER_BLOCK][PAD_PER_CHUNK];
    warp_clear_chunk_type(thread_lane);
    for (vidType i = warp_id; i < size; i += num_warps) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int chunk_id = 0; chunk_id < chunk_padded_rowsize_; chunk_id++) {
        uint32_t pad_size, array_limit;
        if (chunk_id == chunk_padded_rowsize_ - 1) {
          pad_size = ((size - chunk_id * CHUNK_WIDTH) + W - 1) / W;
          array_limit = min(ARRAY_LIMIT,(size - (chunk_id * CHUNK_WIDTH)) / ARRAY_ID_WIDTH);
        }
        else {
          pad_size = PAD_PER_CHUNK;
          array_limit = ARRAY_LIMIT;
        }
        int cnt = 0;
        for (vidType idx = thread_lane; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx += WARP_SIZE) {
          int j = chunk_id * CHUNK_WIDTH + idx;
          unsigned active = __activemask();
          bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
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

        // all bit 0
        if (cnt_shared[warp_lane] == 0) {
          set_chunk_type(i, chunk_id, 0);
        }
        // all bit 11
        else if (cnt_shared[warp_lane] == CHUNK_WIDTH){
          set_chunk_type(i, chunk_id, CHUNK_WIDTH - 1);
        }
        // array chunk (Set limit 0, Not used Now)
        else if (cnt_shared[warp_lane] < array_limit) {
          arrayType* chunk_head = get_chunk_head(i, chunk_id);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, cnt_shared[warp_lane]);
          }
          __syncwarp();
          vidType set_idx = 0;
          if (thread_lane == 0){
            for (vidType idx = 0; idx < CHUNK_WIDTH && (chunk_id * CHUNK_WIDTH + idx) < size; idx++) {
              arrayType chunk_lane = (arrayType)idx;
              if (chunk_buffer[warp_lane][idx / W] & (1 << (idx % W))) {
                chunk_head[set_idx] = chunk_lane;
                set_idx++;
              }
            }
          }
        }
        // bitmap chunk
        else {
          // count += pad_size * sizeof(bitmapType);
          if (thread_lane == 0) {
            set_chunk_type(i, chunk_id, ARRAY_LIMIT);
          }
          for (vidType pad_id = thread_lane; pad_id < pad_size; pad_id += WARP_SIZE) {
            set_pad(i, chunk_id * PAD_PER_CHUNK + pad_id, chunk_buffer[warp_lane][pad_id]);
          }
        }
      }
    }
  }

#elif defined(LOAD_SRAM)
  __device__ void init(T* ptr, T* sram_ptr, uint32_t size) {
    ptr_ = ptr;
    sram_ptr_ = sram_ptr;
    nrow_ = size;
    ncol_ = size;
    padded_rowsize_ = (ncol_+W-1) / W;
    capacity_ = padded_rowsize_ * nrow_;
  }

  __device__ void build(GraphGPU& g, vidType* vlist, vidType size) {
    if (size == 0) return;
    int warp_lane = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    vidType countElement = (size + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
    // how many rows can load in SRAM
    // SRAM space elements / 1 lut row elements
    vidType SRAM_row_num = (WARP_LIMIT / 32) * LOAD_RATE / countElement;
    // if (SRAM_row_num == size) printf("size %d\n", size);
    if (SRAM_row_num >= size) SRAM_row_num = size;
    for (vidType i = 0; i < SRAM_row_num; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set_SRAM(i, j, flag);
      }
    }
    for (vidType i = SRAM_row_num; i < size; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }

  __device__ void build_block(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_lane = threadIdx.x / WARP_SIZE;
    vidType countElement = (size + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
    vidType SRAM_row_num = (BLOCK_LIMIT / 32) * LOAD_RATE / countElement;
    if (SRAM_row_num >= size) SRAM_row_num = size;
    for (vidType i = 0; i < SRAM_row_num; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set_SRAM(i, j, flag);
      }
    }
    for (vidType i = warp_lane; i < size; i += WARPS_PER_BLOCK) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }

  __device__ void build_global(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;    
    int warp_id = thread_id / WARP_SIZE;
    int num_warps = WARPS_PER_BLOCK * gridDim.x;
    for (vidType i = warp_id; i < size; i += num_warps) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }  
#else
  __device__ void build(GraphGPU& g, vidType* vlist, vidType size) {
    int warp_lane = threadIdx.x / WARP_SIZE;
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

  __device__ void build_block(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_lane = threadIdx.x / WARP_SIZE;
    for (vidType i = warp_lane; i < size; i += WARPS_PER_BLOCK) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }

  __device__ void build_global(GraphGPU& g, vidType* vlist, vidType size) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;    
    int warp_id = thread_id / WARP_SIZE;
    int num_warps = WARPS_PER_BLOCK * gridDim.x;
    for (vidType i = warp_id; i < size; i += num_warps) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.getOutDegree(vlist[i]);
      for (int j = thread_lane; j < size; j += WARP_SIZE) {
        bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
        warp_set(i, j, flag);
      }
    }
  }  
#endif

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

#ifdef LOAD_SRAM
  __device__ void warp_set_SRAM(uint32_t x, uint32_t y, bool flag) {
    uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1);
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t activemask = __activemask();
    uint32_t mask = __ballot_sync(activemask, flag);
    if (thread_lane == 0) sram_ptr_[x * padded_rowsize_ + element] = mask;
  }
#endif

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
