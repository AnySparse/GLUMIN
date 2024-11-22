#pragma once

// #include "cutil_subset.h"
#include "common.h"

template <typename T = vidType>
bool binary_search(T *list, T key, T size) {
  int l = 0;
  int r = size-1;
  while (r >= l) { 
    int mid = l + (r - l) / 2; 
    T value = list[mid];
    if (value == key) return true;
    if (value < key) l = mid + 1;
    else r = mid - 1;
  }
  return false;
}

#define THREAD_SIZE 1
template<typename T = bitmapType, int W = BITMAP_WIDTH>
 struct Bitmap1DView {
  T* ptr_ = NULL;
  uint32_t size_;

   Bitmap1DView(){}
   Bitmap1DView(T* ptr, uint32_t size): ptr_(ptr), size_(size) {}

   uint32_t get_numElement(uint32_t size){
    return (size + W-1)/W; // how many BitmapType
  }

   void init(T* ptr, uint32_t size) {
    ptr_ = ptr;
    size_ = size;
  }

   uint32_t size() {
    return size_;
  }

    T load(int i) {
    return ptr_[i];
  }

   void set(int i, T word) {
    ptr_[i] = word;
  }

   bool bit(int i) {
    int idx = i / W;
    int bit_loc = i % W;
    uint32_t mask = (1u << bit_loc);
    return ptr_[idx] & mask;
  }

   void warp_clear(uint32_t lane) {
    uint32_t padded_size_ = (size_ + W - 1) / W;
    for (auto i = lane; i < padded_size_; i += THREAD_SIZE) {
      ptr_[i] = 0;
    }
    
  }

   void AND(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    
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
    *result_size = limit;
    
  }

   void NOT_AND(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    
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
    *result_size = limit;
    
  }

   void AND_NOT(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    
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
    *result_size = limit;
    
  }

   void NOT_AND_NOT(Bitmap1DView<T,W>& ope, Bitmap1DView<T,W>& output, vidType* result_size, int limit) {
    
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
    *result_size = limit;
    
  }

  // convert bitmap to raw vertex list.
   vidType _to_vlist(bool connected, vidType* raw_vlist, vidType * result, vidType* result_size) {
    int thread_lane = 0;
    int limit = size_;
    int remain = limit % THREAD_SIZE;
    if (thread_lane == 0) *result_size = 0;
    

    for (int key = thread_lane; key < size_; key += THREAD_SIZE) {
      bool found = (connected == bit(key) ? 1 : 0);
      if (found) {
        result[*result_size] = raw_vlist[key];
        *result_size += 1;
      }
    }

    // for (int key = thread_lane; key < limit - remain; key += THREAD_SIZE) {
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   if (found) {
    //     result[*result_size] = raw_vlist[key];
    //     *result_size += 1;
    //   }
    //   // unsigned mask = __ballot_sync(FULL_MASK, found);
    //   // auto idx1 = __popc(mask << (THREAD_SIZE - thread_lane - 1));
    //   // if (found) result[*result_size + idx1 - 1] = raw_vlist[key];
    //   // if (thread_lane == 0) *result_size += __popc(mask);
    // }
    // // N.B. This is not clear why CUDA compiler can not generate correct active_mask
    // if (thread_lane < remain) {
    //   int key = thread_lane + limit - remain;
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   unsigned activemask = __activemask();
    //   unsigned mask = __ballot_sync(activemask, found);
    //   auto idx1 = __popc(mask << (THREAD_SIZE - thread_lane - 1));
    //   if (found) result[*result_size + idx1 - 1] = raw_vlist[key];
    //   if (thread_lane == 0) *result_size += __popc(mask);
    // }
    
    return *result_size;
  }

  // convert bitmap to index list.
   vidType _to_index(bool connected, vidType* result, vidType* result_size) {
    int limit = size_;
    int thread_lane = 0;
    int remain = limit % THREAD_SIZE;
    if (thread_lane == 0) *result_size = 0;
    
    // for (int key = thread_lane; key < limit - remain; key += THREAD_SIZE) {
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   unsigned mask = __ballot_sync(FULL_MASK, found);
    //   auto idx1 = __popc(mask << (THREAD_SIZE - thread_lane - 1));
    //   if (found) result[*result_size + idx1 - 1] = key;
    //   if (thread_lane == 0) *result_size += __popc(mask);
    // }
    // // N.B. This is not clear why CUDA compiler can not generate correct active_mask
    // if(thread_lane < remain) {
    //   int key = thread_lane + limit - remain;
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   unsigned activemask = __activemask();
    //   unsigned mask = __ballot_sync(activemask, found);
    //   auto idx1 = __popc(mask << (THREAD_SIZE - thread_lane - 1));
    //   if (found) result[*result_size + idx1 - 1] = key;
    //   if (thread_lane == 0) *result_size += __popc(mask);
    // }

    for (int key = thread_lane; key < size_; key += THREAD_SIZE) {
      bool found = (connected == bit(key) ? 1 : 0);
      if (found) {
        result[*result_size] = key;
        *result_size += 1;
      }
    }
    
    return *result_size;
  }

  // for debug
   vidType _to_index_thread(bool connected, vidType *result, vidType* result_size) {
    *result_size = 0;
    for(int key = 0; key < size_; key++) {
      if(connected == bit(key)) {
        result[*result_size] = key;
        (*result_size)++;
      }
    }
    return *result_size;
  }

   vidType count_warp(bool connected, int limit) {
    int thread_lane = 0;
    limit = limit < 0 ? size_ : (limit < size_ ? limit : size_);
    int remain = limit & (THREAD_SIZE-1);
    int num = 0;
    // for (int key = thread_lane; key < limit - remain; key += THREAD_SIZE) {
    //   unsigned activemask = __activemask();
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   //unsigned activemask = 0xffffffff;
    //   unsigned mask = __ballot_sync(activemask, found);
    //   if (thread_lane == 0) num += __popc(mask);
    // }
    // if (thread_lane < remain) {
    //   unsigned activemask = __activemask();
    //   int key = (limit - remain) + thread_lane;
    //   bool found = (connected == bit(key) ? 1 : 0);
    //   unsigned mask = __ballot_sync(activemask, found);
    //   if (thread_lane == 0) num += __popc(mask);
    // }

     for (int key = thread_lane; key < size_; key += THREAD_SIZE) {
      bool found = (connected == bit(key) ? 1 : 0);
      if (found) {
        num += 1;
      }
    }
    
    
    return num;
  }

   vidType count_thread(bool connected, int limit) {
    limit = limit < 0 ? size_ : (limit < size_ ? limit : size_);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement; i ++) {
      auto element = load(i);
      if(!connected) element = ~element;
      //num += __popc(element);
      num += __builtin_popcount((uint32_t)element);
    }
    if (remain) {
      auto element = load(countElement);
      if(!connected) element = ~element;
      //num += __popc(element & ((1U << remain) - 1));
      num += __builtin_popcount((uint32_t)(element & ((1U << remain) - 1)));
    }
    return num;
  }

   vidType count_AND_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = load(i) & ope.load(i);
      // num += __popc(element);
      num += __builtin_popcount((uint32_t)element);
    }
    if (remain) {
      //printf("step2   \n");
    //   auto printBits = [](uint32_t num) {
    //     for (int bit = 31; bit >= 0; --bit) {
    //         std::cout << ((num >> bit) & 1);
    //     }
    //     std::cout << std::endl;
    // };
    //   printBits(load(countElement));
    //   printBits(ope.load(countElement));

      auto element = load(countElement) & ope.load(countElement);
      // num += __popc(element & ((1U << remain) - 1));
      num += __builtin_popcount((uint32_t)(element & ((1U << remain) - 1)));
    }
    return num;
  }

   vidType count_AND_NOT_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = load(i) & (~ope.load(i));
      // num += __popc(element);
      num += __builtin_popcount((uint32_t)element);
    }
    if (remain) {
      auto element = load(countElement) & (~ope.load(countElement));
      //num += __popc(element & ((1U << remain) - 1));
      num += __builtin_popcount((uint32_t)(element & ((1U << remain) - 1)));
    }
    return num;
  }

   vidType count_NOT_AND_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = (~load(i)) & ope.load(i);
      //num += __popc(element);
      num += __builtin_popcount((uint32_t)element);
    }
    if (remain) {
      auto element = (~load(countElement)) & ope.load(countElement);
      // num += __popc(element & ((1U << remain) - 1));
      num += __builtin_popcount((uint32_t)(element & ((1U << remain) - 1)));
    }
    return num;
  }

   vidType count_NOT_AND_NOT_thread(Bitmap1DView<>& ope, int limit) {
    int valid_size = (size_ < ope.size() ? size_ : ope.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
      auto element = (~load(i)) & (~ope.load(i));
      //num += __popc(element);
      num += __builtin_popcount((uint32_t)element);
    }
    if (remain) {
      auto element = (~load(countElement)) & (~ope.load(countElement));
      //num += __popc(element & ((1U << remain) - 1));
      num += __builtin_popcount((uint32_t)(element & ((1U << remain) - 1)));
    }
    return num;
  }
};

template<typename T = bitmapType, int W = BITMAP_WIDTH>
 struct Bitmap2DView{
  T* ptr_;
  uint32_t nrow_; // how many bits in a row.
  uint32_t ncol_; // how many bits in a col.
  uint32_t padded_rowsize_; // number of T in a row.
  uint32_t capacity_; // how many T used to store data.

  // size_t get_bytes_num(size_t size) {
  //  return sizeof(T) * ((size+W-1) / W) * size;
  //}

   void init(T* ptr, uint32_t size) {
    ptr_ = ptr;
    nrow_ = size;
    ncol_ = size;
    padded_rowsize_ = (ncol_+W-1) / W;
    capacity_ = padded_rowsize_ * nrow_;
  }

   size_t get_valid_bits_num() { return size_t(nrow_) * ncol_; }

   T* row(int x) {
    return ptr_ + x * padded_rowsize_;
  }

  // set all bits in bitmap to 0.
   void clear() {
    int thread_lane = 0;
    for (auto i = thread_lane; i < capacity_; i += THREAD_SIZE) {
      ptr_[i] = 0;
    }
    
  }

   void build(Graph &g, vidType* vlist, vidType size) {
    int thread_lane = 0;
    for (vidType i = 0; i < size; i++) {
      auto search = g.N(vlist[i]);
      vidType search_size = g.get_degree(vlist[i]);
      for (int j = thread_lane; j < size; j += THREAD_SIZE) {
        bool flag = (j!=i) && binary_search(search.data(), vlist[j], search_size);
        //warp_set(i, j, flag);
        thread_set(i, j, flag);
      }
    }
  }

   void build_block(Graph &g, vidType* vlist, vidType size) {
    printf("No implment!\n");
    // int thread_lane = 0;
    // int warp_lane = threadIdx.x / THREAD_SIZE;
    // for (vidType i = warp_lane; i < size; i += WARPS_PER_BLOCK) {
    //   auto search = g.N(vlist[i]);
    //   vidType search_size = g.get_degree(vlist[i]);
    //   for (int j = thread_lane; j < size; j += THREAD_SIZE) {
    //     bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
    //     // warp_set(i, j, flag);
    //     thread_set(i, j, flag);
    //   }
    // }
  }

   void build_global(Graph &g, vidType* vlist, vidType size) {
    printf("No implment!\n");
    // int thread_lane = 0;
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;    
    // int warp_id = thread_id / THREAD_SIZE;
    // int num_warps = WARPS_PER_BLOCK * gridDim.x;
    // for (vidType i = warp_id; i < size; i += num_warps) {
    //   auto search = g.N(vlist[i]);
    //   vidType search_size = g.get_degree(vlist[i]);
    //   for (int j = thread_lane; j < size; j += THREAD_SIZE) {
    //     bool flag = (j!=i) && binary_search(search, vlist[j], search_size);
    //     warp_set(i, j, flag);
    //   }
    // }
  }  

   T warp_load(uint32_t x, uint32_t warp_id) {
    return ptr_[x * padded_rowsize_ + warp_id];
  }

  // set target bits in bitmap[x][y/W].
  // not concurrent-safe, single (warp) writer.
   void warp_set(uint32_t x, uint32_t y, bool flag) {
     printf("No implment!\n");
    // uint32_t thread_lane = threadIdx.x & (THREAD_SIZE-1);
    // uint32_t element = y / W; // should be the same value in this warp
    // uint32_t activemask = __activemask();
    // uint32_t mask = __ballot_sync(activemask, flag);
    // if (thread_lane == 0) ptr_[x * padded_rowsize_ + element] = mask;
  }

  void thread_set(uint32_t x, uint32_t y, bool flag) {
    uint32_t element = y / W; // find the element in the bit array
    uint32_t bit_sft = y % W; // find the position within the element
    T mask = 1 << bit_sft; // create a mask for the bit position
    
    if (flag) { // if flag is true, set the bit to 1
      ptr_[x * padded_rowsize_ + element] |= mask;
    } else { // if flag is false, set the bit to 0
      ptr_[x * padded_rowsize_ + element] &= ~mask;
    }
  }

  // set bitmap[x][y] to 1.
  // not concurrent-safe, single writer.
   void set1(int x, int y) {
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t bit_sft = y % W;
    T mask = 1 << bit_sft;
    ptr_[x * padded_rowsize_ + element] |= mask;
  }

   bool get(int x, int y) {
    uint32_t element = y / W; // should be the same value in this warp
    uint32_t bit_sft = y % W;
    T mask = 1 << bit_sft;
    return ptr_[x * padded_rowsize_ + element] & mask;
  }
};
