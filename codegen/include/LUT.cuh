#pragma once

#include "bitmap.cuh"

// assumption: bitmapType == vidType
template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ __host__ struct LUT{
  T* heap_;
  uint32_t max_size_;

  vidType* vlist_;
  vidType size_;
  Bitmap2DView<T,W> bitmap_;

  __device__ __host__ void init(T* heap_head, uint32_t max_size){
    heap_ = heap_head;
    max_size_ = max_size;
  }

  __device__ __host__ vidType vid(vidType idx) {
    return vlist_[idx];
  }

  __device__ __host__ void build(GraphGPU& g, vidType* vlist, vidType size){
    vlist_ = vlist;
    size_ = size;
    assert(size < max_size_);
    bitmap_.init(heap_, size_);
    bitmap_.clear();
    bitmap_.build(g, vlist, size); 
  }

  __device__ __host__ vidType size() {
    return size_;
  }

  __device__ __host__ Bitmap1DView<T,W> row(int x) {
    Bitmap1DView<T, W> ret;
    ret.init(bitmap_.row(x), size_);
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> row(int x, int upper_bound) {
    Bitmap1DView<T, W> ret;
    ret.init(bitmap_.row(x), upper_bound);
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> row_limit(int x, int upper_bound) {
    Bitmap1DView<T, W> ret;
    int s = 0;
    int len = size_;
    int key = upper_bound;
    while (len > 0) { 
      int half = len >> 1;
      int mid = s + half;
      if (vlist_[mid] < key) {
        s = mid + 1;
        len = len - half - 1;
      } else {
        len = half;
      }
    }
    ret.init(bitmap_.row(x), s);
    return ret;
  }
};

template <typename T = bitmapType, int W = BITMAP_WIDTH>
class LUTManager{
  private:
    T* heap_head_;
    uint32_t LUT_num_;
    uint32_t max_LUT_size_;
    uint32_t max_row_size_;
    bool use_gpu_;
  public:
    LUTManager(uint32_t LUT_num, uint32_t max_nrow, uint32_t max_ncol, bool use_gpu):
      LUT_num_(LUT_num), max_row_size_(max_ncol), use_gpu_(use_gpu) {
        auto max_padded_rowsize_ = (max_ncol + W - 1) / W;
        max_LUT_size_ = max_padded_rowsize_ * max_nrow;
        size_t totalMemSize = max_LUT_size_ * LUT_num_ * sizeof(T);
        if (use_gpu_){
          CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
          CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
          std::cout << "Allocate LUT on GPU-side." << std::endl;
        } else {
          heap_head_ = (T*) malloc(totalMemSize);
          memset(heap_head_, 0, totalMemSize);
          std::cout << "Allocate LUT on CPU-side." << std::endl;
        }
        std::cout << LUT_num << " LUT, each has up to " << max_nrow << " rows, " << max_ncol << " cols,"<<
          " total memory size: " << float(totalMemSize) / (1024*1024) << " MB\n";
      }

    __device__ __host__ LUT<T,W> getEmptyLUT(int lut_id) {
      LUT<T,W> lut;
      assert(lut_id < LUT_num_);
      lut.init(heap_head_ + lut_id * max_LUT_size_, max_row_size_);
      return lut;
    }
};
