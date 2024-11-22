#pragma once

#include "codegen_bitmap.cuh"

// assumption: bitmapType == vidType
template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ __host__ struct LUT{
  T* heap_;
#ifdef ROARING
  typeType* types_;
#endif  
#ifdef LOAD_SRAM
  T* sram_;
#endif
  uint32_t max_size_;

  vidType* vlist_;
  vidType size_;
  Bitmap2DView<T,W> bitmap_;

#ifdef ROARING
  __device__ __host__ void init(T* heap_head, typeType * types_head, uint32_t max_size){
    heap_ = heap_head;
    types_ = types_head;
    max_size_ = max_size;
  }
#endif

#ifdef LOAD_SRAM
  __device__ __host__ void init(T* heap_head, T* sram_head, uint32_t max_size){
    heap_ = heap_head;
    sram_ = sram_head;
    max_size_ = max_size;
  }
#endif
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
    assert(size <= max_size_);
#ifdef ROARING
    bitmap_.init(heap_, types_, size_);
#elif defined(LOAD_SRAM)
    bitmap_.init(heap_, sram_, size_);
#else
    bitmap_.init(heap_, size_);
#endif
    // bitmap_.clear();
    bitmap_.build(g, vlist, size); 
  }

  __device__ __host__ void build_block(GraphGPU& g, vidType* vlist, vidType size){
    vlist_ = vlist;
    size_ = size;
    assert(size <= max_size_);
#ifdef ROARING
    bitmap_.init(heap_, types_, size_);
#else
    bitmap_.init(heap_, size_);
#endif
    // bitmap_.clear();
    bitmap_.build_block(g, vlist, size); 
  }  

  __device__ __host__ void build_global(GraphGPU& g, vidType* vlist, vidType size){
    vlist_ = vlist;
    size_ = size;
    assert(size <= max_size_);
#ifdef ROARING
    bitmap_.init(heap_, types_, size_);
#else
    bitmap_.init(heap_, size_);
#endif
    // bitmap_.clear();
    bitmap_.build_global(g, vlist, size); 
  }    

  __device__ __host__ void set_LUT_para(GraphGPU& g, vidType* vlist, vidType size){
    vlist_ = vlist;
    size_ = size;
    assert(size <= max_size_);
#ifdef ROARING
    bitmap_.init(heap_, types_, size_);
#else
    bitmap_.init(heap_, size_);
#endif
  }  

  __device__ __host__ vidType size() {
    return size_;
  }

  __device__ __host__ Bitmap1DView<T,W> row(int x) {
    Bitmap1DView<T, W> ret;
#ifdef ROARING
    ret.init(bitmap_.row(x), bitmap_.row_types(x), size_);
#else
    ret.init(bitmap_.row(x), size_);
#endif
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> row(int x, int upper_bound) {
    Bitmap1DView<T, W> ret;
#ifdef ROARING
    ret.init(bitmap_.row(x), bitmap_.row_types(x), upper_bound);
#else
    ret.init(bitmap_.row(x), upper_bound);
#endif
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
#ifdef ROARING
    ret.init(bitmap_.row(x), bitmap_.row_types(x), s);
#else
    ret.init(bitmap_.row(x), s);
#endif
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> bitmap(T* bitmap1D) {
    Bitmap1DView<T, W> ret;
    ret.init(bitmap1D, size_);
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> bitmap(T* bitmap1D, int upper_bound) {
    Bitmap1DView<T, W> ret;
    ret.init(bitmap1D, upper_bound);
    return ret;
  }

  __device__ __host__ Bitmap1DView<T,W> bitmap_limit(T* bitmap1D, int upper_bound) {
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
    ret.init(bitmap1D, s);
    return ret;
  }

  __device__ void load_roaring(T* bitmap_ptr, int rowid, bool is_warp, int upper_bound) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    int chunk_size = (upper_bound + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    Bitmap1DView<T, W> lutrow = row(rowid);
    // clear bitmap buffer
    if (is_warp) {
      for (int i = thread_lane; i < bitmap_.padded_rowsize_; i += WARP_SIZE) {
        bitmap_ptr[i] = 0;
      }
    }
    else {
      for (int i = 0; i < bitmap_.padded_rowsize_; i++) {
        bitmap_ptr[i] = 0;
      }
    }
    // each chunk rezip
    for (int chunk_id = 0; chunk_id < chunk_size; chunk_id++) {
      typeType chunk_type =  bitmap_.get_chunk_type(rowid, chunk_id);
      // if (chunk_type != 0) printf("hit %d\n", chunk_type);
      // all 0
      if (chunk_type == 0) {
        continue;
      }
      // all 1
      else if (chunk_type == CHUNK_WIDTH - 1) {
        if (is_warp){
          for (int i = thread_lane; i < PAD_PER_CHUNK && (chunk_id * PAD_PER_CHUNK + i) < bitmap_.padded_rowsize_; i += WARP_SIZE) {
            bitmap_ptr[chunk_id * PAD_PER_CHUNK + i] = FULL_MASK;
          }
        }
        else {
          for (int i = 0; i < PAD_PER_CHUNK && (chunk_id * PAD_PER_CHUNK + i) < bitmap_.padded_rowsize_; i++) {
            bitmap_ptr[chunk_id * PAD_PER_CHUNK + i] = FULL_MASK;
          }
        }
      }
      // bitmap chunk
      else if (chunk_type == ARRAY_LIMIT) {
        if (is_warp) {
          for (int i = thread_lane; i < PAD_PER_CHUNK && (chunk_id * PAD_PER_CHUNK + i) < bitmap_.padded_rowsize_; i += WARP_SIZE) {
            bitmap_ptr[chunk_id * PAD_PER_CHUNK + i] = lutrow.load(chunk_id * PAD_PER_CHUNK + i);
          }
        }
        else {
          for (int i = 0; i < PAD_PER_CHUNK && (chunk_id * PAD_PER_CHUNK + i) < bitmap_.padded_rowsize_; i++) {
            bitmap_ptr[chunk_id * PAD_PER_CHUNK + i] = lutrow.load(chunk_id * PAD_PER_CHUNK + i);
          }
        }
      }
      // array chunk
      else {}
    }
  }
};

template <typename T = bitmapType, int W = BITMAP_WIDTH>
class LUTManager{
  private:
    T* heap_head_;
    uint32_t LUT_num_;
    size_t max_LUT_size_;
    uint32_t max_row_size_;
    bool use_gpu_;
#ifdef ROARING
    typeType* types_head_;
    uint32_t chunk_rowsize_;
    size_t max_type_size_;
#endif
  public:
    LUTManager(){}
    LUTManager(uint32_t LUT_num, uint32_t max_nrow, uint32_t max_ncol, bool use_gpu):
      LUT_num_(LUT_num), max_row_size_(max_ncol), use_gpu_(use_gpu) {
        auto max_padded_rowsize_ = (max_ncol + W - 1) / W;
        max_LUT_size_ = max_padded_rowsize_ * max_nrow;
        size_t totalMemSize = max_LUT_size_ * LUT_num_ * sizeof(T);
#ifdef ROARING
        chunk_rowsize_ = (max_ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
        max_type_size_ = chunk_rowsize_ * max_nrow;
        size_t totalMemSize_chunk_types = max_type_size_ * LUT_num_ * sizeof(typeType);
#endif
        if (use_gpu_){
          CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
          CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
          std::cout << "Allocate LUT on GPU-side." << std::endl;
#ifdef ROARING
          CUDA_SAFE_CALL(cudaMalloc(&types_head_, totalMemSize_chunk_types));
          CUDA_SAFE_CALL(cudaMemset(types_head_, 0, totalMemSize_chunk_types));
          std::cout << LUT_num << " typelist of " << max_nrow << " rows, " << chunk_rowsize_ << " chunks,"<<
          " total memory size: " << float(totalMemSize_chunk_types)/(1024*1024) << " MB\n";
#endif
        } else {
          heap_head_ = (T*) malloc(totalMemSize);
          memset(heap_head_, 0, totalMemSize);
          std::cout << "Allocate LUT on CPU-side." << std::endl;
        }
        std::cout << LUT_num << " LUT, each has up to " << max_nrow << " rows, " << max_ncol << " cols,"<<
          " total memory size: " << float(totalMemSize) / (1024*1024) << " MB\n";
      }
    
    void init(uint32_t LUT_num, uint32_t max_nrow, uint32_t max_ncol, bool use_gpu) {
      LUT_num_ = LUT_num;
      max_row_size_ = max_ncol;
      use_gpu_ = use_gpu;
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

    void recreate(uint32_t LUT_num, uint32_t max_nrow, uint32_t max_ncol, bool use_gpu) {
      LUT_num_ = LUT_num;
      max_row_size_ = max_ncol;
      use_gpu_ = use_gpu;
      CUDA_SAFE_CALL(cudaFree(heap_head_));
#ifdef ROARING
      CUDA_SAFE_CALL(cudaFree(types_head_));
#endif
      auto max_padded_rowsize_ = (max_ncol + W - 1) / W;
      max_LUT_size_ = max_padded_rowsize_ * max_nrow;
      size_t totalMemSize = max_LUT_size_ * LUT_num_ * sizeof(T);
#ifdef ROARING
      chunk_rowsize_ = (max_ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
      max_type_size_ = chunk_rowsize_ * max_nrow;
      size_t totalMemSize_chunk_types = max_type_size_ * LUT_num_ * sizeof(typeType);
#endif
      if (use_gpu_){
        CUDA_SAFE_CALL(cudaMalloc(&heap_head_, totalMemSize));
        CUDA_SAFE_CALL(cudaMemset(heap_head_, 0, totalMemSize));
        std::cout << "Allocate LUT on GPU-side." << std::endl;
#ifdef ROARING
        CUDA_SAFE_CALL(cudaMalloc(&types_head_, totalMemSize_chunk_types));
        CUDA_SAFE_CALL(cudaMemset(types_head_, 0, totalMemSize_chunk_types));
        std::cout << LUT_num << " typelist of " << max_nrow << " rows, " << chunk_rowsize_ << " chunks,"<<
        " total memory size: " << float(totalMemSize_chunk_types)/(1024*1024) << " MB\n";
#endif
      } else {
        heap_head_ = (T*) malloc(totalMemSize);
        memset(heap_head_, 0, totalMemSize);
        std::cout << "Allocate LUT on CPU-side." << std::endl;
      }
      std::cout << LUT_num << " LUT, each has up to " << max_nrow << " rows, " << max_ncol << " cols,"<<
        " total memory size: " << float(totalMemSize) / (1024*1024) << " MB\n";
    }

    void update_para(uint32_t LUT_num, uint32_t max_nrow, uint32_t max_ncol, bool use_gpu) {
      LUT_num_ = LUT_num;
      max_row_size_ = max_ncol;
      use_gpu_ = use_gpu;
      auto max_padded_rowsize_ = (max_ncol + W - 1) / W;
      max_LUT_size_ = max_padded_rowsize_ * max_nrow;
#ifdef ROARING
      chunk_rowsize_ = (max_ncol + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
      max_type_size_ = chunk_rowsize_ * max_nrow;
#endif
    }

    __device__ __host__ LUT<T,W> getEmptyLUT(int lut_id, T* sram_head) {
      LUT<T,W> lut;
      assert(lut_id < LUT_num_);
      lut.init(heap_head_ + lut_id * max_LUT_size_, sram_head, max_row_size_);
      return lut;
    }

    __device__ __host__ LUT<T,W> getEmptyLUT(int lut_id) {
      LUT<T,W> lut;
      assert(lut_id < LUT_num_);
#ifdef ROARING
      lut.init(heap_head_ + lut_id * max_LUT_size_, types_head_ + lut_id * max_type_size_, max_row_size_);
#else
      lut.init(heap_head_ + lut_id * max_LUT_size_, max_row_size_);
#endif
      return lut;
    }
};
