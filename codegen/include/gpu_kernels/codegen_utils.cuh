#pragma once
#include "common.h"
#include "set_operation_ext.cuh"
#include "LUT.cuh"

__device__ struct StorageMeta {
  LUT<> lut;
  vidType* base;
  vidType* base_size;
  bitmapType* bitmap_base;
  vidType* bitmap_base_size;
  vidType nv;
  size_t slot_size;
  size_t bitmap_size;
  size_t capacity;
  size_t bitmap_capacity;
  int global_warp_id;
  int local_warp_id;

  __device__ vidType* buffer(int slot_id) {
    return base
      + slot_size * capacity * global_warp_id // begin of this warp
      + slot_size * slot_id; // begin of this slot
  }

  __device__ vidType* buffer_size_addr(int slot_id) {
    return base_size + WARPS_PER_BLOCK * slot_id + local_warp_id;
  }

  __device__ bitmapType* bitmap(int bitmap_id) {
    return bitmap_base
      + bitmap_size * bitmap_capacity * global_warp_id // begin of this warp
      + bitmap_size * bitmap_id; // begin of this bitmap
  }

  __device__ vidType* bitmap_size_addr(int bitmap_id) {
    return bitmap_base_size + WARPS_PER_BLOCK * bitmap_id + local_warp_id;
  }
};

/******************************************************************************
 * Intermediate Result in Array (get read-only view with negligible overhead)
 *****************************************************************************/

//read-only
__device__ struct VertexArrayView {
  vidType* ptr_ = NULL;
  vidType size_ = 0;

  __device__ VertexArrayView() {}
  __device__ VertexArrayView(vidType* ptr, vidType size): ptr_(ptr), size_(size) {}

  __device__ void init(vidType* ptr, vidType size) {
    ptr_ = ptr;
    size_ = size;
  }

  __device__ vidType size() {
    return size_;
  }

  __device__ vidType* ptr() {
    return ptr_;
  }

  __device__ vidType operator[](size_t i) {
    return ptr_ == NULL ? i : ptr_[i];
  }
};

__device__ VertexArrayView
__get_vlist_from_graph(GraphGPU& g, StorageMeta& meta, vidType vid) {
  vidType* vlist = g.N(vid);
  vidType size = g.getOutDegree(vid);
  return VertexArrayView(vlist, size);
}

__device__ VertexArrayView
__get_vlist_from_heap(GraphGPU& g, StorageMeta& meta, int slot_id) {
  // A hack to represent all vertices
  if (slot_id < 0) {
    return VertexArrayView(NULL, meta.nv);
  }
  vidType* vlist = meta.buffer(slot_id);
  vidType size = meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id];
  return VertexArrayView(vlist, size);
}


/******************************************************************************
 * Intermediate Result in Bitmap (get read-only view with negligible overhead)
 *****************************************************************************/

//read-only
__device__ struct VertexMapView {
  //vidType* vlist_;
  //vidType size_;

  VertexArrayView vlist_;
  Bitmap1DView<> bitmap_;
  VertexArrayView index_;
  bool use_one = true; // watch 1 or 0

  __device__ VertexMapView(VertexArrayView vlist): vlist_(vlist) {}
  __device__ VertexMapView(VertexArrayView vlist, Bitmap1DView<> bitmap): vlist_(vlist), bitmap_(bitmap) {}
  __device__ VertexMapView(VertexArrayView vlist, VertexArrayView index): vlist_(vlist), index_(index) {}
  __device__ VertexMapView(VertexArrayView vlist, Bitmap1DView<> bitmap, VertexArrayView index): vlist_(vlist), bitmap_(bitmap), index_(index) {}

  __device__ void use_zero(){
    use_one = false;
  }

  // TODO(mengke): can we move this from here?
  __device__ uint32_t get_numElement() {
    return bitmap_.get_numElement(vlist_.size());
  }
   
  __device__ Bitmap1DView<> get_bitmap() {
    return bitmap_;
  }

  __device__ vidType* raw_list() {
    return vlist_.ptr();
  }

  __device__ vidType size() {
    return index_.size();
  }

  __device__ vidType* ptr() {
    return index_.ptr();
  }

  __device__ vidType operator[](size_t i) {
    return index_.ptr()==NULL ? i : index_[i];
  }
};

// bitmap only
__device__ VertexMapView
__get_vmap_from_lut(GraphGPU& g, StorageMeta& meta, vidType vidx_rowid, bool connected, int upper_bound=-1) {
  auto lut = meta.lut;
  if (upper_bound < 0){
    auto vmap = VertexMapView(VertexArrayView(lut.vlist_, lut.size_), lut.row(vidx_rowid));
    if(!connected) vmap.use_zero();
    return vmap;
  } else {
    auto vmap = VertexMapView(VertexArrayView(lut.vlist_, lut.size_), lut.row(vidx_rowid, upper_bound));
    if(!connected) vmap.use_zero();
    return vmap;
  }
}

__device__ VertexMapView
__get_vmap_from_lut_vid_limit(GraphGPU& g, StorageMeta& meta, vidType vidx_rowid, bool connected, int upper_bound) {
  auto lut = meta.lut;
  auto vmap = VertexMapView(VertexArrayView(lut.vlist_, lut.size_), lut.row_limit(vidx_rowid, upper_bound));
  if(!connected) vmap.use_zero();
  return vmap;
}

// optional bitmapa and optinalindex
__device__ VertexMapView
__get_vmap_from_heap(GraphGPU& g, StorageMeta& meta, int bitmap_id, int slot_id) {
  auto lut = meta.lut;
  auto raw_list = VertexArrayView(lut.vlist_, lut.size_);
  Bitmap1DView<> bitmap;
  VertexArrayView index;
  if (bitmap_id >= 0) {
    auto bitmap_ptr = meta.bitmap(bitmap_id);
    auto bitmap_size = meta.bitmap_base_size[WARPS_PER_BLOCK * bitmap_id + meta.local_warp_id];
    bitmap = Bitmap1DView<>(bitmap_ptr, bitmap_size);
  }
  if (slot_id >= 0) {
    auto buffer_ptr = meta.buffer(slot_id);
    auto buffer_size = meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id];
    index = VertexArrayView(buffer_ptr, buffer_size);
  }

  if (bitmap_id < 0) {
    if (slot_id < 0) {
      // A hack to represent all vertices in bitmap
      return VertexMapView(raw_list);
    } else {
      return VertexMapView(raw_list, index);
    }
  } else {
    if (slot_id < 0) {
      return VertexMapView(raw_list, bitmap);
    } else {
      return VertexMapView(raw_list, bitmap, index);
    }
  }
}

/******************************************************************************
 * optimization instruction (with considerable cost and potential benefit)
 *****************************************************************************/

__device__ void
__build_LUT(GraphGPU& g, StorageMeta& meta, VertexArrayView target){
  meta.lut.build(g, target.ptr(), target.size());
}

__device__ void 
__build_index_from_vmap(GraphGPU& g, StorageMeta& meta, VertexMapView vmap, int slot_id) {
  vidType* index = meta.buffer(slot_id);
  vidType* index_size_addr = meta.buffer_size_addr(slot_id); // shared_memory
  vmap.bitmap_._to_index(vmap.use_one, index, index_size_addr);
}

__device__ void 
__build_vlist_from_vmap(GraphGPU& g, StorageMeta& meta, VertexMapView vmap, int slot_id) {
  vidType* vlist = meta.buffer(slot_id);
  vidType* vlist_size_addr = meta.buffer_size_addr(slot_id); // shared_memory
  vmap.bitmap_._to_vlist(vmap.use_one, /*raw_list*/vmap.raw_list(), vlist, vlist_size_addr);
}

// warp-level
__device__ void 
__build_bitmap_from_vmap(GraphGPU& g, StorageMeta& meta, VertexMapView vmap, int bitmap_id) {
  vidType* bitmap_result = meta.bitmap(bitmap_id);
  vidType* bitmap_addr_size = meta.bitmap_size_addr(bitmap_id);
  vidType* index_list = vmap.index_.ptr();
  vidType list_size = vmap.index_.size();
  //vidType numElements = vmap.get_numElement();
  vidType numElements = (vmap.vlist_.size() + BITMAP_WIDTH - 1) / BITMAP_WIDTH;
  
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  // clean all possible old bits
  for (vidType iter = thread_lane; iter < numElements; iter += WARP_SIZE) {
    bitmap_result[iter] = 0;
  }
  __syncwarp();
  for (vidType idx = thread_lane; idx < list_size; idx += WARP_SIZE) {
    auto setId = index_list[idx];
    atomicOr(&bitmap_result[setId / BITMAP_WIDTH], 1 << (setId % BITMAP_WIDTH));
  }
  __syncwarp();
  if (thread_lane == 0) {
    *bitmap_addr_size = index_list[list_size-1] + 1;
  }
}

__device__ vidType
__build_vid_from_vidx(GraphGPU& g, StorageMeta& meta, vidType vidx){
  return meta.lut.vid(vidx);
}

/******************************************************************************
 * Set Operation Wrapper (the operation you have to do)
 *****************************************************************************/

////////////////////////// vlist + vlist -> vlist /////////////////////////////
__device__ VertexArrayView
__difference(StorageMeta& meta, VertexArrayView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if (upper_bound < 0) {
    cnt = difference_set(v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = difference_set(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}

__device__ VertexArrayView
__intersect(StorageMeta& meta, VertexArrayView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if(upper_bound < 0) {
    cnt = intersect(v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = intersect(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}

__device__ vidType
__difference_num(VertexArrayView v, VertexArrayView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return difference_num(v.ptr(), v.size(), u.ptr(), u.size());
  } else {
    return difference_num(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound);
  }
}

__device__ vidType
__intersect_num(VertexArrayView v, VertexArrayView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return intersect_num(v.ptr(), v.size(), u.ptr(), u.size());
  } else {
    return intersect_num(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound);
  }
}

////////////////////////// vmap + vmap -> vmap ////////////////////////////////
__device__ VertexMapView
__difference(StorageMeta& meta, VertexMapView v, VertexMapView u, vidType upper_bound, int bitmap_id) {
  vidType* bitmap_ptr = meta.bitmap(bitmap_id);
  vidType* bitmap_addr_size = meta.bitmap_size_addr(bitmap_id);
  auto raw_list = VertexArrayView(meta.lut.vlist_, meta.lut.size_);
  auto output = Bitmap1DView<>(bitmap_ptr, raw_list.size());
  if(v.use_one){
    v.bitmap_.AND_NOT(u.bitmap_, output, bitmap_addr_size, upper_bound);
  } else {
    v.bitmap_.NOT_AND_NOT(u.bitmap_, output, bitmap_addr_size, upper_bound);
  }
  __syncwarp();
  return VertexMapView(raw_list, output);
}

__device__ VertexMapView
__intersect(StorageMeta& meta, VertexMapView v, VertexMapView u, vidType upper_bound, int bitmap_id) {
  vidType* bitmap_ptr = meta.bitmap(bitmap_id);
  vidType* bitmap_addr_size = meta.bitmap_size_addr(bitmap_id);
  auto raw_list = VertexArrayView(meta.lut.vlist_, meta.lut.size_);
  auto output = Bitmap1DView<>(bitmap_ptr, raw_list.size());
  if(v.use_one) {
    v.bitmap_.AND(u.bitmap_, output, bitmap_addr_size, upper_bound);
  } else {
    v.bitmap_.NOT_AND(u.bitmap_, output, bitmap_addr_size, upper_bound);
  }
  __syncwarp();
  return VertexMapView(raw_list, output);
}

__device__ vidType
__difference_num(VertexMapView v, VertexMapView u, vidType upper_bound) {
  if(v.use_one){
    return v.bitmap_.count_AND_NOT_thread(u.bitmap_, upper_bound);
  } else {
    return v.bitmap_.count_NOT_AND_NOT_thread(u.bitmap_, upper_bound);
  }
}

__device__ vidType
__intersect_num(VertexMapView v, VertexMapView u, vidType upper_bound) {
  if(v.use_one){
    return v.bitmap_.count_AND_thread(u.bitmap_, upper_bound);
  } else {
    return v.bitmap_.count_NOT_AND_thread(u.bitmap_, upper_bound);
  }
}

__device__ vidType
__difference_num(VertexMapView v, vidType upper_bound) {
  return v.bitmap_.count_thread(false, upper_bound);
}

__device__ vidType
__intersect_num(VertexMapView v, vidType upper_bound) {
  return v.bitmap_.count_thread(true, upper_bound);
}


////////////////////////// vlist + vmap -> vlist //////////////////////////////
__device__ VertexArrayView
__difference(StorageMeta& meta, VertexArrayView v, VertexMapView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if (upper_bound < 0) {
    cnt = difference_set_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), buffer);
  } else {
    cnt = difference_set_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}

__device__ VertexArrayView
__intersect(StorageMeta& meta, VertexArrayView v, VertexMapView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if(upper_bound < 0) {
    cnt = intersect_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), buffer);
  } else {
    cnt = intersect_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}

__device__ vidType
__difference_num(VertexArrayView v, VertexMapView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return difference_num_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size());
  } else {
    return difference_num_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), upper_bound);
  }
  __syncwarp();
}

__device__ vidType
__intersect_num(VertexArrayView v, VertexMapView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return intersect_num_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size());
  } else {
    return intersect_num_source2(v.ptr(), v.size(), /*u_source=*/u.raw_list(), u.ptr(), u.size(), upper_bound);
  }
  __syncwarp();
}

////////////////////////// vmap + vlist -> vmap ///////////////////////////////
__device__ VertexMapView
__difference(StorageMeta& meta, VertexMapView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if (upper_bound < 0) {
    cnt = difference_set_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = difference_set_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  VertexArrayView index = VertexArrayView(buffer, cnt);
  VertexArrayView raw_list = VertexArrayView(meta.lut.vlist_, meta.lut.size_);
  return VertexMapView(raw_list, index);
}

__device__ VertexMapView
__intersect(StorageMeta& meta, VertexMapView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if(upper_bound < 0) {
    cnt = intersect_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = intersect_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  VertexArrayView index = VertexArrayView(buffer, cnt);
  VertexArrayView raw_list = VertexArrayView(meta.lut.vlist_, meta.lut.size_);
  return VertexMapView(raw_list, index);
}

__device__ vidType
__difference_num(VertexMapView v, VertexArrayView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return difference_num_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size());
  } else {
    return difference_num_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), upper_bound);
  }
  __syncwarp();
}

__device__ vidType
__intersect_num(VertexMapView v, VertexArrayView u, vidType upper_bound) {
  if (upper_bound < 0) {
    return intersect_num_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size());
  } else {
    return intersect_num_source1(/*v_source=*/v.raw_list(), v.ptr(), v.size(), u.ptr(), u.size(), upper_bound);
  }
  __syncwarp();
}
