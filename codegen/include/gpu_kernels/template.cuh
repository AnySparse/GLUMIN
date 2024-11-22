#define __N_LISTS__ {SLOT_CAPACITY}
#define __N_BITMAPS__ {BITMAP_CAPACITY}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
{KERNEL_NAME}(vidType nv, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){{
  __shared__ vidType list_size[WARPS_PER_BLOCK * {SLOT_CAPACITY}];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * {BITMAP_CAPACITY}];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  AccType count = 0;
  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(warp_id);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = nv;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = {SLOT_CAPACITY};
  meta.bitmap_capacity = {BITMAP_CAPACITY};
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  {WARP_LEVEL_DFS_CODE}
  // END OF CODEGEN

  atomicAdd(counter, count);
}}
