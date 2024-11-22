#include "P1.yml.cuh"
#include "P1-GM.yml.cuh"
#include "P2.yml.cuh"
#include "P2-GM.yml.cuh"
#include "P3.yml.cuh"
#include "P3-GM.yml.cuh"
#include "P6.yml.cuh"
#include "P6-GM.yml.cuh"
#include "P7.yml.cuh"
#include "P7-GM.yml.cuh"
#include "P8.yml.cuh"
#include "P8-GM.yml.cuh"
#include "P9.yml.cuh"
#include "P9-GM.yml.cuh"
#include "P10.yml.cuh"
#include "P10-GM.yml.cuh"
#include "P11.yml.cuh"
#include "P11-GM.yml.cuh"
#include "P12.yml.cuh"
#include "P12-GM.yml.cuh"
#include "P13.yml.cuh"
#include "P13-GM.yml.cuh"
#include "P14.yml.cuh"
#include "P14-GM.yml.cuh"
#include "P15.yml.cuh"
#include "P15-GM.yml.cuh"
#include "P16.yml.cuh"
#include "P16-GM.yml.cuh"
#include "P17.yml.cuh"
#include "P17-GM.yml.cuh"
#include "P18.yml.cuh"
#include "P18-GM.yml.cuh"
#include "P19.yml.cuh"
#include "P19-GM.yml.cuh"
#include "P20.yml.cuh"
#include "P20-GM.yml.cuh"
#include "P21.yml.cuh"
#include "P21-GM.yml.cuh"
#include "P22.yml.cuh"
#include "P22-GM.yml.cuh"

#define __N_LISTS 2
#define __N_BITMAPS 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
GM_build_LUT(vidType begin, vidType end, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs,
                  vidType task_id){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 2];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  AccType count = 0;
  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(0);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 2;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();

  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  auto v0 = task_id;
  auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
  __build_LUT_global(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));

}

