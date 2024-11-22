#pragma once

#include "cutil_subset.h"
#include "common.h"
#include "graph_gpu.h"

template <typename T = bitmapType, int W = BITMAP_WIDTH>
class BinaryEncode {
private:
    T* encodeHead;
    int mapNum;
    int encodeNum;
    int bits;
    int numElements;
    int mapSize;
    size_t totalMemSize;

public:
    BinaryEncode (){}
    BinaryEncode (int map_num, int encode_num, int bit) {
        mapNum = map_num;
        encodeNum = encode_num;
        bits = bit;
        numElements = (bits - 1) / W + 1;
        mapSize = size_t(encodeNum) * size_t(numElements);
        totalMemSize = mapSize * size_t(mapNum) * sizeof(T);
        CUDA_SAFE_CALL(cudaMalloc(&encodeHead, totalMemSize));
        CUDA_SAFE_CALL(cudaMemset(encodeHead, 0, totalMemSize));
        std::cout << map_num << " maps of " << encode_num << " sets, " << numElements << " numElements, " << bit 
                << " bits, total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
    }
    ~BinaryEncode(){}

    void recreate (int map_num, int encode_num, int bit) {
        CUDA_SAFE_CALL(cudaFree(encodeHead));
        mapNum = map_num;
        encodeNum = encode_num;
        bits = bit;
        numElements = (bits - 1) / W + 1;
        mapSize = size_t(encodeNum) * size_t(numElements);
        totalMemSize = mapSize * size_t(mapNum) * sizeof(T);
        CUDA_SAFE_CALL(cudaMalloc(&encodeHead, totalMemSize));
        CUDA_SAFE_CALL(cudaMemset(encodeHead, 0, totalMemSize));
        std::cout << map_num << " maps of " << encode_num << " sets, " << numElements << " numElements, " << bit 
                << " bits, total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
    }

    void init (int map_num, int encode_num, int bit) {
        mapNum = map_num;
        encodeNum = encode_num;
        bits = bit;
        numElements = (bits - 1) / W + 1;
        mapSize = size_t(encodeNum) * size_t(numElements);
        totalMemSize = mapSize * size_t(mapNum) * sizeof(T);
        CUDA_SAFE_CALL(cudaMalloc(&encodeHead, totalMemSize));
        CUDA_SAFE_CALL(cudaMemset(encodeHead, 0, totalMemSize));
        std::cout << map_num << " maps of " << encode_num << " sets, " << numElements << " numElements, " << bit 
                << " bits, total memory size: " << float(totalMemSize)/(1024*1024) << " MB\n";
    }

    void update_para (int map_num, int encode_num, int bit) {
        mapNum = map_num;
        encodeNum = encode_num;
        bits = bit;
        numElements = (bits - 1) / W + 1;
        mapSize = size_t(encodeNum) * size_t(numElements);
        totalMemSize = mapSize * size_t(mapNum) * W;
    }
    
    __device__ void warp_clear(int warpMapHead) {
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        T* v = encodeHead + warpMapHead;
        for (auto i = thread_lane; i < mapSize; i += WARP_SIZE) {
            v[i] = 0;
        }
        __syncwarp();
    }

    __device__ void warp_set(int warpMapHead, int x, int y, bool flag) {
        T* v = encodeHead + warpMapHead + x * numElements;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, flag);
        int element = y / W;
        if (thread_lane == 0) v[element] = v[element] | mask;
    }

    __device__ void warp_cover(int warpMapHead, int x, int y, bool flag) {
        T* v = encodeHead + warpMapHead + x * numElements;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, flag);
        int element = y / W;
        if (thread_lane == 0) v[element] = mask;
    }

    __device__ void build_subgraph(int warpMapHead, vidType* sub_list, vidType list_size, GraphGPU g) {
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        for (vidType i = 0; i < list_size; i++) {
            auto search = g.N(sub_list[i]);
            vidType search_size = g.getOutDegree(sub_list[i]);
            for (int j = thread_lane; j < list_size; j += WARP_SIZE) {
                unsigned active = __activemask();
                bool flag = (j!=i) && binary_search(search, sub_list[j], search_size);
                __syncwarp(active);
                warp_cover(warpMapHead, i, j, flag);
            }
        }
    }

    __device__ void build_subgraph(int warpMapHead, vidType* list_a, vidType list_a_size, vidType* list_b, vidType list_b_size, GraphGPU g) {
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        for (vidType i = 0; i < list_a_size; i++) {
            auto search = g.N(list_a[i]);
            vidType search_size = g.getOutDegree(list_a[i]);
            for (int j = thread_lane; j < list_b_size; j += WARP_SIZE) {
                unsigned active = __activemask();
                bool flag = (j!=i) && binary_search(search, list_b[j], search_size);
                __syncwarp(active);
                warp_cover(warpMapHead, i, j, flag);
            }
        }
    }

    __device__ void append_subgraph(int warpMapHead, vidType startIdx, vidType* list_a, vidType list_a_size, vidType* list_b, vidType list_b_size, GraphGPU g) {
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        for (vidType i = 0; i < list_a_size; i++) {
            auto search = g.N(list_a[i]);
            vidType search_size = g.getOutDegree(list_a[i]);
            for (int j = thread_lane; j < list_b_size; j += WARP_SIZE) {
                unsigned active = __activemask();
                bool flag = (j!=i) && binary_search(search, list_b[j], search_size);
                __syncwarp(active);
                warp_cover(warpMapHead, i, j, flag);
            }
        }
    }

    __device__ vidType get_1_index (int warpMapHead, int x, vidType sizex, vidType *result) {
        int warp_lane = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        __shared__ vidType result_size[WARPS_PER_BLOCK];
        if (thread_lane == 0) result_size[warp_lane] = 0;
        __syncwarp();
        vidType size_group = sizex / WARP_SIZE;
        for (int key = 0; key < size_group; key++) {
            bool v1 = 0;
            vidType keyId = (key * WARP_SIZE) + thread_lane;
            if (get(warpMapHead, x, keyId)) v1 = 1;
            unsigned mask1 = __ballot_sync(0xffffffff, v1);
            auto idx1 = __popc(mask1 << (WARP_SIZE - thread_lane - 1));
            if (v1) result[result_size[warp_lane] + idx1 - 1] = keyId;
            if (thread_lane == 0) result_size[warp_lane] += __popc(mask1);
        }
        __syncwarp();
        if (thread_lane < sizex % WARP_SIZE) {
            bool v1 = 0;
            vidType keyId = size_group * WARP_SIZE + thread_lane;
            if (get(warpMapHead, x, keyId)) v1 = 1;
            unsigned active = __activemask();
            unsigned mask1 = __ballot_sync(active, v1);
            auto idx1 = __popc(mask1 << (WARP_SIZE - thread_lane - 1));
            if (v1) result[result_size[warp_lane] + idx1 - 1] = keyId;
            if (thread_lane == 0) result_size[warp_lane] += __popc(mask1);
        }
        return result_size[warp_lane];
    }

    __device__ vidType get_0_index (int warpMapHead, int x, vidType sizex, vidType *result) {
        int warp_lane = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        __shared__ vidType result_size[WARPS_PER_BLOCK];
        if (thread_lane == 0) result_size[warp_lane] = 0;
        __syncwarp();
        vidType size_group = sizex / WARP_SIZE;
        for (int key = 0; key < size_group; key++) {
            bool v0 = 0;
            vidType keyId = (key * WARP_SIZE) + thread_lane;
            if (!get(warpMapHead, x, keyId)) v0 = 1;
            unsigned mask0 = __ballot_sync(0xffffffff, v0);
            auto idx1 = __popc(mask0 << (WARP_SIZE - thread_lane - 1));
            if (v0) result[result_size[warp_lane] + idx1 - 1] = keyId;
            if (thread_lane == 0) result_size[warp_lane] += __popc(mask0);
        }
        __syncwarp();
        if (thread_lane < sizex % WARP_SIZE) {
            bool v0 = 0;
            vidType keyId = size_group * WARP_SIZE + thread_lane;
            if (!get(warpMapHead, x, keyId)) v0 = 1;
            unsigned active = __activemask();
            unsigned mask0 = __ballot_sync(active, v0);
            auto idx1 = __popc(mask0 << (WARP_SIZE - thread_lane - 1));
            if (v0) result[result_size[warp_lane] + idx1 - 1] = keyId;
            if (thread_lane == 0) result_size[warp_lane] += __popc(mask0);
        }
        return result_size[warp_lane];
    }

    __device__ void index2bitmap(vidType *index_list, vidType list_size, vidType *bitmap_result) {
        int thread_lane = threadIdx.x & (WARP_SIZE - 1);
        for (vidType iter = thread_lane; iter < numElements; iter += WARP_SIZE) {
            bitmap_result[iter] = 0;
        }
        __syncwarp();
        for (vidType idx = thread_lane; idx < list_size; idx += WARP_SIZE) {
          auto setId = index_list[idx];
          atomicOr(&bitmap_result[setId / BITMAP_WIDTH], 1 << (setId % BITMAP_WIDTH));
        }
        __syncwarp();
    }
    
    __device__ void set1(int warpMapHead, int x, int y) {
        T* v = encodeHead + warpMapHead + x * numElements;
        int element = y / W;
        int bit_set = y % W;
        T mask = 1 << bit_set;
        v[element] = v[element] | mask;
        // atomicOr(int(v[element]), int(mask));
    }

    __device__ bool get(int warpMapHead, int x, int y) {
        T* v = encodeHead + warpMapHead + x * numElements;
        int element = y / W;
        int bit_set = y % W;
        T mask = 1 << bit_set;
        return v[element] & mask;
    }

    __device__ T* row(size_t warpMapHead, int x) {
        return encodeHead + warpMapHead + x * numElements;
    }

    __device__ int intersect_num_warp(size_t warpMapHead,  int countElement, int x, int y) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < countElement; i+= WARP_SIZE) {
            num += __popc(v1[i] & v2[i]);
        }
        return num;
    }

    __device__ int intersect_num_warp(size_t warpMapHead, int x, int y) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < numElements; i+= WARP_SIZE) {
            num += __popc(v1[i] & v2[i]);
        }
        return num;
    }

    __device__ int intersect_num_thread(size_t warpMapHead,  int countElement, int x, int y) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(v1[i] & v2[i]);
        }
        return num;
    }
    
    __device__ int or_not_num_thread(size_t warpMapHead, int upper, int x, int y) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        for (int i = 0; i < countElement; i++) {
            num += __popc(~(v1[i] | v2[i]));
        }
        if (remain) {
            num += __popc((~(v1[countElement] | v2[countElement])) & ((1U << remain) - 1));
        }
        return num;
    }

    __device__ int and_not_num_thread(size_t warpMapHead, int upper, int x, int y, T* pre) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        // T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        for (int i = 0; i < countElement; i++) {
            num += __popc(pre[i] & ~v2[i]);
        }
        if (remain) {
            num += __popc((pre[countElement] & ~v2[countElement]) & ((1U << remain) - 1));
        }
        return num;
    }

    __device__ int and_not_num_thread(size_t warpMapHead, int upper, int x, int y, vidType* mask_list, vidType mask_size) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        // for (int i = 0; i < countElement; i++) {
        //     num += __popc(~(v1[i] | v2[i]));
        // }
        // if (remain) {
        //     num += __popc((~(v1[countElement] | v2[countElement])) & ((1U << remain) - 1));
        // }
        for (int i = 0; i < mask_size; i++) {
            int check = mask_list[i];
            if (check < upper) {
                if (!(v2[check / BITMAP_WIDTH] & (1 << (check & (BITMAP_WIDTH-1))))){
                    num++;
                }
            }
        }
        return num;
    }

    __device__ int or_not_num_warp(size_t warpMapHead, int upper, int x, int y) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        for (int i = thread_lane; i < countElement; i += WARP_SIZE) {
            num += __popc(~(v1[i] | v2[i]));
        }
        if (thread_lane == 0 && remain) {
            num += __popc((~(v1[countElement] | v2[countElement])) & ((1U << remain) - 1));
        }
        return num;
    }

    __device__ int dif_or_num_thread(size_t warpMapHead,  int countElement, int x, int y) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(v1[i] ^ v2[i]);
        }
        return num;
    }
    __device__ void intersect_thread(size_t warpMapHead, int countElement, int x, int y, T* result) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        for (int i = 0; i < countElement; i++) {
            result[i] =v1[i] & v2[i];
        }
        return;
    }

    __device__ vidType or_not_thread(size_t warpMapHead, int upper, int x, int y, T* result) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        vidType count = 0;
        bitmapType res;
        for (int i = 0; i < countElement; i++) {
            res = ~(v1[i] | v2[i]);
            result[i] = res;
            count += __popc(res);
            // result[i] = ~(v1[i] | v2[i]);
        }
        if (remain) {
            res = (~(v1[countElement] | v2[countElement])) & ((1U << remain) - 1);
            result[countElement] = res;
            count += __popc(res);
            // result[countElement] = (~(v1[countElement] | v2[countElement])) & ((1U << remain) - 1);
        }
        return count;
    }

    __device__ vidType and_not_thread(size_t warpMapHead, int upper, int x, int y, T* result) {
        int countElement = upper / BITMAP_WIDTH;
        int remain = upper % BITMAP_WIDTH;
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        vidType count = 0;
        bitmapType res;
        for (int i = 0; i < countElement; i++) {
            res = v1[i] & ~v2[i];
            result[i] = res;
            count += __popc(res);
            // result[i] = v1[i] & ~v2[i];
        }
        if (remain) {
            res = (v1[countElement] & ~v2[countElement]) & ((1U << remain) - 1);
            result[countElement] = res;
            count += __popc(res);
            // result[countElement] = (v1[countElement] & ~v2[countElement]) & ((1U << remain) - 1);
        }
        return count;
    }

    __device__ void intersect_thread(size_t warpMapHead, int countElement, T* pre_result, int y, T* result) {
        // T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        for (int i = 0; i < countElement; i++) {
            result[i] = pre_result[i] & v2[i];
        }
        return;
    }

    __device__ void intersect_warp(size_t warpMapHead, int countElement, int x, int y, T* result) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < countElement; i += WARP_SIZE) {
            result[i] =v1[i] & v2[i];
        }
        return;
    }

    __device__ void intersect_warp(size_t warpMapHead, int countElement, T* pre_result, int y, T* result) {
        // T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < countElement; i += WARP_SIZE) {
            result[i] =pre_result[i] & v2[i];
        }
        return;
    }

    __device__ int intersect_num_thread_pre(size_t warpMapHead,  int countElement, T* pre_result, int y) {
        register T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(pre_result[i] & v2[i]);
            // T result = pre_result[i] & v2[i];
            // int num1 = 0;
            // while(result) {
            //     if(result & 1)
            //       num1++;
            //     result = result >> 1;
            // }
            // num += num1;
        }
        return num;
    }

    __device__ int intersect_num_warp_pre(size_t warpMapHead,  int countElement, T* pre_result, int y) {
        register T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < countElement; i += WARP_SIZE) {
            num += __popc(pre_result[i] & v2[i]);
        }
        return num;
    }

    __device__ int intersect_thread_pre(size_t warpMapHead,  int countElement, T* pre_result, int y) {
        register T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(pre_result[i] & v2[i]);
        }
        return num;
    }

    __device__ int intersect_num_thread_reg(size_t warpMapHead,  int countElement, T* pre_result1, T* pre_result2) {
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(pre_result1[i] & pre_result2[i]);
        }
        return num;
    }

    __device__ int intersect_3num_thread(size_t warpMapHead,  int countElement, int x, int y, int z) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        T* v3 = encodeHead + warpMapHead + z * numElements;
        int num = 0;
        // int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = 0; i < countElement; i++) {
            num += __popc(v1[i] & v2[i] & v3[i]);
        }
        return num;
    }
    __device__ int intersect_3num_warp(size_t warpMapHead,  int countElement, int x, int y, int z) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        T* v3 = encodeHead + warpMapHead + z * numElements;
        int num = 0;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        for (int i = thread_lane; i < countElement; i+=WARP_SIZE) {
            num += __popc(v1[i] & v2[i] & v3[i]);
        }
        return num;
    }

    __device__ int single_intersect_num(size_t warpMapHead, int countElement,int x, int y) {
        T* v1 = encodeHead + warpMapHead + x * numElements;
        T* v2 = encodeHead + warpMapHead + y * numElements;
        int num = 0;
        for (int i = 0; i < countElement; i++) {
        // for (int i = 0; i < numElements; i++) {
            num += __popc(v1[i] & v2[i]);
            // T result = v1[i] & v2[i];
            // int num1 = 0;
            // while(result) {
            //     if(result & 1)
            //       num1++;
            //     result = result >> 1;
            // }
            // num += num1;
        }
        return num;
    }
};

template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ bool block_get_bitmap2D(T* bitmap2D, int x, int y) {
    int element = y / W;
    int bit_set = y % W;
    T mask = 1 << bit_set;
    return bitmap2D[x][element] & mask;
}

template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ bool block_get_bitmap1D(T* bitmap1D, int x) {
    int element = x / W;
    int bit_set = x % W;
    T mask = 1 << bit_set;
    return bitmap1D[element] & mask;
}

template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ void block_intersect_thread(T* bitmap2D, int countElement, int x, int y, T* result) {
    T* v1 = bitmap2D + x * BITMAP_WIDTH;
    T* v2 = bitmap2D + y * BITMAP_WIDTH;
    for (int i = 0; i < countElement; i++) {
        result[i] =v1[i] & v2[i];
    }
    return;
}

template <typename T = bitmapType, int W = BITMAP_WIDTH>
__device__ void block_intersect_thread_pre(T* bitmap2D, int countElement, T* pre_result, int y, T* result) {
    T* v2 = bitmap2D + y * BITMAP_WIDTH;
    for (int i = 0; i < countElement; i++) {
        result[i] =pre_result[i] & v2[i];
    }
    return;
}

template <typename T = bitmapType, int W = BITMAP_WIDTH>
 __device__ int block_intersect_num_thread_pre(T* bitmap2D, int countElement, T* pre_result, int y) {
    register T* v2 = bitmap2D + y * BITMAP_WIDTH;
    int num = 0;
    // int thread_lane = threadIdx.x & (WARP_SIZE-1);
    for (int i = 0; i < countElement; i++) {
        num += __popc(pre_result[i] & v2[i]);
    }
    return num;
}