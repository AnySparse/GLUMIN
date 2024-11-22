#pragma once
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

typedef float   feat_t;    // vertex feature type
typedef uint8_t patid_t;   // pattern id type
typedef uint8_t mask_t;    // mask type
typedef uint8_t label_t;   // label type
typedef uint8_t vlabel_t;  // vertex label type
typedef uint16_t elabel_t; // edge label type
typedef uint8_t cmap_vt;   // cmap value type
typedef int32_t vidType;   // vertex ID type
typedef int64_t eidType;   // edge ID type
typedef int32_t IndexT;
typedef uint64_t emb_index_t; // embedding index type
typedef unsigned long long AccType;

typedef std::vector<patid_t> PidList;    // pattern ID list
typedef std::vector<vidType> VertexList; // vertex ID list
typedef std::vector<std::vector<vidType>> VertexLists;
typedef std::unordered_map<vlabel_t, int> nlf_map;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define MAX_FSM_PATTERN_SIZE 5
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256 // 128
#define BLOCK_SIZE_DENSE 128
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define WARPS_PER_BLOCK_DENSE (BLOCK_SIZE_DENSE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

#define THREAD_GROUP_SIZE 32 //Set 8 or 16 or 32
#define THREAD_GROUP_PER_BLOCK (BLOCK_SIZE / THREAD_GROUP_SIZE)
#define THREAD_GROUP_PER_WARP (WARP_SIZE / THREAD_GROUP_SIZE)

// typedef uint32_t bitmapType;
#define bitmapType vidType
#define BITMAP_WIDTH 32
#define SUBTASK_SIZE 32

// worklist
#define WORKLIST_SIZE 0x48000
#define WORKLIST_THRESHOLD WORKLIST_SIZE * 0.6
#define LOCAL_SHORT_WORKLIST_SIZE (WARP_SIZE * 2)
#define LOCAL_SHORT_WORKLIST_THRESHOLD (LOCAL_SHORT_WORKLIST_SIZE * 0.8)
// #define LOCAL_BITMAP_LIST_SIZE (LOCAL_SHORT_WORKLIST_SIZE * 0.05)
#define LOCAL_BITMAP_LIST_SIZE 32

#define SPARSE_THRESHOLD 4
#define SMALL_SPLIT 1
#define SPARSE_SPLIT 2
#define DENSE_SPLIT 3

#define BLOCK_GROUP 80

enum Status {
  Idle,
  Extending,
  IteratingEdge,
  Working,
  ReWorking
};

#define OP_INTERSECT 'i'
#define OP_DIFFERENCE 'd'
extern std::map<char,double> time_ops;

const std::string long_separator = "--------------------------------------------------------------------\n";
const std::string short_separator = "-----------------------\n";
