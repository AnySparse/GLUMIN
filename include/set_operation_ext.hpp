#pragma once
// difference

// use source[a[i]], search b, get index
template <typename T = vidType>
T difference_set__(T *a, T size_a, T *b, T size_b, T *c)
{
  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = a[i];
    int found = 0;
    if (!binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}

template <typename T = vidType>
T difference_set__( T *a, T size_a, T *b, T size_b, T upper_bound, T *c)
{
  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}

template <typename T = vidType>
T difference_num__(T *a, T size_a, T *b, T size_b)
{
  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = a[i];
    if (!binary_search(b, key, size_b))
      count++;
  }
  return count;
}

template <typename T = vidType>
T difference_num__(T *a, T size_a, T *b, T size_b, T upper_bound)
{
  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(b, key, size_b))
      count++;
  }
  return count;
}


template <typename T = vidType>
T intersect__(T *a, T size_a, T *b, T size_b, T *c)
{
  vidType *lookup = a;
  vidType *search = b;
  vidType lookup_size = size_a;
  vidType search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  int count = 0;
  for (auto i = 0; i < lookup_size; i++)
  {
    T key = lookup[i];
    int found = 0;
    if (binary_search(search, key, search_size))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}

template <typename T = vidType>
T intersect__(T *a, T size_a, T *b, T size_b, T upper_bound, T *c)
{
  vidType *lookup = a;
  vidType *search = b;
  vidType lookup_size = size_a;
  vidType search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  int count = 0;
  for (auto i = 0; i < lookup_size; i++)
  {
    T key = lookup[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search(search, key, search_size))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}

template <typename T = vidType>
T intersect_num__(T *a, T size_a, T *b, T size_b)
{
  vidType *lookup = a;
  vidType *search = b;
  vidType lookup_size = size_a;
  vidType search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  int count = 0;
  for (auto i = 0; i < lookup_size; i++)
  {
    T key = lookup[i];
    int found = 0;
    if (binary_search(search, key, search_size))
      count++;
  }
  return count;
}

template <typename T = vidType>
T intersect_num__(T *a, T size_a, T *b, T size_b, T upper_bound)
{
  vidType *lookup = a;
  vidType *search = b;
  vidType lookup_size = size_a;
  vidType search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  int count = 0;
  for (auto i = 0; i < lookup_size; i++)
  {
    T key = lookup[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search(search, key, search_size))
      count++;
  }
  return count;
}

// use source[a[i]], search b, get index
template <typename T = vidType>
T difference_set_source1(T *source, T *a, T size_a, T *b, T size_b, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[a[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   if (!binary_search_2phase(b, cache, key, size_b))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = a[i]; // store index
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = source[a[i]];
    int found = 0;
    if (!binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}
template <typename T = vidType>
T difference_set_source1(T *source, T *a, T size_a, T *b, T size_b, T upper_bound, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[a[i]]; // each thread picks a vertex as the key
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   int found = 0;
  //   if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = a[i]; // store index
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  //   mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = source[a[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}

// use source[b[i]], search a, get vid
template <typename T = vidType>
T difference_set_source2(T *a, T size_a, T *source, T *b, T size_b, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[b[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   if (!binary_search_2phase(a, cache, key, size_a))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = key; // store vid
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_b; i++)
  {
    T key = source[b[i]];
    int found = 0;
    if (!binary_search(a, key, size_a))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}
template <typename T = vidType>
T difference_set_source2(T *a, T size_a, T *source, T *b, T size_b, T upper_bound, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[b[i]]; // each thread picks a vertex as the key
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   int found = 0;
  //   if (is_smaller && !binary_search_2phase(a, cache, key, size_a))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = key; // store vid
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  //   mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return count[warp_lane];
  int count = 0;
  for (auto i = 0; i < size_b; i++)
  {
    T key = source[b[i]];
    int found = 0;
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(a, key, size_a))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}

// use source[a[i]], search b, get count
template <typename T = vidType>
T difference_num_source1(T *source, T *a, T size_a, T *b, T size_b)
{
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   auto key = source[a[i]];
  //   if (!binary_search_2phase(b, cache, key, size_b))
  //     num += 1;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_a; i++)
  {
    auto key = source[a[i]];
    if (!binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}
template <typename T = vidType>
T difference_num_source1(T *source, T *a, T size_a, T *b, T size_b, T upper_bound)
{
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   auto key = source[a[i]];
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
  //     num += 1;
  //   unsigned active = __activemask();
  //   unsigned mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_a; i++)
  {
    auto key = source[a[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}

// use source[b[i]], search a, get count
template <typename T = vidType>
T difference_num_source2(T *a, T size_a, T *source, T *b, T size_b)
{
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   auto key = source[b[i]];
  //   if (!binary_search_2phase(a, cache, key, size_a))
  //     num += 1;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_b; i++)
  {
    auto key = source[b[i]];
    if (!binary_search(a, key, size_a))
      num += 1;
  }
  return num;
}
template <typename T = vidType>
T difference_num_source2(T *a, T size_a, T *source, T *b, T size_b, T upper_bound)
{
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   auto key = source[b[i]];
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && !binary_search_2phase(a, cache, key, size_a))
  //     num += 1;
  //   unsigned active = __activemask();
  //   unsigned mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_b; i++)
  {
    auto key = source[b[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(a, key, size_a))
      num += 1;
  }
  return num;
}

// intersection

// use source[a[i]], search b, get index
template <typename T = vidType>
T intersect_source1(T *source, T *a, T size_a, T *b, T size_b, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[a[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   if (binary_search_2phase(b, cache, key, size_b))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = a[i]; // store index
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = source[a[i]];
    int found = 0;
    if (binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}
template <typename T = vidType>
T intersect_source1(T *source, T *a, T size_a, T *b, T size_b, T upper_bound, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[a[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && binary_search_2phase(b, cache, key, size_b))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = a[i]; // store index
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  //   mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_a; i++)
  {
    T key = source[a[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search(b, key, size_b))
      found = 1;
    if (found)
      c[count++] = a[i];
  }
  return count;
}

// use source[b[i]], search a, get vid
template <typename T = vidType>
T intersect_source2(T *a, T size_a, T *source, T *b, T size_b, T *c)
{
  // if (size_a == 0) return 0;
  //  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //  __shared__ T count[WARPS_PER_BLOCK];
  //  __shared__ T cache[BLOCK_SIZE];
  //  cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[b[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   if (binary_search_2phase(a, cache, key, size_a))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = key; // store vid
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_b; i++)
  {
    T key = source[b[i]];
    int found = 0;
    if (binary_search(a, key, size_a))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}
template <typename T = vidType>
T intersect_source2(T *a, T size_a, T *source, T *b, T size_b, T upper_bound, T *c)
{
  // //if (size_a == 0) return 0;
  // int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  // __shared__ T count[WARPS_PER_BLOCK];
  // __shared__ T cache[BLOCK_SIZE];
  // cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // if (thread_lane == 0) count[warp_lane] = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   unsigned active = __activemask();
  //   __syncwarp(active);
  //   T key = source[b[i]]; // each thread picks a vertex as the key
  //   int found = 0;
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && binary_search_2phase(a, cache, key, size_a))
  //     found = 1;
  //   unsigned mask = __ballot_sync(active, found);
  //   auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
  //   if (found) c[count[warp_lane]+idx-1] = key; // store vid
  //   if (thread_lane == 0) count[warp_lane] += __popc(mask);
  //   mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return count[warp_lane];

  int count = 0;
  for (auto i = 0; i < size_b; i++)
  {
    T key = source[b[i]];
    int found = 0;
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && binary_search(a, key, size_a))
      found = 1;
    if (found)
      c[count++] = key;
  }
  return count;
}

// use source[a[i]], search b, get count
template <typename T = vidType>
T intersect_num_source1(T *source, T *a, T size_a, T *b, T size_b)
{
  // //if (size_a == 0) return 0;
  // //assert(size_b != 0);
  // int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  // __shared__ T cache[BLOCK_SIZE];
  // cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   auto key = source[a[i]];
  //   if (binary_search_2phase(b, cache, key, size_b))
  //     num += 1;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_a; i++)
  {
    auto key = source[a[i]];
    if (binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}
template <typename T = vidType>
T intersect_num_source1(T *source, T *a, T size_a, T *b, T size_b, T upper_bound)
{
  // //if (size_a == 0) return 0;
  // //assert(size_b != 0);
  // int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  // __shared__ T cache[BLOCK_SIZE];
  // cache[warp_lane * WARP_SIZE + thread_lane] = b[thread_lane * size_b / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
  //   auto key = source[a[i]];
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && binary_search_2phase(b, cache, key, size_b))
  //     num += 1;
  //   unsigned active = __activemask();
  //   unsigned mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_a; i++)
  {
    auto key = source[a[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}

// use source[b[i]], search a, get count
template <typename T = vidType>
T intersect_num_source2(T *a, T size_a, T *source, T *b, T size_b)
{
  // //if (size_a == 0) return 0;
  // //assert(size_b != 0);
  // int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  // __shared__ T cache[BLOCK_SIZE];
  // cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   auto key = source[b[i]];
  //   if (binary_search_2phase(a, cache, key, size_a))
  //     num += 1;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_b; i++)
  {
    auto key = source[b[i]];
    if (binary_search(a, key, size_a))
      num += 1;
  }
  return num;
}
template <typename T = vidType>
T intersect_num_source2(T *a, T size_a, T *source, T *b, T size_b, T upper_bound)
{
  // //if (size_a == 0) return 0;
  // //assert(size_b != 0);
  // int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  // __shared__ T cache[BLOCK_SIZE];
  // cache[warp_lane * WARP_SIZE + thread_lane] = a[thread_lane * size_a / WARP_SIZE];

  // T num = 0;
  // for (auto i = thread_lane; i < size_b; i += WARP_SIZE) {
  //   auto key = source[b[i]];
  //   int is_smaller = key < upper_bound ? 1 : 0;
  //   if (is_smaller && binary_search_2phase(a, cache, key, size_a))
  //     num += 1;
  //   unsigned active = __activemask();
  //   unsigned mask = __ballot_sync(active, is_smaller);
  //   if (mask != FULL_MASK) break;
  // }
  // return num;

  T num = 0;
  for (auto i = 0; i < size_b; i++)
  {
    auto key = source[b[i]];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && binary_search(a, key, size_a))
      num += 1;
  }
  return num;
}
