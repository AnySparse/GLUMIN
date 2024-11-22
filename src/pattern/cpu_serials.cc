
#include "graph.h"
#include "codegen_LUT.hpp"
#include "codegen_utils.hpp"
#include "5-clique-odag-BS_vertex-serials.h"
#include "5-clique-odag-LUT-serials.h"
#include "4-star-LUT-serials.h"
#include "4-star-BS_vertex-serials.h"

#include "5-star-BSV.yml.h"
#include "5-star-BSE.yml.h"

#include "5-tailed-star-BSE.yml.h"
#include "5-tailed-star-BSV.yml.h"

#include "5-halfsolid-house-BSE.yml.h"
#include "5-halfsolid-house-BSV.yml.h"

#include "5-half-house-BSE.yml.h"
#include "5-half-house-BSV.yml.h"

#include "5-anti-tailed-diamond-BSE.yml.h"
#include "5-anti-tailed-diamond-BSV.yml.h"

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP " << k << "-listing (" << num_threads << " threads)\n";
  
  g.init_edgelist();

  auto md = g.get_max_degree();
  int n_lists = 10;
  int n_bitmaps = 1;
  vidType* bitmaps = (vidType* ) malloc( n_bitmaps*((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType));
  vidType* vlists = (vidType* ) malloc( n_lists* size_t(md) * sizeof(vidType));

  vidType* list_size = (vidType*) malloc(sizeof(vidType)* n_lists);
  vidType* bitmap_size = (vidType*) malloc(sizeof(vidType) * n_bitmaps);

  int max_size = std::max(10000, md);
  LUTManager<> lut_manager(1, max_size, max_size,  false);
  StorageMeta meta;
  meta.lut = lut_manager.getEmptyLUT(0);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = g.V();
  meta.slot_size = md;
  meta.bitmap_size = (md + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 2;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = 0;
  meta.local_warp_id = 0;

  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
  //LUT_5clique(g, meta, accum[0]);
  //cmap_kclique(g, k, total);
  
  
  if(k==1)
    LUT_4star(g, meta, accum[0]);
  else if(k==2)
    BS_vertex_4star(g, meta, accum[0]);
  else if(k==3)
    LUT_5star(g, meta, accum[0]);
  else if(k==4)
    BS_vertex_5star(g, meta, accum[0]);
  else if(k==5)
    LUT_5tailed_star(g, meta, accum[0]);
  else if(k==6)
    BS_vertex_5tailed_star(g, meta, accum[0]);
  else if(k==7)
    LUT_5half_house(g, meta, accum[0]);
  else if(k==8)
    BS_vertex_5half_house(g, meta, accum[0]);
  else if(k==9)
    LUT_5halfsolid_house(g, meta, accum[0]);
  else if(k==10)
    BS_vertex_5harfsolid_house(g, meta, accum[0]);
  else if(k==11)
    LUT_5anti_tailed_diamond(g, meta, accum[0]);
  else if(k==12) 
    BS_vertex_5anti_tailed_diamond(g, meta, accum[0]);
  
  double run_time = omp_get_wtime() - start_time;
  t.Stop();
  std::cout << "runtime [omp_base] = " << run_time << " sec  " << accum[0] <<"\n";
  return;
}

  