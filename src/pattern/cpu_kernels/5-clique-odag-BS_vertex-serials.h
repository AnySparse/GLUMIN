

void BS_vertex_5clique(Graph &g, StorageMeta &meta, uint64_t &total)
{

  uint64_t counter = 0;
  Timer t;
 t.Start();
{
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = 0; v0_idx < candidate_v0.size(); v0_idx += 1){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    for(vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx ++){
      auto v1 = candidate_v1[v1_idx];
      __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
      auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
      for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
        auto v2 = candidate_v2[v2_idx];
        __intersect(meta, __get_vlist_from_heap(g, meta, /*slot_id=*/0), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/1);
        auto candidate_v3 = __get_vlist_from_heap(g, meta, /*slot_id=*/1);
        for(vidType v3_idx = 0; v3_idx < candidate_v3.size(); v3_idx ++){
          auto v3 = candidate_v3[v3_idx];
          counter += __intersect_num(__get_vlist_from_heap(g, meta, /*slot_id=*/1), __get_vlist_from_graph(g, meta, /*vid=*/v3), /*upper_bound=*/-1);
        }
      }
    }
  }
  }
  t.Stop();
  printf("==Serial GEN-CPU:%.4f sec counter:%d\n",t.Seconds(), counter);
 
  total = counter;
}