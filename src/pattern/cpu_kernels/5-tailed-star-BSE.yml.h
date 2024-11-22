
void BS_vertex_5tailed_star(Graph &g, StorageMeta &meta, uint64_t &total){

  uint64_t counter = 0;

  // BEGIN OF CODEGEN
  for(eidType eid = 0; eid < g.E(); eid += 1){
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v1), __get_vlist_from_graph(g, meta, /*vid=*/v0), /*upper_bound=*/-1, /*output_slot=*/0);
    __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/1);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
      auto v2 = candidate_v2[v2_idx];
      __difference(meta, __get_vlist_from_heap(g, meta, /*slot_id=*/1), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1, /*output_slot=*/2);
      auto candidate_v3 = __get_vlist_from_heap(g, meta, /*slot_id=*/2);
      for(vidType v3_idx = 0; v3_idx < candidate_v3.size(); v3_idx ++){
        auto v3 = candidate_v3[v3_idx];
        counter += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/2), __get_vlist_from_graph(g, meta, /*vid=*/v3), /*upper_bound=*/v3);
      }
    }
  }
  // END OF CODEGEN

  total = counter;
}
