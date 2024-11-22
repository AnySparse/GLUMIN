

void BS_vertex_4star(Graph &g, StorageMeta &meta, uint64_t &total)
{

  uint64_t counter = 0;
  Timer t;

  t.Start();
   auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = 0; v0_idx < candidate_v0.size(); v0_idx += 1){
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    for(vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx ++){
      auto v1 = candidate_v1[v1_idx];
      __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/v1, /*output_slot=*/0);
      auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
      for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
        auto v2 = candidate_v2[v2_idx];
        counter += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/0), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v2);
      }
    }
    //printf("counter:%ld\n",counter);
    //if(v0_idx>=10) break;
  }
  t.Stop();
  printf("==GEN-BASE:%.4f sec counter:%ld\n", t.Seconds(), counter);

  total = counter;
}