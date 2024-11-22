

void LUT_4star(Graph &g, StorageMeta &meta, uint64_t &total)
{

  uint64_t counter = 0;
  Timer t;

  t.Start();
   auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  for(vidType v0_idx = 0; v0_idx < candidate_v0.size(); v0_idx += 1){ /*add begin end!!!*/
    auto v0 = candidate_v0[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
    // /*add condition!!!*/
    __build_LUT(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    for (vidType v1_idx = 0; v1_idx < candidate_v1.size(); v1_idx += 1) {
      auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
      __build_index_from_vmap(g, meta, __get_vmap_from_lut_vid_limit(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/v1), /*slot_id=*/1);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
      for(vidType v2_idx_idx = 0; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += 1){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        counter += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
      }
    }
    // printf("counter:%ld\n",counter);
    // if(v0_idx>=10) break;
  }
  t.Stop();
  printf("==Serial LUT:%.4f sec counter:%ld\n", t.Seconds(), counter);

  total = counter;
}