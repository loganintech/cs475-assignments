kernel
void
ArrayMultReduce( global const float *dA, global const float *dB, local float *prods, global float *dC )
{
	int gid = get_global_id( 0 );

	int work_item_id = get_local_id(0);
	int work_items = get_local_size(0);
	int wg_id = get_group_id(0);

	prods[work_item_id] = dA[gid] * dB[gid];
	for (int offset = 1; offset < work_items; offset *= 2) {
		int mask = 2*offset - 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (wg_id & mask == 0) {
			prods[work_item_id] += prods[work_item_id+offset];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if(wg_id == 0)
		dC[wg_id] = prods[0];

}
