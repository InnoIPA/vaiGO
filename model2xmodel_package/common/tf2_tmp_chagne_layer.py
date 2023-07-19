#!/usr/bin/python3

import xir
g = xir.Graph.deserialize("quantize_model.xmodel")
ops = g.toposort()
for op in ops:
	if op.get_name() == "quant_max_pooling2d":
		replace_pool = g.create_op("quant_max_pooling2d_0", op.get_type(), op.get_attrs(), op.get_input_ops())
		[succ.replace_input_ops(op, replace_pool) for succ in op.get_fanout_ops()]
		g.remove_op(op)
g.serialize("renamed_quantize_model.xmodel")
