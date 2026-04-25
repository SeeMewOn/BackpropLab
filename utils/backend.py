from utils.config import USE_GPU

if USE_GPU:
	import cupy as np
	from cupy.lib.stride_tricks import as_strided
else:
	import numpy as np
	from numpy.lib.stride_tricks import as_strided
