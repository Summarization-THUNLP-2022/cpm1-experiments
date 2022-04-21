import torch
import json


import numpy as np
import bmtrain as bmt


class LCSTS_BRIO_Dataset(torch.utils.data.Dataset):
	def __init__(self, path, split, rank, world_size, tokenizer, max_length) -> None:
		pass
