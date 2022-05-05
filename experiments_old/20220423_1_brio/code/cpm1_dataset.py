import torch
import json

import numpy as np
import bmtrain as bmt


class LCSTS_Brio_Dataset(torch.utils.data.Dataset):
	def __init__(self, path, split, rank, world_size, tokenizer, max_length) -> None:
		self.data = []
		path = f"{path}/LCSTS_Brio/{split}.jsonl"
		bmt.print_rank(f"Start loading dataset {path}")
		if split == 'test':
			pass
		else:
			with open(path, encoding='utf8') as fin:
				for i, line in enumerate(fin):
					if i % 5000 == 0:
						bmt.print_rank(i)
					line_json = json.loads(line)
					lef_tokens = line_json['lef_tokens']
					rig_tokens = line_json['rig_tokens']
					cand_rig_tokens:list = line_json['cand_rig_tokens']
					cand_rig_tokens.insert(0, rig_tokens)

					input_tokens, input_length, context, input_span, target, target_length = self.make_input(lef_tokens, cand_rig_tokens, max_length, tokenizer)

					self.data.append({
						"input_tokens": input_tokens,
						"input_length": input_length,
						"input_context": context,
						"input_span": input_span,
						"targets": target,
						"target_length": target_length,
					})

	def make_input(self, lef_tokens, cand_rig_tokens, max_length, tokenizer):
		lef_length = len(lef_tokens)
		cand_num = len(cand_rig_tokens)
		
		input_tokens = torch.zeros((cand_num, max_length), dtype=torch.int32)
		input_length = torch.zeros((cand_num,), dtype=torch.int32)
		context = torch.zeros((cand_num, max_length), dtype=torch.int32)
		input_span = torch.zeros((cand_num, max_length), dtype=torch.int32)
		target = -100 * torch.ones((cand_num, max_length), dtype=torch.int32)
		target_length = torch.zeros((cand_num,), dtype=torch.int32)

		for i in range(len(cand_rig_tokens)):
			rig_tokens = cand_rig_tokens[i]
			rig_length = len(rig_tokens)
			length = lef_length + rig_length
			assert length < max_length
			input_tokens[i][:length] = torch.tensor(lef_tokens + rig_tokens).int()
			input_length[i] = length
			target_length[i] = rig_length
			target[i][lef_length-1:length-1] = torch.tensor(rig_tokens).int()

		context[:cand_num] = torch.from_numpy(np.arange(max_length)).int()
		context = (context < lef_length) | (context >= (target_length+lef_length).unsqueeze(1))

		return input_tokens, input_length, context, input_span, target, target_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]



class LCSTS_Dataset(torch.utils.data.Dataset):
	def __init__(self, path, split, rank, world_size, tokenizer, max_length) -> None:
		self.data = []
		path = f"{path}/LCSTS/{split}.jsonl"
		bmt.print_rank(f"Start loading dataset {path}")
		if split == 'test':
			pass
		else:
			with open(path, encoding='utf8') as fin:
				for i, line in enumerate(fin):
					if i % 5000 == 0:
						bmt.print_rank(i)
					line_json = json.loads(line)
					lef_tokens = line_json['lef_tokens']
					rig_tokens = line_json['rig_tokens']

					input_tokens, input_length, context, input_span, target, target_length = self.make_input(lef_tokens, rig_tokens, max_length, tokenizer)

					self.data.append({
						"input_tokens": input_tokens,
						"input_length": input_length,
						"input_context": context,
						"input_span": input_span,
						"targets": target,
						"target_length": target_length,
					})

	def make_input(self, lef_tokens, rig_tokens, max_length, tokenizer):
		lef_length = len(lef_tokens)
		rig_length = len(rig_tokens)

		input = lef_tokens + rig_tokens

		length = len(input)

		assert length < max_length

		input_tokens = torch.zeros((max_length,), dtype=torch.int32)
		input_tokens[:length] = torch.tensor(input).int()

		input_length = torch.tensor(length, dtype=torch.int32)
		target_length = torch.tensor(rig_length, dtype=torch.int32)
		
		context = np.arange(max_length)
		context = (context < lef_length) | (context >= length)
		context = torch.from_numpy(context).bool()

		target = np.full((max_length,), -100)
		target[lef_length-1:length-1] = rig_tokens
		target = torch.from_numpy(target).int()

		input_span = torch.zeros((max_length,), dtype=torch.int32)

		return input_tokens, input_length, context, input_span, target, target_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


DATASET = {
	"LCSTS": LCSTS_Dataset,
	"LCSTS_train": LCSTS_Brio_Dataset
}
