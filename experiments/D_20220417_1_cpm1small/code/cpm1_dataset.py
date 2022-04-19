import torch
import json

import numpy as np
import bmtrain as bmt


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
					summary = line_json['summary']
					text = line_json['text']

					input_tokens, input_length, context, input_span, target, target_length = self.make_input(summary, text, max_length, tokenizer)

					self.data.append({
						"input_tokens": input_tokens,
						"input_length": input_length,
						"input_context": context,
						"input_span": input_span,
						"targets": target,
						"target_length": target_length,
					})

	def make_input(self, summary, text, max_length, tokenizer):
		lef_tokens = [1] + tokenizer.encode(f'{text}的摘要是:')
		rig_tokens = tokenizer.encode(summary) + [tokenizer.eod_id]

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
	"LCSTS": LCSTS_Dataset
}
