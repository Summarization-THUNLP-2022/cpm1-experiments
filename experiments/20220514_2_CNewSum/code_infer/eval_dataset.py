from typing import Dict
import torch
import json
import numpy as np

from tqdm import tqdm
from model_center.tokenizer import CPM1Tokenizer


def tokenize(text, tokenizer, max_length):
	return tokenizer.encode(text)[:max_length]


def make_input(lef_tokens, cand_rig_tokens, max_length):
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
		max_rig_length = max_length - lef_length
		if rig_length > max_rig_length:
			rig_tokens = rig_tokens[:max_rig_length-1] + rig_tokens[-1:]
			rig_length = len(rig_tokens)
		length = lef_length + rig_length
		assert length <= max_length
		input_tokens[i][:length] = torch.tensor(lef_tokens + rig_tokens).int()
		input_length[i] = length
		target_length[i] = rig_length
		target[i][lef_length-1:length-1] = torch.tensor(rig_tokens).int()

	context[:cand_num] = torch.from_numpy(np.arange(max_length)).int()
	context = (context < lef_length) | (context >= (target_length+lef_length).unsqueeze(1))

	return {
		"input_tokens": input_tokens,
		"input_length": input_length,
		"input_context": context,
		"input_span": input_span,
		"targets": target,
		"target_length": target_length,
	}


class CNewSumEvalDataset:
	def __init__(self, tokenizer: CPM1Tokenizer, file_path: str, candidate_file_path: str, candidate_num: int):
		self.__idx = 0
		self._data = []
		self.tokenizer = tokenizer
		self.file_path = file_path
		self.candidate_num = candidate_num
		self.candidate_file_path = candidate_file_path

	def __len__(self):
		return len(self._data)

	def __iter__(self):
		self.__idx = 0
		return self

	def __next__(self):
		if self.__idx < len(self._data):
			data = self[self.__idx]
			self.__idx += 1
			return data
		else:
			raise StopIteration

	def __getitem__(self, key) -> Dict[str, torch.Tensor]:
		tokenizer = self.tokenizer
		data = self._data[key]
		lef_tokens = data['text']
		cand_rig_tokens = []
		for candidate in data['candidate']:
			rig_tokens = tokenizer.encode(candidate)[:100] + [tokenizer.eod_id]
			cand_rig_tokens.append(rig_tokens)
		return make_input(lef_tokens, cand_rig_tokens, 1024)

	def read_dataset(self, start=0, end=None):
		self._data = []
		with open(self.file_path, 'r') as f:
			for i, line in tqdm(enumerate(f)):
				if i < start or (end is not None and i >= end):
					continue
				line_json = json.loads(line)
				text = line_json['lef_tokens']
				self._data.append({"text": text})

		with open(self.candidate_file_path, 'r') as f:
			for i, line in tqdm(enumerate(f)):
				if i < start * self.candidate_num or (end is not None and i >= end * self.candidate_num):
					continue
				if i % self.candidate_num == 0:
					self._data[i // 16 - start]['candidate'] = [line.strip()]
				else:
					self._data[i // 16 - start]['candidate'].append(line.strip())

