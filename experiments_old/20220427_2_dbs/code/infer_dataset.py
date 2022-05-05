import json
import torch

import bmtrain as bmt

from model_center.tokenizer import CPM1Tokenizer
from tqdm import tqdm
from typing import Tuple


class InferDataset:
	def __init__(self):
		self.__idx = 0
		self._data = []

	def __len__(self):
		return len(self._data)

	def __iter__(self):
		self.__idx = 0
		return self

	def __next__(self):
		if self.__idx < len(self._data):
			data = self._data[self.__idx]
			self.__idx += 1
			return data['text'], data['summary']
		else:
			raise StopIteration

	def __getitem__(self, key) -> Tuple[str, str]:
		if key >= len(self._data):
			return self._data[0]['text'], self._data[0]['summary']
		return	self._data[key]['text'], self._data[key]['summary']


class LCSTSInferDataset(InferDataset):
	def __init__(self, file_path):
		super().__init__()
		self.__file_path = file_path
		self.total_length = 0
		with open(self.file_path, 'r') as f:
			for line in f:
				self.total_length += 1

	def read_dataset(self, start, end):
		with open(self.__file_path, 'r') as f:
			for i, line in tqdm(enumerate(f)):
				if i < start or i >= end:
					continue
				line_data = json.loads(line)
				summary = line_data['summary']
				text = line_data['text']
				self._data.append({'summary': summary, 'text': text})


class CNewSumInferDataset(InferDataset):
	def __init__(self, file_path, max_length=1024):
		super().__init__()
		self.__file_path = file_path
		self.__max_length = max_length
		self.total_length = 0
		with open(self.__file_path, 'r') as f:
			for line in f:
				self.total_length += 1

	def read_dataset(self, start, end):
		with open(self.__file_path, 'r') as f:
			for i, line in tqdm(enumerate(f)):
				if i < start or i >= end:
					continue
				line_data = json.loads(line)
				text = ''.join(line_data['article']).replace(' ', '')[:self.__max_length]
				summary = line_data['summary'].replace(' ', '')
				self._data.append({'text':text, 'summary':summary})


class BatchInferDataset:
	def __init__(self, dataset: InferDataset, tokenizer: CPM1Tokenizer, span_length: int, batch_size: int, total_step: int):
		self.__dataset = dataset
		self.__tokenizer = tokenizer
		self.__span_length = span_length
		self.__batch_size = batch_size
		self.__total_step = total_step
		self.__valid_step = (len(self.__dataset) + self.__batch_size - 1) // self.__batch_size
		self.__idx = 0

	def make_input(self, start, end):
		lef_tokens = []
		for i in range(start, end):
			text, summary = self.__dataset[i]
			lef_tokens.append([1] + self.__tokenizer.encode(f'“{text}”的摘要是:'))
		source_length = max([len(tokens) for tokens in lef_tokens])
		total_length = source_length + self.__span_length
		input_tokens = torch.zeros((self.__batch_size, total_length), dtype=torch.int32)
		input_span = torch.zeros((self.__batch_size, total_length), dtype=torch.int32)
		context = torch.zeros((self.__batch_size, total_length), dtype=bool)
		for i in range(end-start):
			pad_length = source_length - len(lef_tokens[i])
			input_tokens[i, pad_length:source_length] = torch.Tensor(lef_tokens[i]).int()
			input_span[i, pad_length:] = 1
			context[i, pad_length:source_length] = 1
		return {
			"input_tokens": input_tokens,
			"input_span": input_span,
			"context": context,
			"source_length": source_length
		}

	def __len__(self):
		return self.__total_step

	def __iter__(self):
		self.__idx = 0
		return self

	def __next__(self):
		if self.__idx < self.__total_step:
			if self.__idx < self.__valid_step:
				start = self.__idx * self.__batch_size
				end = min(start + self.__batch_size, len(self.__dataset))
				valid = True
				self.__idx += 1
			else:
				start = 0
				end = 1
				valid = False
			input_dict = self.make_input(start, end)
			input_dict['valid'] = valid
			return input_dict
		else:
			raise StopIteration


INFER_DATASET = {
	'LCSTS': LCSTSInferDataset,
	'CNewSum': CNewSumInferDataset
}
