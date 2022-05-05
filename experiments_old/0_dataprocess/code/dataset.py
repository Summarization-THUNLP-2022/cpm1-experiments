import json


from abc import ABC, abstractmethod
from tqdm import tqdm


class Dataset(ABC):
	def __init__(self):
		self._data = []
		self._size = 0
		self._text_len = []
		self._summary_len = []

	@abstractmethod
	def read_data(self, file_path):
		pass

	@property
	def size(self):
		if not self._size:
			self._size = len(self._data)
		return self._size
	
	@property
	def text_len(self):
		if not self._text_len:
			for d in self._data:
				self._text_len.append(len(d['text']))
		return self._text_len

	@property
	def summary_len(self):
		if not self._summary_len:
			for d in self._data:
				self._summary_len.append(len(d['summary']))
		return self._summary_len
	

class LCSTSDataset(Dataset):
	def read_data(self, file_path):
		with open(file_path, 'r', encoding='utf8') as f:
			for line in tqdm(f):
				line_data = json.loads(line)
				summary = line_data['summary']
				text = line_data['text']
				self._data.append({'summary': summary, 'text': text})
	

class CNewSumDataset(Dataset):
	def read_data(self, file_path):
		with open(file_path, 'r', encoding='utf8') as f:
			for line in tqdm(f):
				line_data = json.loads(line)
				summary = line_data['summary'].replace(' ', '')
				text = ''.join(line_data['article']).replace(' ', '')
				self._data.append({'summary': summary, 'text': text})


class PretokenizedDataset(Dataset):
	def read_data(self, file_path):
		with open(file_path, 'r', encoding='utf8') as f:
			for line in tqdm(f):
				line_data = json.loads(line)
				summary = line_data['rig_tokens']
				text = line_data['lef_tokens']
				self._data.append({'summary':summary, 'text':text})


class DatasetFactory:
	__DATASET__ = {
		"LCSTS": LCSTSDataset,
		"CNewSum": CNewSumDataset,
	}

	@classmethod
	def get_dataset(cls, dataset_name) -> Dataset:
		return cls.__DATASET__.get(dataset_name, None)()
	
	@classmethod
	def get_pretokenized_dataset(cls, dataset_name) -> PretokenizedDataset:
		return PretokenizedDataset()
