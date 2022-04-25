# %%

import json
import argparse
import os

import matplotlib.pyplot as plt

from dataset import DatasetFactory


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--data-dir', type=str, default='/home/zhaoxinhao/data1/cpm1-experiments/data')
	parser.add_argument('--dataset', type=str, default='CNewSum')
	parser.add_argument('--file-name', type=str, default='train.simple.label.jsonl')
	# parser.add_argument('--is_tokenized', default=False, action='store_true')

	# tokenizer
	# parser.add_argument('--cache-path', type=str, default='/data2/private/zhaoxinhao/ModelCenter')
	# parser.add_argument('--model-config', type=str, default='cpm1-small')

	return parser.parse_args("")


# %%
args = get_args()

# %%
dataset = DatasetFactory.get_dataset(args.dataset)

file_path = os.path.join(args.data_dir, args.dataset, args.file_name)

dataset.read_data(file_path)
print(dataset.size)

plt.hist(dataset.summary_len)

# %%
plt.hist(dataset.text_len)
# %%
max(dataset.text_len)
# %%
pretokenized_dataset = DatasetFactory.get_pretokenized_dataset(args.dataset)
# %%
pretokenized_dataset.read_data("/home/zhaoxinhao/data1/cpm1-experiments/train_data/CNewSum/train.jsonl")
# %%
plt.hist(pretokenized_dataset.text_len)
# %%
plt.hist(pretokenized_dataset.summary_len)

# %%
pretokenized_dataset = DatasetFactory.get_pretokenized_dataset(args.dataset)
pretokenized_dataset.read_data("/home/zhaoxinhao/data1/cpm1-experiments/train_data/CNewSum/dev.simple.label.jsonl")

# %%
plt.hist(pretokenized_dataset.text_len)

# %%
# %%
