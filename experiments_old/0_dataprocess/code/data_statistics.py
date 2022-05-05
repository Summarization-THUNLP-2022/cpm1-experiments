# %%

import json
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt

from dataset import DatasetFactory


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--data-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')
	parser.add_argument('--dataset', type=str, default='CNewSum')
	parser.add_argument('--file-name', type=str, default='dev.simple.label.jsonl')
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
print(np.average(dataset.summary_len))
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
import numpy as np
# %%
text_len = np.array(dataset.text_len)
summary_len = np.array(dataset.summary_len)
# %%
ratio = text_len / summary_len
print(ratio[:10])
# %%
plt.hist(ratio, bins=40, range=(0, 100))
# %%
np.average(ratio)
# %%
text_lens = []
with open("/home/zhaoxinhao/data2/cpm1/data/CLTS/train.src") as f:
	for line in f:
		line = ''.join(line.strip().split())
		text_lens.append(len(line))
# %%
summary_lens = []
with open("/home/zhaoxinhao/data2/cpm1/data/CLTS/train.tgt") as f:
	for line in f:
		line = ''.join(line.strip().split())
		summary_lens.append(len(line))

# %%
text_lens = np.array(text_lens)
summary_lens = np.array(summary_lens)
# %%
np.average(text_lens / summary_lens)
# %%
ratio = text_lens / summary_lens
# %%
plt.hist(ratio, bins=40, range=(0, 100))