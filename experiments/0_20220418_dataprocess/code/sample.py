import json
import argparse
import os
import random

import numpy as np

from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--data-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')
	parser.add_argument('--dataset', type=str, default='LCSTS')
	parser.add_argument('--file-name', type=str, default='dev.jsonl')
	parser.add_argument('--select-num', type=int, default=500)

	return parser.parse_args()


def main():
	args = get_args()
	random.seed(0)

	file_dir = os.path.join(args.data_dir, args.dataset)
	output_file = os.path.join(file_dir, f'{args.file_name}.{args.select_num}')

	data = []
	with open(os.path.join(file_dir, args.file_name), encoding='utf8') as fin, open(output_file, 'w') as fout:
		for i, line in tqdm(enumerate(fin)):
			data.append(line)
		
		data = random.sample(data, args.select_num)
		for line in data:
			fout.write(line)


if __name__ == '__main__':
	main()
