import argparse
import os
import random
import shutil

import numpy as np

from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--input-file', type=str, default='/home/zhaoxinhao/data2/cpm1/data/CNewSum/train.simple.label.jsonl')
	parser.add_argument('--output-file', type=str, default='/home/zhaoxinhao/data2/cpm1/data/CNewSum/train.sample.jsonl')
	parser.add_argument('--select-num', type=int, default=27000)
	parser.add_argument('--candidate-num', type=int, default=1)
	parser.add_argument('--seed', type=int, default=0)

	return parser.parse_args()


def main():
	args = get_args()
	random.seed(args.seed)

	if os.path.isfile(args.input_file):
		data = []
		with open(args.input_file, encoding='utf8') as fin, open(args.output_file, 'w') as fout:
			for i, line in tqdm(enumerate(fin)):
				data.append(line)
			
			num_list = list(range(len(data)))
			select_nums = random.sample(num_list, args.select_num)
			for num in select_nums:
				for i in range(args.candidate_num):
					fout.write(data[num * args.candidate_num + i])
	elif os.path.isdir(args.input_file):
		file_list = os.listdir(args.input_file)
		num_list = list(range(len(file_list)))
		select_nums = random.sample(num_list, args.select_num)
		for i, num in tqdm(enumerate(select_nums)):
			shutil.copy2(os.path.join(args.input_file, f"{num}.json"), os.path.join(args.output_file, f"{i}.json"))
		


if __name__ == '__main__':
	main()
