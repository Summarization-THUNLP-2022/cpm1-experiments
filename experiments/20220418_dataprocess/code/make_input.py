import json
import argparse

import numpy as np

from model_center.tokenizer import CPM1Tokenizer


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--base-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')
	parser.add_argument('--dataset', type=str, default='LCSTS')
	parser.add_argument('--file_name', type=str, default='train.jsonl')

	# tokenizer
	parser.add_argument('--cache-path', type=str, default='/data2/private/zhaoxinhao/ModelCenter')
	parser.add_argument('--model-config', type=str, default='cpm1-small')


def main():
	args = get_args()

	tokenizer = CPM1Tokenizer.from_pretrained(args.model_config, cache_path=args.cache_path)
	
	
