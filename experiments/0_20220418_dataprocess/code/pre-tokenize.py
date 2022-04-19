import json
import argparse
import os

from tqdm import tqdm
from model_center.tokenizer import CPM1Tokenizer
from multiprocessing import Process


class LCSTSProcess(Process):
	def __init__(self, args, num, start, end):
		super(LCSTSProcess, self).__init__()
		self.args = args
		self.num = num
		self.start_pos = start
		self.end_pos = end

	def run(self):
		args = self.args
		tokenizer = CPM1Tokenizer.from_pretrained_simple(args.model_config, cache_path=args.cache_path)
		file_path = os.path.join(args.data_dir, args.dataset, args.file_name)
		output_file_path = os.path.join(args.output_dir, args.dataset, f'{args.file_name}.{self.num}')
		with open(file_path, encoding='utf8') as fin, open(output_file_path, 'w', encoding='utf8') as fout:
			def work(i, line):
				if self.start_pos <= i and i < self.end_pos:
					line_json = json.loads(line) 
					summary = line_json['summary']
					text = line_json['text']
					lef_tokens, rig_tokens = tokenize(summary, text, tokenizer)
					result = {'lef_tokens': lef_tokens, 'rig_tokens': rig_tokens}
					fout.write(json.dumps(result, ensure_ascii=False) + '\n')

			if self.num == 0:
				for i, line in tqdm(enumerate(fin)):
					work(i, line)
			else:
				for i, line in enumerate(fin):
					work(i, line)


def get_args():
	parser = argparse.ArgumentParser()

	# process number
	parser.add_argument('--process-num', type=int, default=4)

	# input data
	parser.add_argument('--data-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')
	parser.add_argument('--dataset', type=str, default='LCSTS')
	parser.add_argument('--file-name', type=str, default='train.jsonl')

	# tokenizer
	parser.add_argument('--cache-path', type=str, default='/data2/private/zhaoxinhao/ModelCenter')
	parser.add_argument('--model-config', type=str, default='cpm1-small')

	# output
	parser.add_argument('--output-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/train_data')

	return parser.parse_args()


def tokenize(summary, text, tokenizer):
	lef_tokens = [1] + tokenizer.encode(f'“{text}”的摘要是:')
	rig_tokens = tokenizer.encode(summary) + [tokenizer.eod_id]

	return lef_tokens, rig_tokens

def main():
	args = get_args()

	file_path = os.path.join(args.data_dir, args.dataset, args.file_name)
	output_dir= os.path.join(args.output_dir, args.dataset)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	total_line_num = 0
	with open(file_path, encoding='utf8') as fin:
		for line in tqdm(fin):
			total_line_num += 1
	line_num_per_process = (total_line_num + args.process_num - 1) // args.process_num
	process_list = []
	for i in range(args.process_num):
		p = LCSTSProcess(args, i, line_num_per_process * i, line_num_per_process * (i+1))
		p.start()
		process_list.append(p)
	print("Merging files...")
	with open(os.path.join(output_dir, args.file_name), 'w', encoding='utf8') as fout:
		for i, p in enumerate(process_list):
			p.join()
			process_out_file = os.path.join(output_dir, f'{args.file_name}.{i}')
			with open(process_out_file, 'r', encoding='utf8') as fin:
				for line in tqdm(fin):
					fout.write(line)
			os.remove(process_out_file)
	


if __name__ == '__main__':
	main()
