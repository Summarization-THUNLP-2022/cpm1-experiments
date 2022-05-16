import json
import multiprocessing as mp
import os
import sys

from tqdm import tqdm
from model_center.tokenizer import CPM1Tokenizer


def tokenize_list(pid, candidates):
	tokenizer = CPM1Tokenizer.from_pretrained_simple('cpm1-small', os.environ['CPM_CACHE_PATH'])

	def work(candidate):
		return tokenizer.encode(candidate) + [tokenizer.eod_id]
	
	tokenized_list = []
	if pid == 0:
		for candidate in tqdm(candidates):
			tokenized_list.append(work(candidate))
	else:
		for candidate in candidates:
			tokenized_list.append(work(candidate))
	
	return tokenized_list


def main():
	argv = sys.argv
	print(f"原始tokenize文件：{argv[1]}，candidates文件：{argv[2]}，重复次数：{argv[3]}，输出文件：{argv[4]}")
	repeat_times = int(argv[3])
	print("load original tokenized file")
	all_data = []
	with open(argv[1]) as fin:
		for line in tqdm(fin):
			all_data.append(json.loads(line))
	print("load candidates")
	candidates = []
	with open(argv[2]) as fin:
		for line in tqdm(fin):
			candidates.append(line.strip())
	assert len(candidates) == len(all_data) * repeat_times

	process_num = 16
	pool = mp.Pool(processes=process_num)

	print("tokenize")
	line_per_process = (len(candidates) + process_num - 1) // process_num
	tasks = [(i, candidates[i*line_per_process:(i+1)*line_per_process]) for i in range(process_num)]
	result = pool.starmap(tokenize_list, tasks)
	all_tokenized_candidates = [candidate for candidates in result for candidate in candidates]
	pool.close()
	assert len(all_tokenized_candidates) == len(candidates)

	print("save now")
	if not os.path.exists(argv[4]):
		os.makedirs(argv[4])
	for i in tqdm(range(len(all_data))):
		with open(os.path.join(argv[4], f'{i}.json'), 'w') as fout:
			all_data[i]['cand_rig_tokens'] = all_tokenized_candidates[i * repeat_times: (i+1) * repeat_times]
			json.dump(all_data[i], fout)


if __name__ == '__main__':
	main()
