import sys

import multiprocessing as mp

from typing import List
from rouge import Rouge
from tqdm import tqdm


def sentence_process(sentence):
    # 按字分词
    return ' '.join(sentence)


def get_rouge(pid, candidate_list: List[str], reference_list: List[str], rouge_name: str = 'rouge-l', item_name: str = 'f') -> List[float]:
    r = Rouge()
    assert len(candidate_list) == len(reference_list)
    scores = []

    def work(candidate, reference):
        candidate = sentence_process(candidate)
        reference = sentence_process(reference)
        score = r.get_scores(refs=reference, hyps=candidate)[0][rouge_name][item_name]
        scores.append(score)

    if pid == 0:
        for candidate, reference in tqdm(zip(candidate_list, reference_list)):
            work(candidate, reference)
    else:
        for candidate, reference in zip(candidate_list, reference_list):
            work(candidate, reference)

    return scores


def main():
    argv = sys.argv
    print(f"预测结果：{argv[1]}, 测试集: {argv[2]}, 重复次数：{argv[3]}, 输出文件：{argv[4]}")
    repeat_times = int(argv[3])
    candidates = []
    with open(argv[1]) as f:
        for line in tqdm(f):
            candidates.append(line.strip())
    references = []
    with open(argv[2]) as f:
        for line in tqdm(f):
            if line.strip() == "":
                continue
            for i in range(int(argv[3])):
                references.append(line.strip())

    assert len(candidates) == len(references)

    process_num = 32
    pool = mp.Pool(processes=process_num)

    print("compute rouge now")
    line_per_process = (len(candidates) + process_num - 1) // process_num
    tasks = [(i, candidates[i*line_per_process:(i+1)*line_per_process], references[i*line_per_process:(i+1)*line_per_process]) for i in range(process_num)]
    result = pool.starmap(get_rouge, tasks)
    all_scores = [score for scores in result for score in scores]

    pool.close()

    print("output now")
    with open(argv[4], 'w') as fout:
        for i in tqdm(range(len(candidates))):
            fout.write(candidates[i] + '\t' + str(all_scores[i]) + '\n')
    
    print("sort now")
    with open(f'{argv[4]}.sorted', 'w') as fout:
        for i in tqdm(range(len(references))):
            candidates_now = candidates[i * repeat_times: (i+1) * repeat_times]
            scores_now = all_scores[i * repeat_times: (i+1) * repeat_times]
            candidate_score_list = sorted(list(zip(candidates_now, scores_now)), key=lambda x: x[1], reverse=True)
            for candidate_score in candidate_score_list:
                fout.write(candidate_score[0] + '\n')


if __name__ == '__main__':
    main()
