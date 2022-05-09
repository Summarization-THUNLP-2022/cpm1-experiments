BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

INPUT_FILE="${CPM_TRAIN_DATA_PATH}/CNewSum/train.simple.label.jsonl.900"
CANDIDATES_FILE="${CPM_BASE_PATH}/experiments/20220505_1/result_for_brio/6-0.jsonl.sorted.txt.sorted"
OUTPUT_FILE="${CPM_TRAIN_DATA_PATH}/CNewSum/brio.train.simple.label.jsonl.900"

python ${BASE_PATH}/code_data/pretokenize.py \
	${INPUT_FILE} \
	${CANDIDATES_FILE} \
	16 \
	${OUTPUT_FILE}
