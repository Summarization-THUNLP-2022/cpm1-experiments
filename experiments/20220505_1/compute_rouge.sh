BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
TEST_FILE="train.simple.label.jsonl.900"
FILE_NAME="6-0.jsonl.sorted.txt"

INPUT_FILE=${BASE_PATH}/infer_results/${TEST_FILE}/${FILE_NAME}
REFERENCE_FILE="/home/zhaoxinhao/data2/cpm1/reference_data/CNewSum/train.txt"
OUTPUT_FILE=${BASE_PATH}/result_for_brio/${FILE_NAME}

python ${BASE_PATH}/code_infer/compute_rouge.py \
	${INPUT_FILE} \
	${REFERENCE_FILE} \
	16 \
	${OUTPUT_FILE} \
	100
