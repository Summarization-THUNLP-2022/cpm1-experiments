BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
TEST_FILE="test.simple.label.jsonl.900"
EPOCH=2
STEPS=0


INPUT_FILE=${BASE_PATH}/infer_results/${TEST_FILE}/${EPOCH}-${STEPS}.jsonl
OUTPUT_FILE=${BASE_PATH}/infer_results/${TEST_FILE}/${EPOCH}-${STEPS}.sorted.txt

python ${BASE_PATH}/code/sort_prediction.py \
	--file_path ${INPUT_FILE} \
	--output_file_path ${OUTPUT_FILE}
