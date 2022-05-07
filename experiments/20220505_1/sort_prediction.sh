BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
TEST_FILE="test.simple.label.jsonl.900"
FILE_NAME="6-0-LP1-RP1-NP0.jsonl"


INPUT_FILE=${BASE_PATH}/infer_results/${TEST_FILE}/${FILE_NAME}
OUTPUT_FILE=${BASE_PATH}/infer_results/${TEST_FILE}/${FILE_NAME}.sorted.txt

python ${BASE_PATH}/code_infer/sort_prediction.py \
	--file_path ${INPUT_FILE} \
	--output_file_path ${OUTPUT_FILE}
