BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

EPOCH=${1}
REFERENCE_FILE=${CPM_DATA_PATH}/LCSTS/test_private.jsonl.dedup
TEST_FILE=${BASE_PATH}/infer_results/test_private.jsonl.dedup.900/${EPOCH}-0-LP1-RP1-NP0.jsonl.sorted.txt

python ${BASE_PATH}/code_infer/eval.py ${TEST_FILE} ${REFERENCE_FILE} > ${TEST_FILE}.rouge
