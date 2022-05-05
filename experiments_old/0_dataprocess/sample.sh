BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

python $BASE_PATH/code/sample.py \
	--data-dir $CPM_DATA_PATH \
	--file-name dev.jsonl.dedup