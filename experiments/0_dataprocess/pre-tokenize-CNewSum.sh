BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

python $BASE_PATH/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset CNewSum \
	--file-name dev.simple.label.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH \
	--max_length 1800

python $BASE_PATH/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset CNewSum \
	--file-name train.simple.label.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH \
	--max_length 1800
