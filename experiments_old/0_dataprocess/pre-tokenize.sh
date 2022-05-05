BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

python $BASE_PATH/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name dev.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH

python $BASE_PATH/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name train.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH

python $BASE_PATH/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name dev.jsonl.dedup \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH
	