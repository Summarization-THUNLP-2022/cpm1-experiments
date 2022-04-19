python $CPM_BASE_PATH/experiments/0_20220418_dataprocess/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name dev.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH

python $CPM_BASE_PATH/experiments/0_20220418_dataprocess/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name train.jsonl \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH

python $CPM_BASE_PATH/experiments/0_20220418_dataprocess/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir $CPM_DATA_PATH \
	--dataset LCSTS \
	--file-name dev.jsonl.dedup \
	--cache-path $CPM_CACHE_PATH \
	--model-config cpm1-small \
	--output-dir $CPM_TRAIN_DATA_PATH
	