BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)

# python ${BASE_PATH}/code/sample_new.py \
# 	--input-file ${CPM_DATA_PATH}/CNewSum/train.simple.label.jsonl \
# 	--output-file ${CPM_DATA_PATH}/CNewSum/train.sample.jsonl \
# 	--select-num 27000
	
# python ${BASE_PATH}/code/sample_new.py \
# 	--input-file ${CPM_TRAIN_DATA_PATH}/CNewSum/train.simple.label.jsonl.900 \
# 	--output-file ${CPM_TRAIN_DATA_PATH}/CNewSum/train.sample.jsonl.900 \
# 	--select-num 27000
	
python ${BASE_PATH}/code/sample_new.py \
	--input-file ${CPM_TRAIN_DATA_PATH}/CNewSum/brio.train.simple.label.jsonl.900 \
	--output-file ${CPM_TRAIN_DATA_PATH}/CNewSum/brio.train.sample.jsonl.900 \
	--select-num 27000
