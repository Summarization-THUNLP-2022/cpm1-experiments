python experiments/20220418_dataprocess/code/pre-tokenize.py \
	--process-num 16 \
	--data-dir /home/zhaoxinhao/data2/cpm1/data \
	--dataset LCSTS \
	--file-name dev.jsonl \
	--cache-path /data2/private/zhaoxinhao/ModelCenter \
	--model-config cpm1-small \
	--output-dir /home/zhaoxinhao/data2/cpm1/train_data

python experiments/20220418_dataprocess/code/pre-tokenize.py \
	--process-num 32 \
	--data-dir /home/zhaoxinhao/data2/cpm1/data \
	--dataset LCSTS \
	--file-name train.jsonl \
	--cache-path /data2/private/zhaoxinhao/ModelCenter \
	--model-config cpm1-small \
	--output-dir /home/zhaoxinhao/data2/cpm1/train_data
	