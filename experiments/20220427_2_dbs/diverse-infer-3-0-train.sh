#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13583
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
DATASET="CNewSum"
INPUT_FILE="train.simple.label.jsonl"
MODEL_CONFIG_DIR=${CPM_CACHE_PATH}/cpm1-small
EPOCH=3
CKPT_STEPS=0
OUTPUT_FILE=${BASE_PATH}/infer_results/${INPUT_FILE}/${EPOCH}-${CKPT_STEPS}.jsonl

if [ ! -d ${BASE_PATH}/infer_results ]; then
    mkdir ${BASE_PATH}/infer_results
fi

if [ ! -d ${BASE_PATH}/infer_results/${INPUT_FILE} ]; then
    mkdir ${BASE_PATH}/infer_results/${INPUT_FILE}
fi

OPTS=""
OPTS+=" --max-length 1024"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt-${EPOCH}-${CKPT_STEPS}.pt"
OPTS+=" --input-file ${CPM_DATA_PATH}/${DATASET}/${INPUT_FILE}"
OPTS+=" --output-file ${OUTPUT_FILE}"
OPTS+=" --span-length 60"
OPTS+=" --no-repeat-ngram-size 0"
OPTS+=" --repetition-penalty 1"
OPTS+=" --batch-size 2"
OPTS+=" --beam-size 16"
OPTS+=" --beam-group 4"
OPTS+=" --diverse-penalty 0.5"
OPTS+=" --no-repeat-ngram-size 0"
OPTS+=" --repetition-penalty 1.5"
# OPTS+=" --random-sample"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/diverse_infer.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/infer_results/${INPUT_FILE}/infer-${EPOCH}-${CKPT_STEPS}.log

cat ${OUTPUT_FILE}.* > ${OUTPUT_FILE}
rm ${OUTPUT_FILE}.*
