#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13585
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

export CUDA_VISIBLE_DEVICES=0,1,2,3

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
DATASET="CNewSum"
INPUT_FILE="dev.simple.label.jsonl.900"
CANDIDATE_FILE="/home/zhaoxinhao/data2/cpm1/experiments/20220505_1_CNewSum/infer_results/dev.simple.label.jsonl.900/diverse-6-0.jsonl.sorted.txt"
MODEL_CONFIG_DIR=${CPM_CACHE_PATH}/cpm1-small
EPOCH=6
CKPT_STEPS=0
LENGTH_PENALTY=2
OUTPUT_FILE=${BASE_PATH}/eval_results/${INPUT_FILE}/${EPOCH}-${CKPT_STEPS}.jsonl

if [ ! -d ${BASE_PATH}/eval_results ]; then
    mkdir ${BASE_PATH}/eval_results
fi

if [ ! -d ${BASE_PATH}/eval_results/${INPUT_FILE} ]; then
    mkdir ${BASE_PATH}/eval_results/${INPUT_FILE}
fi

OPTS=""
OPTS+=" --max-length 1024"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt-${EPOCH}-${CKPT_STEPS}.pt"
OPTS+=" --input-file ${CPM_TRAIN_DATA_PATH}/${DATASET}/${INPUT_FILE}"
OPTS+=" --output-file ${OUTPUT_FILE}"
OPTS+=" --length-penalty ${LENGTH_PENALTY}"
OPTS+=" --candidate-file ${CANDIDATE_FILE}"
OPTS+=" --candidate-num 16"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code_infer/eval.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/eval_results/${INPUT_FILE}/eval-${EPOCH}-${CKPT_STEPS}.log

cat ${OUTPUT_FILE}.* > ${OUTPUT_FILE}
rm ${OUTPUT_FILE}.*
