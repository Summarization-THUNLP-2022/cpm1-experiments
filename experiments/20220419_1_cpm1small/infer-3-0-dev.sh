#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13583
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
MODEL_CONFIG_DIR=${CPM_CACHE_PATH}/cpm1-small
EPOCH=3
CKPT_STEPS=0

if [ ! -d ${BASE_PATH}/infer_results ]; then
    mkdir ${BASE_PATH}/infer_results
fi

OPTS=""
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt-${EPOCH}-${CKPT_STEPS}.pt"
OPTS+=" --input-file ${CPM_DATA_PATH}/LCSTS/dev.jsonl.dedup.500"
OPTS+=" --output-file ${BASE_PATH}/infer_results/infer-${EPOCH}-${CKPT_STEPS}.jsonl"
OPTS+=" --span-length 40"
OPTS+=" --temperature 1"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0"
OPTS+=" --no-repeat-ngram-size 0"
OPTS+=" --repetition-penalty 1"
OPTS+=" --beam-size 5"
# OPTS+=" --random-sample"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/infer_lcsts.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/infer_results/infer-${EPOCH}-${CKPT_STEPS}.log

cat ${BASE_PATH}/infer_results/infer-${EPOCH}-${CKPT_STEPS}.jsonl.* > ${BASE_PATH}/infer_results/infer-${EPOCH}-${CKPT_STEPS}.jsonl
rm ${BASE_PATH}/infer_results/infer-${EPOCH}-${CKPT_STEPS}.jsonl.*
