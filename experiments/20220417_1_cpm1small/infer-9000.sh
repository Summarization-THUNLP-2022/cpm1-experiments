#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13581
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
MODEL_CONFIG_DIR=/data2/private/zhaoxinhao/ModelCenter/cpm1-small
CKPT_STEPS=9000

OPTS=""
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt-${CKPT_STEPS}.pt"
OPTS+=" --input-file /data2/private/zhaoxinhao/cpm1/data/LCSTS/test.jsonl"
OPTS+=" --output-file ${BASE_PATH}/infer_results/infer-${CKPT_STEPS}.txt"
OPTS+=" --span-length 40"
OPTS+=" --temperature 1"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0"
OPTS+=" --no-repeat-ngram-size 0"
OPTS+=" --repetition-penalty 1"
OPTS+=" --beam-size 5"
# OPTS+=" --random-sample"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/infer_file.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/infer_results/infer-${CKPT_STEPS}.log
