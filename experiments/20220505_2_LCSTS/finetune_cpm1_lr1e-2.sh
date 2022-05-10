#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13576
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
DATASET="LCSTS"
LR=1e-2

OPTS=""
OPTS+=" --cache-path ${CPM_CACHE_PATH}"
OPTS+=" --data-path ${CPM_TRAIN_DATA_PATH}"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CPM_CACHE_PATH}/cpm1-small"
OPTS+=" --batch-size 128"
OPTS+=" --train-iters 600000"
OPTS+=" --save-iters 100000"
OPTS+=" --max-length 256"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-cpm1-ckpt-lr${LR}"
OPTS+=" --lr ${LR}"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
# OPTS+=" --load ${BASE_PATH}/results/cpm1-new.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/finetune_cpm1.py ${OPTS}"
echo ${CMD}

if [ ! -d ${BASE_PATH}/results ]; then
    mkdir ${BASE_PATH}/results
fi

${CMD} 2>&1 | tee ${BASE_PATH}/results/${DATASET}-LR${LR}.log
