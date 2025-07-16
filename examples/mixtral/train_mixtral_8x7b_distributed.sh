#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${1:-"${HOME}/ckpts/mixtral-mcore-TP2PP4EP8"}
TOKENIZER_MODEL=${2:-"${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1/tokenizer.model"}
DATA_PATH=${3:-"${HOME}/datasets/megatron-lm-data/gpt-2-seq_monkey_text_document"}
SAVE_PATH=${4:-"${HOME}/outputs/Megatron-LM-Mixtral-8x7B"}
if [ -d ${SAVE_PATH} ]; then
    rm --recursive --force ${SAVE_PATH}
fi
mkdir -p ${SAVE_PATH}

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --train-data-path ${DATA_PATH}
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --lr 1e-4
    --train-iters 10000
    --lr-decay-iters 5000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --use-torch-optimizer-for-cpu-offload
    # --use-distributed-optimizer
    # --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 50 \
    --eval-interval 100000 \
    --eval-iters 100000 \
    --save ${SAVE_PATH} \
    --load ${CHECKPOINT_PATH} \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} ${HOME}/projects/Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
