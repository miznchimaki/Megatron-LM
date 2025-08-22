#!/bin/bash
# Pretrain a multimodal model.


export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_DIR=${1:-"${HOME}/projects/Megatron-LM"}
MODEL_NAME=${2:-"mcore-llava-mistral-7b-instruct-clip336-pretraining"}
OUTPUT_DIR=${3:-"${HOME}/outputs/${MODEL_NAME}"}
LOG_PATH=${4:-"${OUTPUT_DIR}/output.log"}
TENSORBOARD_DIR=${5:-"${OUTPUT_DIR}/tensorboard"}
PRETRAIN_CKPT_FOLDER=${6:-""}
if [[ -z ${PRETRAIN_CKPT_FOLDER} ]]; then
    PRETRAIN_CKPT_DIR=""
else
    PRETRAIN_CKPT_DIR=${HOME}/ckpts/${PRETRAIN_CKPT_FOLDER}
fi

DATA_CONFIG_PATH=${7:-"${PROJECT_DIR}/examples/multimodal/pretrain_dataset.yaml"}

export TRITON_CACHE_DIR="${HOME}/triton-cache/"
# The following patch to the Triton cache manager is needed for Triton version <= 3.1
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

GLOBAL_BATCH_SIZE=${8:-"1"}
NUM_WORKERS=${9:-"2"}
HIDDEN_DROPOUT=${10:-"0.0"}
LOG_INTERVAL=${11:-"1"}
NONDETERMINISTIC_ATTN=${12:-"1"}
TOKENIZER_MODEL=${13:-"${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1/tokenizer.model"}


NETWORK_SIZE_ARGS=(
    --apply-layernorm-1p
    --normalization RMSNorm
    --group-query-attention
    --num-query-groups 8
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --swiglu
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --max-position-embeddings 4096
    --ffn-hidden-size 14336
)

MIXED_PRECISION_ARGS=(
    --attention-softmax-in-fp32
    --bf16
)

CHECKPOINTING_ARGS=(
    --use-checkpoint-args
    --save-interval 5000
    --save ${OUTPUT_DIR}
    --dataloader-save ${OUTPUT_DIR}/dataloader
    --pretrained-checkpoint ${PRETRAIN_CKPT_DIR}
    --ckpt-format torch_dist
)

# TODO: Need modifications
DISTRIBUTED_ARGS=(
    --use-distributed-optimizer
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --distributed-timeout-minutes 120
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

MULTIMODAL_EXTRA_ARGS=(
    --use-te
    --tokenizer-prompt-format mistral
    --prompt-path ${PROJECT_DIR}/examples/multimodal/manual_prompts.json
    --freeze-LM
    --freeze-ViT
    --language-model-type mistral_7b
    --disable-vision-class-token
    --allow-missing-vision-projection-checkpoint
)

VISION_ARGS=(
    --patch-dim 14
    --img-h 336
    --img-w 336
)

TRAINING_ARGS=(
    --no-masked-softmax-fusion
    --use-flash-attn
    --disable-bias-linear
    --train-iters 100000
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --micro-batch-size 1
    --log-interval ${LOG_INTERVAL}
    --dataloader-type external
    --tensorboard-dir ${TENSORBOARD_DIR}
)

DATA_ARGS=(
    --num-workers ${NUM_WORKERS}
    --seq-length 576
    --decoder-seq-length 1024
    --data-path ${DATA_CONFIG_PATH}
    --split 100,0,0
    --eod-mask-loss
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout ${HIDDEN_DROPOUT}
    --clip-grad 1.0
    --weight-decay 1e-2
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr-decay-iters 20000
    --lr-warmup-fraction .01
    --lr 0.00015
    --min-lr 1.0e-5
    --lr-decay-style cosine
)

VALIDATION_ARGS=(
    --eval-iters 50000
    --eval-interval 50000
)

TOKENIZER_ARGS=(
    --tokenizer-type MultimodalTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
)

INITIALIZATION_ARGS=(
    --init-method-std 0.014
)

LOGGING_ARGS=(
    --log-params-norm
)

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# TODO: Now here
torchrun --nproc_per_node 8 examples/multimodal/train.py ${OPTIONS}
