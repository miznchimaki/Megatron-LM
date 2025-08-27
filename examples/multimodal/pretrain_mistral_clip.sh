#!/bin/bash
# Pretrain a multimodal model.


echo start megatron-lm MLLM training at `date +%Y-%m-%d-%H:%M:%S`
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# distributed params
NPROC_PER_NODE=${1:-"8"}
NNODES=${2:-"1"}
MASTER_ADDR=${3:-"localhost"}
MASTER_PORT=${4:-"23479"}
NODE_RANK=${5:-"0"}

TORCH_DIST_ARGS=(
    --nproc-per-node ${NPROC_PER_NODE}
    --nnodes ${NNODES}
    --master-addr ${MASTER_ADDR}
    --master-port ${MASTER_PORT}
    --node-rank ${NODE_RANK}
)

PROJECT_DIR=${6:-"${HOME}/projects/Megatron-LM"}
MODEL_NAME=${7:-"one-H20-node-731.69b-llave-504-layers-mixtral-and-clip-vit"}
OUTPUT_DIR=${8:-"${HOME}/outputs/${MODEL_NAME}"}
if [ -d ${OUTPUT_DIR} ]; then
    rm -rf ${OUTPUT_DIR}
fi
mkdir ${OUTPUT_DIR}

LOG_PATH=${9:-"${OUTPUT_DIR}/output.log"}
TENSORBOARD_DIR=${10:-"${OUTPUT_DIR}/tensorboard"}
PRETRAIN_CKPT_FOLDER=${11:-""}
if [[ -z "${PRETRAIN_CKPT_FOLDER}" ]]; then
    PRETRAIN_CKPT_DIR=""
else
    PRETRAIN_CKPT_DIR=${HOME}/ckpts/${PRETRAIN_CKPT_FOLDER}
fi

DATA_CONFIG_PATH=${12:-"${PROJECT_DIR}/examples/multimodal/pretrain_dataset.yaml"}

# export TRITON_CACHE_DIR="${HOME}/triton-cache/"
# The following patch to the Triton cache manager is needed for Triton version <= 3.1
# export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

GLOBAL_BATCH_SIZE=${13:-"1"}
NUM_WORKERS=${14:-"2"}
HIDDEN_DROPOUT=${15:-"0.0"}
LOG_INTERVAL=${16:-"1"}
NONDETERMINISTIC_ATTN=${17:-"1"}
TOKENIZER_MODEL=${18:-"${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1"}
WANDB_API_KEY=${19:-""}


# TODO: num-layers: 32 -> (1) LLaVA based on CLIP-ViT/14@336px and Mistral-7B; (2) 7.24B
# TODO: num-layers: 256 -> (1) LLaVA based on CLIP-ViT/14@336px and Mistral-7B (extend attention layers from 32 to 256); (2) 56.10B
# TODO: num-layers: 504 -> (1) LLaVA based on CLIP-ViT/14@336px and Mistral-7B (extend attention layers from 32 to 504); (2) 110.20B

# TODO: num-layers: 32 -> (1) LLaVA based on CLIP-ViT/14@336px and Mixtral; (2) 46.70B
# TODO: num-layers: 504 -> (1) LLaVA based on CLIP-ViT/14@336px and Mixtral; (2) 731.69B
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
    --num-layers 504
    --hidden-size 4096
    --num-attention-heads 32
    --max-position-embeddings 4096
    --ffn-hidden-size 14336
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-pre-softmax
    --expert-model-parallel-size 1
)

MIXED_PRECISION_ARGS=(
    --attention-softmax-in-fp32
    --bf16
)

CHECKPOINTING_ARGS=(
    --save-interval 20000
    --save ${OUTPUT_DIR}
    --dataloader-save ${OUTPUT_DIR}/dataloader
    --ckpt-format torch_dist
    --no-load-optim
    --no-load-rng
    --no-save-optim
)
if [ -n "${PRETRAIN_CKPT_DIR}" ]; then
    CHECKPOINTING_ARGS+=(
        --pretrained-checkpoint ${PRETRAIN_CKPT_DIR}
    )
fi

# TODO: maybe need modifications fro MoE-LLM
MEGATRON_DIST_ARGS=(
    --use-distributed-optimizer
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --distributed-timeout-minutes 120
    --overlap-param-gather
    --overlap-grad-reduce
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
    --use-mcore-models
    --no-masked-softmax-fusion
    --use-flash-attn
    --disable-bias-linear
    --train-iters 100000
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --micro-batch-size 1
    --log-interval ${LOG_INTERVAL}
    --dataloader-type external
    --tensorboard-dir ${TENSORBOARD_DIR}
    --use-torch-optimizer-for-cpu-offload
)
# --sequence-parallel

EXPERIMENTAL_ARGS=(
    --main-grads-dtype bf16
    --main-params-dtype fp16
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
    --use-precision-aware-optimizer
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
    --eval-iters 200000
    --eval-interval 200000
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
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Megatron-LM-1T-VLM"}
        --wandb-exp-name ${WANDB_NAME:-"VLM-1node-pretrain"}
    )
fi

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# export TORCHDYNAMO_VERBOSE=1
# export TORCHDYNAMO_DISABLE=1
cd ${PROJECT_DIR}/examples/multimodal/
torchrun ${TORCH_DIST_ARGS[@]} ./train.py ${NETWORK_SIZE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MEGATRON_DIST_ARGS[@]} \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${MULTIMODAL_EXTRA_ARGS[@]} \
    ${VISION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${VALIDATION_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LOGGING_ARGS[@]} 2>&1 | tee ${LOG_PATH}

echo end megatron-lm MLLM training at `date +%Y-%m-%d-%H:%M:%S`
