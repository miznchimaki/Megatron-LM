#!/usr/bin/bash


TOKENIZER_MODEL=${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1/tokenizer.model
MEGATRON_PATH="${HOME}/projects/Megatron-LM"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE=${1:-"2"}
TARGET_EP_SIZE=${2:-"8"}
TARGET_PP_SIZE=${3:-"4"}

HF_FORMAT_DIR=${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1
MEGATRON_FORMAT_DIR=${HOME}/ckpts/Mixtral-8x7B-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}
if [ -d ${MEGATRON_FORMAT_DIR} ]; then
    rm -rf ${MEGATRON_FORMAT_DIR}
fi
mkdir -p ${MEGATRON_FORMAT_DIR}

echo "start Mixtral huggingface2Megatron-Core at `date '+%Y-%m-%d-%H:%M:%S'`"

cd ${HOME}/projects/Megatron-LM/

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mixtral_hf \
    --saver mcore \
    --target-tensor-parallel-size ${TARGET_TP_SIZE} \
    --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
    --target-expert-parallel-size ${TARGET_EP_SIZE} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} 2>&1 | tee ./Mixtral-hf2megatron-core-output.log
mv ./Mixtral-hf2megatron-core-output.log ${MEGATRON_FORMAT_DIR}/

echo "end Mixtral huggingface2Megatron-Core at `date '+%Y-%m-%d-%H:%M:%S'`"
