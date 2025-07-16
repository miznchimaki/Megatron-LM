#!/usr/bin/bash


INPUT_DATA_FILE=${1:-"${HOME}/datasets/megatron-lm-data/seq_monkey_datawhale.jsonl"}
OUTPUT_PREFIX=${2:-"mixtral-pretrain"}
TOKENIZER_TYPE=${3:-"Llama2Tokenizer"}
TOKENIZER_MODEL=${4:-"${HOME}/ckpts/Mixtral-8x7B-Instruct-v0.1/tokenizer.model"}
JSON_KEYS=${5:-"text"}
WORKERS=${6:-"223"}
LOG_INTERVAL=${7:-"100"}
LOG_FILE=${8:-"mixtral_data_preprocessing.log"}

cd ${HOME}/projects/Megatron-LM/
LOG_PATH=$(pwd)/examples/mixtral/${LOG_FILE}
echo "start mixtral training data preprocessing at $(date '+%Y-%m-%d-%H:%M:%S')" 2>&1 | tee ${LOG_PATH}

python ./tools/preprocess_data.py \
    --input ${INPUT_DATA_FILE} \
    --output-prefix ${OUTPUT_PREFIX} \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --workers ${WORKERS} \
    --log-interval ${LOG_INTERVAL} \
    --json-keys ${JSON_KEYS} 2>&1 | tee --append ${LOG_PATH}

echo "end mixtral training data preprocessing at `date '+%Y-%m-%d-%H:%M:%S'`" 2>&1 | tee --append ${LOG_PATH}
