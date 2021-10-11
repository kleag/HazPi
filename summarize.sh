#!/bin/bash

#set -o nounset
set -o errexit
set -o pipefail

echo "$0"

MODEL_PATH=/home/gael/Documents/Theses/TheseJessicaLopez/JessicaModel

# activate environments
which python3
/usr/bin/env python3 --version

# run script
echo 'Summarizing'

python summarize.py -checkpoint_path ${MODEL_PATH}/checkpoint/ \
  -path_summaries_encoded ${MODEL_PATH}/encoded/ \
  -path_summaries_decoded ${MODEL_PATH}/decoded/ \
  -path_summaries_error ${MODEL_PATH}/error/ \
  -vocab_load_dir ${MODEL_PATH}/vocab/ \
  -encoder_max_vocab 100000 -decoder_max_vocab 100000 -num_layers 4 \
  -batch_size 4 -num_heads 8 -dff 2048 -num_layers 4 -d_model 128 \
  -ngram_size 2 -k 6 -len_summary 216 file.txt

