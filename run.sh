#!bin/bash

DATA_FILE=tmp/dataset_v7w_telling.json
IM_VQA_FILE=tmp/visual7w_image_vqa_results.json
IM_EMBED_FILE=tmp/visual7w_image_embeddings.pkl

LOG_DIR=logs
CACHE_DIR=cache

WORD_CNT_THRESH=5
BATCH_SIZE=256
EMBED_DIM=512
VERBOSE=1

python ./src/main.py \
       --data_file=$DATA_FILE \
       --im_vqa_file=$IM_VQA_FILE \
       --im_embed_file=$IM_EMBED_FILE \
       --emb_dim=$EMBED_DIM \
       --batchsize=$BATCH_SIZE \
       --log_dir=$LOG_DIR \
       --cache_dir=$CACHE_DIR \
       --word_cnt_thresh=$WORD_CNT_THRESH \
       --verbose=$VERBOSE