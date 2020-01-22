#!/bin/bash
task='language_model'
dataset=$1
side=$2 # question, logical_form
# read_model_path=''

num_layers=1
hidden_dim=200
emb_size=100
cell=lstm # lstm, gru
decoder_tied='' # '--decoder_tied', ''

batchSize=16
test_batchSize=128
lr=0.001
dropout=0.5
max_norm=5
l2=1e-5
max_epoch=100
labeled=1.0
deviceId=0

python scripts/language_model.py --task $task --dataset $dataset --side $side \
    --num_layers $num_layers --hidden_dim $hidden_dim --emb_size $emb_size --cell $cell \
    --batchSize $batchSize --test_batchSize $test_batchSize --lr $lr --dropout $dropout --max_norm $max_norm --l2 $l2 \
    --labeled $labeled --max_epoch $max_epoch --deviceId $deviceId $decoder_tied
