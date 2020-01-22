#!/bin/bash

task='pseudo_method'
dataset=$1
if [ "$2" = 'attnptr' ] ; then
    copy='copy__'
else
    copy=''
fi
read_sp_model_path=exp/task_semantic_parsing/dataset_${1}/labeled_${3}/${copy}cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/
read_qg_model_path=exp/task_question_generation/dataset_${1}/labeled_${3}/${copy}cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/

# training paras
reduction=sum # sum, mean
lr=0.001
l2=1e-5
batchSize=16
test_batchSize=128
max_norm=5
max_epoch=100
beam=5
n_best=1

# special paras
discount=0.5
method=constant # constant, linear
labeled=$3
unlabeled=1.0
deviceId="0 1"
seed=999
extra='--extra'

python3 scripts/pseudo_method.py --task $task --dataset $dataset \
    --read_sp_model_path $read_sp_model_path --read_qg_model_path $read_qg_model_path \
    --reduction $reduction --lr $lr --l2 $l2 --batchSize $batchSize --test_batchSize $test_batchSize \
    --discount $discount --method $method --max_norm $max_norm --max_epoch $max_epoch --beam $beam --n_best $n_best \
    --labeled $labeled --unlabeled $unlabeled --seed $seed --deviceId $deviceId $extra
