#!/bin/bash

task='dual_learning'
dataset=$1
if [ "$2" = 'attnptr' ] ; then
    copy='copy__'
else
    copy=''
fi
read_sp_model_path=exp/task_semantic_parsing/dataset_${1}/labeled_${3}/${copy}cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/
read_qg_model_path=exp/task_question_generation/dataset_${1}/labeled_${3}/${copy}cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/
read_qlm_path=exp/task_language_model/dataset_${1}/question__labeled_1.0/cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100/
read_lflm_path=exp/task_language_model/dataset_${1}/logical_form__labeled_1.0/cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100/

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
sample=6
alpha=0.5
beta=0.5
labeled=$3
unlabeled=1.0
cycle=sp+qg
deviceId="0 1"
seed=999
extra='--extra'

python3 scripts/dual_learning.py --task $task --read_sp_model_path $read_sp_model_path --read_qg_model_path $read_qg_model_path \
    --dataset $dataset --read_qlm_path $read_qlm_path --read_lflm_path $read_lflm_path \
    --reduction $reduction --lr $lr --l2 $l2 --batchSize $batchSize --test_batchSize $test_batchSize \
    --cycle $cycle --max_norm $max_norm --max_epoch $max_epoch --beam $beam --n_best $n_best --sample $sample --alpha $alpha --beta $beta \
    --labeled $labeled --unlabeled $unlabeled --deviceId $deviceId --seed $seed $extra
