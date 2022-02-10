#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --account=def-six
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --job-name=GCBcdinfer

source ../../pyvenv/venv/bin/activate
dataset_dir=dataset

python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=${dataset_dir}/train.txt \
    --eval_data_file=${dataset_dir}/valid.txt \
    --test_data_file=${dataset_dir}/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log