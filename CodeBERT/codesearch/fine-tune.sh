#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --account=def-six
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --job-name=CBcsFtphp


lang=php #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
data_dir=/scratch/xuxiaoj3/codeBERT/data/codesearch/
train_dir=$data_dir/train_valid/$lang/
output_dir=./models/$lang

source ../../../pyvenv/venv/bin/activate

python run_classifier.py \
    --model_type roberta \
    --task_name codesearch \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --train_file train.txt \
    --dev_file valid.txt \
    --max_seq_length 200 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --data_dir $train_dir \
    --output_dir $output_dir  \
    --model_name_or_path $pretrained_model