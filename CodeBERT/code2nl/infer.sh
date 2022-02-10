#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --account=def-six
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --job-name=CBc2nlInfer

lang=php #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=/scratch/xuxiaoj3/codeBERT/data/code2nl/CodeSearchNet
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

source ../../pyvenv/venv/bin/activate

python run.py --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path $test_model \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --eval_batch_size $batch_size