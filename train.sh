#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

python -m main \
    --dataloader_workers 2 \
    --seed 42 \
    --epochs 15 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --use_warmup_steps \
    --max_length 256 \
    --pad_mask_id -100 \
    --model vinai/phobert-base-v2 \
    --pin_memory \
    --train_batch_size 16 \
    --valid_batch_size 16 \
    --test_batch_size 16 \
    --train_file dataset/PhoNER_COVID19/train_word.jsonl \
    --valid_file dataset/PhoNER_COVID19/dev_word.jsonl \
    --test_file dataset/PhoNER_COVID19/test_word.jsonl \
    --text_col words \
    --label_col tags \
    --output_dir ./models \
    --record_output_file output.json \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.001 \
    --evaluate_on_accuracy \
