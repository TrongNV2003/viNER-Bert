#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

python -m main \
    --dataloader_num_workers 2 \
    --seed 42 \
    --learning_rate 3e-5 \
    --num_train_epochs 20 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --pad_mask_id -100 \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --model 5CD-AI/visobert-14gb-corpus \
    --pin_memory \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --test_batch_size 64 \
    --train_file dataset/PhoNER_COVID19/train_word.json \
    --valid_file dataset/PhoNER_COVID19/dev_word.json \
    --test_file dataset/PhoNER_COVID19/test_word.json \
    --text_col words \
    --label_col tags \
    --output_dir ./models \
    --record_output_file output.json \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 100 \
    --logging_dir ./models/logs \
    --fp16 \
    --metric_for_best_model eval_f1 \
    --greater_is_better \
    --load_best_model_at_end \
    --report_to mlflow \
    --use_word_splitter \
