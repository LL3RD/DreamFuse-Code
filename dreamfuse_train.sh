#! /bin/bash

save_name=$1

accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 12345 \
    --dynamo_backend "no" \
    --mixed_precision bf16 \
    dreamfuse/trains/train_dreamfuse.py \
    --learning_rate 1 \
    --optimizer_type prodigy \
    --clip_grad_norm 1000 \
    --flux_model_id black-forest-labs/FLUX.1-dev \
    --num_training_steps 10000 \
    --val_interval 1000 \
    --checkpoints_total_limit 5 \
    --save_model_steps 1000 \
    --lr_schedule_type constant \
    --lr_warmup_ratio 0 \
    --train_batch_size 1 \
    --use_lora true \
    --lora_rank 16 \
    --max_sequence_length 256 \
    --guidance_scale 1.0 \
    --work_mode dreamfuse \
    --data_config configs/dreamfuse.yaml \
    --valid_config examples/data_dreamfuse.json \
    --image_ids_offset 0 0 0 \
    --mix_attention_double true \
    --mix_attention_single true \
    --valid_output_dir ./valid/$save_name/valid_output \
    --output_dir ./valid/$save_name/chk/ \
    --debug false \
    --image_tags 0 1 2 \
    --mask_tokenizer False \
    --mask_pos_affine True \
    --prompt_ratio 0.01
