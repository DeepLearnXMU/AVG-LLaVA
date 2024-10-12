export prefix=
export HF_HOME=
export TORCH_HOME=
export TRITON_CACHE_DIR=
export save_name=AVG-LLaVA-Stage3
export dataset=open-llava-next_instruct_mix1M.json

deepspeed $prefix/LLaVA1.6/llava/train/train_mem.py \
    --deepspeed $prefix/LLaVA1.6/scripts/zero3.json \
    --model_name_or_path Lin-Chen/open-llava-next-vicuna-7b \
    --cache_dir $prefix/pretrained_checkpoints \
    --version v1 \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --data_path $prefix/dataset/finetune-next/$dataset \
    --image_folder $prefix/dataset/finetune-next/zip_dir \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $prefix/checkpoints/$save_name \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1595 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vis_token_granularity "36,72,144,288,576" \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-5 \
    --report_to none