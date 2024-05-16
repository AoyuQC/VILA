#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

n_nodes=1
bs=1
# OUTPUT of stage 2 script
# STAGE2_PATH=$1
BASE_MODEL_PATH=$1
# Final output checkpoint path
OUTPUT=$2

gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU name: $gpu_info"

if [[ "$gpu_info" == *"A100"* ]]; then
    echo "The GPU is an A100 GPU."
    # a100
    /home/ec2-user/SageMaker/conda_env/vila/bin/torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
        --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
        llava/train/train_mem.py \
        --deepspeed ./scripts/zero3_offload.json \
        --model_name_or_path $BASE_MODEL_PATH \
        --version llama_3 \
        --data_mixture ltcc \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir ./checkpoints/$OUTPUT \
        --num_train_epochs 1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --vflan_no_system_prompt True \
        --report_to wandb
elif [[ "$gpu_info" == *"A10"* ]]; then
    echo "The GPU is an A10 GPU."
    /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/anaconda3/envs/vila/bin/torchrun --nnodes=$n_node --nproc_per_node=1 --master_port=25001 \
        --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
        llava/train/train_mem.py \
        --deepspeed ./scripts/zero3_offload.json \
        --model_name_or_path $BASE_MODEL_PATH \
        --version llama_3 \
        --data_mixture ltcc \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir ./checkpoints/$OUTPUT \
        --num_train_epochs 1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --vflan_no_system_prompt True \
        --report_to wandb
else
    echo "The GPU is neither A10 nor A100."
    echo "GPU name: $gpu_info"
fi

