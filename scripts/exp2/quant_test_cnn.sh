#!/bin/bash

#-------------------------------
# Configurable Parameters
#-------------------------------
# data_config
declare -a train_strategy_options=("personalized")
declare -a target_person_options=("person1")
declare -a window_size_options=(50)
declare -a stride_ratio_options=(0.25)
declare -a downsampling_rate_options=(2) 

# model settings
model_type="cnn" 
declare -a num_blocks_options=(3)

# exp settings
num_exps=1
num_epochs=100
exp_mode="test" 
# change the following values depending on your experiment
given_timestamp="2025-06-19_00-06-19" 
batch_size=48
lr=0.0003367
# quantization settings
declare -a quant_bits_options=(4)

# HW simulation settings
subset_size=1
target_hw="lattice"

# wandb settings
wandb_mode="disabled"    # "disabled" or "online" or "offline"
wandb_project_prefix="EXP2_Quant"

#-------------------------------
# Function to run experiment
#-------------------------------

run_experiment() {
    local train_strategy="$1"
    local window_size="$2"
    local downsampling_rate="$3"
    local person="$4"
    local num_blocks="$5"
    local bits="$6"
    local stride_ratio="$7"

    exp_base_dir="exp_records_exp1/${model_type}"
    wandb_project_name="${wandb_project_prefix}Watch"

    python main.py \
        --train_strategy="$train_strategy" \
        --target_person="$person" \
        --window_size="$window_size" \
        --stride_ratio="$stride_ratio" \
        --downsampling_rate="$downsampling_rate" \
        --model_type="$model_type" \
        --num_blocks="$num_blocks" \
        --exp_mode="$exp_mode" \
        --exp_base_dir="$exp_base_dir" \
        --batch_size="$batch_size" \
        --num_epochs="$num_epochs" \
        --lr="$lr" \
        --wandb_mode="$wandb_mode" \
        --wandb_project_name="$wandb_project_name" \
        --quant_bits="$bits" \
        --subset_size="$subset_size" \
        --target_hw="$target_hw" \
        --given_timestamp="$given_timestamp" \
        --enable_qat \
        --enable_hw_simulation 
        
}

#-------------------------------
# Main Execution Loop
#-------------------------------
for ((i=1; i<=num_exps; i++)); do
    for train_strategy in "${train_strategy_options[@]}"; do
        for person in "${target_person_options[@]}"; do
            for window_size in "${window_size_options[@]}"; do
                for stride_ratio in "${stride_ratio_options[@]}"; do
                    for downsampling_rate in "${downsampling_rate_options[@]}"; do
                        for num_blocks in "${num_blocks_options[@]}"; do
                            for bits in "${quant_bits_options[@]}"; do
                                run_experiment "$train_strategy" "$window_size" "$downsampling_rate" "$person" "$num_blocks" "$bits" "$stride_ratio" 
                            done
                        done
                    done
                done
            done
        done
    done
done
