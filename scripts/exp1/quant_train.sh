#!/bin/bash

#-------------------------------
# Configurable Parameters
#-------------------------------
# data_config
declare -a train_strategy_options=("finetuning") 
declare -a target_person_options=("person1")
declare -a window_size_options=(50)
declare -a stride_ratio_options=(0.25)
declare -a downsampling_rate_options=(2) 

# exp settings
exp_mode="train" 
num_epochs=100

# model settings
model_type="rnn" # ("cnn" "sepcnn" "rnn" "transformer")

# HW simulation settings
subset_size=1
target_hw="amd"

# wandb settings
wandb_mode="online"    
wandb_project_prefix="EXP1_Quant"

# optuna search settings
n_trials=200
optuna_hw_target="energy" 

#-------------------------------
# Function to run experiment
#-------------------------------

run_experiment() {
    local train_strategy="$1"
    local window_size="$2"
    local downsampling_rate="$3"
    local person="$4"
    local stride_ratio="$5"

    exp_base_dir="exp_records_exp1/${model_type}"
    wandb_project_name="${wandb_project_prefix}Watch_${model_type}"

    python optuna_search.py \
        --train_strategy="$train_strategy" \
        --target_person="$person" \
        --window_size="$window_size" \
        --stride_ratio="$stride_ratio" \
        --downsampling_rate="$downsampling_rate" \
        --exp_mode="$exp_mode" \
        --exp_base_dir="$exp_base_dir" \
        --model_type="$model_type" \
        --num_epochs="$num_epochs" \
        --wandb_mode="$wandb_mode" \
        --wandb_project_name="$wandb_project_name" \
        --subset_size="$subset_size" \
        --target_hw="$target_hw" \
        --enable_qat \
        --enable_fused_ffn \
        --enable_hw_simulation \
        --optuna_hw_target="$optuna_hw_target" \
        --n_trials="$n_trials" 
}

#-------------------------------
# Main Execution Loop
#-------------------------------
for train_strategy in "${train_strategy_options[@]}"; do
    for person in "${target_person_options[@]}"; do
        for window_size in "${window_size_options[@]}"; do
            for downsampling_rate in "${downsampling_rate_options[@]}"; do
                for stride_ratio in "${stride_ratio_options[@]}"; do
                    run_experiment "$train_strategy" "$window_size" "$downsampling_rate" "$person" "$stride_ratio"
                done
            done
        done
    done
done

