#!/bin/bash

#-------------------------------
# Configurable Parameters
#-------------------------------
# data_config
declare -a train_strategy_options=("finetuning")
declare -a target_person_options=("person1" "person2" "person3" "person4" "person5" "person6" "person7" "person8" "person9" "person10" "person11" "person12")
declare -a window_size_options=(50)
declare -a stride_ratio_options=(0.25)
declare -a downsampling_rate_options=(2) 

# model settings
model_type="sepcnn" 
declare -a num_blocks_options=(3)

# exp settings
num_exps=50
exp_mode="train" 
batch_size=24
lr=0.0005274
num_epochs=100

# quantization settings
declare -a quant_bits_options=(6) 

# wandb settings
wandb_mode="online"    # "disabled" or "online" or "offline"
wandb_project_prefix="EXP3_Quant"

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

    exp_base_dir="exp_records_exp3/${person}/${model_type}"  
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
        --enable_qat 
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
