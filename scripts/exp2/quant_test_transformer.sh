#!/bin/bash

#-------------------------------
# Configurable Parameters
#-------------------------------
# data_config
declare -a train_strategy_options=("personalized") # "personalized" "generalized" "finetuning"
declare -a target_person_options=("person1")
declare -a window_size_options=(50)
declare -a stride_ratio_options=(0.25)
declare -a downsampling_rate_options=(2) 

# model settings
model_type="transformer" 
declare -a d_model_options=(8)
declare -a ffn_dim_options=(32)
declare -a nhead_options=(1)
declare -a num_enc_layers_options=(1)

# exp settings
num_exps=1
num_epochs=100
exp_mode="test"    
# change the following values depending on your experiment
given_timestamp="2025-06-19_07-18-10" #
batch_size=16
lr=0.00043249
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
    local dmodel="$5"
    local ffn_dim="$6" 
    local nhead="$7"
    local nlayer="$8"
    local bits="$9"
    local stride_ratio="${10}"
    
    exp_base_dir="exp_records_exp1/${model_type}"
    wandb_project_name="${wandb_project_prefix}Watch"

    echo "▶ Running experiment: $train_strategy for $person with window_size=$window_size, downsampling_rate=$downsampling_rate, d_model=$dmodel, ffn_dim=$ffn_dim, nhead=$nhead, num_enc_layers=$nlayer, quant_bits=$bits stride_ratio=$stride_ratio"
    python main.py \
        --train_strategy="$train_strategy" \
        --target_person="$person" \
        --window_size="$window_size" \
        --stride_ratio="$stride_ratio" \
        --downsampling_rate="$downsampling_rate" \
        --model_type="$model_type" \
        --given_timestamp="$given_timestamp" \
        --d_model="$dmodel" \
        --ffn_dim="$ffn_dim" \
        --nhead="$nhead" \
        --num_enc_layers="$nlayer" \
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
        --enable_qat \
        --enable_fused_ffn \
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
                        for dmodel in "${d_model_options[@]}"; do
                            for ffn_dim in "${ffn_dim_options[@]}"; do
                                for nhead in "${nhead_options[@]}"; do
                                    for nlayer in "${num_enc_layers_options[@]}"; do
                                        for bits in "${quant_bits_options[@]}"; do
                                            run_experiment "$train_strategy" "$window_size" "$downsampling_rate" "$person" "$dmodel" "$ffn_dim" "$nhead" "$nlayer" "$bits" "$stride_ratio"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
