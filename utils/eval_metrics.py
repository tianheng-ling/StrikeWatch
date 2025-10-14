import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def analyze_model_memory(model_config: dict, exp_config: dict, model: torch.nn.Module):
    # bit width
    unit_bits = (
        model_config["quant_bits"]
        if exp_config["enable_qat"] == True
        and model_config["enable_int_forward"] == True
        else 32
    )
    unit_bytes = unit_bits / 8

    # weights
    param_size = sum(p.numel() * unit_bytes for p in model.parameters()) / 1e3  # KB
    # activations
    buffer_size = sum(b.numel() * unit_bytes for b in model.buffers()) / 1e3  # KB

    # MACs estimation
    total_macs = 0
    window_size = model_config["seq_len"]
    d_model = model_config["d_model"]
    num_layers = model_config["num_enc_layers"]
    dim_feedforward = model_config["d_model"] * 4
    num_heads = model_config["nhead"]

    qkv_proj = 3 * window_size * d_model * d_model
    attention_scores = window_size * d_model * window_size
    output_proj = window_size * d_model * d_model
    ffn = 2 * window_size * d_model * dim_feedforward

    layer_macs = qkv_proj + attention_scores + output_proj + ffn
    total_macs = layer_macs * num_layers

    weights_size = round(param_size, 2)
    activations_size = round(buffer_size, 2)
    total_macs = int(total_macs)
    return weights_size, activations_size, total_macs


def get_model_complexity(
    model_config: dict, model: torch.nn.Module, exp_config: dict, prefix: str
):

    # determine the data type bit widths
    unit_bits = (
        model_config["quant_bits"]
        if exp_config["enable_qat"] == True
        and model_config["enable_int_forward"] == True
        else 32
    )
    unit_bytes = unit_bits / 8  # default FP32 (4 bytes)

    # calculate the parameters
    param_size = 0
    param_amount = 0
    param_tensors = 0
    for param in model.parameters():
        param_size += param.numel() * unit_bytes  # Calculate parameter size
        param_amount += param.numel()  # Calculate parameter amount
        param_tensors += 1  # Calculate parameter tensors
    param_size /= 1e3  # Convert to kilobytes(KB)

    return {
        f"{prefix}param_size (KB)": f"{param_size:.2f}",
        f"{prefix}param_amount": int(param_amount),
    }


def get_classification_metrics(
    phase: str, targets: list, preds: list, pos_label: int = 1
):
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return {
        f"{phase}_acc": accuracy_score(targets, preds),
        f"{phase}_precision": precision_score(
            targets, preds, pos_label=pos_label, average="binary", zero_division=0
        ),
        f"{phase}_recall": recall_score(
            targets, preds, pos_label=pos_label, average="binary", zero_division=0
        ),
        f"{phase}_f1": f1_score(
            targets, preds, pos_label=pos_label, average="binary", zero_division=0
        ),
    }
