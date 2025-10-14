import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_config = {
    "data_file_path": "data/csv",
    "feature_cols": ["x", "y", "z"],
    "num_in_features": 3,
    "num_out_features": 2,  # two classes: forefoot and heel
}

cnn_defaul_config = {"num_blocks": 2}
sepcnn_defaul_config = {"num_blocks": 2}

rnn_default_config = {
    "hidden_size": 20,
    "num_rnn_layers": 1,
    "cell_type": "lstm",
}

transformer_default_config = {
    "d_model": 12,
    "num_enc_layers": 1,
    "nhead": 1,
}

search_space = {
    "quant_bits": {"low": 8, "high": 8, "step": 2},
    "batch_size": {"low": 16, "high": 16, "step": 8},
    "lr": {"low": 1e-5, "high": 1e-3, "log": True},
    "num_blocks": {
        "low": 1,
        "high": 5,
        "step": 1,
    },  # depending on the downsampling rate
    "hidden_size": {"low": 8, "high": 64, "step": 8},
    "d_model": {"low": 8, "high": 32, "step": 8},
}

# per quantization bits, the expected F1-score threshold
f1_thresholds = {
    8: 0.8,
    6: 0.7,
    4: 0.6,
}

# hardware constraints for optuna pruning
STOP_TIME = "125ms"
LATENCY_THRESHOLD = 125  # ms, should be same as STOP_TIME
POWER_THRESHOLD = 500  # mW
ENERGY_THRESHOLD = 100  # muJ
