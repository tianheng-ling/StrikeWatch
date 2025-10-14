import wandb
import argparse

from config import data_config
from run_one_experiment import run_one_experiment


def main(args):

    # set training strategy
    if args.train_strategy == "personalized":
        data_split_approach = "PP"
        train_mode = "scratch"
    elif args.train_strategy == "generalized":
        data_split_approach = "LOPO"
        train_mode = "scratch"
    elif args.train_strategy == "finetuning":
        data_split_approach = None
        train_mode = "finetuning"
    else:
        raise ValueError(f"Unsupported training strategy: {args.train_strategy}")

    # set data config
    data_config.update(
        {
            "data_split_approach": data_split_approach,
            "target_person": args.target_person,
            "window_size": args.window_size,
            "downsampling_rate": args.downsampling_rate,
            "stride": int(args.stride_ratio * args.window_size),
        }
    )

    # set exp_config
    exp_config = {
        "exp_mode": args.exp_mode,
        "train_mode": train_mode,
        "exp_base_dir": args.exp_base_dir,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "enable_qat": args.enable_qat,
        "given_timestamp": None if args.exp_mode == "train" else args.given_timestamp,
        "enable_hw_simulation": args.enable_hw_simulation,
    }

    # set model config
    model_config = {
        "num_in_features": data_config["num_in_features"],
        "num_out_features": data_config["num_out_features"],
        "seq_len": int(args.window_size / args.downsampling_rate),
        "model_type": args.model_type,
    }
    if model_config["model_type"] == "cnn" or model_config["model_type"] == "sepcnn":
        model_config.update(
            {
                "num_blocks": args.num_blocks,
            }
        )
    elif model_config["model_type"] == "rnn":
        model_config.update(
            {
                "hidden_size": args.hidden_size,
                "num_rnn_layers": args.num_rnn_layers,
                "cell_type": args.cell_type,
                "batch_size": exp_config["batch_size"],
            }
        )
    elif model_config["model_type"] == "transformer":
        model_config.update(
            {
                "d_model": args.d_model,
                "ffn_dim": args.ffn_dim,
                "num_enc_layers": args.num_enc_layers,
                "nhead": args.nhead,
            }
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    quant_config = None
    if exp_config["enable_qat"]:
        quant_config = {
            "model_name": "network",
            "quant_bits": args.quant_bits,
            "qat_warmup_epochs": args.qat_warmup_epochs,
        }
        model_config.update(
            {
                "name": quant_config["model_name"],
                "quant_bits": quant_config["quant_bits"],
                "enable_int_forward": False,
                "enable_fused_ffn": args.enable_fused_ffn,
            }
        )

    wandb_config = {
        "name": args.wandb_project_name,
        "mode": args.wandb_mode,
        "config": {},
    }
    if exp_config["train_mode"] == "finetuning":
        assert (
            data_config["data_split_approach"] is None
        ), "data_split_approach should be None for finetuning"

        # phase 1: run LOPO
        data_config["data_split_approach"] = "LOPO"
        exp_config["enable_qat"] = False
        wandb_config["config"].update(
            {
                **data_config,
                **model_config,
                **exp_config,
                **(quant_config or {}),
            }
        )
        (
            timestamp,
            _,
            _,
            _,
            _,
        ) = run_one_experiment(
            data_config=data_config,
            model_config=model_config,
            exp_config=exp_config,
            wandb_config=wandb_config,
            quant_config=quant_config,
            hw_config=None,
        )
        wandb.finish()
        exp_config["given_timestamp"] = timestamp
        data_config["data_split_approach"] = "PP"

    hw_config = None
    exp_config["enable_qat"] = args.enable_qat
    if exp_config["enable_qat"] and exp_config["enable_hw_simulation"]:
        hw_config = {
            "top_module": quant_config["model_name"],
            "subset_size": args.subset_size,
            "target_hw": args.target_hw,
            "fpga_type": args.fpga_type,
        }
        wandb_config["config"].update(hw_config)

    wandb_config = {
        "name": args.wandb_project_name,
        "mode": args.wandb_mode,
        "config": {
            **data_config,
            **model_config,
            **exp_config,
            **(quant_config or {}),
            **(hw_config or {}),
        },
    }
    run_one_experiment(
        data_config=data_config,
        model_config=model_config,
        exp_config=exp_config,
        wandb_config=wandb_config,
        quant_config=quant_config,
        hw_config=hw_config,
    )

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # wandb config
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
    )

    # data config
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--stride_ratio", type=float, default=0.5)
    parser.add_argument("--downsampling_rate", type=int, default=None)
    parser.add_argument("--target_person", type=str, default=None)

    # model_config
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["transformer", "cnn", "sepcnn", "rnn"],
        required=True,
    )
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_rnn_layers", type=int, default=1)
    parser.add_argument(
        "--cell_type", type=str, choices=["lstm", "gru"], default="lstm"
    )
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--ffn_dim", type=int)
    parser.add_argument("--num_enc_layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--enable_fused_ffn", action="store_true")

    # experiment config
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument("--given_timestamp", type=str, default=None)
    parser.add_argument(
        "--train_strategy",
        type=str,
        choices=["personalized", "generalized", "finetuning"],
        required=True,
    )
    parser.add_argument("--exp_base_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--enable_qat", action="store_true")
    parser.add_argument("--enable_hw_simulation", action="store_true")

    # quantization config
    parser.add_argument("--quant_bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--qat_warmup_epochs", type=int, default=0)

    # hw simulation config
    parser.add_argument("--subset_size", type=int, default=1)
    parser.add_argument(
        "--target_hw", type=str, choices=["amd", "lattice"], default="amd"
    )
    parser.add_argument(
        "--fpga_type",
        type=str,
        choices=["xc7s15ftgb196-2", "xc7s25ftgb196-2", "xc7s50ftgb196-2"],
        help="FPGA type for HW simulation",
        default="xc7s15ftgb196-2",
    )
    main(args=parser.parse_args())
