import os
import wandb
import optuna
import argparse
from functools import partial
import optuna.visualization as vis
from optuna.storages import RDBStorage
from optuna.samplers import NSGAIISampler

from config import (
    data_config,
    search_space,
    f1_thresholds,
    LATENCY_THRESHOLD,
    POWER_THRESHOLD,
    ENERGY_THRESHOLD,
)
from run_one_experiment import run_one_experiment
from utils import (
    save_trials_records,
    plot_pareto_from_json,
    safe_wandb_log,
)


def objective(trial, args):
    try:
        # optuna search space
        quant_bits = (
            trial.suggest_int("quant_bits", **search_space["quant_bits"])
            if args.enable_qat
            else None
        )
        batch_size = trial.suggest_int("batch_size", **search_space["batch_size"])
        lr = trial.suggest_float("lr", **search_space["lr"])

        if args.model_type == "cnn" or args.model_type == "sepcnn":
            num_blocks = trial.suggest_int("num_blocks", **search_space["num_blocks"])
        elif args.model_type == "rnn":
            hidden_size = trial.suggest_int(
                "hidden_size", **search_space["hidden_size"]
            )

        elif args.model_type == "transformer":
            d_model = trial.suggest_int("d_model", **search_space["d_model"])
            num_enc_layers = 1
            nhead = 1
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        # set optuna config
        optuna_config = {
            "optuna_hw_target": args.optuna_hw_target,
            "f1_thresholds": f1_thresholds,
        }

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

        # set model config
        model_config = {
            "num_in_features": data_config["num_in_features"],
            "num_out_features": data_config["num_out_features"],
            "seq_len": int(args.window_size / args.downsampling_rate),
            "model_type": args.model_type,
        }
        if (
            model_config["model_type"] == "cnn"
            or model_config["model_type"] == "sepcnn"
        ):
            model_config.update(
                {
                    "num_blocks": num_blocks,
                }
            )
        elif model_config["model_type"] == "rnn":
            model_config.update(
                {
                    "hidden_size": hidden_size,
                    "num_rnn_layers": 1,
                    "cell_type": args.cell_type,
                    "batch_size": batch_size,
                }
            )
        elif model_config["model_type"] == "transformer":
            model_config.update(
                {
                    "d_model": d_model,
                    "ffn_dim": d_model * 4,  # typically 4x d_model
                    "num_enc_layers": num_enc_layers,
                    "nhead": nhead,
                }
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        # set exp_config
        exp_config = {
            "exp_mode": args.exp_mode,
            "train_mode": train_mode,
            "exp_base_dir": args.exp_base_dir,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": args.num_epochs,
            "enable_qat": args.enable_qat,
            "given_timestamp": None,
            "enable_hw_simulation": args.enable_hw_simulation,
        }

        quant_config = None
        if exp_config["enable_qat"]:
            quant_config = {
                "model_name": "network",
                "quant_bits": quant_bits,
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
            exp_config["enable_hw_simulation"] = False
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
        exp_config["enable_hw_simulation"] = args.enable_hw_simulation
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
        timestamp, val_metrics, test_metrics, int_test_metrics, hw_metrics = (
            run_one_experiment(
                data_config=data_config,
                model_config=model_config,
                exp_config=exp_config,
                wandb_config=wandb_config,
                quant_config=quant_config,
                hw_config=hw_config,
                use_optuna=True,
                optuna_config=optuna_config,
            )
        )
        acc_target = val_metrics["best_val_f1"]

        user_attrs = {
            "timestamp": timestamp,
            **val_metrics,
            **test_metrics,
        }

        if not args.enable_qat:
            return acc_target

        user_attrs.update(**int_test_metrics)  # of quantized model

        if not args.enable_hw_simulation:
            trial.set_user_attr("user_attrs", user_attrs)
            return acc_target

        # --- Log all hardware metrics ---
        for k, v in hw_metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    user_attrs[f"{k}/{kk}"] = vv
            else:
                user_attrs[k] = v

        # ------- Deployability Condition ------
        if hw_metrics["res_info"].get("is_deployable") == False:
            print(
                f"[PRUNE] Resource Utilization {hw_metrics['res_info']} is not deployable"
            )
            hw_metrics["failure_type"] = "deployability_failure"
            safe_wandb_log(hw_metrics)
            trial.set_user_attr(user_attrs)
            raise optuna.exceptions.TrialPruned()

        # ------- Latency Condition ------
        if hw_metrics["time(ms)"] is None or float(hw_metrics["time(ms)"]) > float(
            LATENCY_THRESHOLD
        ):
            print(
                f"[PRUNE] Latency {hw_metrics['time(ms)']} > LATENCY_THRESHOLD {LATENCY_THRESHOLD}"
            )
            hw_metrics["failure_type"] = "latency_failure"
            safe_wandb_log(hw_metrics)
            trial.set_user_attr(user_attrs)
            return optuna.exceptions.TrialPruned()

        #  ------- Power Condition ------
        if (
            hw_metrics["power_info"].get("total_power(mW)") is None
            or float(hw_metrics["power_info"].get("total_power(mW)")) > POWER_THRESHOLD
        ):
            print("[PRUNE] Power info is not available")
            hw_metrics["failure_type"] = "power_failure"
            safe_wandb_log(hw_metrics)
            trial.set_user_attr(user_attrs)
            return optuna.exceptions.TrialPruned()

        # ------- Energy Condition ------
        if (
            hw_metrics["energy(muJ)"] is None
            or float(hw_metrics["energy(muJ)"]) > ENERGY_THRESHOLD
        ):
            print("[PRUNE] Energy info is not available")
            hw_metrics["failure_type"] = "energy_failure"
            safe_wandb_log(hw_metrics)
            trial.set_user_attr(user_attrs)
            return optuna.exceptions.TrialPruned()

        # ------- All constraints passed -------
        safe_wandb_log(hw_metrics)
        target_map = {
            "power": hw_metrics["power_info"].get("total_power(mW)"),
            "latency": hw_metrics["time(ms)"],
            "energy": hw_metrics["energy(muJ)"],
        }

        if args.optuna_hw_target not in target_map:
            raise ValueError(f"Unsupported optuna_hw_target: {args.optuna_hw_target}")
        trial.set_user_attr("user_attrs", user_attrs)

        return acc_target, target_map[args.optuna_hw_target]

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # wandb config
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_mode", type=str)

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
    parser.add_argument(
        "--cell_type", type=str, choices=["lstm", "gru"], default="lstm"
    )
    parser.add_argument("--enable_fused_ffn", action="store_true")

    # experiment config
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument(
        "--train_strategy",
        type=str,
        choices=["personalized", "generalized", "finetuning"],
        required=True,
    )
    parser.add_argument("--exp_base_dir", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--enable_qat", action="store_true")
    parser.add_argument("--enable_hw_simulation", action="store_true")

    # quantization config
    parser.add_argument("--qat_warmup_epochs", type=int, default=0)

    # hw simulation config
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--target_hw", type=str, choices=["amd", "lattice"])
    parser.add_argument(
        "--fpga_type",
        type=str,
        choices=["xc7s15ftgb196-2", "xc7s25ftgb196-2", "xc7s50ftgb196-2"],
        help="FPGA type for HW simulation",  # only for AMD
        default="xc7s15ftgb196-2",
    )

    # optuna configs
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument(
        "--optuna_hw_target",
        type=str,
        choices=["power", "latency", "energy", None],
        default=None,
    )
    args = parser.parse_args()

    # ------------------------- setup optuna db -------------------------
    os.makedirs(args.exp_base_dir, exist_ok=True)
    study_name = f"{args.train_strategy}_{args.enable_qat}"
    db_path = os.path.join(args.exp_base_dir, f"{study_name}.db")
    storage = RDBStorage(f"sqlite:///{db_path}")

    if args.enable_qat and args.enable_hw_simulation:
        directions = ["maximize", "minimize"]  # [val_f1, target_hw_metric]
    else:
        directions = ["maximize"]  # [val_f1]

    study = optuna.create_study(
        directions=directions,
        sampler=NSGAIISampler(),
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    study.optimize(
        partial(objective, args=args), n_trials=args.n_trials, catch=(Exception,)
    )

    json_all_path = f"{args.exp_base_dir}/all_trials.json"
    json_pareto_path = f"{args.exp_base_dir}/pareto_trials.json"
    save_trials_records(json_path=json_all_path, study=study, only_best=False)
    save_trials_records(json_path=json_pareto_path, study=study, only_best=True)

    if len(directions) == 2:
        plot_pareto_from_json(
            json_all_path,
            json_pareto_path,
            save_path=f"{args.exp_base_dir}/pareto_plot.pdf",
        )
    elif len(directions) == 1:
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig1.write_html(f"{args.exp_base_dir}/optimization_history.html")
        fig2.write_html(f"{args.exp_base_dir}/param_importances.html")
    else:
        raise ValueError("Invalid number of objectives. Must be 1 or 2.")
