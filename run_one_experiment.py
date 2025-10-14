import wandb
import optuna

from data import get_data
from train_val import train_val
from test import test
from hw_converter import convert2hw, run_hw_simulation
from utils import safe_print, set_base_paths, safe_wandb_log


def run_one_experiment(
    data_config: dict,
    model_config: dict,
    exp_config: dict,
    wandb_config: dict,
    quant_config: dict = None,
    hw_config: dict = None,
    use_optuna: bool = False,
    optuna_config: dict = None,
):

    # get datasets
    train_dataset, val_dataset, test_dataset = get_data(data_config)

    # set exp_save_path
    model_save_dir, fig_save_dir, log_save_dir = set_base_paths(
        exp_config["exp_base_dir"], exp_config["given_timestamp"]
    )
    exp_config["model_save_dir"] = model_save_dir
    exp_config["fig_save_dir"] = fig_save_dir
    exp_config["log_save_dir"] = log_save_dir
    timestamp = str(exp_config["model_save_dir"]).split("/")[-1]

    val_metrics = {}
    if exp_config["exp_mode"] == "train":
        # set up wandb
        wandb.init(
            project=wandb_config["name"],
            mode=wandb_config["mode"],
            config=wandb_config["config"],
        )
        val_metrics = train_val(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_config=model_config,
            exp_config=exp_config,
            quant_config=quant_config,
        )
    test_metrics = test(
        test_dataset=test_dataset, model_config=model_config, exp_config=exp_config
    )
    if exp_config["exp_mode"] == "train":
        wandb.log({"timestamp": timestamp})

    int_test_metrics, hw_metrics = {}, {}
    hw_metrics["did_hw_simulation"] = False

    if quant_config is not None and exp_config["enable_qat"]:
        model_config["enable_int_forward"] = True
        int_test_metrics = test(
            test_dataset=test_dataset,
            model_config=model_config,
            exp_config=exp_config,
        )

        if use_optuna:
            threshold = optuna_config["f1_thresholds"][quant_config["quant_bits"]]
            if val_metrics["best_val_f1"] < threshold:
                print(
                    f"[PRUNE] Accuracy {val_metrics['best_val_f1']:.3f} < threshold {threshold:.3f}"
                )
                hw_metrics["failure_type"] = "accuracy_failure"
                wandb.log(hw_metrics)
                raise optuna.exceptions.TrialPruned()

        if hw_config is not None and exp_config["enable_hw_simulation"]:

            # generate necessary files for hardware simulation
            convert2hw(
                model_type=model_config["model_type"],
                test_dataset=test_dataset,
                subset_size=hw_config["subset_size"],
                model_config=model_config,
                model_save_dir=exp_config["model_save_dir"],
                target_hw=hw_config["target_hw"],
            )

            # run hardware simulation
            hw_metrics = run_hw_simulation(
                model_save_dir=exp_config["model_save_dir"],
                hw_config=hw_config,
            )
            hw_metrics["did_hw_simulation"] = True
            safe_print("Resource Utilization", hw_metrics["res_info"])
            safe_print("Time", hw_metrics["time(ms)"], " (ms)")
            safe_print("Power Consumption", hw_metrics["power_info"])
            safe_print("Energy Consumption", hw_metrics["energy(muJ)"], " (muJ)")

            # log hw metrics to wandb
            if use_optuna == False and exp_config["exp_mode"] == "train":
                safe_wandb_log(hw_metrics)

        model_config["enable_int_forward"] = False  # necessary for finetuning

    return (timestamp, val_metrics, test_metrics, int_test_metrics, hw_metrics)
