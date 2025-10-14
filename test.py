import os
import torch
import wandb
import torch.nn as nn
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset

from config import DEVICE
from models.build_model import build_model
from utils.exp_utils import setup_logger
from utils.plots import plot_confusion_matrix
from utils.eval_metrics import get_model_complexity, get_classification_metrics


def test(
    test_dataset: Dataset,
    model_config: dict,
    exp_config: dict,
):
    batch_size = exp_config["batch_size"]
    exp_mode = exp_config["exp_mode"]
    model_save_dir = exp_config["model_save_dir"]
    fig_save_dir = exp_config["fig_save_dir"]
    log_save_dir = exp_config["log_save_dir"]

    # get dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    # build model and load weights

    model = build_model(model_config, exp_config["enable_qat"]).to(DEVICE)
    checkpoint = torch.load(model_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    # set up logging
    logger = setup_logger("test_logger", os.path.join(log_save_dir, "test_logfile.log"))

    # set criterion
    criterion = nn.CrossEntropyLoss()

    # set prefix
    prefix = (
        "int_"
        if exp_config["enable_qat"] == True
        and model_config["enable_int_forward"] == True
        else ""
    )

    # calculate model complexity
    model_complexity = get_model_complexity(model_config, model, exp_config, prefix)

    # test
    model.eval()
    sum_batch_losses = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for _, (features, target) in enumerate(test_dataloader):
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            pred = model(features)

            pred = pred.squeeze(1)
            pred = pred.to(target.device)
            test_batch_loss = criterion(pred, target)
            sum_batch_losses += test_batch_loss.item()

            pred = pred.argmax(dim=1)

            test_preds.append(pred.view(-1, 1).detach().cpu().numpy())
            test_targets.append(target.view(-1, 1).detach().cpu().numpy())

    test_loss = sum_batch_losses / len(test_dataloader)

    metrics = {f"{prefix}test_loss": test_loss}
    metrics.update(
        get_classification_metrics(f"{prefix}test", test_preds, test_targets)
    )

    # log test results
    print(f"---------------- {prefix}Test Results ----------------")

    headers = list(metrics.keys())[1:] + list(model_complexity.keys())
    row = [
        [f"{metrics[k]:.4f}" for k in list(metrics.keys())[1:]]
        + [
            (
                model_complexity[k]
                if isinstance(model_complexity[k], (int, float))
                else str(model_complexity[k])
            )
            for k in model_complexity.keys()
        ]
    ]
    logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
    if exp_mode == "train":

        wandb.log({**metrics, **model_complexity})

    plot_confusion_matrix(
        labels=[item for sublist in test_targets for item in sublist],
        preds=[item for sublist in test_targets for item in sublist],
        save_path=fig_save_dir,
        prefix=prefix,
    )

    return metrics
