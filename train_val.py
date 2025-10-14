import os
import torch
import wandb
import torch.nn as nn
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

from config import DEVICE
from models.build_model import build_model
from utils.exp_utils import setup_logger
from utils.EarlyStopping import EarlyStopping
from utils.plots import plot_learning_curve
from utils.eval_metrics import get_classification_metrics


def train_val(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_config: dict,
    exp_config: dict,
    quant_config: dict = None,
):
    batch_size = exp_config["batch_size"]
    lr = exp_config["lr"]
    num_epochs = exp_config["num_epochs"]
    model_save_dir = exp_config["model_save_dir"]
    fig_save_dir = exp_config["fig_save_dir"]
    log_save_dir = exp_config["log_save_dir"]

    # get dataloaders
    print("Number of samples in train_dataset:", len(train_dataset))
    print("Number of samples in val_dataset:", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # build model
    model = build_model(model_config, exp_config["enable_qat"]).to(DEVICE)
    wandb.log(model_config)

    # set up logging
    logger = setup_logger(
        "train_val_logger", os.path.join(log_save_dir, "train_val_logfile.log")
    )

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(
        patience=10, verbose=True, path=model_save_dir, monitor="val_f1", mode="max"
    )

    all_train_epoch_losses = []
    all_val_epoch_losses = []
    all_val_epoch_f1_scores = []

    qat_warmup_epochs = quant_config.get("qat_warmup_epochs")
    for epoch in range(num_epochs):

        # training phase
        model.train()
        sum_train_loss = 0
        train_preds, train_targets = [], []

        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if exp_config["enable_qat"]:
                pred = model(inputs=x, enable_simquant=epoch > qat_warmup_epochs)
            else:
                pred = model(x)
            pred = pred.squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            sum_train_loss += loss.item()
            train_preds.append(pred.argmax(1).cpu())
            train_targets.append(y.cpu())

        epoch_train_loss = sum_train_loss / len(train_dataloader)
        all_train_epoch_losses.append(epoch_train_loss)

        # validation phase
        model.eval()
        sum_val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if exp_config["enable_qat"]:
                    pred = model(inputs=x, enable_simquant=epoch > qat_warmup_epochs)
                else:
                    pred = model(x)
                pred = pred.squeeze(1)
                loss = criterion(pred, y)
                sum_val_loss += loss.item()

                val_preds.append(pred.argmax(1).cpu())
                val_targets.append(y.cpu())

        epoch_val_loss = sum_val_loss / len(val_dataloader)
        all_val_epoch_losses.append(epoch_val_loss)

        # Compute evaluation metrics
        metrics = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            **get_classification_metrics("train", train_preds, train_targets),
            **get_classification_metrics("val", val_preds, val_targets),
        }

        # Early stopping check
        early_stopping(metrics["val_f1"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        all_val_epoch_f1_scores.append(metrics["val_f1"])

        # Log metrics
        headers = ["epoch"] + list(metrics.keys())
        row = [[epoch + 1] + [f"{metrics[k]:.4f}" for k in metrics]]
        logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
        wandb.log(metrics)

    wandb.log({"timestamp": str(model_save_dir).split("/")[-1]})
    print("Training finished. Please find the checkpoint at:", model_save_dir)

    plot_learning_curve(
        epochs=range(1, len(all_train_epoch_losses) + 1),
        train_losses=all_train_epoch_losses,
        val_losses=all_val_epoch_losses,
        save_path=fig_save_dir,
        prefix=exp_config["enable_qat"],
    )
    val_metrics = {
        "best_val_loss": min(all_val_epoch_losses),
        "best_val_f1": max(all_val_epoch_f1_scores),
    }
    wandb.log(val_metrics)
    return val_metrics
