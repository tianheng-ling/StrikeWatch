import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_learning_curve(
    epochs: list, train_losses: list, val_losses: list, save_path: str, prefix: bool
):
    prefix = "QAT_" if prefix else ""
    save_path = os.path.join(save_path, f"{prefix}learning_curve.pdf")

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_losses, label=f"Train Loss")
    ax.plot(epochs, val_losses, label=f"Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()


def plot_preds_truths(
    preds: list, truths: list, plot_len: int, save_path: str, prefix=None
):
    plt.plot(range(plot_len), truths[:plot_len], color="green", label="Ground Truths")
    plt.plot(range(plot_len), preds[:plot_len], color="red", label="Predictions")
    plt.ylabel("Target Value")
    plt.xlabel("Timesteps")
    plt.title("Partial preds on test set")
    plt.legend()
    plt.savefig(
        str(save_path) + f"/{prefix}preds_truths.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.clf()
    plt.close()


def plot_confusion_matrix(labels, preds, save_path, prefix):
    all_labels = range(0, 2)
    cm = confusion_matrix(labels, preds, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(save_path, f"{prefix}confusion_matrix.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()


def plot_pareto_from_json(
    json_all_path: str,
    json_pareto_path: str,
    save_path: str,
    x_key="objective1",
    y_key="objective2",
    x_label="Val MSE",
    y_label="Energy (mW)",
):
    with open(json_all_path, "r") as f:
        all_data = json.load(f)
    with open(json_pareto_path, "r") as f:
        pareto_data = json.load(f)

    all_points = {
        (round(t[x_key], 6), round(t[y_key], 6))
        for t in all_data
        if t[x_key] is not None and t[y_key] is not None
    }
    pareto_points = {
        (round(t[x_key], 6), round(t[y_key], 6))
        for t in pareto_data
        if t[x_key] is not None and t[y_key] is not None
    }

    non_pareto_points = all_points - pareto_points

    non_pareto_losses, non_pareto_hw = (
        zip(*non_pareto_points) if non_pareto_points else ([], [])
    )
    pareto_losses, pareto_hw = zip(*pareto_points) if pareto_points else ([], [])

    plt.figure(figsize=(4, 3))
    if non_pareto_points:
        plt.scatter(
            non_pareto_losses,
            non_pareto_hw,
            c="blue",
            label="Non-Pareto Front",
            alpha=0.6,
        )
    if pareto_points:
        plt.scatter(pareto_losses, pareto_hw, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
