import os
import numpy as np
import torch


class EarlyStopping:
    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0.0,
        path=".",
        trace_func=print,
        monitor="val_acc",  # or "val_loss"
        mode="max",  # "max" for accuracy, "min" for loss
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_best = -np.inf if mode == "max" else np.inf
        self.delta = delta
        self.path = os.path.join(path, "best_model.pth")
        self.trace_func = trace_func
        self.monitor = monitor
        self.mode = mode

    def __call__(self, current_metric, model):
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(current_metric, model)
        elif (self.mode == "min" and score < self.best_score - self.delta) or (
            self.mode == "max" and score > self.best_score + self.delta
        ):
            self.best_score = score
            self._save_checkpoint(current_metric, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, current_metric, model):
        """Save model when monitored metric improves."""
        if self.verbose:
            self.trace_func(
                f"{self.monitor} improved to {current_metric:.6f}. Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.metric_best = current_metric
        print(f"Model saved to {self.path}")
