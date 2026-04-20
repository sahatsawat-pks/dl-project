"""
dl_utils.py — Core training and evaluation utilities.
Matches the style of the course example (trainer.py + dl_utils.py pattern).
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def train_one_epoch(
        dataloader, model, loss_fn,
        optimizer, epoch,
        device, writer,
        log_step_interval=50):
    """Run one training epoch, logging batch loss to TensorBoard."""
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.
    last_loss = 0.

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % log_step_interval == 0:
            last_loss = running_loss / log_step_interval
            current   = (i + 1) * len(X)
            print(f"  loss: {last_loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar(
                'Loss/train_running',
                last_loss,
                epoch * len(dataloader) + i
            )
            running_loss = 0.

    return last_loss


def test(dataloader, model, loss_fn, device):
    """Evaluate model on a dataloader. Returns loss, predictions tensor, true labels tensor."""
    num_batches = len(dataloader)
    model.eval()

    loss = 0.
    y_preds, y_trues, y_probs = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred     = model(X)
            y_pred   = pred.argmax(1)
            probs    = F.softmax(pred, dim=1)[:, 1]

            loss    += loss_fn(pred, y).item()
            y_preds.append(y_pred)
            y_trues.append(y)
            y_probs.append(probs)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    y_probs = torch.cat(y_probs)

    return loss / num_batches, y_preds, y_trues, y_probs


def compute_performance(y_preds, y_trues, y_probs=None):
    """Compute accuracy, F1, precision, recall (and AUC if probs given)."""
    yp = y_preds.cpu().numpy()
    yt = y_trues.cpu().numpy()

    perf = {
        'accuracy' : accuracy_score(yt, yp),
        'f1'       : f1_score(yt, yp, zero_division=0),
        'precision': precision_score(yt, yp, zero_division=0),
        'recall'   : recall_score(yt, yp, zero_division=0),
    }
    if y_probs is not None:
        perf['auc'] = roc_auc_score(yt, y_probs.cpu().numpy())
    return perf
