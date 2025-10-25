import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from model import CNNAutoencoder
from dataset import get_dataloader


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss.

    From the paper: "we use a weighted cross-entropy loss and explore a range
    of different weights on the 'fire' labels to take into account the class imbalance."

    They use a weight of 3 on the "fire" class.
    """

    def __init__(self, pos_weight=3.0):
        """
        Args:
            pos_weight: Weight for positive class (fire). Default: 3.0 from the paper.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: Model predictions, shape [batch, 1, H, W]
            targets: Ground truth labels, shape [batch, 1, H, W]
            mask: Optional mask for uncertain labels, shape [batch, 1, H, W]
                  1 = valid, 0 = ignore (uncertain labels like cloud coverage)

        Returns:
            Scalar loss value
        """
        # Create mask for valid labels (0 or 1 only)
        valid_mask = ((targets == 0) | (targets == 1)).float()

        # Clip targets to be in [0, 1] range (in case there are invalid values)
        targets = torch.clamp(targets, 0, 1)

        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Clip predictions for numerical stability
        eps = 1e-7
        predictions = torch.clamp(predictions, eps, 1 - eps)

        # Compute binary cross-entropy manually with pos_weight
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        # Weighted BCE = -[y * pos_weight * log(p) + (1-y) * log(1-p)]
        bce = -(targets * self.pos_weight * torch.log(predictions) +
                (1 - targets) * torch.log(1 - predictions))

        # Apply valid mask
        bce = bce * valid_mask

        # Apply additional mask if provided
        if mask is not None:
            bce = bce * mask
            combined_mask = valid_mask * mask
            return bce.sum() / (combined_mask.sum() + 1e-7)
        else:
            return bce.sum() / (valid_mask.sum() + 1e-7)


def compute_metrics(predictions, targets, mask=None):
    """Compute precision, recall, and AUC-PR.

    Args:
        predictions: Model predictions, shape [batch, 1, H, W]
        targets: Ground truth labels, shape [batch, 1, H, W]
        mask: Optional mask for uncertain labels

    Returns:
        Dictionary with metrics
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(predictions)

    # Flatten
    probs_flat = probs.reshape(-1).cpu().numpy()
    targets_flat = targets.reshape(-1).cpu().numpy()

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1).cpu().numpy().astype(bool)
        probs_flat = probs_flat[mask_flat]
        targets_flat = targets_flat[mask_flat]

    # Filter out invalid/uncertain labels (anything not 0 or 1)
    valid_mask = (targets_flat == 0) | (targets_flat == 1)
    probs_flat = probs_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]

    # Convert to int for binary classification
    targets_flat = targets_flat.astype(int)

    # Check if we have both classes
    if len(np.unique(targets_flat)) < 2:
        # If only one class, return default metrics
        return {
            'auc_pr': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    # Compute precision-recall curve and AUC
    precision, recall, thresholds = precision_recall_curve(targets_flat, probs_flat)
    auc_pr = auc(recall, precision)

    # Compute precision and recall at default threshold (0.5)
    predictions_binary = (probs_flat > 0.5).astype(int)
    tp = np.sum((predictions_binary == 1) & (targets_flat == 1))
    fp = np.sum((predictions_binary == 1) & (targets_flat == 0))
    fn = np.sum((predictions_binary == 0) & (targets_flat == 1))

    precision_at_50 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_at_50 = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'auc_pr': auc_pr,
        'precision': precision_at_50,
        'recall': recall_at_50
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch.

    Args:
        model: The neural network model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, split="eval"):
    """Evaluate the model.

    Args:
        model: The neural network model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        split: Name of the split (for logging)

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{split.capitalize()}]")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            # Store for metrics
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

            pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)

    metrics['loss'] = total_loss / num_batches

    return metrics


def train(
    num_epochs=10,
    batch_size=32,
    learning_rate=0.0001,
    pos_weight=3.0,
    device=None,
    checkpoint_dir="models",
    log_dir="logs"
):
    """Main training function.

    Args:
        num_epochs: Number of epochs to train (paper uses 1000)
        batch_size: Batch size (paper uses 32-256, we default to 32)
        learning_rate: Learning rate (paper uses 0.0001)
        pos_weight: Weight for positive class (paper uses 3.0)
        device: Device to train on (auto-detect if None)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    writer = SummaryWriter(os.path.join(log_dir, run_name))

    # Create model
    print("Creating model...")
    model = CNNAutoencoder(in_channels=12, out_channels=1)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create loss and optimizer
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataloaders
    print("Loading datasets...")
    train_loader = get_dataloader("train", batch_size=batch_size, augment=True)
    eval_loader = get_dataloader("eval", batch_size=batch_size, augment=False)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Run name: {run_name}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    best_auc_pr = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Evaluate on validation set
        eval_metrics = evaluate(model, eval_loader, criterion, device, split="eval")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/eval", eval_metrics['loss'], epoch)
        writer.add_scalar("Metrics/AUC_PR", eval_metrics['auc_pr'], epoch)
        writer.add_scalar("Metrics/Precision", eval_metrics['precision'], epoch)
        writer.add_scalar("Metrics/Recall", eval_metrics['recall'], epoch)

        # Print metrics to console
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss:  {eval_metrics['loss']:.4f}")
        print(f"  AUC-PR:     {eval_metrics['auc_pr']:.4f}")
        print(f"  Precision:  {eval_metrics['precision']:.4f}")
        print(f"  Recall:     {eval_metrics['recall']:.4f}")

        # Prepare checkpoint metadata
        checkpoint_metadata = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_metrics['loss'],
            'auc_pr': eval_metrics['auc_pr'],
            'precision': eval_metrics['precision'],
            'recall': eval_metrics['recall'],
            'hyperparameters': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'pos_weight': pos_weight,
            },
            'timestamp': timestamp,
            'run_name': run_name,
        }

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(checkpoint_metadata, checkpoint_path)

        # Save best model separately
        if eval_metrics['auc_pr'] > best_auc_pr:
            best_auc_pr = eval_metrics['auc_pr']
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_metadata, best_model_path)
            print(f"  ✓ New best model! (AUC-PR: {best_auc_pr:.4f})")

        # Also save latest model
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint_metadata, latest_path)

    writer.close()
    print("\n✓ Training complete!")
    print(f"Best AUC-PR: {best_auc_pr:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train wildfire spread prediction model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Weight for fire class")
    parser.add_argument("--checkpoint-dir", type=str, default="models", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pos_weight=args.pos_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
