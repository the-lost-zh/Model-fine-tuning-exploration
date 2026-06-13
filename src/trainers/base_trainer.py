"""Training and evaluation loop with logging and checkpointing."""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device,
                    sparsity_lambda=0.0, scaler=None):
    """Train for one epoch.

    Args:
        model: the model to train
        loader: training DataLoader
        optimizer: optimizer
        criterion: loss function
        device: torch device
        sparsity_lambda: L1 sparsity weight (for SSF-Sparse, 0 otherwise)
        scaler: GradScaler for AMP (optional)

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                ce_loss = criterion(logits, y)
                sp_loss = _collect_sparsity_loss(model) if sparsity_lambda > 0 else 0.0
                loss = ce_loss + sparsity_lambda * sp_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            ce_loss = criterion(logits, y)
            sp_loss = _collect_sparsity_loss(model) if sparsity_lambda > 0 else 0.0
            loss = ce_loss + sparsity_lambda * sp_loss
            loss.backward()
            optimizer.step()

        total_loss += ce_loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def _collect_sparsity_loss(model):
    """Collect L1 sparsity loss from SparseScaleShift modules."""
    total = 0.0
    from ..models.ssf_sparse import SparseScaleShift
    for m in model.modules():
        if isinstance(m, SparseScaleShift):
            total += m.sparsity_loss()
    return total


def train_model(model, train_loader, val_loader, test_loader,
                lr=5e-3, weight_decay=1e-4, epochs=100,
                patience=10, device="cuda", sparsity_lambda=0.0,
                use_amp=True, save_dir=None, tag=""):
    """Full training loop with early stopping and checkpointing.

    Args:
        model: the model to train
        train_loader, val_loader, test_loader: DataLoaders
        lr: learning rate
        weight_decay: AdamW weight decay
        epochs: max epochs
        patience: early stopping patience
        device: torch device
        sparsity_lambda: L1 sparsity weight for SSF-Sparse
        use_amp: use automatic mixed precision
        save_dir: directory for checkpoints
        tag: experiment tag for checkpoint naming

    Returns:
        results dict with keys: best_val_acc, test_acc, train_time,
        best_epoch, train_acc, gpu_memory_mb
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    total_train_time = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            sparsity_lambda=sparsity_lambda, scaler=scaler,
        )
        train_time = time.time() - t0
        total_train_time += train_time

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {train_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model and evaluate on test set
    model.load_state_dict(best_state)
    model = model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Save checkpoint
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{tag}_best.pth" if tag else "best.pth"
        torch.save(best_state, os.path.join(save_dir, fname))

    # GPU memory
    gpu_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if "cuda" in str(device) else 0
    torch.cuda.reset_peak_memory_stats(device)

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time": total_train_time,
        "best_epoch": best_epoch,
        "train_acc": train_acc,
        "gpu_memory_mb": gpu_mem,
    }
