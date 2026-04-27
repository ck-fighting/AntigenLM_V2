import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DEFAULT_TRAIN_FASTA = os.path.join(CURRENT_DIR, "data", "BP3C50ID_training_set.fasta")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "LLM", "esm2_650M")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "trained_model")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a residue-level B-cell epitope predictor with frozen AntigenLM_V2 features."
    )
    parser.add_argument("--train-fasta", type=str, default=DEFAULT_TRAIN_FASTA)
    parser.add_argument("--antigenlm-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for MLP.")
    parser.add_argument("--llm-lr", type=float, default=1e-5, help="Learning rate for unfrozen LLM layers.")
    parser.add_argument("--llm-unfreeze-layers", type=int, default=5, help="Number of last LLM layers to unfreeze.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta_samples(fasta_path):
    samples = []
    current_id = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    samples.append(build_sample(current_id, "".join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        samples.append(build_sample(current_id, "".join(current_seq)))

    if not samples:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    return samples


def build_sample(sample_id, labeled_sequence):
    sequence = labeled_sequence.upper()
    labels = [1 if residue.isupper() else 0 for residue in labeled_sequence]
    if len(sequence) != len(labels):
        raise ValueError(f"Length mismatch in sample {sample_id}")
    return {
        "id": sample_id,
        "sequence": sequence,
        "labels": labels,
    }


def split_samples(samples, val_fraction, seed):
    if val_fraction <= 0 or len(samples) < 2:
        return samples, []

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(len(samples) * val_fraction))
    val_indices = set(indices[:val_count])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples


class ResidueMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class WeightedBCEFocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, labels):
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probabilities, 1 - probabilities)
        focal_weight = (1 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


class ResidueDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_roc_curve(labels, scores):
    labels = labels.to(torch.int64).cpu()
    scores = scores.to(torch.float64).cpu()
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None, None

    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order]
    tps = torch.cumsum(sorted_labels, dim=0, dtype=torch.float64)
    fps = torch.cumsum(1 - sorted_labels, dim=0, dtype=torch.float64)

    threshold_indices = torch.nonzero(sorted_scores[1:] != sorted_scores[:-1], as_tuple=False).flatten()
    threshold_indices = torch.cat(
        [threshold_indices, torch.tensor([sorted_labels.numel() - 1], dtype=torch.long)]
    )

    tps = tps[threshold_indices]
    fps = fps[threshold_indices]
    tps = torch.cat([torch.tensor([0.0], dtype=torch.float64), tps])
    fps = torch.cat([torch.tensor([0.0], dtype=torch.float64), fps])

    tpr = tps / positive_count
    fpr = fps / negative_count
    return fpr, tpr


def compute_auc(labels, scores):
    fpr, tpr = compute_roc_curve(labels, scores)
    if fpr is None or tpr is None:
        return 0.0
    return torch.trapz(tpr, fpr).item()


def compute_auc10(labels, scores, max_fpr=0.1):
    fpr, tpr = compute_roc_curve(labels, scores)
    if fpr is None or tpr is None:
        return 0.0

    cutoff = float(max_fpr)
    cutoff_tensor = torch.tensor(cutoff, dtype=fpr.dtype)
    if cutoff <= 0 or cutoff > 1:
        raise ValueError("max_fpr must be in (0, 1].")

    exact_matches = torch.nonzero(fpr == cutoff_tensor, as_tuple=False).flatten()
    if exact_matches.numel() > 0:
        end_idx = int(exact_matches[0].item()) + 1
        truncated_fpr = fpr[:end_idx]
        truncated_tpr = tpr[:end_idx]
    else:
        insertion_idx = int(torch.searchsorted(fpr, cutoff_tensor, right=False).item())
        left_idx = max(insertion_idx - 1, 0)
        right_idx = min(insertion_idx, fpr.numel() - 1)

        left_fpr = fpr[left_idx]
        right_fpr = fpr[right_idx]
        left_tpr = tpr[left_idx]
        right_tpr = tpr[right_idx]

        if right_fpr.item() == left_fpr.item():
            interpolated_tpr = right_tpr
        else:
            slope = (right_tpr - left_tpr) / (right_fpr - left_fpr)
            interpolated_tpr = left_tpr + slope * (cutoff_tensor - left_fpr)

        truncated_fpr = torch.cat([fpr[:insertion_idx], cutoff_tensor.unsqueeze(0)])
        truncated_tpr = torch.cat([tpr[:insertion_idx], interpolated_tpr.unsqueeze(0)])

    partial_auc = torch.trapz(truncated_tpr, truncated_fpr).item()
    return partial_auc / cutoff


def compute_metrics(logits, labels, threshold=0.5):
    probabilities = torch.sigmoid(logits).cpu()
    labels = labels.cpu()
    predictions = (probabilities >= threshold).long()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())

    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_divide(tp * tn - fp * fn, mcc_denominator)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auc": compute_auc(labels, probabilities),
        "auc10": compute_auc10(labels, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(encoder, model, loader, criterion, device, tokenizer, max_length, accumulation_steps=1, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        encoder.train()
        model.train()
        optimizer.zero_grad()
    else:
        encoder.eval()
        model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []
    total_residues = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, batch in enumerate(tqdm(loader, desc="Training" if is_train else "Validating")):
            sequences = [sample["sequence"] for sample in batch]
            batch_labels = [sample["labels"] for sample in batch]
            
            encoded = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = encoder(**encoded)
            hidden_states = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]

            residue_features = []
            residue_labels = []

            for i, labels in enumerate(batch_labels):
                token_count = int(attention_mask[i].sum().item())
                residue_count = max(token_count - 2, 0)
                if residue_count == 0:
                    continue

                res_labels = torch.tensor(labels[:residue_count], dtype=torch.long, device=device)
                features = hidden_states[i, 1 : 1 + residue_count]
                
                residue_features.append(features)
                residue_labels.append(res_labels)

            if not residue_features:
                continue

            features = torch.cat(residue_features, dim=0)
            labels = torch.cat(residue_labels, dim=0)

            logits = model(features)
            loss = criterion(logits, labels)

            if is_train:
                loss = loss / accumulation_steps
                loss.backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

            loss_val = loss.item() * (accumulation_steps if is_train else 1) * labels.size(0)
            total_loss += loss_val
            total_residues += labels.size(0)
            
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(total_residues, 1)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading FASTA: {args.train_fasta}")
    all_samples = read_fasta_samples(args.train_fasta)
    train_samples, val_samples = split_samples(all_samples, args.val_fraction, args.seed)

    print(f"Total sequences: {len(all_samples)}")
    print(f"Train sequences: {len(train_samples)}")
    print(f"Validation sequences: {len(val_samples)}")

    print(f"Loading AntigenLM_V2 from: {args.antigenlm_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.antigenlm_path)
    encoder = AutoModel.from_pretrained(args.antigenlm_path).to(device)
    
    for param in encoder.parameters():
        param.requires_grad = False
        
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
        for layer in layers[-args.llm_unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        print("Warning: Could not identify layers to unfreeze. Defaulting to full freezing.")

    train_dataset = ResidueDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b)

    val_loader = None
    if val_samples:
        val_dataset = ResidueDataset(val_samples)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)

    input_dim = encoder.config.hidden_size if hasattr(encoder, "config") else args.hidden_dim
    model = ResidueMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    positive_count = 0
    negative_count = 0
    for sample in train_samples:
        positive_count += sum(sample["labels"])
        negative_count += len(sample["labels"]) - sum(sample["labels"])
    pos_weight = negative_count / max(positive_count, 1)

    criterion = WeightedBCEFocalLoss(pos_weight=pos_weight, gamma=args.focal_gamma).to(device)
    
    llm_params = [p for p in encoder.parameters() if p.requires_grad]
    mlp_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": llm_params, "lr": args.llm_lr},
        {"params": mlp_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    best_score = -1.0
    best_metrics = None
    model_path = os.path.join(args.output_dir, "bcell_epitope_esm2_mlp.pt")
    finetuned_encoder_path = os.path.join(args.output_dir, "fine_tuned_encoder_esm2")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            encoder, model, train_loader, criterion, device, tokenizer, args.max_length, args.gradient_accumulation_steps, optimizer=optimizer
        )
        epoch_record = {"epoch": epoch, "train": train_metrics}

        if val_loader is not None:
            val_metrics = run_epoch(
                encoder, model, val_loader, criterion, device, tokenizer, args.max_length, args.gradient_accumulation_steps, optimizer=None
            )
            epoch_record["val"] = val_metrics
            score = val_metrics["f1"]
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_mcc={train_metrics['mcc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_mcc={val_metrics['mcc']:.4f}"
            )
        else:
            score = train_metrics["f1"]
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_mcc={train_metrics['mcc']:.4f}"
            )

        history.append(epoch_record)

        if score > best_score:
            best_score = score
            best_metrics = epoch_record
            torch.save(
                {
                    "mlp_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "encoder_model_path": finetuned_encoder_path,
                    "label_rule": "uppercase residue -> 1, lowercase residue -> 0",
                    "loss_name": "weighted_bce_focal",
                    "pos_weight": pos_weight,
                    "focal_gamma": args.focal_gamma,
                    "best_epoch": epoch,
                    "best_metrics": epoch_record,
                },
                model_path,
            )
            encoder.save_pretrained(finetuned_encoder_path)
            tokenizer.save_pretrained(finetuned_encoder_path)

    summary_path = os.path.join(args.output_dir, "bcell_epitope_antigenlm_v2_mlp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_fasta": args.train_fasta,
                "encoder_model_path": args.antigenlm_path,
                "num_sequences": len(all_samples),
                "train_sequences": len(train_samples),
                "validation_sequences": len(val_samples),
                "train_residues": positive_count + negative_count,
                "validation_residues": 0,
                "train_positive_residues": positive_count,
                "train_negative_residues": negative_count,
                "pos_weight": pos_weight,
                "focal_gamma": args.focal_gamma,
                "history": history,
                "best_metrics": best_metrics,
            },
            handle,
            indent=2,
        )

    print(f"Best model saved to: {model_path}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
