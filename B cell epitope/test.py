import argparse
import csv
import json
import math
import os
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DEFAULT_TEST_FASTA = os.path.join(CURRENT_DIR, "data", "BP3C50ID_external_test_set.fasta")
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "trained_model", "bcell_epitope_esm2_mlp.pt")
DEFAULT_ENCODER_PATH = os.path.join(PROJECT_ROOT, "LLM", "esm2_650M")
DEFAULT_RESULT_DIR = os.path.join(CURRENT_DIR, "result")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the residue-level B-cell epitope predictor on an external FASTA set."
    )
    parser.add_argument("--test-fasta", type=str, default=DEFAULT_TEST_FASTA)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--antigenlm-path", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for AntigenLM_V2 feature extraction.")
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


def compute_metrics(probabilities, labels, threshold=0.5):
    probabilities = probabilities.cpu()
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


def extract_test_features(samples, tokenizer, encoder, device, batch_size, max_length):
    residue_features = []
    residue_labels = []
    residue_metadata = []
    truncated_sequences = 0

    for start in tqdm(range(0, len(samples), batch_size), desc="Extracting test features"):
        batch = samples[start : start + batch_size]
        sequences = [sample["sequence"] for sample in batch]
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = encoder(**encoded)
            hidden_states = outputs.last_hidden_state

        attention_mask = encoded["attention_mask"]
        for i, sample in enumerate(batch):
            token_count = int(attention_mask[i].sum().item())
            residue_count = max(token_count - 2, 0)
            if residue_count == 0:
                continue

            if residue_count < len(sample["sequence"]):
                truncated_sequences += 1

            features = hidden_states[i, 1 : 1 + residue_count].detach().cpu()
            labels = torch.tensor(sample["labels"][:residue_count], dtype=torch.long)

            residue_features.append(features)
            residue_labels.append(labels)

            for residue_index in range(residue_count):
                residue_metadata.append(
                    {
                        "sample_id": sample["id"],
                        "residue_index": residue_index + 1,
                        "residue": sample["sequence"][residue_index],
                        "true_label": sample["labels"][residue_index],
                    }
                )

    if not residue_features:
        raise ValueError("No residue features were extracted. Check tokenizer/model settings.")

    return (
        torch.cat(residue_features, dim=0),
        torch.cat(residue_labels, dim=0),
        residue_metadata,
        truncated_sequences,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found: {args.model_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_dir, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    encoder_path = args.antigenlm_path or checkpoint.get("encoder_model_path") or DEFAULT_ENCODER_PATH

    print(f"Loading test FASTA: {args.test_fasta}")
    test_samples = read_fasta_samples(args.test_fasta)
    print(f"Total test sequences: {len(test_samples)}")

    print(f"Loading AntigenLM_V2 from: {encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"Loading classifier from: {args.model_path}")
    model = ResidueMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["mlp_state_dict"])
    model.eval()

    test_features, test_labels, residue_metadata, truncated_sequences = extract_test_features(
        test_samples,
        tokenizer,
        encoder,
        device,
        args.batch_size,
        args.max_length,
    )

    with torch.no_grad():
        logits = model(test_features.to(device)).cpu()
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

    metrics = compute_metrics(probabilities, test_labels)
    metrics["num_sequences"] = len(test_samples)
    metrics["num_residues"] = int(test_labels.numel())
    metrics["truncated_sequences"] = truncated_sequences
    metrics["model_path"] = args.model_path
    metrics["encoder_model_path"] = encoder_path
    metrics["best_epoch"] = checkpoint.get("best_epoch")
    metrics["loss_name"] = checkpoint.get("loss_name")
    metrics["pos_weight"] = checkpoint.get("pos_weight")
    metrics["focal_gamma"] = checkpoint.get("focal_gamma")

    prediction_csv_path = os.path.join(
        args.result_dir, "bcell_epitope_esm2_mlp_external_test_predictions.csv"
    )
    with open(prediction_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "residue_index",
                "residue",
                "true_label",
                "pred_label",
                "prob_epitope",
            ],
        )
        writer.writeheader()
        for meta, pred, prob in zip(residue_metadata, predictions.tolist(), probabilities.tolist()):
            writer.writerow(
                {
                    "sample_id": meta["sample_id"],
                    "residue_index": meta["residue_index"],
                    "residue": meta["residue"],
                    "true_label": meta["true_label"],
                    "pred_label": pred,
                    "prob_epitope": f"{prob:.6f}",
                }
            )

    metrics_json_path = os.path.join(args.result_dir, "bcell_epitope_esm2_mlp_external_test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        "External test metrics | "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"mcc={metrics['mcc']:.4f} "
        f"auc={metrics['auc']:.4f} "
        f"auc10={metrics['auc10']:.4f}"
    )
    print(f"Prediction CSV saved to: {prediction_csv_path}")
    print(f"Metrics JSON saved to: {metrics_json_path}")


if __name__ == "__main__":
    main()
