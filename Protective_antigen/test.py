import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "LLM", "esm2_650M")
ANTIGENLM_V2_MODEL_PATH = os.path.join(PROJECT_ROOT, "LLM", "AntigenLM_2", "merged_model")
DATA_DIR = os.path.join(CURRENT_DIR, "data", "PLDGL")
SEQUENCE_COL = "Sequence"
LABEL_COL = "Label"

FEATURE_MODEL_CHOICES = ("esm2_650m", "antigenlm_v2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate protective antigen classifiers with selectable feature extractors."
    )
    parser.add_argument(
        "--feature-model",
        choices=FEATURE_MODEL_CHOICES,
        default="antigenlm_v2",
        help="Feature extractor to use: plain ESM2 650M or AntigenLM_V2 LoRA model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for sequence feature extraction.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length passed to the tokenizer.",
    )
    return parser.parse_args()


def get_feature_suffix(feature_model):
    return feature_model.lower()


def load_feature_extractor(feature_model, device):
    if feature_model == "esm2_650m":
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        model = AutoModel.from_pretrained(BASE_MODEL_PATH)
    elif feature_model == "antigenlm_v2":
        tokenizer = AutoTokenizer.from_pretrained(ANTIGENLM_V2_MODEL_PATH)
        model = AutoModel.from_pretrained(ANTIGENLM_V2_MODEL_PATH)
    else:
        raise ValueError(f"Unsupported feature model: {feature_model}")

    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool_embeddings(token_embeddings, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def extract_features(sequences, tokenizer, model, batch_size, max_length, device):
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting features"):
            batch_sequences = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs)
            batch_features = mean_pool_embeddings(outputs.last_hidden_state, inputs["attention_mask"])
            features.append(batch_features.cpu().numpy())

    return np.vstack(features)


def safe_auc(y_true, y_score):
    if y_score is None or len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_aupr(y_true, y_score):
    if y_score is None or len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def evaluate_model(y_true, y_pred, y_score=None):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": safe_auc(y_true, y_score),
        "PR-AUC (AUPR)": safe_aupr(y_true, y_score),
    }


def get_model_score(clf, x_test):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x_test)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(x_test)
    return None


def display_model_name(model_file, feature_suffix):
    name = model_file.replace(".joblib", "")
    suffix = f"_{feature_suffix}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.replace("_", " ")


def write_results(results_dict, save_path):
    if not results_dict:
        raise RuntimeError("No evaluation results were generated. Please train models first.")

    with pd.ExcelWriter(save_path) as writer:
        if "All" in results_dict and not results_dict["All"].empty:
            results_dict["All"].to_excel(writer, sheet_name="Overall_ALL", index=False)
            print("\nOverall results:")
            print(results_dict["All"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        for domain_safe, df_res in results_dict.items():
            if domain_safe == "All" or df_res.empty:
                continue
            sheet_title = domain_safe[:31]
            df_res.to_excel(writer, sheet_name=sheet_title, index=False)


def main():
    args = parse_args()
    feature_suffix = get_feature_suffix(args.feature_model)
    trained_model_dir = os.path.join(CURRENT_DIR, "trained_model_test", feature_suffix)
    result_dir = os.path.join(CURRENT_DIR, "result", feature_suffix)
    save_path = os.path.join(result_dir, f"Evaluation_Metrics_MultiSheet_{feature_suffix}.xlsx")
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.isdir(trained_model_dir):
        raise FileNotFoundError(f"Trained model directory not found: {trained_model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using feature extractor: {args.feature_model}")
    print(f"Device: {device}")
    print(f"Loading trained models from: {trained_model_dir}")
    print(f"Using existing test files from: {DATA_DIR}")
    print(f"Results will be saved to: {save_path}")

    tokenizer, feature_model = load_feature_extractor(args.feature_model, device)
    domain_folders = [
        d for d in os.listdir(trained_model_dir) if os.path.isdir(os.path.join(trained_model_dir, d))
    ]
    results_dict = {}

    for domain_safe in sorted(domain_folders):
        print(f"\nEvaluating domain: {domain_safe}")
        test_data_path = os.path.join(DATA_DIR, f"test_set_{domain_safe}.xlsx")
        if not os.path.exists(test_data_path):
            print(f"Skipping {domain_safe}: missing test split {test_data_path}")
            continue

        test_df = pd.read_excel(test_data_path)
        x_test = extract_features(
            test_df[SEQUENCE_COL].astype(str).tolist(),
            tokenizer,
            feature_model,
            args.batch_size,
            args.max_length,
            device,
        )
        y_test = test_df[LABEL_COL].to_numpy().astype(int)

        domain_model_dir = os.path.join(trained_model_dir, domain_safe)
        model_files = [
            f
            for f in os.listdir(domain_model_dir)
            if f.endswith(f"_{feature_suffix}.joblib")
        ]
        if not model_files:
            print(f"Skipping {domain_safe}: no model files ending with _{feature_suffix}.joblib")
            continue

        branch_results = []
        for model_file in sorted(model_files):
            clf = joblib.load(os.path.join(domain_model_dir, model_file))
            y_pred = clf.predict(x_test)
            y_score = get_model_score(clf, x_test)
            metrics = evaluate_model(y_test, y_pred, y_score)
            branch_results.append(
                {
                    "Model": display_model_name(model_file, feature_suffix),
                    **metrics,
                }
            )

        results_dict[domain_safe] = pd.DataFrame(branch_results).sort_values(
            by="ROC-AUC",
            ascending=False,
            na_position="last",
        )

    write_results(results_dict, save_path)
    print(f"\nEvaluation completed. Report saved to: {save_path}")


if __name__ == "__main__":
    main()
