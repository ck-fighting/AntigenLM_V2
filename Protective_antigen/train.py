import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "LLM", "esm2_650M")
ANTIGENLM_V2_MODEL_PATH = os.path.join(PROJECT_ROOT, "LLM", "AntigenLM_2", "merged_model")
DATA_DIR = os.path.join(CURRENT_DIR, "data", "PLDGL")
LABEL_COL = "Label"
SEQUENCE_COL = "Sequence"

FEATURE_MODEL_CHOICES = ("esm2_650m", "antigenlm_v2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train protective antigen classifiers with selectable feature extractors."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for classifiers.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_suffix(feature_model):
    return feature_model.lower()


def get_output_paths(feature_suffix):
    trained_model_dir = os.path.join(CURRENT_DIR, "trained_model_test", feature_suffix)
    os.makedirs(trained_model_dir, exist_ok=True)
    return trained_model_dir


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


def build_classifiers(seed):
    return {
        "Ridge_Regression": RidgeClassifier(class_weight="balanced"),
        "Balanced_Bagging": BalancedBaggingClassifier(random_state=seed),
        "Linear_SVC": LinearSVC(class_weight="balanced", max_iter=10000, random_state=seed),
        "Random_Forest": RandomForestClassifier(class_weight="balanced", random_state=seed),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=seed,
        ),
    }


def train_and_save_classifiers(x_train, y_train, trained_model_dir, domain_safe, feature_suffix, seed):
    domain_model_dir = os.path.join(trained_model_dir, domain_safe)
    os.makedirs(domain_model_dir, exist_ok=True)

    for name, clf in build_classifiers(seed).items():
        print(f"Training {name} for {domain_safe} with {feature_suffix} features...")
        clf.fit(x_train, y_train)
        model_path = os.path.join(domain_model_dir, f"{name}_{feature_suffix}.joblib")
        joblib.dump(clf, model_path)
        print(f"Saved: {model_path}")


def list_train_files():
    train_files = []
    for file_name in sorted(os.listdir(DATA_DIR)):
        if file_name.startswith("train_set_") and file_name.endswith(".xlsx"):
            domain_safe = file_name.removeprefix("train_set_").removesuffix(".xlsx")
            train_files.append((domain_safe, os.path.join(DATA_DIR, file_name)))
    return train_files


def main():
    args = parse_args()
    set_seed(args.seed)

    feature_suffix = get_feature_suffix(args.feature_model)
    trained_model_dir = get_output_paths(feature_suffix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using feature extractor: {args.feature_model}")
    print(f"Device: {device}")
    print(f"Trained models will be saved under: {trained_model_dir}")
    print(f"Using existing train files from: {DATA_DIR}")

    tokenizer, feature_model = load_feature_extractor(args.feature_model, device)
    train_files = list_train_files()
    if not train_files:
        raise FileNotFoundError(f"No train_set_*.xlsx files found under: {DATA_DIR}")

    for domain_safe, train_path in train_files:
        print(f"\nProcessing domain: {domain_safe}")
        train_df = pd.read_excel(train_path)

        if train_df[LABEL_COL].nunique() < 2:
            print(f"Skipping {domain_safe}: only one class is present.")
            continue

        sequences = train_df[SEQUENCE_COL].astype(str).tolist()
        labels = train_df[LABEL_COL].to_numpy()
        x_train = extract_features(
            sequences,
            tokenizer,
            feature_model,
            args.batch_size,
            args.max_length,
            device,
        )

        train_and_save_classifiers(
            x_train,
            labels,
            trained_model_dir,
            domain_safe,
            feature_suffix,
            args.seed,
        )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
