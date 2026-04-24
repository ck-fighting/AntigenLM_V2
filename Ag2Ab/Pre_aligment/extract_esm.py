import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class AgDataset(Dataset):
    def __init__(self, csv_path, sequence_column="antigen_seq"):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        if sequence_column not in self.data.columns:
            if "Antigen_AA" in self.data.columns:
                sequence_column = "Antigen_AA"
            else:
                raise ValueError(f"Cannot find antigen sequence column in {csv_path}")
        self.sequences = self.data[sequence_column].astype(str).tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class CollateFn:
    def __init__(self, model_path, max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length

    def __call__(self, batches):
        return self.tokenizer(
            list(batches),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )


class AntigenLMEncoder(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
        summed = (hidden * expanded_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return summed / counts


@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        embedding = model(batch["input_ids"], batch["attention_mask"])
        embeddings.append(embedding.cpu().numpy())
    return np.concatenate(embeddings)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AgDataset(args.dataset_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=CollateFn(args.antigen_model_path, args.max_length),
        shuffle=False,
    )
    model = AntigenLMEncoder(args.antigen_model_path).to(device)
    embeddings = get_embeddings(model, loader, device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, embeddings)
    print("Embeddings shape:", embeddings.shape)
    print("Saved to:", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract antigen representations with AntigenLM_V2")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "data", "MAGE.csv"),
    )
    parser.add_argument(
        "--antigen_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "LLM", "AntigenLM_2", "merged_model"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "Pre_aligment", "features", "antigenlm_v2_antigen_embeddings.npy"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    main(parser.parse_args())
