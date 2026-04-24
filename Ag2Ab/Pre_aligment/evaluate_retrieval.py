import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import AgAbCollateFn
from models import AlignmentModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestDataset(Dataset):
    COLUMN_MAPS = {
        "mage": {
            "ag": "antigen_seq",
            "vh": "VH_AA",
            "vl": "VL_AA",
            "m_ag": None,
            "m_vh": None,
            "m_vl": None,
        },
        "sabdab": {
            "ag": "Antigen_AA",
            "vh": "Antibody_VH_AA",
            "vl": "Antibody_VL_AA",
            "m_ag": "Antigen_Epitope",
            "m_vh": "Antibody_VH_Paratope",
            "m_vl": "Antibody_VL_Paratope",
        },
    }

    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)

        cols = set(self.data.columns)
        if "antigen_seq" in cols:
            col_map = self.COLUMN_MAPS["mage"]
        elif "Antigen_AA" in cols:
            col_map = self.COLUMN_MAPS["sabdab"]
        else:
            raise ValueError(f"Cannot auto-detect CSV format. Columns: {list(self.data.columns)}")

        self.data = self.data.dropna(subset=[col_map["ag"], col_map["vh"], col_map["vl"]])
        self.ag = self.data[col_map["ag"]].astype(str).tolist()
        self.heavy = self.data[col_map["vh"]].astype(str).tolist()
        self.light = self.data[col_map["vl"]].astype(str).tolist()
        self.m_ag = self.data[col_map["m_ag"]].tolist() if col_map["m_ag"] in self.data.columns else [None] * len(self.ag)
        self.m_vh = self.data[col_map["m_vh"]].tolist() if col_map["m_vh"] in self.data.columns else [None] * len(self.ag)
        self.m_vl = self.data[col_map["m_vl"]].tolist() if col_map["m_vl"] in self.data.columns else [None] * len(self.ag)
        print(f"Test set: {len(self.ag)} pairs")

    def __len__(self):
        return len(self.ag)

    def __getitem__(self, index):
        return (
            self.ag[index],
            self.heavy[index],
            self.light[index],
            self.m_ag[index],
            self.m_vh[index],
            self.m_vl[index],
        )


def get_metrics(sim_mat):
    n = sim_mat.shape[0]
    ranks = []
    for i in range(n):
        sorted_indices = torch.argsort(sim_mat[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)
    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mean_r = ranks.mean().item()
    median_r = ranks.median().item()
    return r1, r5, r10, mean_r, median_r


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    dataset = TestDataset(args.data_path)
    collate_fn = AgAbCollateFn(
        antigen_model_path=args.antigen_model_path,
        antigen_max_length=args.antigen_max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = AlignmentModel(
        antigen_model_path=args.antigen_model_path,
        proj_dim=args.proj_dim,
    ).to(device)

    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}")
        return

    print(f"Loading checkpoint {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    all_ag_embs = []
    all_ab_embs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            (
                ag_input_ids,
                ag_attention_mask,
                heavy_sequences,
                light_sequences,
                ag_pool_mask,
                vh_raw_masks,
                vl_raw_masks,
            ) = batch
            ag_input_ids = ag_input_ids.to(device)
            ag_attention_mask = ag_attention_mask.to(device)
            ag_pool_mask = ag_pool_mask.to(device)

            with torch.cuda.amp.autocast():
                ag_emb, ab_emb = model(
                    ag_input_ids=ag_input_ids,
                    ag_attention_mask=ag_attention_mask,
                    heavy_sequences=heavy_sequences,
                    light_sequences=light_sequences,
                    ag_pool_mask=ag_pool_mask,
                    vh_raw_masks=vh_raw_masks,
                    vl_raw_masks=vl_raw_masks,
                )

            all_ag_embs.append(ag_emb.float().cpu())
            all_ab_embs.append(ab_emb.float().cpu())

    ag_embs = torch.cat(all_ag_embs, dim=0)
    ab_embs = torch.cat(all_ab_embs, dim=0)
    sim_matrix = torch.matmul(ag_embs, ab_embs.T)

    ag_r1, ag_r5, ag_r10, ag_mean, ag_med = get_metrics(sim_matrix)
    ab_r1, ab_r5, ab_r10, ab_mean, ab_med = get_metrics(sim_matrix.T)

    print("\n" + "=" * 50)
    print("Retrieval Evaluation Results")
    print("=" * 50)
    print("Ag2Ab (Antigen queries Antibody):")
    print(f"  Recall@1:  {ag_r1:>5.2f}%")
    print(f"  Recall@5:  {ag_r5:>5.2f}%")
    print(f"  Recall@10: {ag_r10:>5.2f}%")
    print(f"  Mean Rank: {ag_mean:>5.2f}")
    print(f"  Med. Rank: {ag_med:>5.2f}")
    print("-" * 50)
    print("Ab2Ag (Antibody queries Antigen):")
    print(f"  Recall@1:  {ab_r1:>5.2f}%")
    print(f"  Recall@5:  {ab_r5:>5.2f}%")
    print(f"  Recall@10: {ab_r10:>5.2f}%")
    print(f"  Mean Rank: {ab_mean:>5.2f}")
    print(f"  Med. Rank: {ab_med:>5.2f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "data", "SABDab", "sabdab_test_80.csv"),
    )
    parser.add_argument(
        "--antigen_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "LLM", "esm2_650M"),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "Pre_aligment", "checkpoints_esm2_antiberty", "alignment_model_epoch_30.pt"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--antigen_max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    evaluate(parser.parse_args())
