import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AgAbDataset(Dataset):
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
            format_name = "mage"
        elif "Antigen_AA" in cols:
            col_map = self.COLUMN_MAPS["sabdab"]
            format_name = "sabdab"
        else:
            raise ValueError(f"Cannot auto-detect CSV format. Columns found: {list(self.data.columns)}")

        self.data = self.data.dropna(subset=[col_map["ag"], col_map["vh"], col_map["vl"]])
        self.ag = self.data[col_map["ag"]].astype(str).tolist()
        self.heavy = self.data[col_map["vh"]].astype(str).tolist()
        self.light = self.data[col_map["vl"]].astype(str).tolist()
        self.m_ag = self._load_optional_column(col_map["m_ag"], len(self.ag))
        self.m_vh = self._load_optional_column(col_map["m_vh"], len(self.ag))
        self.m_vl = self._load_optional_column(col_map["m_vl"], len(self.ag))

        print(f"Loaded {len(self.ag)} pairs (format: {format_name})")

    def _load_optional_column(self, column_name, size):
        if column_name and column_name in self.data.columns:
            return self.data[column_name].tolist()
        return [None] * size

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


class AgAbCollateFn:
    def __init__(self, antigen_model_path, antigen_max_length=1024):
        self.ag_tokenizer = AutoTokenizer.from_pretrained(antigen_model_path)
        self.antigen_max_length = antigen_max_length

    def __call__(self, batches):
        ag_chains, heavy_chains, light_chains, ag_masks_raw, vh_masks_raw, vl_masks_raw = zip(*batches)

        ag_inputs = self.ag_tokenizer(
            list(ag_chains),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.antigen_max_length,
        )

        ag_pool_mask = self._build_mask_tensor(ag_masks_raw, ag_inputs["input_ids"], ag_inputs["attention_mask"])

        return (
            ag_inputs["input_ids"],
            ag_inputs["attention_mask"],
            list(heavy_chains),
            list(light_chains),
            ag_pool_mask,
            list(vh_masks_raw),
            list(vl_masks_raw),
        )

    def _build_mask_tensor(self, raw_masks, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        mask_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)

        for i, raw_mask in enumerate(raw_masks):
            if raw_mask is not None and isinstance(raw_mask, str):
                bits = [float(bit) for bit in raw_mask.strip()]
                limit = min(len(bits), max(seq_len - 2, 0))
                if limit > 0:
                    mask_tensor[i, 1 : 1 + limit] = torch.tensor(bits[:limit], dtype=torch.float32)
            else:
                mask_tensor[i] = self._default_residue_mask(input_ids[i], attention_mask[i]).float()

        fallback_rows = mask_tensor.sum(dim=1) == 0
        if fallback_rows.any():
            for idx in torch.nonzero(fallback_rows, as_tuple=True)[0]:
                mask_tensor[idx] = self._default_residue_mask(input_ids[idx], attention_mask[idx]).float()

        return mask_tensor

    def _default_residue_mask(self, input_ids, attention_mask):
        special_ids = set(self.ag_tokenizer.all_special_ids)
        valid = attention_mask.bool()
        non_special = torch.ones_like(valid, dtype=torch.bool)
        for token_id in special_ids:
            non_special &= input_ids.ne(token_id)
        return (valid & non_special).float()
