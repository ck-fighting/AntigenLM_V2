import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANTIBERTY_ROOT = os.path.join(PROJECT_ROOT, "LLM", "antiberty")
if ANTIBERTY_ROOT not in sys.path:
    sys.path.insert(0, ANTIBERTY_ROOT)

from antiberty import AntiBERTyRunner  # noqa: E402


class LocalAntiBERTyRunner(AntiBERTyRunner):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trained_models_dir = os.path.join(ANTIBERTY_ROOT, "antiberty", "trained_models")
        checkpoint_path = os.path.join(trained_models_dir, "AntiBERTy_md_smooth")
        vocab_file = os.path.join(trained_models_dir, "vocab.txt")

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"AntiBERTy checkpoint directory not found: {checkpoint_path}")
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"AntiBERTy vocab file not found: {vocab_file}")

        self.model = transformers.AutoModelForMaskedLM.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()
        self.tokenizer = transformers.BertTokenizer(vocab_file=vocab_file, do_lower_case=False)


class AlignmentModel(nn.Module):
    def __init__(self, antigen_model_path, proj_dim=512):
        super().__init__()

        self.ag_encoder = AutoModel.from_pretrained(antigen_model_path)
        if hasattr(self.ag_encoder, "gradient_checkpointing_enable"):
            self.ag_encoder.gradient_checkpointing_enable()

        self.ab_runner = LocalAntiBERTyRunner()
        self.ab_runner.model.eval()
        for param in self.ab_runner.model.parameters():
            param.requires_grad = False

        ag_hidden_size = getattr(self.ag_encoder.config, "hidden_size", 1280)
        self.ag_proj = nn.Sequential(
            nn.Linear(ag_hidden_size, ag_hidden_size),
            nn.ReLU(),
            nn.Linear(ag_hidden_size, proj_dim),
        )
        self.ab_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim),
        )

    def masked_mean_pooling(self, last_hidden_state, mask):
        mask = mask.float()
        expanded_mask = mask.unsqueeze(-1).expand_as(last_hidden_state)
        summed = (last_hidden_state * expanded_mask).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return summed / counts

    def encode_antigen(self, input_ids, attention_mask, pool_mask=None):
        outputs = self.ag_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if pool_mask is None:
            pool_mask = attention_mask.float()
        ag_global = self.masked_mean_pooling(hidden, pool_mask)
        ag_emb = self.ag_proj(ag_global)
        return F.normalize(ag_emb, p=2, dim=1)

    def _pool_antiberty_embedding(self, embedding, raw_mask=None):
        residue_embedding = embedding[1:-1]
        if residue_embedding.numel() == 0:
            return embedding.mean(dim=0)

        if raw_mask is not None and isinstance(raw_mask, str):
            raw_bits = [float(bit) for bit in raw_mask.strip()]
            limit = min(len(raw_bits), residue_embedding.size(0))
            if limit > 0:
                bits = torch.tensor(
                    raw_bits[:limit],
                    device=residue_embedding.device,
                    dtype=residue_embedding.dtype,
                )
                if bits.sum() > 0:
                    return (residue_embedding[:limit] * bits.unsqueeze(-1)).sum(dim=0) / bits.sum().clamp(min=1.0)

        return residue_embedding.mean(dim=0)

    def encode_antibody_chain(self, sequences, raw_masks):
        embeddings = self.ab_runner.embed([seq.replace("J", "L") for seq in sequences])
        device = next(self.ag_proj.parameters()).device
        pooled = [self._pool_antiberty_embedding(embedding.to(device), raw_mask) for embedding, raw_mask in zip(embeddings, raw_masks)]
        return torch.stack(pooled, dim=0)

    def encode_antibody(self, heavy_sequences, light_sequences, vh_raw_masks, vl_raw_masks):
        with torch.no_grad():
            heavy_global = self.encode_antibody_chain(heavy_sequences, vh_raw_masks)
            light_global = self.encode_antibody_chain(light_sequences, vl_raw_masks)
            ab_global = torch.cat([heavy_global, light_global], dim=-1)

        ab_emb = self.ab_proj(ab_global)
        return F.normalize(ab_emb, p=2, dim=1)

    def forward(
        self,
        ag_input_ids,
        ag_attention_mask,
        heavy_sequences,
        light_sequences,
        ag_pool_mask=None,
        vh_raw_masks=None,
        vl_raw_masks=None,
    ):
        ag_emb = self.encode_antigen(ag_input_ids, ag_attention_mask, ag_pool_mask)
        ab_emb = self.encode_antibody(heavy_sequences, light_sequences, vh_raw_masks, vl_raw_masks)
        return ag_emb, ab_emb
