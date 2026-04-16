import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
import pandas as pd
import numpy as np
import logging

class AntigenEpitopeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024, soft_labels_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_aa_len = max_length - 2 # Reserve space for <cls> and <eos>
        self.soft_labels_dir = soft_labels_dir
        
        # Read the CSV
        logging.info(f"Loading dataset from {csv_file} ...")
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Antigen_Sequence'])
        
        self.chunks = []
        step = self.max_aa_len // 2
        
        logging.info("Applying sliding window chunking to preserve long sequence data...")
        for row in df[['Antigen_Sequence', 'Matched_Epitopes']].itertuples(index=False):
            sequence = str(row[0])
            epitopes_str = row[1]
            
            # Find the global character-level mask before slicing
            global_char_mask = self._find_epitope_indices(sequence, epitopes_str)
            seq_len = len(sequence)
            
            if seq_len <= self.max_aa_len:
                self.chunks.append((sequence, global_char_mask))
            else:
                # Sequence is too long: slide a window across it
                for i in range(0, seq_len - self.max_aa_len + 1, step):
                    sub_seq = sequence[i:i + self.max_aa_len]
                    sub_mask = global_char_mask[i:i + self.max_aa_len]
                    self.chunks.append((sub_seq, sub_mask))
                    
                # Ensure the very tail of the sequence is captured if step doesn't align perfectly
                if (seq_len - self.max_aa_len) % step != 0:
                    sub_seq = sequence[-self.max_aa_len:]
                    sub_mask = global_char_mask[-self.max_aa_len:]
                    self.chunks.append((sub_seq, sub_mask))
                    
        logging.info(f"Processed {len(df)} original sequences into {len(self.chunks)} overlapping chunks.")

    def __len__(self):
        return len(self.chunks)

    def _find_epitope_indices(self, sequence, epitopes_str):
        """Find character-level 0-based indices for all epitopes in the sequence."""
        epitope_mask = np.zeros(len(sequence), dtype=bool)
        
        # Robustly handle Pandas NaN, None, empty strings, and string literal 'nan'
        if pd.isna(epitopes_str) or str(epitopes_str).strip() == "" or str(epitopes_str).strip().lower() == "nan":
            return epitope_mask
            
        epitopes = str(epitopes_str).split(';')
        for ep in epitopes:
            ep = ep.strip()
            if not ep: continue
            
            # Find all occurrences of the epitope in the sequence
            start = 0
            while True:
                idx = sequence.find(ep, start)
                if idx == -1:
                    break
                epitope_mask[idx : idx + len(ep)] = True
                start = idx + 1
        return epitope_mask

    def __getitem__(self, idx):
        sequence, char_epitope_mask = self.chunks[idx]

        # Tokenize (ESM2 uses <cls> at start and <eos> at end)
        # REMOVED padding="max_length" for extreme performance boosts! 
        # Sequences are dynamically padded inside the DataCollator per batch.
        encoded = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ESM2 tokenizes amino acids 1-to-1.
        # Token 0 is <cls>, Token 1 is 1st AA, ..., Token N+1 is <eos>
        token_epitope_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        seq_len = min(len(sequence), self.max_length - 2) # Account for cls and eos
        
        for i in range(seq_len):
            if char_epitope_mask[i]:
                token_epitope_mask[i + 1] = True

        # Create labels: full true sequence ids. Masking ignores (-100) will be handled in the DataCollator.
        labels = input_ids.clone()
        
        # Structure mask: True for structural backbone (unmasked, valid AAs, non-epitopes)
        special_token_mask = (input_ids == self.tokenizer.cls_token_id) | \
                             (input_ids == self.tokenizer.eos_token_id) | \
                             (input_ids == self.tokenizer.pad_token_id) | \
                             (attention_mask == 0)
                             
        structure_mask = (~token_epitope_mask) & (~special_token_mask)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,  # Only epitope regions have valid token IDs
            "structure_mask": structure_mask,
            "epitope_mask": token_epitope_mask
        }
        
        if self.soft_labels_dir is not None:
            import os
            soft_labels_path = os.path.join(self.soft_labels_dir, f"seq_{idx}.pt")
            if os.path.exists(soft_labels_path):
                # Float32 casting optional here, but good for precise KD matching
                item["soft_labels_target"] = torch.load(soft_labels_path, map_location="cpu", weights_only=True)

        return item

class EpitopeMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Biased Dynamic Masking Generator.
    """
    def __init__(self, *args, **kwargs):
        self.epitope_bias = kwargs.pop("epitope_bias", 0.80)
        super().__init__(*args, **kwargs)

    def torch_call(self, examples):
        # 1. Dynamic Padding Logic (VRAM and Compute Saver!)
        # Pad only to the longest sequence in the *current batch*, not 1024!
        max_len_in_batch = max(len(ex["input_ids"]) for ex in examples)
        
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "epitope_mask": [], "structure_mask": []}
        
        for ex in examples:
            pad_len = max_len_in_batch - len(ex["input_ids"])
            
            # ESM2 pad_token is 1. Attention mask pad is 0. False for masks. -100 for labels.
            batch["input_ids"].append(torch.cat([ex["input_ids"], torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)]))
            batch["attention_mask"].append(torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["labels"].append(torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            batch["epitope_mask"].append(torch.cat([ex["epitope_mask"], torch.zeros(pad_len, dtype=torch.bool)]))
            batch["structure_mask"].append(torch.cat([ex["structure_mask"], torch.zeros(pad_len, dtype=torch.bool)]))
            
        batch = {k: torch.stack(v) for k, v in batch.items()}
        
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"].clone()
        epitope_mask = batch.pop("epitope_mask")
        structure_mask = batch["structure_mask"]

        bsz, seq_len = input_ids.shape
        masked_indices = torch.zeros(input_ids.shape, dtype=torch.bool)
        
        for i in range(bsz):
            # Active tokens: union of epitopes and structural backbone (ignores padding/special)
            active_mask = (epitope_mask[i] | structure_mask[i])
            active_count = active_mask.sum().item()
            
            if active_count == 0:
                continue
                
            total_mask_count = int(self.mlm_probability * active_count)
            # Epiope masking target configured from parameter
            target_ep_mask_count = int(self.epitope_bias * total_mask_count)
            
            ep_indices = torch.where(epitope_mask[i])[0]
            bb_indices = torch.where(structure_mask[i])[0]
            
            # If there are fewer epitope tokens than the target, mask all of them
            # and shift the remaining quota to the backbone to hit exactly 20% masking globally
            actual_ep_mask_count = min(target_ep_mask_count, len(ep_indices))
            actual_bb_mask_count = min(total_mask_count - actual_ep_mask_count, len(bb_indices))
            
            # Randomly select indices
            if actual_ep_mask_count > 0:
                selected_ep = ep_indices[torch.randperm(len(ep_indices))[:actual_ep_mask_count]]
                masked_indices[i, selected_ep] = True
                
            if actual_bb_mask_count > 0:
                selected_bb = bb_indices[torch.randperm(len(bb_indices))[:actual_bb_mask_count]]
                masked_indices[i, selected_bb] = True
                
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest 10% of the time we keep the masked input tokens unchanged
        
        batch["input_ids"] = input_ids
        
        # Cross Entropy Loss is ONLY computed on the dynamically masked tokens
        labels[~masked_indices] = -100
        batch["labels"] = labels
        
        return batch
