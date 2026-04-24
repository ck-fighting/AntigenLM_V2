"""
Split SABDab dataset into train/test by antigen clustering at 80% sequence identity.

Strategy:
1. Extract all unique antigen sequences
2. Run CD-HIT at 80% identity to form antigen clusters
3. Randomly split CLUSTERS (not individual samples) into train/test (~80/20)
4. All pairs belonging to a train-cluster go to train; all pairs of a test-cluster go to test
   => Guarantees that no antigen in the test set shares >80% identity with any antigen in the train set
"""

import pandas as pd
import subprocess
import os
import random
import tempfile

random.seed(42)

INPUT_CSV = "/home/dataset-local/Ag2AbLM/Data/SABDab/calm_sabdab_dataset_filtered.csv"
TRAIN_CSV = "/home/dataset-local/Ag2AbLM/Data/SABDab/sabdab_train_80.csv"
TEST_CSV  = "/home/dataset-local/Ag2AbLM/Data/SABDab/sabdab_test_80.csv"
TEST_RATIO = 0.2  # Target ~20% of pairs for test

# ── Step 1: Load data ────────────────────────────────────────────────
print("Step 1: Loading dataset...")
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['Antigen_AA', 'Antibody_VH_AA', 'Antibody_VL_AA'])
print(f"  Complete triplets: {len(df)}")

unique_antigens = df['Antigen_AA'].unique()
print(f"  Unique antigen sequences: {len(unique_antigens)}")

# ── Step 2: Write unique antigens to FASTA ───────────────────────────
print("\nStep 2: Writing antigens to FASTA...")
fasta_file = "sabdab_antigens.fasta"
cdhit_out  = "sabdab_antigens_clustered"

with open(fasta_file, 'w') as f:
    for i, seq in enumerate(unique_antigens):
        f.write(f">ag_{i}\n{seq}\n")

# ── Step 3: Run CD-HIT at 80% identity ──────────────────────────────
print("\nStep 3: Running CD-HIT at 80% sequence identity...")
cmd = f"cd-hit -i {fasta_file} -o {cdhit_out} -c 0.80 -n 5 -M 0 -T 0"
subprocess.run(cmd, shell=True, check=True)

# ── Step 4: Parse .clstr to get cluster assignments ──────────────────
print("\nStep 4: Parsing CD-HIT clusters...")
seq_to_cluster = {}
current_cluster_id = -1

with open(f"{cdhit_out}.clstr", 'r') as f:
    for line in f:
        if line.startswith(">Cluster"):
            current_cluster_id = int(line.strip().split()[-1])
        else:
            parts = line.strip().split(">")
            seq_id = parts[1].split("...")[0]
            seq_idx = int(seq_id.split("_")[1])
            seq_to_cluster[unique_antigens[seq_idx]] = current_cluster_id

n_clusters = current_cluster_id + 1
print(f"  Total clusters: {n_clusters}")

# ── Step 5: Split CLUSTERS into train/test ───────────────────────────
print("\nStep 5: Splitting clusters...")

# Map cluster_id -> list of antigen sequences
from collections import defaultdict
cluster_to_seqs = defaultdict(list)
for seq, cid in seq_to_cluster.items():
    cluster_to_seqs[cid].append(seq)

# Map cluster_id -> number of pairs in the dataframe
df['ag_cluster_id'] = df['Antigen_AA'].map(seq_to_cluster)
cluster_pair_counts = df.groupby('ag_cluster_id').size().to_dict()

# Sort clusters by size (descending) for a more balanced split
cluster_ids = sorted(cluster_pair_counts.keys(), key=lambda c: cluster_pair_counts[c], reverse=True)
random.shuffle(cluster_ids)

total_pairs = len(df)
target_test = int(total_pairs * TEST_RATIO)

test_clusters = set()
test_count = 0

for cid in cluster_ids:
    if test_count >= target_test:
        break
    test_clusters.add(cid)
    test_count += cluster_pair_counts[cid]

train_clusters = set(cluster_pair_counts.keys()) - test_clusters

print(f"  Train clusters: {len(train_clusters)}, Test clusters: {len(test_clusters)}")

# ── Step 6: Create the splits ────────────────────────────────────────
print("\nStep 6: Creating train/test DataFrames...")
df_train = df[df['ag_cluster_id'].isin(train_clusters)].copy()
df_test  = df[df['ag_cluster_id'].isin(test_clusters)].copy()

# Drop the helper column
df_train = df_train.drop(columns=['ag_cluster_id'])
df_test  = df_test.drop(columns=['ag_cluster_id'])

print(f"\n{'='*60}")
print(f"  FINAL SPLIT SUMMARY")
print(f"{'='*60}")
print(f"  Total pairs:   {len(df)}")
print(f"  Train pairs:   {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Test pairs:    {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
print(f"  Train unique Ag: {df_train['Antigen_AA'].nunique()}")
print(f"  Test unique Ag:  {df_test['Antigen_AA'].nunique()}")
print(f"{'='*60}")

# ── Step 7: Save ─────────────────────────────────────────────────────
df_train.to_csv(TRAIN_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)
print(f"\n  Train saved to: {TRAIN_CSV}")
print(f"  Test saved to:  {TEST_CSV}")

# ── Cleanup ──────────────────────────────────────────────────────────
os.remove(fasta_file)
os.remove(cdhit_out)
os.remove(f"{cdhit_out}.clstr")
print("\nDone!")
