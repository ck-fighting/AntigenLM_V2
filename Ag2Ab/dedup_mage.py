import pandas as pd
import subprocess
import os

input_csv = "/home/dataset-local/Ag2AbLM/Data/MAGE.csv"
output_csv = "/home/dataset-local/Ag2AbLM/Data/MAGE_dedup_90.csv"
fasta_file = "unique_antigens.fasta"
cdhit_out = "clustered_antigens"

print(f"Loading {input_csv}...")
df = pd.read_csv(input_csv, low_memory=False)
original_len = len(df)

# 1. Get unique antigens and save to fasta
unique_antigens = df['antigen_seq'].dropna().unique()
print(f"Found {len(unique_antigens)} unique antigen sequences.")

with open(fasta_file, 'w') as f:
    for i, seq in enumerate(unique_antigens):
        f.write(f">ag_{i}\n{seq}\n")

# 2. Run cd-hit
print("Running CD-HIT at 90% identity...")
cmd = f"cd-hit -i {fasta_file} -o {cdhit_out} -c 0.90 -n 5 -M 0 -T 0"
subprocess.run(cmd, shell=True, check=True)

# 3. Parse .clstr file to map each sequence to its representative
print("Parsing CD-HIT clusters...")
ag_map = {}
current_rep = None
cluster_seqs = []

with open(f"{cdhit_out}.clstr", 'r') as f:
    for line in f:
        if line.startswith(">Cluster"):
            if current_rep is not None:
                for seq in cluster_seqs:
                    ag_map[seq] = current_rep
            cluster_seqs = []
            current_rep = None
        else:
            # e.g., "0	123aa, >ag_0... *"
            parts = line.strip().split(">")
            seq_id = parts[1].split("...")[0]
            seq_idx = int(seq_id.split("_")[1])
            real_seq = unique_antigens[seq_idx]
            cluster_seqs.append(real_seq)
            if "*" in line:
                current_rep = real_seq

# Last cluster
if current_rep is not None:
    for seq in cluster_seqs:
        ag_map[seq] = current_rep

# 4. Map to dataframe and drop duplicates
print(f"Mapping {len(ag_map)} sequences to their cluster representatives...")
df['ag_cluster_rep'] = df['antigen_seq'].map(ag_map)

subset_for_dedup = ['ag_cluster_rep', 'VH_AA', 'VL_AA']
df_dedup = df.drop_duplicates(subset=subset_for_dedup).copy()
df_dedup = df_dedup.drop(columns=['ag_cluster_rep'])
new_len = len(df_dedup)

print(f"Original pairs: {original_len}")
print(f"Deduplicated pairs: {new_len}")
print(f"Removed {original_len - new_len} redundant pairs.")

df_dedup.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

# Cleanup
os.remove(fasta_file)
os.remove(cdhit_out)
os.remove(f"{cdhit_out}.clstr")
