import os
import time

# Enforce mirror & aggressive Rust parallelism
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

def main():
    repo = "facebook/esm2_t33_650M_UR50D"
    local_dir = "/home/dataset-local/AntigenLM_V2/Training/Data/esm2_650M"
    
    print(f"Bypassing CDN Throttling. Initiating hf_transfer multi-threaded burst pull for {repo}...")
    start = time.time()
    
    # Download everything (including model.safetensors, config, tokenizer) directly to a local hard drive folder
    snapshot_download(repo_id=repo, local_dir=local_dir)
    
    end = time.time()
    print(f"SUCCESS! 2.6GB Payload downloaded in {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()
