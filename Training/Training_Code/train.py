import os
# Auto-route all Hugging Face downloads to the official domestic high-speed mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Extreme Speedup: Enable Rust-based Multi-Threaded Chunk Downloader
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import torch.nn as nn
from transformers import EsmTokenizer, Trainer, TrainingArguments
from dataset import AntigenEpitopeDataset, EpitopeMaskingDataCollator
from model import DualEngineAntigenLM

class DualEngineTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom compute_loss that cleanly extracts our DualEngine custom losses.
        """
        # Model returns a dict containing standard loss and intermediate scaling losses
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # Inject custom Granular dual-engine loss into Hugging Face's global logging history
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "mlm_loss_(escape)": outputs["mlm_loss"].mean().item(), 
                "kd_loss_(anchor)": outputs["kd_loss"].mean().item()
            })
            
        return (loss, outputs) if return_outputs else loss

def main():
    config_file = "config.json"
    
    # 0. Load Master Configuration
    import json
    with open(config_file, "r") as f:
        config = json.load(f)
        
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    
    csv_file = dataset_cfg.get("csv_file", "/home/dataset-local/AntigenLM_V2/Training/Data/Antigen_all.csv")
    output_dir = train_cfg.get("output_dir", "./AntigenLM_Output")
    
    # 1. Tokenizer Configuration
    print(f"-> Loading EsmTokenizer from {config['base_model_name']}...")
    tokenizer = EsmTokenizer.from_pretrained(config['base_model_name'])
    
    # 2. Dataset & DataCollator
    print(f"-> Preparing dataset from {csv_file}")
    max_length = dataset_cfg.get("max_length", 1024)
    full_dataset = AntigenEpitopeDataset(csv_file=csv_file, tokenizer=tokenizer, max_length=max_length)
    
    mlm_prob = dataset_cfg.get("mlm_probability", 0.20)
    bias = dataset_cfg.get("epitope_bias", 0.80)
    data_collator = EpitopeMaskingDataCollator(tokenizer=tokenizer, mlm_probability=mlm_prob, epitope_bias=bias)

    # 3. Robust Train/Validation Split
    train_ratio = dataset_cfg.get("train_split_ratio", 0.95)
    print(f"-> Splitting Dataset ({train_ratio * 100}% Train / {(1-train_ratio) * 100}% Validation)...")
    train_size = int(train_ratio * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
    )

    # 4. Model Configuration
    print("-> Initializing DualEngine Model Configuration...")
    model = DualEngineAntigenLM(config_file)
    
    # 5. Trainer Configuration (TrainingArguments)
    print("-> Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 10),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        logging_dir="./logs",
        logging_steps=train_cfg.get("logging_steps", 20),
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=600,
        eval_steps=600,
        save_safetensors=False,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=True,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        ddp_find_unused_parameters=True
    )
    
    # 6. Initialize Custom Trainer class
    trainer = DualEngineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 6. Ignite the Training Pipeline!
    print("-> Starting Hugging Face Trainer Pipeline...")
    
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        
    if last_checkpoint is not None:
        print(f"-> Resuming from checkpoint detected at {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("-> No existing checkpoint found. Starting training from scratch...")
        trainer.train()
    
    # 7. Custom PEFT Checkpointing Logic
    print(f"-> Saving lightweight PEFT adapters exclusively to {output_dir}/final_model ...")
    os.makedirs(os.path.join(output_dir, "final_model"), exist_ok=True)
    # Target only the student sub-network adapters avoiding deep copying frozen 650M anchors
    trainer.model.student.save_pretrained(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    main()
