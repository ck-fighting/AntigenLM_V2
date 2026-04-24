import argparse
import glob
import os

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import AgAbCollateFn, AgAbDataset
from losses import InfoNCELoss
from models import AlignmentModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_model_to_save(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, epoch, step, loss, save_path):
    state_dict = {
        key: value
        for key, value in get_model_to_save(model).state_dict().items()
        if not key.startswith("ab_runner.")
    }
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )
    print(f"\nCheckpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer, device, is_main):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        if is_main and checkpoint_path:
            print(f"No checkpoint found at '{checkpoint_path}'.")
        return 0, 0

    if is_main:
        print(f"Loading checkpoint '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = get_model_to_save(model)

    if "model_state_dict" in checkpoint:
        model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

    model_to_load.load_state_dict(checkpoint, strict=False)
    return 0, 0


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]

    if local_rank != -1:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"Using device: {device}")

    dataset = AgAbDataset(args.data_path)
    collate_fn = AgAbCollateFn(
        antigen_model_path=args.antigen_model_path,
        antigen_max_length=args.antigen_max_length,
    )
    sampler = DistributedSampler(dataset) if local_rank != -1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
    )

    if is_main_process:
        print(f"Loaded {len(dataset)} samples from {args.data_path}")

    model = AlignmentModel(
        antigen_model_path=args.antigen_model_path,
        proj_dim=args.proj_dim,
    ).to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = InfoNCELoss(temperature=args.temperature)
    scaler = GradScaler()

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir):
        checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "alignment_model_epoch_*.pt")))
        if checkpoints:
            args.resume_from_checkpoint = checkpoints[-1]

    start_epoch, start_step = load_checkpoint(args.resume_from_checkpoint, model, optimizer, device, is_main_process)

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("Starting Training...")

    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}") if is_main_process else dataloader
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            if epoch == start_epoch and batch_idx < start_step:
                continue

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

            with autocast():
                ag_emb, ab_emb = model(
                    ag_input_ids=ag_input_ids,
                    ag_attention_mask=ag_attention_mask,
                    heavy_sequences=heavy_sequences,
                    light_sequences=light_sequences,
                    ag_pool_mask=ag_pool_mask,
                    vh_raw_masks=vh_raw_masks,
                    vl_raw_masks=vl_raw_masks,
                )
                loss = criterion(ag_emb, ab_emb) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = loss.item() * args.grad_accum_steps
            epoch_loss += loss_val
            if is_main_process:
                progress.set_postfix({"loss": f"{loss_val:.4f}"})

        if is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                0,
                avg_loss,
                os.path.join(args.output_dir, f"alignment_model_epoch_{epoch + 1}.pt"),
            )
        start_step = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "data", "SABDab", "sabdab_train_80.csv"),
    )
    parser.add_argument(
        "--antigen_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "LLM", "esm2_650M"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Ag2Ab", "Pre_aligment", "checkpoints_esm2_antiberty"),
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--antigen_max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())
