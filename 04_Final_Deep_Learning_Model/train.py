import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import RoadNetworkModel
from dataset.road_dataset import CityScaleDataset, Sentinel2Dataset


def train_epoch(model, loader, optimizer, scaler, accum_steps, device):
    model.train()
    total_loss = 0.0
    valid_steps = 0
    bce = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")

    for step, batch in enumerate(pbar):
        imgs = batch["image"].to(device)
        gt_masks = batch["masks"].to(device)
        src_coords = batch["src_coords"].to(device)
        tgt_coords = batch["tgt_coords"].to(device)
        edge_labels = batch["edge_labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        # Mixed precision for 4GB VRAM
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # 1. SAM Encoder
            feat_map = model.forward_encoder(imgs)

            # 2. Geometry loss
            pred_masks = model.forward_geometry(feat_map)
            loss_geom = bce(pred_masks, gt_masks)

            # 3. Topology loss
            edge_logits = model.forward_topology(feat_map, src_coords, tgt_coords, valid_mask)
            if valid_mask.sum() > 0:
                loss_topo = bce(edge_logits[valid_mask], edge_labels[valid_mask])
            else:
                loss_topo = torch.tensor(0.0, device=device)

            step_loss = loss_geom + loss_topo

            if torch.isnan(step_loss) or torch.isinf(step_loss):
                continue

            loss = step_loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        curr_loss = (loss.item() * accum_steps)
        total_loss += curr_loss
        valid_steps += 1
        pbar.set_postfix({"Loss": f"{curr_loss:.4f}", "Geom": f"{loss_geom.item():.4f}", "Topo": f"{loss_topo.item():.4f}"})

    if valid_steps == 0:
        return 0.0
    return total_loss / valid_steps


def main():
    parser = argparse.ArgumentParser(description="Train Custom SAM-Road")
    parser.add_argument("--dataset", type=str, default="sentinel2", choices=["cityscale", "sentinel2"], help="Which dataset to train on")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default="checkpoints/best_road_model.pth", help="Checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start from")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=16)
    parser.add_argument("--crops_per_epoch", type=int, default=3, help="Crops per region per epoch for CityScale")
    parser.add_argument("--steps_per_epoch", type=int, default=450, help="Max steps per epoch for fast Sentinel-2 fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--sam_weights", type=str, default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--save_dir", type=str, default="checkpoints_sentinel2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model
    model = RoadNetworkModel(
        sam_checkpoint=args.sam_weights,
        freeze_encoder_blocks=8,
        img_size=512,
    )

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"--> Resuming cleanly from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    else:
        print("--> Training from scratch")

    model = model.to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.geometry_decoder.parameters()) + list(model.topology_decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler()

    # Datasets
    if args.dataset == "cityscale":
        train_dataset = CityScaleDataset(
            root_dir="../City_Scale Dataset/data/20cities",
            split_path="../City_Scale Dataset/data/data_split.json",
            split="train",
            patch_size=512,
            is_train=True,
            crops_per_epoch=args.crops_per_epoch,
        )
    elif args.dataset == "sentinel2":
        train_dataset = Sentinel2Dataset(
            base_dir="../Sentinel 2 Dataset/15765738",
            use_enhanced=True,
            target_size=512,
            split="train",
            max_samples=args.steps_per_epoch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"=== LAUNCHING {args.dataset.upper()} TRAINING ({len(train_loader)} steps/epoch) ===")
    best_loss = float("inf")

    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        avg_loss = train_epoch(model, train_loader, optimizer, scaler, args.accum_steps, device)
        print(f"Epoch {epoch} finished | Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, "best_road_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"--> Saved new BEST Sentinel-2 checkpoint: {best_path} (Loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
