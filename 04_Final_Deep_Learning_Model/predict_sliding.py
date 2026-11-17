import os
import argparse
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import RoadNetworkModel, NMSExtractor


def predict_sliding_window(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_road_model.pth",
    output_json: str = "sliding_window_graph.json",
    output_vis: str = "sliding_window_overlay.png",
    device: str = "cuda",
    window_size: int = 512,
    stride: int = 256,   # 50% overlap for dense road detection
    r_nbr: float = 64.0,
    n_nbr: int = 16,
):
    """
    Sliding-Window Inference as defined in SAM-Road paper Section 3.6.
    Evaluates at 100% full native resolution with overlapping tiles and probability fusion,
    detecting narrow alleys and small paths without shrinking the image.
    """
    # 1. Load full resolution image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    print(f"Loaded full-res image: {image_path} ({W}x{H})")

    # 2. Load trained model
    model = RoadNetworkModel(
        sam_checkpoint="weights/sam_vit_b_01ec64.pth",
        freeze_encoder_blocks=8,
        img_size=window_size,
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # 3. Create global probability accumulation maps (SAM-Road Sec 3.6)
    road_accum = np.zeros((H, W), dtype=np.float32)
    inter_accum = np.zeros((H, W), dtype=np.float32)
    count_accum = np.zeros((H, W), dtype=np.float32)

    # Pad image to ensure all borders are covered
    pad_h = max(0, window_size - H)
    pad_w = max(0, window_size - W)
    if pad_h > 0 or pad_w > 0:
        img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        Hp, Wp, _ = img_padded.shape
    else:
        img_padded = img_rgb
        Hp, Wp = H, W

    # Generate sliding window coordinates
    y_steps = list(range(0, max(1, Hp - window_size + 1), stride))
    if y_steps[-1] != max(0, Hp - window_size):
        y_steps.append(max(0, Hp - window_size))

    x_steps = list(range(0, max(1, Wp - window_size + 1), stride))
    if x_steps[-1] != max(0, Wp - window_size):
        x_steps.append(max(0, Wp - window_size))

    print(f"--> Running sliding window across {len(y_steps) * len(x_steps)} overlapping tiles (at 100% full resolution)...")

    with torch.no_grad():
        for y0 in y_steps:
            for x0 in x_steps:
                patch = img_padded[y0 : y0 + window_size, x0 : x0 + window_size]
                t_patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                feat = model.forward_encoder(t_patch)
                geom = model.forward_geometry(feat)
                p_road = torch.sigmoid(geom[0, 0]).cpu().numpy()
                p_inter = torch.sigmoid(geom[0, 1]).cpu().numpy()

                # Accumulate fused probabilities within image bounds
                h_slice = min(H - y0, window_size)
                w_slice = min(W - x0, window_size)

                road_accum[y0 : y0 + h_slice, x0 : x0 + w_slice] += p_road[:h_slice, :w_slice]
                inter_accum[y0 : y0 + h_slice, x0 : x0 + w_slice] += p_inter[:h_slice, :w_slice]
                count_accum[y0 : y0 + h_slice, x0 : x0 + w_slice] += 1.0

    # Average the multi-observation fused heatmaps
    count_accum = np.maximum(count_accum, 1.0)
    fused_road = road_accum / count_accum
    fused_inter = inter_accum / count_accum

    # 4. Extract dense vertices across the fused high-res heatmap
    # Sensitive threshold for narrow dirt alleys and village paths
    road_thresh = max(0.20, np.percentile(fused_road, 85))
    inter_thresh = max(0.12, np.percentile(fused_inter, 85))
    print(f"Fused heatmaps: road_thresh={road_thresh:.3f}, inter_thresh={inter_thresh:.3f}")

    nms = NMSExtractor(
        road_threshold=road_thresh,
        inter_threshold=inter_thresh,
        window_size=15,
        min_distance=16,
    )
    nodes = nms.extract_vertices(torch.from_numpy(fused_road), torch.from_numpy(fused_inter))
    print(f"--> Dense vertices detected at full resolution: {len(nodes)} nodes")

    # 5. Connect edges between nearby vertices
    edges = []
    if len(nodes) > 1:
        # Distance-based topology connectivity along fused road probability
        min_dist = 12.0
        max_dist = r_nbr
        for i in range(len(nodes)):
            dists = np.sqrt(np.sum((nodes - nodes[i]) ** 2, axis=1))
            candidates = np.where((dists >= min_dist) & (dists <= max_dist))[0]
            for c in candidates:
                mid_x = int((nodes[i][0] + nodes[c][0]) / 2)
                mid_y = int((nodes[i][1] + nodes[c][1]) / 2)
                if 0 <= mid_x < W and 0 <= mid_y < H:
                    # If fused road confidence between the two nodes is solid
                    if fused_road[mid_y, mid_x] >= (road_thresh * 0.75):
                        edges.append(tuple(sorted([i, int(c)])))

        edges = list(set(edges))

    print(f"--> Extracted {len(edges)} connected road edges!")

    # 6. Save JSON
    with open(output_json, "w") as f:
        json.dump({
            "image_file": os.path.basename(image_path),
            "dimensions": {"width": W, "height": H},
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes.tolist(),
            "edges": [list(e) for e in edges],
        }, f, indent=2)
    print(f"Saved full-res JSON to: {output_json}")

    # 7. Save High-Res Overlay Plot
    plt.figure(figsize=(16, 8))
    plt.imshow(img_rgb)
    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=2.5, zorder=2)
    for x, y in nodes:
        plt.scatter(x, y, color="red", s=25, zorder=3)
    plt.title(f"Sliding-Window Full Resolution Extraction: {len(nodes)} Nodes, {len(edges)} Edges", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_vis, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved full-res visual overlay to: {output_vis}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../user_upload2_nadir.jpg")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_road_model.pth")
    parser.add_argument("--output_json", type=str, default="user_upload_dense_graph.json")
    parser.add_argument("--output_vis", type=str, default="user_upload_dense_overlay.png")
    args = parser.parse_args()

    predict_sliding_window(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        output_vis=args.output_vis,
    )

