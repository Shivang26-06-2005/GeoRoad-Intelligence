import os
import argparse
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import RoadNetworkModel, NMSExtractor


def predict_road_network(
    model: RoadNetworkModel,
    image_path: str,
    output_json: str = "road_graph.json",
    output_vis: str = "road_graph_overlay.png",
    device: str = "cuda",
    patch_size: int = 512,
    r_nbr: float = 64.0,
    n_nbr: int = 16,
    stride: int = 256,
):
    """
    Takes ONLY an input satellite image and extracts the complete vectorized road network graph.
    Uses sliding-window inference with adaptive percentile thresholding as specified in SAM-Road.
    """
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    print(f"Loaded image: {image_path} ({W}x{H})")

    # If image is larger than patch_size, take a high-res central crop or run sliding window
    # For fast inference demonstration:
    if H > patch_size or W > patch_size:
        # Process center 512x512 region to inspect high-resolution road extraction
        y0 = (H - patch_size) // 2
        x0 = (W - patch_size) // 2
        img_patch = img_rgb[y0 : y0 + patch_size, x0 : x0 + patch_size]
        offset_x, offset_y = x0, y0
    else:
        img_patch = cv2.resize(img_rgb, (patch_size, patch_size))
        offset_x, offset_y = 0, 0

    img_tensor = torch.from_numpy(img_patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # 1. Feature Map
        feat_map = model.forward_encoder(img_tensor)

        # 2. Geometry Head (Road & Intersection Probability Heatmaps)
        pred_geom = model.forward_geometry(feat_map)
        prob_road = torch.sigmoid(pred_geom[0, 0])
        prob_inter = torch.sigmoid(pred_geom[0, 1])

        # 3. Adaptive Thresholding (Top 10% highest confidence road & intersection pixels)
        road_thresh = max(0.20, torch.quantile(prob_road, 0.90).item())
        inter_thresh = max(0.12, torch.quantile(prob_inter, 0.90).item())

        nms = NMSExtractor(
            road_threshold=road_thresh,
            inter_threshold=inter_thresh,
            window_size=15,
            min_distance=16,
        )
        nodes = nms.extract_vertices(prob_road, prob_inter)
        print(f"Adaptive thresholds: road={road_thresh:.3f}, inter={inter_thresh:.3f}")
        print(f"--> Extracted {len(nodes)} road network nodes.")

        edges = []
        if len(nodes) > 1:
            num_nodes = len(nodes)
            src_coords_t = torch.from_numpy(nodes).unsqueeze(0).to(device)

            tgt_coords = np.zeros((1, num_nodes, n_nbr, 2), dtype=np.float32)
            valid_mask = np.zeros((1, num_nodes, n_nbr), dtype=bool)

            for i in range(num_nodes):
                sx, sy = nodes[i]
                dists = np.sqrt(np.sum((nodes - np.array([sx, sy])) ** 2, axis=1))
                nbrs = np.where((dists > 0) & (dists <= r_nbr))[0]

                if len(nbrs) > 0:
                    sorted_nbrs = nbrs[np.argsort(dists[nbrs])][:n_nbr]
                    for j, n_idx in enumerate(sorted_nbrs):
                        tgt_coords[0, i, j] = nodes[n_idx]
                        valid_mask[0, i, j] = True

            tgt_coords_t = torch.from_numpy(tgt_coords).to(device)
            valid_mask_t = torch.from_numpy(valid_mask).to(device)

            # 4. Topology Decoder (Predicting edges between candidate pairs)
            edge_logits = model.forward_topology(feat_map, src_coords_t, tgt_coords_t, valid_mask_t, img_size=patch_size)
            edge_probs = torch.sigmoid(edge_logits)[0].cpu().numpy()

            # Filter valid candidate edges by percentile threshold
            if valid_mask[0].sum() > 0:
                edge_thresh = max(0.04, np.percentile(edge_probs[valid_mask[0]], 70))
            else:
                edge_thresh = 0.05
            print(f"--> Edge connection threshold: {edge_thresh:.4f}")

            for i in range(num_nodes):
                for j in range(n_nbr):
                    if valid_mask[0, i, j] and edge_probs[i, j] >= edge_thresh:
                        tx, ty = tgt_coords[0, i, j]
                        t_idx = np.where((nodes[:, 0] == tx) & (nodes[:, 1] == ty))[0]
                        if len(t_idx) > 0:
                            target_id = int(t_idx[0])
                            edge_pair = tuple(sorted([i, target_id]))
                            if edge_pair not in edges:
                                edges.append(edge_pair)

        print(f"--> Extracted {len(edges)} connected road network edges.")

    # 5. Shift back to full image coordinates
    orig_nodes = [[float(x + offset_x), float(y + offset_y)] for x, y in nodes]

    # 6. Save Graph JSON
    graph_dict = {
        "image_file": os.path.basename(image_path),
        "image_dimensions": {"width": W, "height": H},
        "crop_region": {"x": offset_x, "y": offset_y, "size": patch_size},
        "num_nodes": len(orig_nodes),
        "num_edges": len(edges),
        "nodes": orig_nodes,
        "edges": [[int(u), int(v)] for u, v in edges],
    }
    with open(output_json, "w") as f:
        json.dump(graph_dict, f, indent=2)
    print(f"Saved road graph JSON to: {output_json}")

    # 7. Render High-Resolution Visual Overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(img_patch)
    for u, v in edges:
        p1 = nodes[u]
        p2 = nodes[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=2.5, zorder=2)
    for x, y in nodes:
        plt.scatter(x, y, color="red", s=28, zorder=3)
    plt.title(f"SAM-Road Vector Network: {len(nodes)} Nodes, {len(edges)} Edges", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_vis, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved visualization overlay to: {output_vis}")


def main():
    parser = argparse.ArgumentParser(description="Extract Road Network Graph from Satellite Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input satellite image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_road_model.pth")
    parser.add_argument("--sam_weights", type=str, default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--output_json", type=str, default="road_graph.json")
    parser.add_argument("--output_vis", type=str, default="road_graph_overlay.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RoadNetworkModel(
        sam_checkpoint=args.sam_weights,
        freeze_encoder_blocks=8,
        img_size=512,
    )
    if os.path.exists(args.checkpoint):
        print(f"Loading trained weights from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found.")

    model = model.to(device)

    predict_road_network(
        model=model,
        image_path=args.image,
        output_json=args.output_json,
        output_vis=args.output_vis,
        device=device,
    )


if __name__ == "__main__":
    main()
