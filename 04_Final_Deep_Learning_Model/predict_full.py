import os
import argparse
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import RoadNetworkModel, NMSExtractor


def predict_entire_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_road_model.pth",
    output_json: str = "test2_full_graph.json",
    output_vis: str = "test2_full_overlay.png",
    device: str = "cuda",
    target_dim: int = 512,
    r_nbr: float = 64.0,
    n_nbr: int = 16,
):
    """
    Processes the 100% ENTIRE image without any center-cropping.
    Resizes the whole image to 512x512 for the SAM-Road pipeline,
    and maps the predicted road graph directly back to original image dimensions.
    """
    # 1. Load entire image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = img_rgb.shape
    print(f"Loaded ENTIRE image: {image_path} ({orig_w}x{orig_h})")

    # 2. Resize ENTIRE image to 512x512 (zero cropping)
    img_resized = cv2.resize(img_rgb, (target_dim, target_dim), interpolation=cv2.INTER_LINEAR)
    scale_x = orig_w / float(target_dim)
    scale_y = orig_h / float(target_dim)

    # 3. Load Model
    model = RoadNetworkModel(
        sam_checkpoint="weights/sam_vit_b_01ec64.pth",
        freeze_encoder_blocks=8,
        img_size=target_dim,
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # 4. Predict Geometry (Road & Intersection heatmaps)
    t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        feat = model.forward_encoder(t)
        geom = model.forward_geometry(feat)
        p_road = torch.sigmoid(geom[0, 0])
        p_inter = torch.sigmoid(geom[0, 1])

        # 5. Peak NMS Extraction across full image
        road_th = max(0.22, torch.quantile(p_road, 0.90).item())
        inter_th = max(0.14, torch.quantile(p_inter, 0.90).item())

        nms = NMSExtractor(
            road_threshold=road_th,
            inter_threshold=inter_th,
            window_size=15,
            min_distance=16,
        )
        nodes_512 = nms.extract_vertices(p_road, p_inter)
        print(f"Thresholds across full image: road={road_th:.3f}, inter={inter_th:.3f}")
        print(f"Extracted {len(nodes_512)} road network vertices.")

        # 6. Predict Topology Edges
        edges = []
        if len(nodes_512) > 1:
            num_nodes = len(nodes_512)
            src_t = torch.from_numpy(nodes_512).unsqueeze(0).to(device)
            tgt = np.zeros((1, num_nodes, n_nbr, 2), dtype=np.float32)
            vmask = np.zeros((1, num_nodes, n_nbr), dtype=bool)

            for i in range(num_nodes):
                sx, sy = nodes_512[i]
                dists = np.sqrt(np.sum((nodes_512 - np.array([sx, sy])) ** 2, axis=1))
                nbrs = np.where((dists > 0) & (dists <= r_nbr))[0]
                if len(nbrs) > 0:
                    sorted_nbrs = nbrs[np.argsort(dists[nbrs])][:n_nbr]
                    for j, n_idx in enumerate(sorted_nbrs):
                        tgt[0, i, j] = nodes_512[n_idx]
                        vmask[0, i, j] = True

            tgt_t = torch.from_numpy(tgt).to(device)
            vmask_t = torch.from_numpy(vmask).to(device)

            edge_logits = model.forward_topology(feat, src_t, tgt_t, vmask_t, img_size=target_dim)
            edge_probs = torch.sigmoid(edge_logits)[0].cpu().numpy()

            if vmask[0].sum() > 0:
                edge_thresh = max(0.08, np.percentile(edge_probs[vmask[0]], 72))
            else:
                edge_thresh = 0.10
            print(f"Edge connection threshold: {edge_thresh:.4f}")

            for i in range(num_nodes):
                for j in range(n_nbr):
                    if vmask[0, i, j] and edge_probs[i, j] >= edge_thresh:
                        tx, ty = tgt[0, i, j]
                        t_idx = np.where((nodes_512[:, 0] == tx) & (nodes_512[:, 1] == ty))[0]
                        if len(t_idx) > 0:
                            edges.append(tuple(sorted([i, int(t_idx[0])])))
            edges = list(set(edges))

        print(f"Extracted {len(edges)} connecting road edges across entire image.")

    # 7. Map nodes back to original 1280x720 coordinates
    orig_nodes = [[float(x * scale_x), float(y * scale_y)] for x, y in nodes_512]

    # 8. Save Vector Graph JSON
    with open(output_json, "w") as f:
        json.dump({
            "image_file": os.path.basename(image_path),
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "num_nodes": len(orig_nodes),
            "num_edges": len(edges),
            "nodes": orig_nodes,
            "edges": [list(e) for e in edges],
        }, f, indent=2)
    print(f"Saved full road graph JSON to: {output_json}")

    # 9. Render Full Overlay on the uncropped original 1280x720 image
    plt.figure(figsize=(14, 8))
    plt.imshow(img_rgb)
    for u, v in edges:
        p1 = orig_nodes[u]
        p2 = orig_nodes[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=2.5, zorder=2)
    for x, y in orig_nodes:
        plt.scatter(x, y, color="red", s=30, zorder=3)
    plt.title(f"Entire Image Road Graph ({orig_w}x{orig_h}): {len(orig_nodes)} Nodes, {len(edges)} Edges", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_vis, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved full visualization overlay to: {output_vis}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../test2.jpg")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_road_model.pth")
    parser.add_argument("--output_json", type=str, default="test2_full_graph.json")
    parser.add_argument("--output_vis", type=str, default="test2_full_overlay.png")
    args = parser.parse_args()

    predict_entire_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        output_vis=args.output_vis,
    )

