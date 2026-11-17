import os
import sys
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure Model directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "..", "04_Final_Deep_Learning_Model")
if not os.path.exists(FINAL_MODEL_DIR):
    FINAL_MODEL_DIR = os.path.join(BASE_DIR, "..", "FinalModel")
if os.path.exists(FINAL_MODEL_DIR) and FINAL_MODEL_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(FINAL_MODEL_DIR))

from models.road_model import RoadNetworkModel
from models.nms_extractor import NMSExtractor

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="GeoRoad Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static web app
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_standalone")
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Locate Weights and Checkpoint with multi-directory fallback
SAM_WEIGHTS = os.path.join(FINAL_MODEL_DIR, "weights", "sam_vit_b_01ec64.pth")
if not os.path.exists(SAM_WEIGHTS):
    alt_sam = os.path.join(BASE_DIR, "..", "..", "FinalModel", "weights", "sam_vit_b_01ec64.pth")
    if os.path.exists(alt_sam):
        SAM_WEIGHTS = os.path.abspath(alt_sam)

CHECKPOINT = os.path.join(FINAL_MODEL_DIR, "checkpoints", "best_road_model.pth")
if not os.path.exists(CHECKPOINT):
    alt_chk = os.path.join(BASE_DIR, "..", "..", "FinalModel", "checkpoints", "best_road_model.pth")
    if os.path.exists(alt_chk):
        CHECKPOINT = os.path.abspath(alt_chk)

print(f"Loading Model on {DEVICE}...")
model = RoadNetworkModel(sam_checkpoint=SAM_WEIGHTS, freeze_encoder_blocks=8, img_size=512)
if os.path.exists(CHECKPOINT):
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    print(f"Loaded weights from {CHECKPOINT}")
else:
    print(f"WARNING: Checkpoint {CHECKPOINT} not found!")

model.to(DEVICE)
model.eval()

# Normalization constants (ImageNet standard)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


@app.get("/health")
def health():
    return {
        "status": "online",
        "device": DEVICE,
        "checkpoint": os.path.basename(CHECKPOINT),
        "sam_weights": os.path.basename(SAM_WEIGHTS)
    }


def process_image(img_bytes: bytes, mode: str = "crop", road_th: float = 0.30, inter_th: float = 0.16):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image file provided.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    if mode == "crop":
        # Extract 512x512 crop
        if H < 512 or W < 512:
            img_patch = cv2.resize(img_rgb, (512, 512))
            offset_x, offset_y = 0, 0
        else:
            offset_x = (W - 512) // 2
            offset_y = (H - 512) // 2
            img_patch = img_rgb[offset_y : offset_y + 512, offset_x : offset_x + 512]
        
        target_img_for_display = img_patch
        curr_w, curr_h = 512, 512
        scale_x, scale_y = 1.0, 1.0
    else:
        # Full Canvas Mode (Resize to 512x512 for encoder, map back)
        img_patch = cv2.resize(img_rgb, (512, 512))
        target_img_for_display = img_rgb
        curr_w, curr_h = W, H
        scale_x = W / 512.0
        scale_y = H / 512.0
        offset_x, offset_y = 0, 0

    # Tensor preprocessing
    tensor_img = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.to(DEVICE)
    tensor_img = (tensor_img - MEAN) / STD
    tensor_img = tensor_img.unsqueeze(0)

    with torch.no_grad():
        feat_map = model.forward_encoder(tensor_img)
        geom_preds = model.forward_geometry(feat_map)
        p_road = torch.sigmoid(geom_preds[0, 0])
        p_inter = torch.sigmoid(geom_preds[0, 1])

        # Adaptive thresholding
        r_thresh = max(road_th, float(torch.quantile(p_road, 0.85).item()))
        i_thresh = max(inter_th, float(torch.quantile(p_inter, 0.85).item()))

        nms = NMSExtractor(
            road_threshold=r_thresh,
            inter_threshold=i_thresh,
            window_size=15,
            min_distance=16,
        )
        local_nodes = nms.extract_vertices(p_road.cpu(), p_inter.cpu())

        edges = []
        n_nbr = 16
        r_nbr = 64.0

        if len(local_nodes) > 1:
            num_nodes = len(local_nodes)
            src_coords_t = torch.from_numpy(local_nodes).unsqueeze(0).to(DEVICE)
            tgt_coords = np.zeros((1, num_nodes, n_nbr, 2), dtype=np.float32)
            valid_mask = np.zeros((1, num_nodes, n_nbr), dtype=bool)

            for i in range(num_nodes):
                sx, sy = local_nodes[i]
                dists = np.sqrt(np.sum((local_nodes - np.array([sx, sy])) ** 2, axis=1))
                nbrs = np.where((dists > 0) & (dists <= r_nbr))[0]
                if len(nbrs) > 0:
                    sorted_nbrs = nbrs[np.argsort(dists[nbrs])][:n_nbr]
                    for j, n_idx in enumerate(sorted_nbrs):
                        tgt_coords[0, i, j] = local_nodes[n_idx]
                        valid_mask[0, i, j] = True

            tgt_coords_t = torch.from_numpy(tgt_coords).to(DEVICE)
            valid_mask_t = torch.from_numpy(valid_mask).to(DEVICE)

            edge_logits = model.forward_topology(feat_map, src_coords_t, tgt_coords_t, valid_mask_t, img_size=512)
            edge_probs = torch.sigmoid(edge_logits)[0].cpu().numpy()

            if valid_mask[0].sum() > 0:
                edge_thresh = max(0.04, float(np.percentile(edge_probs[valid_mask[0]], 70)))
            else:
                edge_thresh = 0.05

            for i in range(num_nodes):
                for j in range(n_nbr):
                    if valid_mask[0, i, j] and edge_probs[i, j] >= edge_thresh:
                        tx, ty = tgt_coords[0, i, j]
                        t_idx = np.where((local_nodes[:, 0] == tx) & (local_nodes[:, 1] == ty))[0]
                        if len(t_idx) > 0:
                            target_id = int(t_idx[0])
                            edge_pair = tuple(sorted([i, target_id]))
                            if edge_pair not in edges:
                                edges.append(edge_pair)

    # Scale back to canvas display coordinates
    final_nodes = []
    for x, y in local_nodes:
        gx = float(x * scale_x + offset_x)
        gy = float(y * scale_y + offset_y)
        final_nodes.append([round(gx, 2), round(gy, 2)])

    # Generate Overlay Plot as Base64 Image
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(target_img_for_display)
    
    # Plot edges
    for u, v in edges:
        p1 = final_nodes[u]
        p2 = final_nodes[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00FF66", linewidth=2.5, zorder=2)

    # Plot nodes
    if len(final_nodes) > 0:
        fn_arr = np.array(final_nodes)
        ax.scatter(fn_arr[:, 0], fn_arr[:, 1], color="#FF2222", s=22, zorder=3, edgecolors="white", linewidths=0.5)

    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Generate Probability Heatmap Preview as Base64
    heat_np = (p_road.cpu().numpy() * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
    _, heat_buf = cv2.imencode(".png", heat_color)
    heatmap_b64 = base64.b64encode(heat_buf).decode("utf-8")

    return {
        "dimensions": {"width": curr_w, "height": curr_h},
        "mode": mode,
        "num_nodes": len(final_nodes),
        "num_edges": len(edges),
        "nodes": final_nodes,
        "edges": [[int(u), int(v)] for u, v in edges],
        "overlay_base64": f"data:image/png;base64,{overlay_b64}",
        "heatmap_base64": f"data:image/png;base64,{heatmap_b64}"
    }


@app.post("/extract-roads")
async def extract_roads(
    file: UploadFile = File(...),
    mode: str = Query("crop", regex="^(crop|full)$"),
    road_thresh: float = Query(0.30, ge=0.05, le=0.95),
    inter_thresh: float = Query(0.16, ge=0.05, le=0.95),
):
    try:
        contents = await file.read()
        res = process_image(contents, mode=mode, road_th=road_thresh, inter_th=inter_thresh)
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

