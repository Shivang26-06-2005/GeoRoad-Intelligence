import os
import json
import cv2
import numpy as np
from config import DATA_DIR, SPLIT_PATH, RANDOM_SEED, FEATURE_NAMES
from feature_extraction import extract_pixel_features

def prepare_region_splits():
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)
    return splits.get("train", []), splits.get("valid", []), splits.get("test", [])

def sample_dataset_from_regions(region_ids, samples_per_region=2000, max_regions=20, seed=RANDOM_SEED):
    np.random.seed(seed)
    X_list, y_list = [], []

    count = 0
    for rid in region_ids[:max_regions]:
        prefix = f"region_{rid}"
        sat_p = os.path.join(DATA_DIR, f"{prefix}_sat.png")
        mask_p = os.path.join(DATA_DIR, f"{prefix}_gt.png")

        if not os.path.exists(sat_p) or not os.path.exists(mask_p):
            continue

        sat = cv2.imread(sat_p)
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        label = (mask > 128).astype(np.uint8)

        # Center crop 512x512 to prevent RAM exhaustion and preserve standard 1m/px resolution
        H, W, _ = sat.shape
        cy, cx = (H - 512) // 2, (W - 512) // 2
        sat_crop = sat[cy:cy+512, cx:cx+512]
        label_crop = label[cy:cy+512, cx:cx+512].flatten()

        feats, _ = extract_pixel_features(sat_crop)

        pos_idx = np.where(label_crop == 1)[0]
        neg_idx = np.where(label_crop == 0)[0]

        n_pos = min(len(pos_idx), samples_per_region // 2)
        if n_pos == 0:
            continue
        n_neg = min(len(neg_idx), n_pos * 3)  # Balanced class weighting 1:3 road-to-background ratio

        c_pos = np.random.choice(pos_idx, n_pos, replace=False)
        c_neg = np.random.choice(neg_idx, n_neg, replace=False)
        chosen = np.concatenate([c_pos, c_neg])

        X_list.append(feats[chosen])
        y_list.append(label_crop[chosen])
        count += 1

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def load_canonical_test_crop(region_id=9):
    sat_p = os.path.join(DATA_DIR, f"region_{region_id}_sat.png")
    mask_p = os.path.join(DATA_DIR, f"region_{region_id}_gt.png")

    sat = cv2.imread(sat_p)
    sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    label = (mask > 128).astype(np.uint8)

    H, W, _ = sat.shape
    cy, cx = (H - 512) // 2, (W - 512) // 2
    crop_rgb = sat[cy:cy+512, cx:cx+512]
    crop_gt = label[cy:cy+512, cx:cx+512]

    feats, (h, w) = extract_pixel_features(crop_rgb)
    return crop_rgb, crop_gt, feats, (h, w)

