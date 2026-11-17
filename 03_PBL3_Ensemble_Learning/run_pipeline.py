import os
import json
import time
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from config import (
    MODELS_DIR, OUTPUTS_DIR, PBL3_OUTPUT_DIR,
    RANDOM_SEED, FEATURE_NAMES, TEST_REGION_ID
)
from dataset_preparation import prepare_region_splits, sample_dataset_from_regions, load_canonical_test_crop


def plot_confusion_matrix(cm, model_name, save_paths):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    classes = ["Background", "Road"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

    ax.set_title(f"Confusion Matrix: {model_name}", pad=20, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    plt.tight_layout()

    for p in save_paths:
        plt.savefig(p, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("   GEOROAD INTELLIGENCE: UNIT 5 ENSEMBLE LEARNING BENCHMARK")
    print("   PBL-III Review Implementation")
    print("   Algorithms: CART, SVM, Logistic Regression, Hard/Soft Voting, Random Forest, HistGradientBoosting")
    print("=" * 80)

    train_ids, val_ids, test_ids = prepare_region_splits()
    print(f"Dataset split verification: {len(train_ids)} train regions, {len(val_ids)} val regions, {len(test_ids)} test regions.")

    # 1. Dataset sampling
    print("\n[Step 1/5] Extracting geographic training and testing data...")
    X_train, y_train = sample_dataset_from_regions(train_ids, samples_per_region=2000, max_regions=16, seed=RANDOM_SEED)
    X_test, y_test = sample_dataset_from_regions(test_ids, samples_per_region=1200, max_regions=10, seed=RANDOM_SEED)

    print(f"--> Training Set: {X_train.shape[0]:,} samples | Road: {np.sum(y_train==1):,}, Background: {np.sum(y_train==0):,}")
    print(f"--> Held-out Test Set: {X_test.shape[0]:,} samples | Road: {np.sum(y_test==1):,}, Background: {np.sum(y_test==0):,}")

    # 2. Model Definitions
    print("\n[Step 2/5] Defining Base and Ensemble Classifiers...")

    # Individual Models
    cart = DecisionTreeClassifier(max_depth=12, min_samples_split=20, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_SEED)
    
    # Fast Linear SVM with Probability Calibration
    svm_base = LinearSVC(C=1.0, dual="auto", max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(svm_base, cv=3))
    ])

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))
    ])

    # Ensembles
    voting_hard = VotingClassifier(
        estimators=[("cart", cart), ("svm", svm), ("lr", lr)],
        voting="hard"
    )

    voting_soft = VotingClassifier(
        estimators=[("cart", cart), ("svm", svm), ("lr", lr)],
        voting="soft"
    )

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=15, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
    gb = HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, random_state=RANDOM_SEED)

    all_models = {
        "CART Decision Tree": cart,
        "Support Vector Machine": svm,
        "Logistic Regression": lr,
        "Voting (Hard)": voting_hard,
        "Voting (Soft)": voting_soft,
        "Random Forest (Bagging)": rf,
        "Gradient Boosting (Boosting)": gb
    }

    # 3. Training & Evaluation
    print("\n[Step 3/5] Training and Evaluating Models...")
    metrics_summary = {}
    preds_on_test_region = {}

    # Load canonical full 2D spatial evaluation tile
    crop_rgb, crop_gt, crop_feats, (tH, tW) = load_canonical_test_crop(region_id=TEST_REGION_ID)

    for name, model in all_models.items():
        print(f"\n---> Training: {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        print(f"     Testing: {name} on {X_test.shape[0]:,} samples...")
        t1 = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - t1

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"     [Result] Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}% | Train Time: {train_time:.1f}s")

        metrics_summary[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "train_time_sec": round(float(train_time), 2),
            "infer_time_sec": round(float(infer_time), 4)
        }

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_save_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_save_path)

        # Plot & save confusion matrix
        cm_name = f"cm_{safe_name}.png"
        cm_p1 = os.path.join(OUTPUTS_DIR, "confusion_matrices", cm_name)
        cm_p2 = os.path.join(PBL3_OUTPUT_DIR, "confusion_matrices", cm_name)
        plot_confusion_matrix(cm, name, [cm_p1, cm_p2])

        # Generate 2D spatial prediction on canonical image
        if hasattr(model, "predict_proba"):
            tile_prob = model.predict_proba(crop_feats)[:, 1].reshape(tH, tW)
        else:
            tile_prob = model.predict(crop_feats).reshape(tH, tW)
        preds_on_test_region[name] = tile_prob

    # 4. Save Metrics
    json_path1 = os.path.join(OUTPUTS_DIR, "metrics", "performance_metrics.json")
    json_path2 = os.path.join(PBL3_OUTPUT_DIR, "metrics", "performance_metrics.json")
    with open(json_path1, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    shutil.copy(json_path1, json_path2)

    # 5. Visualizations & Charts
    print("\n[Step 4/5] Generating Comparative Presentation Charts...")

    model_names = list(metrics_summary.keys())
    f1_scores = [metrics_summary[m]["f1_score"] * 100 for m in model_names]
    precisions = [metrics_summary[m]["precision"] * 100 for m in model_names]
    recalls = [metrics_summary[m]["recall"] * 100 for m in model_names]

    # Chart 1: F1-Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(model_names, f1_scores, color=["#64748b", "#3b82f6", "#0ea5e9", "#f59e0b", "#10b981", "#06b6d4", "#8b5cf6"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1-Score (%)", fontweight="bold")
    ax.set_title("Unit 5 PBL-III: F1-Score Comparison Across Models", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "charts", "f1_score_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PBL3_OUTPUT_DIR, "charts", "f1_score_comparison.png"), bbox_inches="tight")
    plt.close()

    # Chart 2: Precision vs Recall Comparison
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, precisions, width, label="Precision", color="#06b6d4")
    ax.bar(x + width/2, recalls, width, label="Recall", color="#10b981")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Unit 5 PBL-III: Precision and Recall Trade-off", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "charts", "precision_recall_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PBL3_OUTPUT_DIR, "charts", "precision_recall_comparison.png"), bbox_inches="tight")
    plt.close()

    # Chart 3: Feature Importance (Random Forest)
    rf_model = all_models["Random Forest (Bagging)"]
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.barh(range(len(idx_sorted)), importances[idx_sorted], color="#06b6d4")
    ax.set_yticks(range(len(idx_sorted)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in idx_sorted], fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontweight="bold")
    ax.set_title("Random Forest - Feature Importance Ranking", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "charts", "feature_importance.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PBL3_OUTPUT_DIR, "charts", "feature_importance.png"), bbox_inches="tight")
    plt.close()

    # Chart 4: Full Multi-Model 2D Prediction Overlays
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=150)
    axes[0, 0].imshow(crop_rgb)
    axes[0, 0].set_title("Input Satellite Image (Region 9)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(crop_gt, cmap="gray")
    axes[0, 1].set_title("Ground Truth Road Mask", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    display_keys = [
        ("CART Decision Tree", (0, 2)),
        ("Support Vector Machine", (0, 3)),
        ("Logistic Regression", (1, 0)),
        ("Voting (Soft)", (1, 1)),
        ("Random Forest (Bagging)", (1, 2)),
        ("Gradient Boosting (Boosting)", (1, 3))
    ]

    for k, (r, c) in display_keys:
        pred_map = preds_on_test_region[k]
        axes[r, c].imshow(pred_map, cmap="jet")
        axes[r, c].set_title(f"{k}\nF1: {metrics_summary[k]['f1_score']*100:.2f}% | AUC: {metrics_summary[k]['roc_auc']*100:.2f}%", fontsize=10, fontweight="bold")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "prediction_masks", "full_model_spatial_predictions.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PBL3_OUTPUT_DIR, "prediction_masks", "full_model_spatial_predictions.png"), bbox_inches="tight")
    plt.close()

    print("\n[Step 5/5] All results, confusion matrices, charts, and models saved successfully!")


if __name__ == "__main__":
    main()
