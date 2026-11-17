# PBL-III: Unit 5 Ensemble Learning Component for GeoRoad Intelligence

## 1. Overview
Unit 5 conducted an empirical benchmark of classical machine learning algorithms and ensemble techniques on pixel-level road vs. background classification using a 13-dimensional spatial-spectral feature extraction pipeline.

## 2. Models Evaluated
1. **CART Decision Tree**: Single tree with Gini impurity, max_depth=12.
2. **Support Vector Machine (SVM)**: Calibrated Linear SVC with probability output.
3. **Logistic Regression**: L2-regularized linear baseline with StandardScaler.
4. **Hard Voting Classifier**: Majority rule combination of CART + SVM + LR.
5. **Soft Voting Classifier**: Weighted probability averaging across base models.
6. **Random Forest (Bagging)**: 100 bootstrap-aggregated decision trees with Gini feature importance.
7. **Gradient Boosting (Boosting)**: HistGradientBoosting sequential additive error correction.

## 3. Empirical Results (Held-Out Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 83.27% | 29.23% | 38.25% | 33.14% | 0.002 ms/sample |
| CART Decision Tree | 83.65% | 31.25% | 40.76% | 35.38% | 0.005 ms/sample |
| Support Vector Machine (SVM) | 84.12% | 32.18% | 41.90% | 36.40% | 0.012 ms/sample |
| Hard Voting Classifier | 84.55% | 33.80% | 42.15% | 37.51% | 0.019 ms/sample |
| Soft Voting Classifier | 84.92% | 34.65% | 43.20% | 38.45% | 0.021 ms/sample |
| HistGradientBoosting | 86.84% | 39.12% | 48.24% | 43.19% | 0.035 ms/sample |
| **Random Forest (100 Trees)** | **87.42%** | **40.58%** | **49.15%** | **44.47%** | **0.048 ms/sample** |

## 4. Key Architectural Takeaway
While Random Forest achieved the highest classical F1-score (44.47%), classical pixel models hit a ceiling because spectral reflectance cannot disambiguate asphalt roofs/parking lots from roads and cannot model topological graph connectivity across occluded tree canopies. This motivated the deep SAM-Road architecture.
