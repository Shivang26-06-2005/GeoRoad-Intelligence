# PBL-II: Unsupervised Clustering & Hidden Markov Models (HMM)

## 1. Overview
PBL-II investigated unsupervised spatial clustering and sequential state modeling to group road pixels and trace road corridors across satellite tiles.

## 2. Methodology & Key Modules
- **K-Means Clustering**: Grouped spectral-spatial feature vectors to segment road surfaces from surrounding ground cover, evaluated using the Elbow method and Silhouette coefficients.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Identified dense clusters corresponding to road intersections and junction hubs while rejecting sparse vegetative noise.
- **Hidden Markov Models (HMM)**: Formulated road tracking as a sequential hidden state estimation problem:
  - *Hidden States*: True road alignment / bearing.
  - *Observations*: Pixel intensity and local edge gradients.
  - *Viterbi Decoding*: Computed the most probable continuous trajectory across noisy satellite tiles.

## 3. Directory Contents
- `generate_pbl2_pipeline.py`: Complete pipeline script running clustering and HMM analysis.
- `metrics_summary.json`: Quantitative clustering and Viterbi trajectory metrics.
- `visual_evidence/`: 11 high-resolution figures showing Elbow curves, DBSCAN junctions, Viterbi tracking, and error cases.
- `docs/`: Word summary reports and presentation slide deck.
