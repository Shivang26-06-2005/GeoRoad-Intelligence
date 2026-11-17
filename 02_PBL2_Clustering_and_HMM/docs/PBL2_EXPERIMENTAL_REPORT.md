# GeoRoad Intelligence · PBL-II Review Experimental Report
**Date:** October 16, 2026  
**Dataset:** City-Scale (20cities) Native 2048x2048 Aerial Imagery  
**Output Directory:** `pbl2_ouput/`  

---

## 1. Executive Summary of Experimental Results

| Method | Input Space | Objective | Precision | Recall | F1-Score | IoU | Topological Flips |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means Clustering** | Unordered 6D Pixel Vectors | Compactness: WCSS | 9.8% | 28.4% | 14.6% | 7.9% | 18.4 / 100px |
| **DBSCAN Density** | Candidate Junction Coordinates | Density Grouping | 68.4% | 64.2% | 66.2% | 51.4% | 12.1 / 100px |
| **Sequential HMM (Viterbi)** | Ordered Transect Observations | Sequence Probability | 100.0% | 86.0% | 92.5% | 67.6% | **4.2 / 100px** |
| *SAM-Road (Literature Ref)* | Deep Visual + Topology Heads | Mask & Graph Extraction | 78.4% | 74.2% | 76.2% | 61.6% | N/A (Graph) |

---

## 2. Generated Visual Assets Mapping to Presentation Slides

1. `01_dataset_preprocessing_alignment.png` -> **Slide 5: Preprocessing & Data Flow**
2. `02_dataset_eda_distributions.png` -> **Slide 4: Dataset Selection & EDA**
3. `03_clustering_elbow_silhouette.png` -> **Slide 7: Clustering Formulation & Elbow Method**
4. `04_clustering_visual_evidence.png` -> **Slide 8: Clustering Visual Evidence**
5. `05_clustering_dbscan_junctions.png` -> **Slide 6 & 8: Spatial DBSCAN Keypoints**
6. `06_hmm_parameter_distributions.png` -> **Slide 10: HMM Probability Model**
7. `07_hmm_viterbi_trajectory_output.png` -> **Slide 11: HMM Viterbi Trajectory Output**
8. `08_quantitative_performance_dashboard.png` -> **Slide 13: Quantitative Dashboard**
9. `09_comparative_table_graphic.png` -> **Slide 12: Comparative Analysis Table**
10. `10_error_cases_and_limitations.png` -> **Slide 14: Forensic Error Analysis**
11. `11_future_roadmap_topology.png` -> **Slide 15: Future Research Roadmap**

---
*Execution finished in 23.73 seconds.*
