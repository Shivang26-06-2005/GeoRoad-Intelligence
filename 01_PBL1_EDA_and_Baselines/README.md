# PBL-I: Problem Identification, Dataset Selection & Exploratory Data Analysis (EDA)

## 1. Overview
In the initial PBL-I phase of the **GeoRoad Intelligence** project, we formulated the core challenge: **Autonomous road network extraction from high-resolution satellite imagery**.

## 2. Key Objectives & Accomplishments
- **Dataset Selection**: Identified the **CityScale** benchmark dataset (20 cities, 1.0 m/px resolution) and **Sentinel-2** multi-spectral satellite imagery (10 m/px resolution).
- **Exploratory Data Analysis (EDA)**: Analyzed spectral reflectance distributions (Red, Green, Blue bands), vegetation indices (NDVI proxies), pixel class imbalance (road pixels represent <12% of typical scenes), and spatial contrast variations across diverse urban topologies.
- **Baseline Modeling**: Formulated baseline linear regression and decision-tree classification baselines to assess raw pixel separability.

## 3. Directory Contents
- `eda_analysis.py`: Python script performing statistical analysis, color histograms, and pixel distribution plots.
- `reports/20cities_eda_report.png`: Graphical summary of the 20-city spatial and spectral distributions.
- `reports/Dataset_Information.docx`: Dataset specification and acquisition documentation.
