<div align="center">

# 🌿 Indian Medicinal Leaves Classification Benchmark

### An Empirical Multi-Feature Computer Vision & Machine Learning Benchmark

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-2A75D3?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Kaggle-Medicinal--Leaf--Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Systematic 155-experiment benchmark evaluating 31 feature combinations across 5 ML classifiers on 80 Indian Medicinal Leaf species (6,900 images).</b>
</p>

[Key Results](#-key-results) •
[Pipeline Architecture](#-pipeline-architecture) •
[Feature Extraction](#-feature-extraction-167-dimensions) •
[Installation & Quick Start](#-installation--quick-start) •
[Benchmark Leaderboard](#-benchmark-leaderboard) •
[Citation & Acknowledgments](#-citation--acknowledgments)

</div>

---

## 📌 Overview

Botanical species identification through leaf image analysis poses significant computer vision challenges due to subtle inter-class variations and high intra-class variations in leaf geometry, pigmentation, micro-texture, and venation.

This repository presents a **comprehensive 155-run benchmark evaluation** comparing 5 machine learning classifiers across all $2^5 - 1 = 31$ possible non-empty combinations of 5 domain-specific Computer Vision feature extraction families:

1. 📐 **Shape Geometry & Hu Invariant Moments** (14 dims)
2. 🎨 **HSV Color Histograms & Color Moments** (57 dims)
3. 🔬 **GLCM, LBP, and Gabor Filter Bank Textures** (32 dims)
4. 🍃 **Canny, Skeleton, Vascular Nodes & Edge Topography** (14 dims)
5. 🔍 **ORB Keypoint Bag-of-Visual-Words (BoVW)** (50 dims)

> [!IMPORTANT]
> **Key Finding**: The combination of **`Shape + Color + Texture + Vein`** features trained on an **SVM (RBF Kernel)** achieved the overall highest benchmark performance with **68.48% Accuracy** and **69.19% Macro F1-score** across all 80 leaf classes.

---

## 🏆 Key Results

| Metric | Overall Best Configuration |
| :--- | :--- |
| **Winning Feature Combination** | `shape + color + texture + vein` (117 dimensions) |
| **Winning Classifier** | **Support Vector Machine (RBF Kernel, C=10)** |
| **Accuracy** | **68.48%** (0.6848) |
| **Macro Precision** | **70.92%** (0.7092) |
| **Macro Recall** | **69.16%** (0.6916) |
| **Macro F1-Score** | **69.19%** (0.6919) |
| **Dataset Size** | **6,900 images** across **80 classes** (5,520 train / 1,380 test) |

### 📈 Best Performance by Feature Combination Size (1 to 5 Groups)

| Feature Count | Best Feature Combination | Top Classifier | Dimensions | Accuracy | Macro F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1 Group** | `color` | Random Forest | 57 | 58.48% | 58.30% |
| **2 Groups** | `color + texture` | SVM | 89 | 67.90% | 68.17% |
| **3 Groups** | `shape + color + texture` | SVM | 103 | 67.25% | 68.43% |
| **4 Groups** | **`shape + color + texture + vein`** 🏆 | **SVM** | **117** | **68.48%** | **69.19%** |
| **5 Groups** | `shape + color + texture + vein + keypoint` | Random Forest | 167 | 64.93% | 65.60% |

---

## 🏗️ Pipeline Architecture

```mermaid
graph TD
    A["Raw Leaf Image (256x256)"] --> B["Otsu Thresholding & Morphology"]
    B --> C["Largest Connected Component Mask & CLAHE"]
    
    C --> D1["1. Shape Features (14D)<br>Contour Geometry + Hu Moments"]
    C --> D2["2. Color Features (57D)<br>HSV Histograms + Color Moments"]
    C --> D3["3. Texture Features (32D)<br>GLCM + LBP + Gabor Filters"]
    C --> D4["4. Vein Topology (14D)<br>Skeleton, Nodes, Thickness"]
    C --> D5["5. BoVW Keypoints (50D)<br>ORB Descriptors + MiniBatchKMeans"]
    
    D1 --> E["Per-Group StandardScaler<br>(Fitted on Train Set Only)"]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    
    E --> F["Combinatorial Feature Fusion<br>(31 Combinations)"]
    F --> G["5 ML Classifiers<br>RF, LightGBM, SVM, Naive Bayes, KNN"]
    G --> H["Macro F1 & Accuracy Leaderboard"]
```

---

## 🔬 Feature Extraction (167 Dimensions)

Feature extraction targets 5 distinct domain properties of leaf imagery:

### 1. 📐 Shape Descriptors (14 Dimensions)
* **Contour Geometry (7 features)**: Area, Perimeter, Aspect Ratio, Solidity ($\frac{\text{Area}}{\text{Hull Area}}$), Circularity ($\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$), Extent, Equivalent Diameter.
* **Hu Invariant Moments (7 features)**: 7 log-scaled, sign-preserving Hu moments derived from central moments for scale, rotation, and translation invariance:
  $$\text{Hu}_{\text{log}} = -\text{sign}(h_i) \cdot \log_{10}(|h_i| + 10^{-30})$$

### 2. 🎨 Color Descriptors (57 Dimensions)
* **HSV Histograms (48 features)**: 16-bin normalized color intensity histograms computed over leaf pixels only across Hue, Saturation, and Value channels ($16 \times 3 = 48$).
* **Color Moments (9 features)**: Mean, Standard Deviation, and Skewness per channel ($H, S, V$).

### 3. 🔬 Texture Descriptors (32 Dimensions)
* **GLCM - Gray-Level Co-occurrence Matrix (6 features)**: Haralick texture properties (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) averaged across 4 angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
* **LBP - Local Binary Patterns (10 features)**: 10-bin uniform LBP histogram ($P=8, R=1$) capturing local surface micro-patterns.
* **Gabor Filter Bank (16 features)**: Mean and variance responses across 4 orientations ($0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}$) and 2 frequencies ($0.1, 0.3$).

### 4. 🍃 Vein & Edge Topography (14 Dimensions)
* **Canny Edge Density**: Ratio of edge pixels to total leaf area.
* **Skeleton Vein Density**: Ratio of skeletonized binary leaf silhouette pixels to total leaf area.
* **Vascular Nodes**: Endpoint count (1 skeleton neighbor) and Branch point count ($\ge 3$ skeleton neighbors).
* **Vein Thickness Profile**: Mean and Standard Deviation of distance transform values sampled along the skeleton line.
* **Edge Orientation Histogram**: 8-bin Sobel gradient direction histogram within the leaf boundary.

### 5. 🔍 Salient Keypoint Bag-of-Visual-Words (50 Dimensions)
* **ORB Descriptors**: Fast ORB keypoint detection (capped at 300 keypoints/image).
* **BoVW Vocabulary**: MiniBatchKMeans ($k=50$) vocabulary built **exclusively on training set descriptors** to prevent data leakage.
* **Visual Word Histograms**: Normalized frequency distribution of assigned visual words per image.

---

## 💡 Technical Insights & Discussion

> [!NOTE]
> **Color & Texture Synergy**: Standalone `color` achieves **58.30% F1** and `texture` achieves **49.32% F1**. Fusing both boosts the Macro F1 score dramatically to **68.17%**, showing that leaf color distribution and micro-texture are strongly complementary.

> [!TIP]
> **Optimal Feature Fusion**: Adding `shape` and `vein` features to `color + texture` yields the peak benchmark score of **69.19% Macro F1**. Vascular node counts and contour geometry provide critical structural boundaries that distinguish leaves with similar coloration.

> [!WARNING]
> **Keypoint Quantization Effect**: Adding ORB Bag-of-Visual-Words features slightly reduced peak accuracy (from 68.48% down to 64.93%). Quantizing local keypoints into 50 visual words introduced high-frequency noise relative to dense global color/texture descriptors on unaligned leaves.

---

## 💻 Installation & Quick Start

### Prerequisites
* Python $\ge 3.8$
* OpenCV, Scikit-Image, Scikit-Learn, LightGBM, SciPy, Pandas, NumPy

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/indian-medicinal-leaves-benchmark.git
cd indian-medicinal-leaves-benchmark
pip install -r requirements.txt
```

### 2. Download Dataset
Download the **Indian Medicinal Leaves Image Datasets** from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset) and place the dataset folder in your workspace.

### 3. Run Benchmark Notebook
Launch Jupyter and execute `cv-final.ipynb`:
```bash
jupyter notebook cv-final.ipynb
```

> [!NOTE]
> Feature extraction automatically utilizes Python `multiprocessing.Pool` across all available CPU cores.

---

## 📊 Benchmark Leaderboard (Top 10 Configurations)

| Rank | Feature Combination | Classifier | Dimensions | Accuracy | Precision | Recall | Macro F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`shape+color+texture+vein`** | **SVM** | **117** | **0.6848** | **0.7092** | **0.6916** | **0.6919** |
| 🥈 **2** | `shape+color+texture` | SVM | 103 | 0.6725 | 0.7001 | 0.6835 | 0.6843 |
| 🥉 **3** | `color+texture+vein` | SVM | 103 | 0.6768 | 0.6965 | 0.6833 | 0.6825 |
| **4** | `color+texture` | SVM | 89 | 0.6790 | 0.6964 | 0.6830 | 0.6817 |
| **5** | `color+texture+vein` | Random Forest | 103 | 0.6652 | 0.7156 | 0.6727 | 0.6792 |
| **6** | `shape+color+texture` | Random Forest | 103 | 0.6652 | 0.7103 | 0.6754 | 0.6785 |
| **7** | `shape+color+texture+vein` | Random Forest | 117 | 0.6674 | 0.7173 | 0.6715 | 0.6774 |
| **8** | `color+texture` | Random Forest | 89 | 0.6659 | 0.7074 | 0.6677 | 0.6742 |
| **9** | `color+texture+vein+keypoint` | Random Forest | 153 | 0.6464 | 0.7152 | 0.6500 | 0.6588 |
| **10** | `color+texture+keypoint` | Random Forest | 139 | 0.6435 | 0.7036 | 0.6520 | 0.6576 |

---

## 📁 Repository Structure

```
.
├── cv-final.ipynb             # Complete Jupyter Notebook with code, logs & reports
├── README.md                  # Comprehensive GitHub repository documentation
├── requirements.txt           # Python dependency requirements
├── all_features_raw.csv       # Saved 169-column feature matrix (6,900 x 169)
├── all_combo_results.csv      # Complete metrics log for all 155 benchmark runs
└── leaderboard_sorted.csv     # Ranked leaderboard sorted by Macro F1-score
```

---

## 📜 Citation & Acknowledgments

If you find this benchmark dataset analysis or pipeline useful for your research or coursework, please cite or reference this repository:

```bibtex
@misc{indian_medicinal_leaves_benchmark_2026,
  author = {Parvathy Krishnaa},
  title = {Indian Medicinal Leaves — Multi-Feature Computer Vision Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/indian-medicinal-leaves-benchmark}}
}
```

### Dataset Attribution
* **Dataset Creator**: Arya Shah ([Kaggle Dataset Link](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset))
* **Dataset Name**: Indian Medicinal Leaves Image Datasets (80 Classes)

---

<div align="center">
  <sub>Built with ❤️ for Computer Vision & Species Identification Research</sub>
</div>
