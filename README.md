<h1 align="center">Code for VIL-SOW</h1>

<h3 align="center">
Multi-view Multi-label Feature Selection By View-specific Implicit Label Information and Second-order Weight Matrix
</h3>

<p align="center">
  <a href="https://doi.org/10.1109/TMM.2026.3701532">
    <img src="https://img.shields.io/badge/IEEE-TMM%202026-blue">
  </a>
  <img src="https://img.shields.io/badge/Language-MATLAB-orange">
  <a href="https://doi.org/10.1109/TMM.2026.3701532">
    <img src="https://img.shields.io/badge/DOI-10.1109%2FTMM.2026.3701532-green">
  </a>
</p>

> Official MATLAB implementation of **"Multi-view Multi-label Feature Selection By View-specific Implicit Label Information and Second-order Weight Matrix"**, published in *IEEE Transactions on Multimedia (TMM), 2026*.

---

## 🌟 Overview

This repository provides the MATLAB implementation of **VIL-SOW**, a multi-view multi-label feature selection method based on view-specific implicit label information and a second-order weight matrix.

### 🔑 Key Highlights

- **View-specific Implicit Label Information:** Explores implicit label information within individual views to facilitate feature selection.
- **Second-order Weight Matrix:** Incorporates second-order weight information into the feature selection framework.
- **MATLAB Implementation:** Provides code and an executable demonstration using the `emotions` dataset.

---

## 🗂️ Datasets

The `emotions` dataset is included in the `datasets` directory for demonstration purposes.

---

## ⚙️ Requirements

- MATLAB R2018a or later (recommended).
- Dataset: `emotions.mat` (included in `./datasets`).
- ML-KNN functions for classification-based evaluation.

---

## 🚀 Quick Start

**Step 1.** Clone the repository:

```bash
git clone https://github.com/gammaxwei/VIL-SOW.git
cd VIL-SOW
```

**Step 2.** Open MATLAB and navigate to the repository directory.

**Step 3.** Run the demonstration:

```matlab
demo
```

The script can report the following metrics when the ML-KNN functions are available:

- Average Precision (AP)
- One Error (OE)
- Ranking Loss (RL)
- Hamming Loss (HL)
- Coverage (CV)

The convergence plot is saved to the `results` directory when generated.

---

## 📝 Corrections

This section documents corrections and clarifications to **Table III** and **Figs. 6 and 7** of the published paper.

### 📊 Table III: Numerical Corrections

Several numerical entries in Table III have been corrected. The reporting precision of the Ranking Loss (RL) and Hamming Loss (HL) results has also been standardized to two decimal places.

The following corrections apply to the **scene** dataset.

| Metric | Method | Original | Corrected |
|:---:|:---:|---:|---:|
| AP (%) ↑ | GRRO | 85.61 ± 0.61 | **85.61 ± 0.24** |
| OE (%) ↓ | GRRO | 3.41 ± 3.41 | **3.41 ± 0.82** |
| RL (%) ↓ | GRRO | 6.10 ± 6.10 | **6.12 ± 0.45** |
| HL (%) ↓ | GRRO | 8.80 ± 8.85 | **8.85 ± 0.15** |
| CV ↓ | GRRO | 11.24 ± 11.24 | **11.24 ± 0.18** |
| OE (%) ↓ | SSFS | 1.37 ± 3.61 | **1.37 ± 0.92** |

*↑ Higher is better; ↓ Lower is better. Corrected values are highlighted in bold.*

### 🔍 Corrected Table III: Full Results

The complete corrected performance comparison is provided below.

<details>
<summary><b>📄 Click to view the corrected Table III</b></summary>

<br>

<p align="center">
  <a href="https://github.com/user-attachments/assets/b0a446c6-ee31-45c1-9c98-d63dc6673d4a">
    <img src="https://github.com/user-attachments/assets/b0a446c6-ee31-45c1-9c98-d63dc6673d4a" alt="Corrected Table III" width="100%">
  </a>
</p>

<p align="center">
  <em>Table III. Performance comparison of different methods on different metrics.</em>
</p>

</details>

### 🔢 Numerical Precision

Some RL and HL values were previously rounded to one decimal place, with a trailing zero added as a placeholder for consistent formatting. For example, the original values 0.06123 and 0.08847 were displayed as 6.10% and 8.80%, respectively, rather than 6.12% and 8.85%.

The corrected values are rounded directly to two decimal places from the original experimental results. These adjustments only standardize numerical precision without changing the underlying experiments.

Corrections to incorrectly reported standard deviations are documented separately in the numerical correction table above.

### 📈 Figures 6 and 7: Metric Labels

The metric labels **OE** and **CV** were inadvertently interchanged in Figs. 6 and 7.

The corresponding labels have been corrected.

These corrections do not affect the proposed method, the experimental setup, or the overall conclusions of the paper.

> **Official Correction Notice:** A link to the correction notice will be added once it becomes available on IEEE Xplore.

---

## 🔗 Links

- **Paper:** https://doi.org/10.1109/TMM.2026.3701532
- **Code:** https://github.com/gammaxwei/VIL-SOW
