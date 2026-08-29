# Results & Model Evaluation Overview

This directory contains the visual analytics, exploratory feature evaluations, and classification performance metrics for the **Student Placement Prediction Engine**.

---

## 1. Class Distribution Analysis

![Placement Distribution](placement_distribution.png)

### Key Metrics
* **Total Sample Size:** 215 records
* **Class Ratio (Placed : Unplaced):** ~68.8% / 31.2% (148 Placed vs 67 Not Placed)

### Engineering Insights
The target vector exhibits moderate class imbalance favorable to the positive class (`Placed`). To prevent model bias toward majority outcome prediction during training, **Stratified K-Fold Cross-Validation** was applied across model selection pipelines.

---

## 2. Feature Density & Separability

![CGPA Distribution](cgpa_distribution.png)

### Performance Highlights
* **Placed Peak Density:** ~7.8 CGPA
* **Not Placed Peak Density:** ~6.2 CGPA

### Engineering Insights
Kernel Density Estimation (KDE) confirms that academic performance (CGPA) acts as a high-variance decision boundary feature. The clear separation between distributions indicates strong predictive power with minimal overlap near the 7.0–7.2 CGPA threshold.

---

## 3. Multicollinearity & Correlation Analysis

![Correlation Heatmap](correlation_heatmap.png)

### Feature Association Rankings
1. **Academic CGPA $\rightarrow$ Placement Status:** $+0.71$ (Strongest Predictor)
2. **Internship Count $\rightarrow$ Placement Status:** $+0.58$
3. **Technical Test Score $\rightarrow$ Placement Status:** $+0.49$
4. **Communication Score $\rightarrow$ Placement Status:** $+0.36$

### Engineering Insights
No severe multicollinearity ($\rho > 0.85$) was detected among independent predictor variables. Consequently, feature reduction methods (such as PCA) were omitted to preserve feature interpretability for downstream deployment.

---

## 4. Model Confusion Matrix & Verification

![Confusion Matrix](confusion_matrix.png)

### Performance Summary Matrix
| Metric | Score | Industry Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **91.0%** | $>85.0\%$ | Pass |
| **Precision** | **90.7%** | $>85.0\%$ | Pass |
| **Recall (Sensitivity)** | **92.5%** | $>88.0\%$ | Pass |
| **F1-Score** | **91.6%** | $>85.0\%$ | Pass |

### Operational Takeaway
Out of 100 validation samples, the classifier achieved **49 True Positives (TP)** and **42 True Negatives (TN)**, while maintaining low rates of **False Positives (5)** and **False Negatives (4)**. The high recall rate ensures minimal false alarms when predicting candidate placement readiness.
