# Student Placement Prediction Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end classification engine designed to evaluate student profile metrics—including academic transcripts, technical test evaluations, work experience, and communication scores—to predict campus placement likelihood. 

Built using modular scikit-learn pipelines, cross-validated classification models, and comprehensive diagnostic visualization suites, this repository provides actionable data insights for both candidates and institution career desks.

---

## 📌 Project Overview

Student placement outcome modeling is a binary decision task where the target output $y \in \{0, 1\}$ represents whether a student secures campus placement ($1 = \text{Placed}, 0 = \text{Not Placed}$).

This framework implements an automated pipeline to handle tabular data processing, feature transformation, model selection, hyperparameter validation, and metric tracking. Starting with robust baseline benchmarks (Dummy Classifier & Logistic Regression), the pipeline evaluates predictive performance against strict production thresholds.

### Business & Operational Impact
* **For Students:** Identifies critical performance bottlenecks (e.g., threshold CGPA vs. technical assessment requirements) prior to campus recruitment drives.
* **For Institutions:** Enables early intervention by highlighting candidates requiring targeted skill development, interview prep, or academic support.

---

## 🎯 Project Objectives

1. **Automated Pipeline Design:** Build a reusable, data-leakage-free preprocessing and classification pipeline.
2. **Exploratory Analytics:** Inspect statistical distributions, collinearity, and decision boundaries between predictor features.
3. **Data Preprocessing & Encoding:** Apply feature-specific standardization and nominal encoding via `ColumnTransformer`.
4. **Baseline & Comparative Modeling:** Train Logistic Regression models alongside baseline benchmarking classifiers.
5. **Cross-Validation Reliability:** Evaluate model stability across folds using 5-Fold Stratified Cross-Validation.
6. **Performance Auditing:** Inspect model classification metrics utilizing Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

---

## 📊 Dataset Architecture

The primary data source is the **Campus Placement Prediction Dataset** (Kaggle). Each record captures individual academic credentials, employability test results, and post-grad specializations.

### Feature Dictionary

| Feature Name | Type | Category | Description / Encoding |
| :--- | :---: | :---: | :--- |
| `Gender` | Nominal | Demographics | Candidate gender (`M` / `F`) |
| `SSC Marks` | Continuous | Academic | Secondary School Certificate Percentage ($10^{\text{th}}$ Grade) |
| `HSC Marks` | Continuous | Academic | Higher Secondary Certificate Percentage ($12^{\text{th}}$ Grade) |
| `Degree Percentage` | Continuous | Academic | Undergraduate Degree aggregate score |
| `Work Experience` | Nominal | Experience | Prior industrial work experience (`Yes` / `No`) |
| `Employability Test Score` | Continuous | Skill Metric | Technical & aptitude test score conducted by placement cell |
| `MBA Percentage` | Continuous | Academic | Post-graduate MBA aggregate percentage |
| `Specialization` | Nominal | Academic | Core MBA focus stream (`Mkt&Fin` / `Mkt&HR`) |
| `Salary` | Continuous | Post-Placement | Compensation package offered (Target feature for regression; dropped for placement status classification) |
| **`Placement Status`** | **Binary Target** | **Target** | Placement classification outcome: **`Placed` (1)** vs. **`Not Placed` (0)** |

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.10+
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Pipelines:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `DummyClassifier`, `StratifiedKFold`)
* **Development & Control:** VS Code / Jupyter Notebooks, Git / GitHub

---

## 🔄 Machine Learning Pipeline Workflow
