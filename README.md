# Student Retention Analysis

Exploratory analysis and predictive modeling of student dropout, using a longitudinal panel of 20,000 students followed for up to 8 semesters (79,239 student-semester records).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mihaighise/student_retention_analysis/blob/main/Student_Retention.ipynb)

## Questions
1. Which factors are associated with dropping out, and how strongly?
2. When do students drop out, and which groups are at higher risk?
3. How well can next-semester dropout be predicted, and which model works best?

## Dataset
[Student Retention and Academic Performance Data](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data) (Kaggle, synthetic), included as `academic_survival_longitudinal.csv`.

Each row is one student in one semester: demographics, family income, scholarship, tuition, work hours, GPA, attendance, LMS logins, advising visits, failed courses and financial stress. The target `Target_Dropout_Next_Sem` is 1 when the student does not return for the next semester. Students whose record ends while still enrolled are **censored** (outcome unknown).

## Report
The full write-up is in [`Student_Retention_Report.pdf`](Student_Retention_Report.pdf).

## Notebook contents
| Section | Contents |
|---|---|
| 2. Data structure | How each student's record ends (dropped out / graduated / censored) and why censored rows are excluded |
| 3. Cleaning | Attendance clipping, gender normalisation, missing values (median for skewed income, mean for symmetric LMS logins, plus missing flags) |
| 4. EDA | Dropout risk by semester, correlations with dropout, risk curves per feature, group comparisons, GPA × financial stress, trajectories before dropout |
| 5. Modeling | Grouped CV by student, 5 models + baseline, Optuna / grid-search tuning, threshold selection, held-out test evaluation, odds ratios and SHAP |
| 6. Findings | Insights, model results, recommendations, limitations |

## Key findings
- **Attrition is high**: 34.6 % of students are observed dropping out, and this understates the true rate because 56 % of students are censored before their outcome is known.
- **Risk grows over time**: the per-semester dropout rate rises from 7.4 % (semester 1) to 14.7 % (semester 7).
- **Grades and financial stress add up**: GPA ≤ 2.0 with high financial stress → ~55 % dropout risk; GPA > 3.0 with low stress → ~2 %.
- **First-generation students are at higher risk** and **scholarship holders at lower risk** (7.3 % vs 11.2 % dropout rate), and both effects remain strong in the model after controlling for income, stress and grades.
- **Dropout is gradual**: rising financial stress and a faster GPA decline are visible 2-3 semesters before a student leaves.
- Gender, age and emergency expenses carry no meaningful signal.

## Model results
All splits are grouped by `Student_ID`, so no student appears in both training and test data.

| Model (5-fold grouped CV) | ROC AUC | PR AUC |
|---|---|---|
| Logistic Regression (tuned, L1) | 0.802 | 0.402 |
| LightGBM (tuned, Optuna) | 0.801 | 0.401 |
| XGBoost | 0.799 | 0.396 |
| Random Forest | 0.790 | 0.378 |
| Baseline (always "stays") | 0.500 | 0.101 |

**Final model (tuned logistic regression) on held-out students:** ROC AUC 0.809, PR AUC 0.396 (random: 0.5 / 0.10). Its predicted probabilities are well calibrated.

All models perform alike because every relationship in the data is smooth and additive, so logistic regression already captures the available signal. The notebook explains this in section 5.3.

## Running the notebook
**Colab:** open the badge link above and run all cells; the first cell installs the packages Colab is missing.

**Locally** (Python 3.10+):
```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna shap matplotlib seaborn jupyter
jupyter notebook Student_Retention.ipynb
```
A full run takes about 5-10 minutes, most of it the Optuna hyperparameter search.

## Limitations
The data are synthetic, so relationships are cleaner than in real student records. All findings are associations, not causal effects.
