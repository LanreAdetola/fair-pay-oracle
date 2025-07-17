# 🧠 Fair Pay Oracle – Salary Prediction with Machine Learning

This project predicts salary based on years of experience using regression techniques in Python. It follows a full machine learning pipeline — from data cleaning and exploratory analysis to model training, evaluation, and cloud deployment with Azure.


---

## Project Status (Current Snapshot)

* ✅ Data loaded (`salary_data.csv`).
* ✅ Basic cleaning: dropped missing + duplicate rows.
* ✅ Exploratory Data Analysis (EDA): distributions, outliers, correlations, group summaries.
* ✅ Feature engineering: one‑hot encoding for categoricals (`pd.get_dummies(..., drop_first=True)`), numeric scaling for `Age` & `Years of Experience`.
* ✅ Baseline & advanced regression models trained + evaluated on an **80/20 train/test split**.
* ✅ Performance summary table & comparison plots created.
* 🔄 Hyperparameter tuning in progress (**KNN & Ridge** focus).
* ⏭️ Next: model interpretability + fairness diagnostics + deployment endpoint/UI.

---

## Why This Project?

Compensation discussions are often emotional and opaque. **Fair Pay Oracle** explores whether we can build a *data‑informed* salary estimator given a small structured dataset containing:

* Age
* Gender
* Education Level
* Job Title
* Years of Experience
* Salary (target)

> ⚠️ This is an educational / exploratory dataset. Do **not** use for real compensation decisions without domain validation, larger data, and fairness audits.

---

## Repository Layout

```
fair-pay-oracle/
├── README.md                # (this file)
├── main.ipynb               # interactive notebook: EDA + modeling
├── salary_data.csv          # input dataset
├── requirements.txt         # Python deps (core libs)
└── model_performance_summary.csv  # optional output you generate
```

---

## Quick Start

Clone (or open) the project folder in VS Code, then in the integrated terminal:

```bash
# 1. Create & activate virtual env (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell equivalent
# .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# Optional: install extras used during experiments
pip install xgboost

# 3. Launch Jupyter
jupyter notebook  # open main.ipynb
```

---

## Data Overview

**File:** `salary_data.csv`

| Column                | Type                         | Notes                                             |
| --------------------- | ---------------------------- | ------------------------------------------------- |
| `Age`                 | numeric                      | Candidate age in years.                           |
| `Gender`              | categorical                  | e.g., Male / Female / Other (depends on dataset). |
| `Education Level`     | categorical                  | Bachelor's, Master's, PhD, etc.                   |
| `Job Title`           | categorical high‑cardinality | Many roles; becomes many one‑hot columns.         |
| `Years of Experience` | numeric                      | Total professional experience.                    |
| `Salary`              | numeric (target)             | Annual compensation (currency unspecified).       |

### Cleaning Steps Performed

* Dropped rows with any missing values: `df.dropna()`.
* Dropped duplicate rows: `df.drop_duplicates()`.
* Confirmed column dtypes; converted where needed.

### Encoding & Scaling

```python
# One-hot encode categoricals; drop_first reduces dummy trap risk
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

# Target
y = df_encoded["Salary"]

# Features
X = df_encoded.drop("Salary", axis=1).copy()

# Scale numeric columns only
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X.loc[:, ["Age", "Years of Experience"]] = scaler.fit_transform(X[["Age", "Years of Experience"]])
```

---

## EDA Highlights

**Run `main.ipynb` to reproduce. Key checks included:**

* Histograms for numeric features (age, experience, salary) → range, skew, gaps.
* Boxplots by category (e.g., Salary by Gender, Education) → outliers & spread.
* Scatter: Years of Experience vs Salary → broad upward trend; variance grows at higher experience.
* Categorical counts → class imbalance (some job titles rare).
* Correlation heatmap (numeric only) → experience & age both relate to salary; collinearity expected.
* Grouped summaries: salary means by Education Level, Gender, Job Title clusters.

> **Tip:** High‑cardinality job titles can dominate models; consider grouping titles into families (Engineering, Finance, Data, Product, etc.) in future iterations.

---

## Modeling Workflow (Baseline → Advanced)

1. **Train/Test Split** (80/20, `random_state=42`).
2. **Baseline Linear Regression** (with & w/o scaling) for reference.
3. **Regularized Linear Models**: Ridge (L2), Lasso (L1).
4. **Tree & Ensemble Models**: Decision Tree, Random Forest, Gradient Boosting, XGBoost.
5. **Non‑parametric / Other**: KNN, SVR, MLP Neural Net.
6. **Comparison Metrics**: RMSE (lower better), R² (higher better; 1.0 is perfect).

---

## Current Test Results

*(Values rounded for readability; taken from most recent run.)*

| Model               | RMSE          | R²         |
| ------------------- | ------------- | ---------- |
| **KNN Regressor**   | **12,989.30** | **0.9107** |
| Ridge Regression    | 14,061.14     | 0.8954     |
| Lasso Regression    | 14,565.05     | 0.8877     |
| Random Forest       | 14,934.15     | 0.8820     |
| Gradient Boosting   | 15,067.14     | 0.8799     |
| Linear (No Scaling) | 15,851.15     | 0.8670     |
| Linear (Scaled)     | 15,851.15     | 0.8670     |
| XGBoost             | 15,561.76     | 0.8719     |
| SVR                 | 43,448.99     | 0.0010     |
| MLP Regressor       | 103,055.05    | -4.6199    |

### Interpretation

* **KNN currently leads** in both RMSE & R² on the test split. May indicate local structure in feature space. Tune `k`, distance metric.
* **Ridge** is the best *parametric / interpretable* model. Good tradeoff if you need coefficients.
* **Lasso** slightly behind; useful for feature selection.
* Tree ensembles competitive but not dominant — possibly due to dataset size, noise, or sparse one‑hot features.
* **SVR & MLP underperformed** with default settings; would need tuning, scaling (done), and maybe dimensionality reduction.

---

## Hyperparameter Tuning

Focused on **KNN** & **Ridge**.

### KNN Grid Example

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

knn_param_grid = {
    "n_neighbors": [3,5,7,9,11,15],
    "weights": ["uniform", "distance"],
    "p": [1,2]  # Manhattan vs Euclidean
}

knn_grid = GridSearchCV(KNeighborsRegressor(), knn_param_grid,
                        scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
knn_grid.fit(X_train_s, y_train_s)
print(knn_grid.best_params_)
```

### Ridge Grid Example

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge_param_grid = {"alpha": np.logspace(-3, 3, 13)}

ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_param_grid,
                          scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
ridge_grid.fit(X_train_s, y_train_s)
print(ridge_grid.best_params_)
```

> 🔁 For *gold‑standard* tuning, wrap preprocessing + model inside a `Pipeline` + `ColumnTransformer` and tune end‑to‑end under cross‑validation (see notebook section: **Pro Pipeline**).

---

## Save & Reuse Best Model

```python
import joblib
joblib.dump(best_knn, "best_model_knn.pkl")
# or
joblib.dump(best_ridge, "best_model_ridge.pkl")
```

Load later:

```python
model = joblib.load("best_model_knn.pkl")
pred = model.predict(new_X)
```

---

## Reproduce Full Workflow

1. Activate virtual env.
2. Install deps.
3. Open `main.ipynb`.
4. Run notebook top‑to‑bottom (EDA → encoding → scaling → split → models).
5. Record metrics (auto‑collected into DataFrame if you use the comparison cell).
6. Export summary:

   ```python
   summary_df_sorted.to_csv("model_performance_summary.csv", index=False)
   ```

---

## Next Steps / Roadmap

* [ ] Finish KNN & Ridge hyperparameter tuning.
* [ ] Add grouped job‑family feature to reduce one‑hot explosion.
* [ ] Fairness slices: compare errors & predicted salary gaps by Gender & Education.
* [ ] Model explainability: SHAP for top models (KNN via surrogate, Ridge via coefficients).
* [ ] Simple prediction API (FastAPI) + lightweight UI (Streamlit or Blazor front‑end calling API).
* [ ] Batch scoring script for HR CSV uploads.

---

## Notes on Fairness & Ethics

Salary data can encode historic bias. Before deployment:

* Validate feature usage with stakeholders (legal/HR).
* Consider excluding sensitive attributes (Gender) from final model and audit downstream impact.
* Report group performance metrics.

---

## Contributing

Open to collaboration! Please:

1. Create a new branch.
2. Add notebooks / scripts under `experiments/`.
3. Include environment diffs in `requirements-dev.txt` if needed.
4. Submit a pull request with short model summary + metrics.

---

## Contact

Maintainer: **Lanre**
Questions? Open an issue or reach out via Teams / email.

---

*This README is a working draft. Update as results evolve.*
