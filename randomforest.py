

# ---------------------------------------------
# Random-Forest regression pipeline (no log-target)
# ---------------------------------------------
import time, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

start = time.time()

# 1 ─── LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv("ai_job_dataset.csv")        # adjust path if necessary

# 2 ─── MINIMAL HYGIENE ─────────────────────────────────────────────────────
cat_clean_cols = [
    "experience_level", "employment_type", "company_location",
    "company_size", "employee_residence", "salary_currency"
]
for c in cat_clean_cols:
    df[c] = df[c].str.strip().str.upper()

# 3 ─── FEATURE ENGINEERING ────────────────────────────────────────────────
df["is_remote"] = (df["remote_ratio"] >= 50).astype(int)

df["posting_date"]         = pd.to_datetime(df["posting_date"])
df["application_deadline"] = pd.to_datetime(df["application_deadline"])
df["days_open"]            = (df["application_deadline"]
                              - df["posting_date"]).dt.days.clip(lower=0)

# 4 ─── FEATURES & TARGET ──────────────────────────────────────────────────
target = "salary_usd"           # ←── direct USD target
numeric_cols = ["years_experience", "benefits_score",
                "remote_ratio", "job_description_length", "days_open"]
categorical_cols = [
    "job_title", "experience_level", "employment_type",
    "company_location", "company_size", "employee_residence",
    "industry", "salary_currency", "is_remote"
]

X, y = df[numeric_cols + categorical_cols], df[target]

# 5 ─── SPLIT ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.20,
    random_state=42,
    stratify=df["experience_level"]
)

# 6 ─── PREPROCESSOR ────────────────────────────────────────────────────────
pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
     ("num", "passthrough", numeric_cols)]
)

# 7 ─── RANDOM-FOREST & PIPELINE ────────────────────────────────────────────
rf   = RandomForestRegressor(random_state=42, verbose=1)
pipe = Pipeline([("pre", pre), ("rf", rf)])

# 8 ─── RANDOM SEARCH (coarser, “steppier” trees) ──────────────────────────
param_space = {
    # ▸ BIGGER LEAVES  → coarser plateaus
    "rf__min_samples_leaf":  [10, 20, 40],

    # ▸ LESS AGGRESSIVE SPLITTING at internal nodes
    "rf__min_samples_split": [10, 20],

    # ▸ SHALLOW-ISH TREES  (None lets tree stop only at leaf-size)
    "rf__max_depth":         [None, 25],

    # ▸ MORE TREES to stabilise variance
    "rf__n_estimators":      [300, 500],

    # other common knobs left unchanged
    "rf__max_features":      ["sqrt"],
    "rf__bootstrap":         [True]
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_space,
    n_iter=8,                        # 8 random combos – adjust to taste
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=2
)
search.fit(X_train, y_train)

# 9 ─── FINAL METRICS ───────────────────────────────────────────────────────
best_model = search.best_estimator_
y_pred     = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\nBest CV parameters : {search.best_params_}")
print(f"Test RMSE          : {rmse:,.0f} USD")
print(f"Test MAE           : {mae:,.0f} USD")
print(f"Test R²            : {r2:.3f}")

# 10 ─── FEATURE IMPORTANCE (top-15) ───────────────────────────────────────
importances   = best_model.named_steps["rf"].feature_importances_
encoded_names = best_model.named_steps["pre"].get_feature_names_out()
fi = (pd.Series(importances, index=encoded_names)
      .sort_values(ascending=False)
      .head(15))
print("\nTop-15 feature importances:")
print(fi)

# 11 ─── SAVE ───────────────────────────────────────────────────────────────
joblib.dump(best_model, "rf_regressor_pipeline.pkl")

