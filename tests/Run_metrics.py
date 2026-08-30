"""
Reproducible MAE benchmark for whyanalyst.ai
Uses the ACTUAL, unmodified source files:
  - loader.py      (Dataloader)
  - inspector.py   (describe_data)
  - pipeline.py    (MLPipeline.prepare_data)
  - predictor.py   (Predictor: XGBRegressor wrapper)

This mirrors exactly what main.py's run_analysis_pipeline() does for the
'prediction' branch, so the number below is what the live app would produce
for this dataset and target column.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from src.core.loader import Dataloader
from src.core.inspector import describe_data
from src.ml.pipeline import MLPipeline
from src.ml.predictor import Predictor

DATA_PATH = "data/test_data_large.csv"

# ---- 1. Load data (same as app.py /upload step) ----
loader = Dataloader(DATA_PATH)
df = loader.load_data()
assert df is not None, "Failed to load CSV"
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ---- 2. Build metadata (same as app.py /upload step) ----
metadata = describe_data(df)

# ---- 3. Pick target exactly like main.py does ----
target_col = "Profit" if "Profit" in df.columns else "Sales"
print(f"Target column selected (main.py logic): '{target_col}'\n")

# ---- 4. Prepare data with the real MLPipeline ----
pipeline = MLPipeline(metadata)
Xtrain, Xvalid, ytrain, yvalid, preprocessor = pipeline.prepare_data(df, target_col)
print(f"Train rows: {len(Xtrain)}  |  Validation rows: {len(Xvalid)}")
print(f"Features used after ID/leakage drop: {list(Xtrain.columns)}\n")

# ---- 5. Train with the real Predictor (XGBRegressor) ----
predictor = Predictor(preprocessor)
predictor.train(Xtrain, ytrain, Xvalid, yvalid)

# ---- 6. Evaluate on the held-out validation split ----
preds, mae = predictor.predictvalid(Xvalid, yvalid)
r2 = r2_score(yvalid, preds)

full_target = np.concatenate([ytrain.values, yvalid.values])
target_mean = full_target.mean()
target_std = full_target.std()
target_min = full_target.min()
target_max = full_target.max()
target_range = target_max - target_min

print("\n================ RESULTS ================")
print(f"MAE (validation set)        : {mae:.4f}")
print(f"R^2 (validation set)        : {r2:.4f}")
print(f"Target mean ({target_col})  : {target_mean:.4f}")
print(f"Target std                  : {target_std:.4f}")
print(f"Target range                : [{target_min:.2f}, {target_max:.2f}]  (span={target_range:.2f})")
print(f"MAE as % of target mean     : {100*mae/target_mean:.2f}%")
print(f"MAE as % of target range    : {100*mae/target_range:.2f}%")
print("===========================================")

# ---- 7. Sanity check: compare against a naive baseline ----
naive_pred = np.full_like(yvalid.values, fill_value=ytrain.mean(), dtype=float)
naive_mae = mean_absolute_error(yvalid, naive_pred)
print(f"\nBaseline check -> naive 'always predict train mean' MAE: {naive_mae:.4f}")
print(f"Your model beats the naive baseline by: {100*(1 - mae/naive_mae):.2f}%")