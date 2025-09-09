import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Base Paths ===
base_dir = os.path.dirname(__file__)
base_output_path = os.path.join(base_dir, "outputs")
os.makedirs(base_output_path, exist_ok=True)

# === Predictions folder ===
# (parallel to example_data/, inside repo, not absolute C:/)
prediction_dir = os.path.join(base_dir, "predictions")

# === Output file ===
output_csv = os.path.join(base_output_path, "summary_metrics.csv")

# === Collect All Prediction Files ===
csv_files = glob.glob(os.path.join(prediction_dir, "*_predictions.csv"))

# === Initialize Summary Table ===
summary_rows = []

for file in csv_files:
    filename = os.path.basename(file)
    df = pd.read_csv(file)

    # --- Extract metadata from filename ---
    name_lower = filename.lower()
    model_type = (
        "RF" if "rf" in name_lower else
        "SVM" if "svm" in name_lower else
        "XGB" if "xgb" in name_lower else
        "UNKNOWN"
    )

    descriptor_type = (
        "MOE" if "moe" in name_lower else
        "MORGAN" if "morgan" in name_lower else
        "MORDRED" if "mordred" in name_lower else
        "UNKNOWN"
    )

    cv_type = (
        "LOSO" if "loso" in name_lower else
        "10-Fold" if "kfold" in name_lower else
        "UNKNOWN"
    )

    tuned = "Tuned" if "tuned" in name_lower else "Untuned"

    # --- Compute Metrics ---
    y_true = df['solubility_g_100g_log']
    y_pred = df['prediction']
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # --- Append to summary ---
    summary_rows.append({
        "Model": model_type,
        "Descriptor": descriptor_type,
        "CV_Type": cv_type,
        "Tuned": tuned,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "File": filename
    })

# === Save Summary Table ===
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_csv, index=False)
print(f" Summary saved to: {output_csv}")