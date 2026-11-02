"""Diagnostic script for the latent-defect labels in the DC test dataset.

Run from the project folder with:
    python check_anomaly.py

Expected dataset location:
    data/dc_test_results.csv
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dc_test_results.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure dc_test_results.csv is inside the project's data folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "sort_pass", "is_latent_defect_truth",
    "IIL_A", "IIL_B",
    "IDDQ_V1", "IDDQ_V2", "IDDQ_V3",
    "IDDQ_V4", "IDDQ_V5", "IDD_STATIC"
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError("Missing required dataset columns: " + ", ".join(missing))

print("Dataset shape:", df.shape)

print("\nLatent defect distribution:")
print(df["is_latent_defect_truth"].value_counts(dropna=False).sort_index())

passed = df.loc[df["sort_pass"] == 1].copy()

print("\nDevices that passed initial sort:", len(passed))
print("\nDefects among passed devices:")
print(passed["is_latent_defect_truth"].value_counts(dropna=False).sort_index())

features = [
    "IIL_A", "IIL_B",
    "IDDQ_V1", "IDDQ_V2",
    "IDDQ_V3", "IDDQ_V4",
    "IDDQ_V5", "IDD_STATIC"
]

print("\nMeasurement comparison (group 0 = normal, group 1 = latent defect):")
for col in features:
    print("\nFeature:", col)
    print(
        passed.groupby("is_latent_defect_truth")[col]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

print("\nDiagnostic complete. These statistics describe the supplied synthetic dataset;")
print("they do not establish production defect-detection performance.")
