"""Prepare StudentDOA data for multiclass student-status prediction.

Flow: dataset.csv -> validation -> stratified train/test split.
Run from any directory with: python data_prep.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset.csv"
TARGET = "Target"
EXPECTED_CLASSES = {"Dropout", "Enrolled", "Graduate"}

# These fields are category codes, not quantities. Keep this list aligned with
# the Student Dropout and Academic Success dataset codebook.
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read the source CSV and normalize column names."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data


def validate_data(data: pd.DataFrame) -> None:
    """Check required columns and target labels before data preparation."""
    if TARGET not in data.columns:
        raise ValueError(f"Dataset wajib memiliki kolom target '{TARGET}'.")
    if data.empty:
        raise ValueError("Dataset tidak memiliki baris data.")
    if data.columns.duplicated().any():
        raise ValueError("Dataset memiliki nama kolom duplikat.")
    if data[TARGET].isna().any():
        raise ValueError(f"Kolom '{TARGET}' memiliki nilai kosong.")
    found_classes = set(data[TARGET].astype(str).unique())
    if found_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Nilai target tidak sesuai. Ditemukan {sorted(found_classes)}, "
            f"diharapkan {sorted(EXPECTED_CLASSES)}."
        )
    non_numeric = data.drop(columns=[TARGET]).select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Fitur harus numerik; periksa kolom: {non_numeric}")
    absent_categorical = sorted(set(CATEGORICAL_COLUMNS) - set(data.columns))
    if absent_categorical:
        raise ValueError(f"Kolom kategori yang diharapkan tidak ada: {absent_categorical}")


def main() -> None:
    data = load_data()
    validate_data(data)
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[TARGET]
    )
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)
    data.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)

    report = {
        "source_file": DATA_FILE.name,
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "feature_count": int(data.shape[1] - 1),
        "target": TARGET,
        "classes": {str(k): int(v) for k, v in data[TARGET].value_counts().items()},
        "missing_values": {str(k): int(v) for k, v in data.isna().sum().items() if v},
        "duplicate_rows": int(data.duplicated().sum()),
        "categorical_features": CATEGORICAL_COLUMNS,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_seed": 42,
    }
    (BASE_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Persiapan data selesai.")
    print(f"Baris train/test: {len(train):,} / {len(test):,}")
    print("Distribusi target:\n" + data[TARGET].value_counts().to_string())
    print(f"Baris duplikat: {report['duplicate_rows']:,}")
    print("Dibuat: student_train.csv, student_test.csv, student_ml_dataset.csv")
    print("Dibuat: data_quality_report.json")


if __name__ == "__main__":
    main()
