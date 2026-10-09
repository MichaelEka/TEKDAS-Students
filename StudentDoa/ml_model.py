"""Train and evaluate a multiclass student-status model.

Run data_prep.py first, then: python ml_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import CATEGORICAL_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_status_pipeline.pkl"


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build preprocessing and classifier together to prevent train/test leakage."""
    categorical = [name for name in CATEGORICAL_COLUMNS if name in feature_columns]
    numeric = [name for name in feature_columns if name not in categorical]
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def main() -> None:
    train = pd.read_csv(BASE_DIR / "student_train.csv")
    test = pd.read_csv(BASE_DIR / "student_test.csv")
    feature_columns = [column for column in train.columns if column != TARGET]
    X_train, y_train = train[feature_columns], train[TARGET]
    X_test, y_test = test[feature_columns], test[TARGET]

    pipeline = build_pipeline(feature_columns)
    print("Melatih Random Forest untuk klasifikasi status mahasiswa...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    labels = pipeline.classes_.tolist()
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)

    # Majority-class baseline clarifies whether the ML model adds value over a simple rule.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "macro_f1": float(f1_score(y_test, baseline_predictions, average="macro")),
    }
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "baseline_most_frequent": baseline_metrics,
        "classes": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "categorical_features_one_hot_encoded": [name for name in CATEGORICAL_COLUMNS if name in feature_columns],
        "random_state": 42,
    }
    print("\nLaporan klasifikasi:\n")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print(f"Akurasi model: {metrics['accuracy']:.3f} | Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Baseline akurasi: {baseline_metrics['accuracy']:.3f} | Macro F1: {baseline_metrics['macro_f1']:.3f}")

    joblib.dump(pipeline, MODEL_FILE)
    transformed_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": transformed_names,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(BASE_DIR / "feature_importance.csv", index=False)

    prediction_file = test[[TARGET]].copy()
    prediction_file.insert(0, "row_index", test.index)
    prediction_file["predicted_status"] = predictions
    probabilities = pipeline.predict_proba(X_test)
    for index, label in enumerate(labels):
        prediction_file[f"probability_{str(label).lower()}"] = probabilities[:, index]
    prediction_file.to_csv(BASE_DIR / "student_predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).rename_axis("actual").to_csv(
        BASE_DIR / "confusion_matrix.csv"
    )
    (BASE_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModel disimpan: {MODEL_FILE.name}")
    print("Artefak dibuat: feature_importance.csv, student_predictions.csv,")
    print("confusion_matrix.csv, model_metrics.json")


if __name__ == "__main__":
    main()
