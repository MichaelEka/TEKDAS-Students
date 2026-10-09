"""Streamlit dashboard for StudentDOA data, analysis, ML, and vAI interpretation."""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
TARGET = "Target"

st.set_page_config(page_title="StudentDOA | Analisis Status Mahasiswa", page_icon="🎓", layout="wide")
st.title("StudentDOA | Analisis Status Mahasiswa")
st.caption("Eksplorasi data, evaluasi model klasifikasi, dan interpretasi hasil secara bertanggung jawab.")


@st.cache_data
def load_data():
    frame = pd.read_csv(DATA_PATH)
    frame.columns = frame.columns.str.strip()
    return frame


@st.cache_resource
def fit_evaluation(frame):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from data_prep import CATEGORICAL_COLUMNS

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    labels = pipeline.classes_.tolist()
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    importance = pd.DataFrame({
        "Fitur": pipeline.named_steps["preprocessing"].get_feature_names_out(),
        "Importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "labels": labels,
        "report": report,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "confusion": confusion_matrix(y_test, predictions, labels=labels),
        "importance": importance,
    }


if not DATA_PATH.exists():
    st.error(f"Dataset tidak ditemukan: {DATA_PATH}")
    st.stop()

data = load_data()
if TARGET not in data.columns:
    st.error(f"Kolom target '{TARGET}' tidak ditemukan.")
    st.stop()

counts = data[TARGET].value_counts().reindex(["Dropout", "Enrolled", "Graduate"]).dropna()
quality = {
    "Nilai kosong": int(data.isna().sum().sum()),
    "Baris duplikat": int(data.duplicated().sum()),
    "Jumlah fitur": int(data.shape[1] - 1),
}

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mahasiswa", f"{len(data):,}")
m2.metric("Fitur", quality["Jumlah fitur"])
m3.metric("Kelas target", data[TARGET].nunique())
m4.metric("Status terbanyak", str(counts.idxmax()))

st.sidebar.header("Navigasi")
tab_overview, tab_data, tab_model, tab_analyst = st.tabs([
    "Ringkasan", "Data & Eksplorasi", "Evaluasi Model", "vAI Analyst"
])

with tab_overview:
    st.subheader("Tujuan proyek")
    st.write("Menganalisis faktor yang berkaitan dengan status akhir studi mahasiswa dan mengevaluasi model klasifikasi tiga kelas.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Distribusi status akhir**")
        chart = pd.DataFrame({"Jumlah": counts, "Persentase (%)": (counts / len(data) * 100).round(1)})
        st.bar_chart(chart["Jumlah"])
        st.dataframe(chart, use_container_width=True)
    with right:
        st.markdown("**Ringkasan kualitas data**")
        st.dataframe(pd.DataFrame([quality]), hide_index=True, use_container_width=True)
        st.markdown("**Status berdasarkan gender**")
        if "Gender" in data.columns:
            st.bar_chart(pd.crosstab(data["Gender"], data[TARGET]))
    st.info(
        "Batasan waktu prediksi: dataset mencakup hasil akademik semester 1 dan 2. "
        "Karena itu, model ini mengklasifikasikan status dengan informasi tersebut; "
        "hasilnya tidak boleh ditafsirkan sebagai prediksi dini sebelum semester 2."
    )

with tab_data:
    st.subheader("Jelajahi dataset")
    selected_status = st.multiselect("Filter status", counts.index.tolist(), default=counts.index.tolist())
    filtered = data[data[TARGET].isin(selected_status)]
    st.write(f"Menampilkan **{len(filtered):,}** dari **{len(data):,}** baris.")
    st.dataframe(filtered.head(200), use_container_width=True)
    st.download_button(
        "Unduh data hasil filter", filtered.to_csv(index=False).encode("utf-8"),
        "student_data_filtered.csv", "text/csv",
    )
    with st.expander("Kamus kolom dan tipe data"):
        schema = pd.DataFrame({"Kolom": data.columns, "Tipe": data.dtypes.astype(str).values})
        st.dataframe(schema, use_container_width=True, hide_index=True)
    if "Age at enrollment" in data.columns:
        st.markdown("**Usia saat pendaftaran menurut status**")
        age = data.groupby(TARGET)["Age at enrollment"].agg(
            Rata_rata="mean", Median="median", Jumlah="count"
        ).round(1)
        st.dataframe(age, use_container_width=True)
        st.bar_chart(age["Rata_rata"])

with tab_model:
    st.subheader("Evaluasi Random Forest")
    st.caption("Test set 20%, stratified, random state 42. Fitur kategori dikodekan dengan one-hot encoding.")
    with st.spinner("Melatih dan mengevaluasi model..."):
        result = fit_evaluation(data)
    a, b = st.columns(2)
    a.metric("Akurasi test", f"{result['accuracy']:.1%}")
    b.metric("Macro F1", f"{result['macro_f1']:.3f}", help="Rata-rata F1 antar kelas; setiap kelas mendapat bobot setara.")
    report = result["report"]
    class_metrics = pd.DataFrame([
        {"Status": label, "Precision": report[label]["precision"],
         "Recall": report[label]["recall"], "F1-score": report[label]["f1-score"],
         "Support": int(report[label]["support"])}
        for label in result["labels"]
    ]).set_index("Status")
    st.markdown("**Metrik per kelas**")
    st.dataframe(class_metrics.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}), use_container_width=True)
    st.markdown("**Confusion matrix · baris = aktual, kolom = prediksi**")
    st.dataframe(pd.DataFrame(result["confusion"], index=result["labels"], columns=result["labels"]), use_container_width=True)
    st.markdown("**Fitur paling berpengaruh menurut Random Forest**")
    st.bar_chart(result["importance"].set_index("Fitur"))

    st.markdown("**Periksa prediksi pada sampel test set**")
    row_index = st.selectbox("Pilih indeks sampel", result["X_test"].index.tolist())
    sample = result["X_test"].loc[[row_index]]
    actual = result["y_test"].loc[row_index]
    prediction = result["pipeline"].predict(sample)[0]
    probabilities = result["pipeline"].predict_proba(sample)[0]
    p1, p2 = st.columns(2)
    p1.metric("Prediksi", str(prediction))
    p2.metric("Aktual", str(actual))
    st.bar_chart(pd.Series(probabilities, index=result["labels"], name="Probabilitas"))
    st.dataframe(sample, use_container_width=True)
    st.caption("Feature importance menunjukkan pola yang digunakan model, bukan penyebab status mahasiswa.")

with tab_analyst:
    st.subheader("Interpretasi berbantuan Ollama")
    st.caption("vAI merangkum keluaran model. Periksa bukti dan konteks langsung sebelum mengambil tindakan.")
    with st.spinner("Menyiapkan prediksi sampel test..."):
        result = fit_evaluation(data)
    row_index = st.selectbox("Sampel untuk dianalisis", result["X_test"].index.tolist(), key="analyst_row")
    sample = result["X_test"].loc[[row_index]]
    prediction = result["pipeline"].predict(sample)[0]
    probabilities = result["pipeline"].predict_proba(sample)[0]
    probability_map = dict(zip(result["labels"], probabilities))
    st.write(f"Prediksi model: **{prediction}**")
    st.write({label: f"{probability:.1%}" for label, probability in probability_map.items()})
    if st.button("Buat interpretasi", type="primary"):
        try:
            from vai_analyst import generate_student_analysis
        except ImportError as exc:
            st.error(f"Paket Ollama belum tersedia. Instal requirements.txt. Detail: {exc}")
        else:
            with st.spinner("Menghubungi model Ollama lokal..."):
                answer = generate_student_analysis(
                    student_profile=sample.iloc[0].to_dict(),
                    predicted_status=str(prediction),
                    probabilities={key: float(value) for key, value in probability_map.items()},
                    model_factors=result["importance"].head(8).to_dict("records"),
                )
            st.markdown(answer)
    st.info("Jangan gunakan keluaran otomatis ini sebagai dasar tunggal keputusan akademik terhadap individu.")

st.caption("Sumber data: StudentDOA/dataset.csv · Target: Dropout, Enrolled, Graduate")
