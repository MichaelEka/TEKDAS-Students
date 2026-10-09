"""LLM layer for explaining student-status model output with Ollama.

Standalone examples:
    ollama serve
    ollama pull llama3.2
    python vai_analyst.py --prompt-only
    python vai_analyst.py
"""
import argparse
import os
import sys
from typing import Any

import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(
    student_profile: dict[str, Any],
    predicted_status: str,
    probabilities: dict[str, float],
    model_factors: list[dict[str, Any]] | None = None,
) -> str:
    """Create a prompt grounded in the supplied student data and model output."""
    # Exclude the known target from the profile so the model only sees input evidence.
    profile = {key: value for key, value in student_profile.items() if key != "Target"}
    probability_text = {key: f"{value:.1%}" for key, value in probabilities.items()}
    return f"""
Anda adalah asisten analisis akademik. Gunakan hanya data yang diberikan.
Prediksi adalah keluaran statistik model, bukan kepastian atau bukti penyebab.
Jangan menyimpulkan karakter, motivasi, kondisi keluarga, atau keadaan pribadi mahasiswa.

DATA FITUR MAHASISWA
{profile}

KELUARAN MODEL
Status prediksi: {predicted_status}
Probabilitas per status: {probability_text}
Faktor global model teratas (bukan sebab-akibat): {model_factors or 'Tidak disediakan.'}

TUGAS
Jawab dalam bahasa Indonesia secara ringkas:
- Jelaskan prediksi dan tingkat ketidakpastiannya.
- Sebutkan maksimal dua pengamatan yang didukung fitur yang tersedia.
- Berikan dua saran tindak lanjut yang suportif dan dapat diverifikasi oleh pihak kampus.
- Tegaskan bahwa keputusan penting harus mempertimbangkan konteks dan komunikasi langsung dengan mahasiswa.
""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def generate_student_analysis(
    student_profile: dict[str, Any],
    predicted_status: str,
    probabilities: dict[str, float],
    model_factors: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate an Indonesian interpretation using the local Ollama model."""
    prompt = build_prompt(student_profile, predicted_status, probabilities, model_factors)
    try:
        return call_ollama(prompt, model)
    except Exception as exc:
        return (
            f"Tidak dapat menghubungi Ollama model '{model}'. Pastikan `ollama serve` "
            f"aktif dan model sudah diunduh. Detail: {exc}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi prompt vAI Analyst StudentDOA.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nama model Ollama.")
    parser.add_argument("--prompt-only", action="store_true", help="Cetak prompt tanpa memanggil Ollama.")
    args = parser.parse_args()

    demo_profile = {
        "Age at enrollment": 20,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (grade)": 12.5,
        "Curricular units 2nd sem (approved)": 4,
        "Curricular units 2nd sem (grade)": 11.8,
        "Tuition fees up to date": 1,
        "Scholarship holder": 0,
    }
    demo_probabilities = {"Dropout": 0.62, "Enrolled": 0.23, "Graduate": 0.15}
    prompt = build_prompt(demo_profile, "Dropout", demo_probabilities)
    print(f"=== Validasi StudentDOA vAI Analyst | Model: {args.model} ===")
    if args.prompt_only:
        print(prompt)
        return 0
    try:
        print(call_ollama(prompt, args.model))
        return 0
    except Exception as exc:
        print(
            f"Gagal menghubungi Ollama: {exc}\n"
            f"Pastikan `ollama serve` aktif dan jalankan `ollama pull {args.model}`.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
