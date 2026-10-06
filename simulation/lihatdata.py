import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "customers.csv"
df = pd.read_csv("customer.csv")

print("Ukuran data:", df.shape)
print("\nTipe kolom:\n", df.dtypes)
print("\n5 baris pertama:\n", df.head())
print("\nPersentase nilai kosong tertinggi:\n",
    df.isna().mean().sort_values(ascending=False).head() * 100)