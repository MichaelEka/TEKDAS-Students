from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_FILE = Path(__file__).resolve().parent / "data_prepared" / "coffee_sales_prepared.csv"
COLORS = ["#087F8C", "#7656B5", "#D98B39"]


@st.cache_data
def load_data(path: str, modified_time: int) -> pd.DataFrame:
    """Muat data yang sudah lolos audit dan parse kolom tanggal."""
    data = pd.read_csv(path)
    data["transaction_date"] = pd.to_datetime(data["transaction_date"])
    return data


def format_angka(value: float, decimals: int = 2) -> str:
    """Format angka dengan titik ribuan dan koma desimal."""
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def build_monthly_summary(data: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Susun ringkasan bulanan, termasuk bulan yang tidak memiliki catatan."""
    monthly = data.groupby("month").agg(
        total_sales=("sales", "sum"),
        total_transactions=("transaction_id", "nunique"),
        total_units=("transaction_qty", "sum"),
    ).reindex(months)

    previous = monthly["total_sales"].shift(1)
    monthly["growth_pct"] = (
        (monthly["total_sales"] - previous) / previous * 100
    ).where(previous.notna() & previous.ne(0))
    monthly["average_transaction_value"] = (
        monthly["total_sales"] / monthly["total_transactions"]
    )
    return monthly.reset_index().rename(columns={"index": "month"})


def render_monthly_summary(monthly: pd.DataFrame, last_month: pd.Period) -> None:
    st.subheader("Ringkasan penjualan")
    st.caption(
        "Total, transaksi, dan rata-rata: seluruh periode. "
        f"Pertumbuhan: {last_month} terhadap {last_month - 1}."
    )

    total_sales = monthly["total_sales"].sum()
    total_transactions = monthly["total_transactions"].sum()
    total_units = monthly["total_units"].sum()
    average = total_sales / total_transactions if total_transactions else None
    growth = monthly["growth_pct"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total penjualan", format_angka(total_sales))
    col2.metric("Transaksi tercatat", format_angka(total_transactions, 0))
    col3.metric(
        "Rata-rata nilai transaksi",
        format_angka(average) if average is not None else "Belum tersedia",
    )
    col4.metric(
        f"Pertumbuhan {last_month} terhadap {last_month - 1}",
        f"{format_angka(growth)}%" if pd.notna(growth) else "Belum tersedia",
    )

    st.subheader("Tren penjualan bulanan")
    fig = px.line(
        monthly,
        x="month",
        y="total_sales",
        markers=True,
        labels={"month": "Bulan", "total_sales": "Nilai penjualan"},
    )
    fig.update_traces(line_color="#087F8C", connectgaps=False)
    fig.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, key="trend")

    with st.expander("Lihat angka ringkasan bulanan"):
        table = monthly.rename(columns={
            "month": "Bulan",
            "total_sales": "Total penjualan",
            "total_transactions": "Transaksi tercatat",
            "total_units": "Unit terjual",
            "growth_pct": "Pertumbuhan (%)",
            "average_transaction_value": "Rata-rata nilai transaksi",
        })
        st.dataframe(
            table.style.format(
                {
                    "Total penjualan": "{:,.2f}",
                    "Transaksi tercatat": "{:,.0f}",
                    "Unit terjual": "{:,.0f}",
                    "Pertumbuhan (%)": "{:,.2f}",
                    "Rata-rata nilai transaksi": "{:,.2f}",
                },
                na_rep="Belum tersedia",
                decimal=",",
                thousands=".",
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_store_and_category(data: pd.DataFrame) -> None:
    st.header("Perbandingan gerai dan kategori")
    left, right = st.columns(2)

    with left:
        st.subheader("Penjualan per gerai")
        sales_by_store = (
            data.groupby("store_location", as_index=False)
            .agg(total_sales=("sales", "sum"))
            .sort_values("total_sales", ascending=False)
        )
        fig_store = px.bar(
            sales_by_store,
            x="store_location",
            y="total_sales",
            color="store_location",
            text_auto=".2s",
            labels={"store_location": "Gerai", "total_sales": "Nilai penjualan"},
            color_discrete_sequence=COLORS,
        )
        fig_store.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_store, use_container_width=True, key="stores")
        st.caption("Hanya gerai yang dipilih muncul pada grafik.")

    with right:
        st.subheader("Kontribusi penjualan kategori")
        sales_by_category = (
            data.groupby("product_category", as_index=False)
            .agg(total_sales=("sales", "sum"))
            .sort_values("total_sales", ascending=False)
        )
        fig_category = px.pie(
            sales_by_category,
            names="product_category",
            values="total_sales",
            hole=0.55,
        )
        fig_category.update_traces(
            textinfo="none",
            hovertemplate=(
                "%{label}<br>Penjualan: %{value:,.2f}<br>"
                "Kontribusi: %{percent}<extra></extra>"
            ),
        )
        fig_category.update_layout(height=400, legend_title_text="Kategori")
        st.plotly_chart(fig_category, use_container_width=True, key="categories")
        st.caption("Kontribusi terhadap total dalam pilihan gerai yang sama.")


def render_product_and_time(data: pd.DataFrame) -> None:
    st.header("Produk dan pola waktu transaksi")
    left, right = st.columns(2)

    with left:
        st.subheader("10 produk dengan penjualan tertinggi")
        top_products = (
            data.groupby(["product_id", "product_detail"], as_index=False)
            .agg(total_sales=("sales", "sum"))
            .nlargest(10, "total_sales")
            .sort_values("total_sales", ascending=True)
        )
        top_products["label"] = (
            top_products["product_detail"]
            + " ("
            + top_products["product_id"].astype(str)
            + ")"
        )
        fig_products = px.bar(
            top_products,
            x="total_sales",
            y="label",
            orientation="h",
            color_discrete_sequence=["#7656B5"],
            labels={"total_sales": "Nilai penjualan", "label": "Produk"},
        )
        fig_products.update_layout(
            height=470,
            yaxis={
                "categoryorder": "array",
                "categoryarray": top_products["label"].tolist(),
            },
        )
        st.plotly_chart(fig_products, use_container_width=True, key="products")
        st.caption("Peringkat nilai penjualan, bukan unit atau keuntungan.")

    with right:
        st.subheader("Transaksi tercatat menurut jam")
        by_hour = (
            data.groupby("hour", as_index=False)
            .agg(total_transactions=("transaction_id", "nunique"))
            .sort_values("hour")
        )
        by_hour["label"] = (
            by_hour["hour"].astype(int).astype(str).str.zfill(2) + ":00"
        )
        fig_hour = px.bar(
            by_hour,
            x="label",
            y="total_transactions",
            color_discrete_sequence=["#D98B39"],
            labels={"label": "Jam", "total_transactions": "Transaksi tercatat"},
        )
        fig_hour.update_layout(height=470, xaxis_type="category")
        st.plotly_chart(fig_hour, use_container_width=True, key="hours")
        st.caption("Akumulasi sepanjang periode, belum mengukur antrean layanan.")


def main() -> None:
    st.set_page_config(page_title="Coffee Shop Sales", layout="wide")
    st.title("Dashboard Coffee Shop Sales")
    st.caption("Januari-Juni 2023. Mata uang belum dinyatakan pada file.")

    if not DATA_FILE.exists():
        st.error("Jalankan audit. File coffee_sales_prepared.csv belum ada.")
        st.stop()

    data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime_ns)
    st.sidebar.header("Filter data")
    stores = sorted(data["store_location"].dropna().unique().tolist())
    selected_stores = st.sidebar.multiselect(
        "Pilih gerai", stores, default=stores
    )
    if not selected_stores:
        st.info("Pilih minimal satu gerai.")
        st.stop()

    filtered = data[data["store_location"].isin(selected_stores)].copy()
    st.sidebar.caption("Semua grafik mengikuti pilihan gerai.")

    month_range = pd.period_range(
        data["transaction_date"].min().to_period("M"),
        data["transaction_date"].max().to_period("M"),
        freq="M",
    )
    months = month_range.astype(str).tolist()
    monthly = build_monthly_summary(filtered, months)

    render_monthly_summary(monthly, month_range[-1])
    st.divider()
    render_store_and_category(filtered)
    render_product_and_time(filtered)

    st.caption(
        "Transaksi tercatat adalah transaction_id unik, bukan pelanggan unik. "
        "Rata-rata nilainya belum dipastikan sebagai seluruh isi struk pelanggan."
    )


if __name__ == "__main__":
    main()
