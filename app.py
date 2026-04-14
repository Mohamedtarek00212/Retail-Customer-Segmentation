"""
Online Retail — Customer Segmentation Web App
=============================================
Run with:  streamlit run app.py

Dependencies:
    pip install streamlit pandas numpy scikit-learn plotly openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Segment definitions (label, colour, insight, recommendation)
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_INFO = {
    "💎 Champions": {
        "color": "#F4C542",
        "insight": (
            "These customers have purchased very recently, very often, and "
            "spend the most. They are your most profitable and loyal cohort. "
            "Losing even a handful is a material revenue event — their churn "
            "is usually silent and fast."
        ),
        "recommendations": [
            "🎁 **VIP Early Access** — invite them to product launches 48 h before the public.",
            "🤝 **Referral Programme** — 'Give £10 / Get £10'; their referrals convert at premium rates.",
            "🔁 **Predictive Replenishment** — auto-trigger reorder reminders based on purchase cadence.",
            "📞 **Dedicated Account Manager** (B2B) — deepen relationships and surface upsells organically.",
        ],
    },
    "🛒 Loyal Regulars": {
        "color": "#4C9BE8",
        "insight": (
            "Steady, predictable buyers with good order sizes. They are the most "
            "stable revenue cohort but can feel invisible without recognition — "
            "risking a slow drift into the At-Risk segment."
        ),
        "recommendations": [
            "🏆 **Loyalty Tier Programme** — a visible 'progress bar' to the next tier raises frequency 15–25%.",
            "📦 **Personalised Bundle Offers** — 2–3 items from their history at a 10% bundle discount.",
            "📝 **NPS Survey + £5 Credit** — 3 questions; regulars respond well and the data improves targeting.",
            "📅 **Seasonal Priority Stock** — guarantee availability on top SKUs during peak seasons.",
        ],
    },
    "🌱 New / Promising": {
        "color": "#5CB85C",
        "insight": (
            "Recent first or second-time buyers whose lifetime value is entirely "
            "uncaptured. Over-discounting at this stage trains price sensitivity — "
            "the first 60 days are the highest-leverage window in the customer lifecycle."
        ),
        "recommendations": [
            "✉️ **30-Day Email Onboarding** — Day 1 welcome, Day 7 top sellers, Day 20 personalised rec, Day 30 nudge.",
            "🔄 **Free Returns / Flexible Exchange** — reduce perceived risk; biggest predictor of a repeat visit.",
            "🛍️ **'Complete the Look' Cross-sell** — surface complementary items at order confirmation.",
            "❓ **New Customer Survey** — 14 days post-delivery: 'What almost stopped you from buying?'",
        ],
    },
    "😴 At-Risk / Hibernating": {
        "color": "#E8735A",
        "insight": (
            "Once-active customers who have gone quiet. Blanket win-back discounts "
            "are costly and can attract customers back *only* for the discount. "
            "Customers with >200 days recency and a single order are likely "
            "economically unviable to re-engage."
        ),
        "recommendations": [
            "📧 **Personalised Win-Back (No Discount First)** — 'Here's what's new' with 3 relevant product picks.",
            "💸 **Recency-Tiered Discount** — 60–90 days → 10% off; 90–180 days → 15%; >180 days → 20%.",
            "📱 **SMS Re-engagement** — a single well-timed SMS can outperform 3 emails for this cohort.",
            "🚫 **Sunset >200-Day Customers** — suppress from active campaigns to protect cost and sender rep.",
        ],
    },
}

SEGMENT_ORDER = list(SEGMENT_INFO.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers  (mirrors the notebook pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Online Retail transaction data."""
    df = df.copy()

    # ── 1. Invoice: keep only 6-digit numeric invoices (drop cancellations etc.)
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[df["Invoice"].str.match(r"^\d{6}$")]

    # ── 2. StockCode: keep product codes (5-digit, 5-digit+letters, or PADS*)
    df["StockCode"] = df["StockCode"].astype(str)
    mask = (
        df["StockCode"].str.match(r"^\d{5}$")
        | df["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$")
        | df["StockCode"].str.contains(r"^PADS", na=False)
    )
    df = df[mask]

    # ── 3. Drop rows with missing Customer ID
    df.dropna(subset=["Customer ID"], inplace=True)

    # ── 4. Remove non-positive prices and quantities
    df = df[df["Price"] > 0]
    df = df[df["Quantity"] > 0]

    # ── 5. Feature engineering
    df["total_sales"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["InvoiceDate"], inplace=True)

    return df


@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data into per-customer RFM metrics."""
    agg = df.groupby("Customer ID", as_index=False).agg(
        MonetaryValue=("total_sales", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max"),
    )
    max_date = agg["LastInvoiceDate"].max()
    agg["Recency"] = (max_date - agg["LastInvoiceDate"]).dt.days
    return agg


@st.cache_data(show_spinner=False)
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove IQR outliers on MonetaryValue and Frequency (same as notebook)."""
    def iqr_mask(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)

    mask = iqr_mask(df["MonetaryValue"]) & iqr_mask(df["Frequency"])
    return df[mask].copy()


@st.cache_data(show_spinner=False)
def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    """Scale RFM features and fit K-Means (k=4). Returns rfm with 'Segment' column."""
    features = ["MonetaryValue", "Frequency", "Recency"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # ── Assign business labels based on composite RFM score
    stats = rfm.groupby("Cluster")[features].mean()
    stats["RecencyScore"] = 1 - (stats["Recency"] / stats["Recency"].max())
    stats["FreqScore"]    = stats["Frequency"]     / stats["Frequency"].max()
    stats["MoneyScore"]   = stats["MonetaryValue"] / stats["MonetaryValue"].max()
    stats["Score"]        = (stats["RecencyScore"] + stats["FreqScore"] + stats["MoneyScore"]) / 3
    stats = stats.sort_values("Score", ascending=False)
    stats["Segment"] = SEGMENT_ORDER[: len(stats)]

    cluster_map = dict(zip(stats.index, stats["Segment"]))
    rfm["Segment"] = rfm["Cluster"].map(cluster_map)
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shopping-cart.png",
        width=64,
    )
    st.title("Customer Segmentation")
    st.caption("Online Retail · RFM + K-Means (k = 4)")
    st.divider()

    uploaded = st.file_uploader(
        "Upload transaction data",
        type=["csv", "xlsx", "xls"],
        help=(
            "Expected columns: Invoice, StockCode, Description, Quantity, "
            "InvoiceDate, Price, Customer ID"
        ),
    )

    st.divider()
    st.markdown("**Expected columns**")
    for col in ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID"]:
        st.markdown(f"- `{col}`")

    st.divider()
    st.caption("Preprocessing mirrors the Online Retail II notebook: "
               "6-digit invoice filter · StockCode regex · IQR outlier removal · StandardScaler")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🛍️ Customer Segmentation Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:grey;'>Upload your Online Retail transactions to "
    "automatically segment customers into behavioural archetypes.</p>",
    unsafe_allow_html=True,
)

if uploaded is None:
    # ── Landing state ──────────────────────────────────────────────────────
    st.info("👈  Upload a CSV or Excel file in the sidebar to get started.", icon="📁")

    col1, col2, col3, col4 = st.columns(4)
    for col, seg in zip([col1, col2, col3, col4], SEGMENT_ORDER):
        info = SEGMENT_INFO[seg]
        col.metric(label=seg, value="—", delta=None)

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
raw_df = None

if uploaded.name.lower().endswith(('.xlsx', '.xls')):
    try:
        # Load ExcelFile to get sheet names without reading the whole file into a DataFrame
        xls = pd.ExcelFile(uploaded)
        sheet_names = xls.sheet_names
        
        if len(sheet_names) > 1:
            st.info(f"📂 Multiple sheets found in **{uploaded.name}**")
            selected_sheet = st.selectbox("Select the sheet containing transaction data:", sheet_names)
        else:
            selected_sheet = sheet_names[0]
            
        with st.spinner(f"Reading sheet '{selected_sheet}' …"):
            raw_df = pd.read_excel(uploaded, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"❌ **Error reading Excel file:** {e}")
        st.stop()
else:
    with st.spinner("Reading CSV file …"):
        try:
            raw_df = pd.read_csv(uploaded, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            try:
                raw_df = pd.read_csv(uploaded, encoding="latin-1", low_memory=False)
            except Exception as e:
                st.error(f"❌ **Error reading CSV file:** {e}")
                st.stop()
        except Exception as e:
            st.error(f"❌ **Error reading CSV file:** {e}")
            st.stop()

# Validate required columns
required = {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"}
missing = required - set(raw_df.columns)
if missing:
    st.error(f"❌ **Invalid file structure.** Missing required columns:\n`{', '.join(missing)}`")
    st.info("💡 Please ensure your file has the exact column names listed in the sidebar.")
    st.stop()

with st.spinner("Cleaning data …"):
    clean_df = preprocess(raw_df)

with st.spinner("Computing RFM …"):
    rfm_df = compute_rfm(clean_df)
    rfm_no_outliers = remove_outliers(rfm_df)

with st.spinner("Running K-Means …"):
    segmented = segment_customers(rfm_no_outliers)

# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Raw Transactions", f"{len(raw_df):,}")
k2.metric("After Cleaning",   f"{len(clean_df):,}")
k3.metric("Unique Customers", f"{len(rfm_df):,}")
k4.metric("Segmented (no outliers)", f"{len(segmented):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Segment distribution
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Segment Distribution")

seg_counts = (
    segmented.groupby("Segment")
    .size()
    .reset_index(name="Customers")
    .assign(Pct=lambda d: (d["Customers"] / d["Customers"].sum() * 100).round(1))
)
# Keep SEGMENT_ORDER ordering
seg_counts["Segment"] = pd.Categorical(seg_counts["Segment"], categories=SEGMENT_ORDER, ordered=True)
seg_counts = seg_counts.sort_values("Segment")
colors = [SEGMENT_INFO[s]["color"] for s in seg_counts["Segment"]]

col_pie, col_bar = st.columns(2)

with col_pie:
    fig_pie = px.pie(
        seg_counts,
        names="Segment",
        values="Customers",
        color="Segment",
        color_discrete_map={s: SEGMENT_INFO[s]["color"] for s in SEGMENT_ORDER},
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    fig_bar = px.bar(
        seg_counts,
        x="Segment",
        y="Customers",
        color="Segment",
        color_discrete_map={s: SEGMENT_INFO[s]["color"] for s in SEGMENT_ORDER},
        text="Pct",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="# Customers",
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RFM summary table
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Segment RFM Summary")

summary = (
    segmented.groupby("Segment")
    .agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("MonetaryValue", "mean"),
        Total_Revenue=("MonetaryValue", "sum"),
    )
    .reset_index()
)
summary["Segment"] = pd.Categorical(summary["Segment"], categories=SEGMENT_ORDER, ordered=True)
summary = summary.sort_values("Segment")
summary.columns = [
    "Segment", "Customers",
    "Avg Recency (days)", "Avg Frequency",
    "Avg Monetary (£)", "Total Revenue (£)",
]

# Style it
styled = (
    summary.style
    .format({
        "Customers": "{:,}",
        "Avg Recency (days)": "{:.1f}",
        "Avg Frequency": "{:.1f}",
        "Avg Monetary (£)": "£{:,.0f}",
        "Total Revenue (£)": "£{:,.0f}",
    })
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
    ])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# RFM scatter (Recency vs. Monetary, size = Frequency)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔵 RFM Scatter — Recency vs. Monetary Value")
st.caption("Bubble size = Frequency  ·  Click legend items to toggle segments")

fig_scatter = px.scatter(
    segmented,
    x="Recency",
    y="MonetaryValue",
    size="Frequency",
    color="Segment",
    color_discrete_map={s: SEGMENT_INFO[s]["color"] for s in SEGMENT_ORDER},
    hover_data={"Customer ID": True, "Recency": True, "Frequency": True, "MonetaryValue": ":.0f"},
    opacity=0.65,
    size_max=24,
    labels={"MonetaryValue": "Monetary Value (£)", "Recency": "Recency (days)"},
)
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Insights + Recommendations (accordion)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Insights & Recommendations by Segment")

for seg in SEGMENT_ORDER:
    info = SEGMENT_INFO[seg]
    n = int(segmented[segmented["Segment"] == seg].shape[0])
    pct = n / len(segmented) * 100

    with st.expander(f"{seg}  ·  {n:,} customers  ({pct:.1f}%)", expanded=False):
        tab_insight, tab_reco = st.tabs(["💡 Insight", "📋 Recommendations"])

        with tab_insight:
            st.markdown(
                f"""
                <div style="background:{info['color']}22; border-left:4px solid {info['color']};
                            padding:12px 16px; border-radius:6px;">
                {info['insight']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_reco:
            for rec in info["recommendations"]:
                st.markdown(f"- {rec}")

# ─────────────────────────────────────────────────────────────────────────────
# Download segmented Excel
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⬇️ Download Segmented Data")

if 'segmented' not in locals() or segmented is None or segmented.empty:
    st.warning("⚠️ No segmented data available to download.")
else:
    import io
    import base64

    @st.cache_data(show_spinner=False)
    def make_download_link(df: pd.DataFrame, filename: str = "segmented_customers.xlsx") -> str:
        """
        Build a base64-encoded HTML <a> download link for Excel.

        Why not st.download_button?
        In sandboxed / iframe-based environments (e.g. Antigravity) the browser
        intercepts the blob URL that Streamlit generates and replaces the
        content-disposition filename with a random UUID.  A plain <a href="data:…">
        tag is handled entirely by the browser's own download machinery and
        therefore respects the `download` attribute filename correctly.
        """
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Segmented Customers")
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        href = (
            f'<a href="data:{mime};base64,{b64}" '
            f'download="{filename}" '
            f'style="display:inline-block; padding:0.45em 1.1em; '
            f'background:#1D6F42; color:white; border-radius:6px; '
            f'text-decoration:none; font-weight:600; font-size:0.95rem;">'
            f'📥 Download {filename}</a>'
        )
        return href

    download_df = segmented[
        ["Customer ID", "Recency", "Frequency", "MonetaryValue", "Segment"]
    ].copy()

    st.markdown(
        make_download_link(download_df, "segmented_customers.xlsx"),
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "K-Means clustering (k=4) · StandardScaler · "
    "Segment labels assigned dynamically via RFM composite score."
)
