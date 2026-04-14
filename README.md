# 🛍️ Customer Segmentation Dashboard
### *Turn raw transactions into revenue strategy — in seconds.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🤔 The Problem

You have **thousands of customers** and one budget. Who gets the loyalty reward? Who needs a win-back campaign? Who's about to churn without a single warning?

Most businesses answer these questions with gut instinct. This project answers them with **data**.

---

## 💡 The Solution

An end-to-end machine learning pipeline that ingests raw e-commerce transactions, engineers **RFM features** (Recency, Frequency, Monetary Value), and clusters customers into **4 actionable business archetypes** — all wrapped in a beautiful, interactive Streamlit dashboard.

Upload your data. Get your segments. Download your results. Done.

---

## ✨ What You Get

| Segment | Who They Are | What To Do |
|---|---|---|
| 💎 **Champions** | High spenders, buy often, bought recently | Reward them. They're your revenue engine. |
| 🛒 **Loyal Regulars** | Steady, predictable, reliable | Keep them visible. Prevent silent drift. |
| 🌱 **New / Promising** | Recent first-timers with untapped potential | Nurture fast — the first 60 days are everything. |
| 😴 **At-Risk / Hibernating** | Once active, now quiet | Win them back before they're gone for good. |

> **Why does this matter?** A Champions cohort that represents 10% of customers can easily drive 40–50% of revenue. Identifying them is the difference between a generic newsletter and a targeted strategy.

---

## 🎬 Live Demo

```
Upload Excel/CSV  →  Instant Cleaning  →  RFM Computation  →  K-Means Clustering  →  Interactive Charts  →  Download Results
```

The dashboard features:
- 📊 **Donut + Bar charts** — visual segment distribution at a glance
- 🔵 **RFM Bubble Scatter** — Recency vs. Monetary Value with Frequency as bubble size
- 📋 **Segment Summary Table** — avg. recency, frequency, spend, and total revenue per segment
- 🔍 **Insights & Recommendations** — per-segment accordion with business intelligence baked in
- 📥 **One-click Export** — download your segmented customers as a clean `.xlsx` file

---

## 🏗️ Project Structure

```
📦 Retail-Customer-Segmentation/
├── 🐍 app.py                   # Full EDA + modelling notebook
├── 📓 online_retail.ipynb      # Streamlit web application
├── 📊 online_retail_II.xlsx    # Source dataset (UCI Online Retail II)
└── 📄 requirements.txt         # Python dependencies
```

---

## ⚙️ How It Works

### 1. Data Cleaning
Raw transactions are filtered aggressively:
- Only valid 6-digit invoice numbers (cancellations dropped)
- StockCodes validated via regex (product codes only)
- Null Customer IDs and non-positive prices/quantities removed

### 2. RFM Feature Engineering
Each customer is reduced to three numbers:
- **Recency** — days since last purchase (lower = better)
- **Frequency** — number of unique invoices
- **Monetary Value** — total spend (£)

### 3. Outlier Removal
IQR-based filtering on Monetary Value and Frequency ensures extreme power users don't distort the cluster centroids.

### 4. K-Means Clustering (k=4)
Features are standardised with `StandardScaler` before fitting. The optimal `k=4` was selected via Elbow Method + Silhouette Score analysis (see notebook).

### 5. Automatic Business Labelling
Clusters are ranked by a composite RFM score and mapped to human-readable business segment names — no manual label assignment required.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mohamedtarek00212/Retail-Customer-Segmentation.git
cd Retail-Customer-Segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Using the App

1. **Upload** your transaction file (`.xlsx` or `.csv`) via the sidebar
2. **Select** the sheet containing your data (if multi-sheet Excel)
3. **Watch** the pipeline run automatically
4. **Explore** the charts and insights
5. **Download** `segmented_customers.xlsx` with one click

### Expected Data Format

Your file must contain these columns:

| Column | Description |
|---|---|
| `Invoice` | Transaction ID |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units purchased |
| `InvoiceDate` | Date & time of transaction |
| `Price` | Unit price (£) |
| `Customer ID` | Unique customer identifier |

---

## 📓 Notebook Walkthrough

The Jupyter notebook (`online_retail.ipynb`) is the full research story behind the app:

- ✅ Exploratory Data Analysis with distribution plots and box plots
- ✅ Data quality audit (nulls, cancellations, invalid codes)
- ✅ 3D scatter visualisation of RFM space (before and after scaling)
- ✅ Elbow Method + Silhouette Score to justify k=4
- ✅ Violin plots of RFM distributions per cluster
- ✅ Deep-dive insights and actionable recommendations per segment
- ✅ Production readiness notes (model persistence, data pipeline, monitoring)

---

## 🧰 Tech Stack

| Tool | Role |
|---|---|
| `pandas` | Data wrangling and aggregation |
| `scikit-learn` | StandardScaler + KMeans |
| `plotly` | Interactive charts |
| `streamlit` | Web application framework |
| `openpyxl` | Excel file I/O |

---

## 📊 Dataset

**UCI Online Retail II Dataset**
- Transactions from a UK-based online retailer
- Period: December 2009 – December 2011
- ~1 million rows across 2 yearly sheets
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

---

## 🗺️ Roadmap

- [ ] Add a date-range filter to analyse segment evolution over time
- [ ] Persist the trained model with `joblib` for instant predictions
- [ ] Add cohort retention analysis
- [ ] Deploy to Streamlit Community Cloud with one-click demo

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT — do whatever you want with it, just don't blame us if your marketing budget disappears into the Champions segment.

---

<p align="center">
  Built with ❤️ using Streamlit · scikit-learn · Plotly
</p>
