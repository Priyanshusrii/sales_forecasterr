import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sales Forecaster", page_icon="📈", layout="centered")

dataset_info = {
    "1":  {"name": "Monthly Revenue",    "csv": "monthly_revenue.csv",    "x_col": "month",            "y_col": "total_revenue",       "input_label": "Month Number (1-12)",  "data_type": "Linear, Clean"},
    "2":  {"name": "Marketing Spend",    "csv": "marketing_spend.csv",    "x_col": "ad budget",        "y_col": "units sold",          "input_label": "Ad Budget (₹)",        "data_type": "Linear, Messy"},
    "3":  {"name": "Product Decay",      "csv": "product_decay.csv",      "x_col": "product_age",      "y_col": "daily_sales_count",   "input_label": "Product Age (days)",   "data_type": "Exponential, Clean"},
    "4":  {"name": "Store Visitors",     "csv": "store_visitors.csv",     "x_col": "number_ofvisitor", "y_col": "total_daily_billing", "input_label": "Number of Visitors",   "data_type": "Linear, Clean"},
    "5":  {"name": "Subscription Users", "csv": "subscription_users.csv", "x_col": "week_number",      "y_col": "active_subscribers",  "input_label": "Week Number",          "data_type": "Logarithmic, Messy"},
    "6":  {"name": "Price Impact",       "csv": "price_impact.csv",       "x_col": "Unit_Price",       "y_col": "Units_Sold",          "input_label": "Unit Price (₹)",       "data_type": "Polynomial, Clean"},
    "7":  {"name": "Inventory Waste",    "csv": "inventory_waste.csv",    "x_col": "Items_Produced",   "y_col": "Amount_of_Waste",     "input_label": "Items Produced",       "data_type": "Linear, Clean"},
    "8":  {"name": "App Downloads",      "csv": "app_downloads.csv",      "x_col": "Ad_Impressions",   "y_col": "App_Installs",        "input_label": "Ad Impressions",       "data_type": "Square Root, Messy"},
    "9":  {"name": "Holiday Bonus",      "csv": "holiday_bonus.csv",      "x_col": "Company_Profit",   "y_col": "Bonus_Amount",        "input_label": "Company Profit (₹)",   "data_type": "Linear, Very Clean"},
    "10": {"name": "Festival Rush",      "csv": "festival_rush.csv",      "x_col": "Days_to_Festival", "y_col": "Sales_Volume",        "input_label": "Days to Festival",     "data_type": "Exponential, Messy"},
}

models = {
    "Linear Regression": "LR",
    "Decision Tree":     "DT",
    "Random Forest":     "RF"
}

# ---- Sidebar ----
st.sidebar.title("⚙️ Settings")
dataset_id = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_info.keys()),
    format_func=lambda x: f"{x}. {dataset_info[x]['name']}"
)
model_type = st.sidebar.radio("Select algorithm", list(models.keys()))
model_key = models[model_type]

info = dataset_info[dataset_id]

# ---- Load Data ----
df = pd.read_csv(f"datasets/{info['csv']}")
X = df[[info["x_col"]]].values
y = df[info["y_col"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Main ----
st.title("📈 Sales Forecaster")
st.caption(f"Dataset: **{info['name']}** | Model: **{model_type}**")
st.divider()

# ---- Prediction ----
st.subheader("Make a Prediction")
input_value = st.number_input(info["input_label"], step=1.0)

if st.button("Predict", use_container_width=True):
    model_path = Path(f"models/dataset_{dataset_id}_{model_key}.pkl")
    if not model_path.exists():
        st.error("Model not found. Run train.py first.")
    else:
        model = joblib.load(model_path)
        prediction = model.predict(np.array([[input_value]]))[0]
        st.success(f"Predicted **{info['y_col'].replace('_', ' ').title()}**: `{round(prediction, 2):,}`")

st.divider()

# ---- Actual vs Predicted Chart ----
st.subheader("Actual vs Predicted")
model_path = Path(f"models/dataset_{dataset_id}_{model_key}.pkl")
if model_path.exists():
    model = joblib.load(model_path)
    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    predicted_line = model.predict(x_range)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, color="#0400FF", alpha=0.2, s=10, label="Actual Data")
    ax.plot(x_range, predicted_line, color="#e11d48", linewidth=2, label="Predicted")
    ax.set_xlabel(info["x_col"].replace("_", " ").title())
    ax.set_ylabel(info["y_col"].replace("_", " ").title())
    ax.set_title(f"{info['name']} — {model_type}")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.divider()

# ---- Model Comparison ----
st.subheader("Model Comparison")

r2_scores = {}
mae_scores = {}
best_model = None
best_r2 = -999

for mname, mkey in models.items():
    path = Path(f"models/dataset_{dataset_id}_{mkey}.pkl")
    if path.exists():
        m = joblib.load(path)
        r2  = m.score(X_test, y_test) * 100
        mae = mean_absolute_error(y_test, m.predict(X_test))
        r2_scores[mname]  = round(r2, 2)
        mae_scores[mname] = round(mae, 2)
        if r2 > best_r2:
            best_r2 = r2
            best_model = mname

# ---- Best Model Badge ----
st.success(f"🏆 Best algorithm for **{info['name']}**: **{best_model}** — R²: {round(best_r2, 2)}%")

# ---- Metrics ----
col1, col2, col3 = st.columns(3)
for col, mname in zip([col1, col2, col3], models.keys()):
    is_best = mname == best_model
    col.metric(
        label=f"{'🥇 ' if is_best else ''}{mname}",
        value=f"{r2_scores[mname]}%",
        delta=f"MAE: {mae_scores[mname]}"
    )

