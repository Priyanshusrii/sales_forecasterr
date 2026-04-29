import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path

Path("models").mkdir(exist_ok=True)
Path("graphs").mkdir(exist_ok=True)

datasets = {
    "1":  ["monthly_revenue.csv",    "month",             "total_revenue"],
    "2":  ["marketing_spend.csv",    "ad budget",         "units sold"],
    "3":  ["product_decay.csv",      "product_age",       "daily_sales_count"],
    "4":  ["store_visitors.csv",     "number_ofvisitor",  "total_daily_billing"],
    "5":  ["subscription_users.csv", "week_number",       "active_subscribers"],
    "6":  ["price_impact.csv",       "Unit_Price",        "Units_Sold"],
    "7":  ["inventory_waste.csv",    "Items_Produced",    "Amount_of_Waste"],
    "8":  ["app_downloads.csv",      "Ad_Impressions",    "App_Installs"],
    "9":  ["holiday_bonus.csv",      "Company_Profit",    "Bonus_Amount"],
    "10": ["festival_rush.csv",      "Days_to_Festival",  "Sales_Volume"],
}

dataset_names = {
    "1": "Monthly Revenue",     "2": "Marketing Spend",
    "3": "Product Decay",       "4": "Store Visitors",
    "5": "Subscription Users",  "6": "Price Impact",
    "7": "Inventory Waste",     "8": "App Downloads",
    "9": "Holiday Bonus",       "10": "Festival Rush",
}

print("--- TRAINING MODELS ---")

for i, info in datasets.items():
    name, x_col, y_col = info
    df = pd.read_csv(f"datasets/{name}")
    X = df[[x_col]].values
    y = df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Linear Regression ──
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, f"models/dataset_{i}_LR.pkl")

    # ── Decision Tree ──
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    joblib.dump(dt, f"models/dataset_{i}_DT.pkl")

    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"models/dataset_{i}_RF.pkl")

    # ── Scores ──
    lr_r2  = lr.score(X_test, y_test) * 100
    dt_r2  = dt.score(X_test, y_test) * 100
    rf_r2  = rf.score(X_test, y_test) * 100

    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    dt_mae = mean_absolute_error(y_test, dt.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

    print(f"Success: Dataset {i} trained.")
    print(f"   > Linear Regression  — R²: {round(lr_r2, 2)}%  | MAE: {round(lr_mae, 2)}")
    print(f"   > Decision Tree      — R²: {round(dt_r2, 2)}%  | MAE: {round(dt_mae, 2)}")
    print(f"   > Random Forest      — R²: {round(rf_r2, 2)}%  | MAE: {round(rf_mae, 2)}")
    print("-" * 20)

    # ── Graph ──
    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    lr_line = lr.predict(x_range)
    dt_line = dt.predict(x_range)
    rf_line = rf.predict(x_range)

    plt.figure(figsize=(9, 5))
    plt.scatter(X, y, color="#0400FF", alpha=0.2, s=10, label="Actual Data")
    plt.plot(x_range, lr_line, color="#e11d48", linewidth=2, label=f"Linear (R²: {round(lr_r2, 1)}%)")
    plt.plot(x_range, dt_line, color="#f59e0b", linewidth=2, linestyle=":", label=f"Decision Tree (R²: {round(dt_r2, 1)}%)")
    plt.plot(x_range, rf_line, color="#16a34a", linewidth=2, linestyle="--", label=f"Random Forest (R²: {round(rf_r2, 1)}%)")

    plt.title(f"Dataset {i} — {dataset_names[i]}", fontsize=14, fontweight="bold")
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=11)
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"graphs/dataset_{i}.png", dpi=150)
    plt.close()
    print(f"   > Graph saved: graphs/dataset_{i}.png")

print("\nTraining complete, models and graphs saved!")