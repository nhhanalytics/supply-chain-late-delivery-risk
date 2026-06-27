
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Late Delivery Risk Dashboard",
    layout="wide"
)

st.title("Supply Chain Late Delivery Risk Dashboard")

st.write(
    "This dashboard supports supply chain stakeholders in monitoring late delivery risk "
    "using the original DataCo supply chain dataset and the best model from Q2."
)

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "DataCoSupplyChainDataset.csv"
model_path = BASE_DIR / "models" / "hgb_model.pkl"
features_path = BASE_DIR / "models" / "model_features.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(data_path, encoding="latin1")

    object_columns = df.select_dtypes(include="object").columns
    for col in object_columns:
        df[col] = df[col].astype(str).str.strip()

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce"
    )

    df["order_month"] = df["order date (DateOrders)"].dt.month
    df["order_dayofweek"] = df["order date (DateOrders)"].dt.dayofweek
    df["is_weekend_order"] = df["order_dayofweek"].isin([5, 6]).astype(int)

    return df

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
    return model, model_features

df = load_data()
model, model_features = load_model()

target = "Late_delivery_risk"

categorical_features = [
    "Type",
    "Customer Segment",
    "Market",
    "Order Region",
    "Category Name",
    "Shipping Mode"
]

selected_features = [
    "Type",
    "Customer Segment",
    "Market",
    "Order Region",
    "Category Name",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Total",
    "Order Item Profit Ratio",
    "order_month",
    "order_dayofweek",
    "is_weekend_order"
]

# Sidebar filters
st.sidebar.header("Filter Options")

market_filter = st.sidebar.multiselect(
    "Select Market",
    options=sorted(df["Market"].dropna().unique()),
    default=sorted(df["Market"].dropna().unique())
)

shipping_filter = st.sidebar.multiselect(
    "Select Shipping Mode",
    options=sorted(df["Shipping Mode"].dropna().unique()),
    default=sorted(df["Shipping Mode"].dropna().unique())
)

filtered_df = df[
    (df["Market"].isin(market_filter)) &
    (df["Shipping Mode"].isin(shipping_filter))
].copy()

# Summary metrics
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

total_orders = len(filtered_df)
late_risk_rate = filtered_df[target].mean() * 100 if total_orders > 0 else 0
avg_scheduled_days = filtered_df["Days for shipment (scheduled)"].mean() if total_orders > 0 else 0

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Late Delivery Risk Rate", f"{late_risk_rate:.2f}%")
col3.metric("Average Scheduled Shipping Days", f"{avg_scheduled_days:.2f}")

# Visualization 1
st.subheader("Visualization 1: Late Delivery Risk Rate by Shipping Mode")

if not filtered_df.empty:
    risk_by_shipping = (
        filtered_df.groupby("Shipping Mode")[target]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    st.bar_chart(risk_by_shipping)
else:
    st.warning("No data available for the selected filters.")

# Visualization 2
st.subheader("Visualization 2: Late Delivery Risk Rate by Market")

if not filtered_df.empty:
    risk_by_market = (
        filtered_df.groupby("Market")[target]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    st.bar_chart(risk_by_market)

# Visualization 3
st.subheader("Visualization 3: Late Delivery Risk Distribution")

if not filtered_df.empty:
    risk_distribution = (
        filtered_df[target]
        .map({0: "No Late Risk", 1: "Late Risk"})
        .value_counts()
    )
    st.bar_chart(risk_distribution)

# Predictive output
st.subheader("Predictive Output: Late Delivery Risk Prediction")

if not filtered_df.empty:
    row_number = st.number_input(
        "Select an order row from the filtered data",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0
    )

    selected_order = filtered_df.iloc[[row_number]]

    st.write("Selected Order Features")
    st.dataframe(selected_order[selected_features])

    model_input = selected_order[selected_features].copy()

    model_input = pd.get_dummies(
        model_input,
        columns=categorical_features,
        drop_first=True
    )

    model_input = model_input.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]

    prediction_label = "Late Risk" if prediction == 1 else "No Late Risk"

    col4, col5 = st.columns(2)
    col4.metric("Predicted Late Delivery Risk", prediction_label)
    col5.metric("Predicted Late Risk Probability", f"{probability * 100:.2f}%")

# Data preview
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100))
