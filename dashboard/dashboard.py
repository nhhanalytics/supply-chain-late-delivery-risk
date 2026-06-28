
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

market_options = sorted(df["Market"].dropna().unique().tolist())
shipping_options = sorted(df["Shipping Mode"].dropna().unique().tolist())
region_options = sorted(df["Order Region"].dropna().unique().tolist())
category_options = sorted(df["Category Name"].dropna().unique().tolist())

market_filter = st.sidebar.multiselect(
    "Select Market",
    options=market_options,
    default=market_options
)

shipping_filter = st.sidebar.multiselect(
    "Select Shipping Mode",
    options=shipping_options,
    default=shipping_options
)

region_filter = st.sidebar.multiselect(
    "Select Order Region",
    options=region_options,
    default=region_options
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=category_options,
    default=category_options
)

scheduled_min = int(df["Days for shipment (scheduled)"].min())
scheduled_max = int(df["Days for shipment (scheduled)"].max())

scheduled_range = st.sidebar.slider(
    "Select Scheduled Shipping Days",
    min_value=scheduled_min,
    max_value=scheduled_max,
    value=(scheduled_min, scheduled_max)
)

late_risk_only = st.sidebar.checkbox(
    "Show only late-risk orders",
    value=False
)

show_top_segments = st.sidebar.checkbox(
    "Show top 10 high-risk segments",
    value=True
)

show_data_preview = st.sidebar.checkbox(
    "Show filtered data preview",
    value=True
)

# Apply filters
filtered_df = df[
    (df["Market"].isin(market_filter)) &
    (df["Shipping Mode"].isin(shipping_filter)) &
    (df["Order Region"].isin(region_filter)) &
    (df["Category Name"].isin(category_filter)) &
    (df["Days for shipment (scheduled)"].between(
        scheduled_range[0],
        scheduled_range[1]
    ))
].copy()

if late_risk_only:
    filtered_df = filtered_df[filtered_df[target] == 1].copy()

# Summary metrics
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

total_orders = len(filtered_df)
late_risk_rate = filtered_df[target].mean() * 100 if total_orders > 0 else 0
if total_orders > 0:
    avg_scheduled_days = filtered_df["Days for shipment (scheduled)"].mean()
    avg_actual_days = filtered_df["Days for shipping (real)"].mean()
    avg_delay_days = (
        filtered_df["Days for shipping (real)"] -
        filtered_df["Days for shipment (scheduled)"]
    ).clip(lower=0).mean()
else:
    avg_scheduled_days = 0
    avg_actual_days = 0
    avg_delay_days = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Late Delivery Risk Rate", f"{late_risk_rate:.2f}%")
col3.metric("Average Scheduled Days", f"{avg_scheduled_days:.2f}")
col4.metric("Average Delay Days", f"{avg_delay_days:.2f}")

# Monitoring metrics
st.subheader("Monitoring Metrics")

if total_orders > 0:
    missing_value_rate = (
        filtered_df.isnull().sum().sum() /
        (filtered_df.shape[0] * filtered_df.shape[1])
    ) * 100

    duplicate_records = filtered_df.duplicated().sum()

    col_m1, col_m2, col_m3 = st.columns(3)

    col_m1.metric(
        "Business Metric: Late Delivery Risk Rate",
        f"{late_risk_rate:.2f}%"
    )

    col_m2.metric(
        "Data Quality Metric: Missing Value Rate",
        f"{missing_value_rate:.2f}%"
    )

    col_m3.metric(
        "Data Quality Metric: Duplicate Records",
        f"{duplicate_records:,}"
    )

    if missing_value_rate > 5:
        st.warning(
            "Data quality warning: missing value rate is above 5%. "
            "The data should be reviewed before further analysis."
        )
    else:
        st.success(
            "Data quality status: missing value rate is within an acceptable level."
        )

else:
    st.info("Monitoring metrics are not available because no data matches the selected filters.")

# Analytical risk alert
st.subheader("Analytical Output: Delivery Risk Alert")

if total_orders > 0:
    if late_risk_rate >= 60:
        st.error(
            f"High late delivery risk detected. The current filtered data shows a "
            f"late delivery risk rate of {late_risk_rate:.2f}%."
        )
    elif late_risk_rate >= 40:
        st.warning(
            f"Moderate late delivery risk detected. The current filtered data shows a "
            f"late delivery risk rate of {late_risk_rate:.2f}%."
        )
    else:
        st.success(
            f"Lower late delivery risk detected. The current filtered data shows a "
            f"late delivery risk rate of {late_risk_rate:.2f}%."
        )
else:
    st.info("No data available for the selected filters.")

# Interactive visualization selector
st.subheader("Interactive Visualization")

chart_option = st.selectbox(
    "Select visualization to display",
    [
        "Late Delivery Risk Rate by Shipping Mode",
        "Late Delivery Risk Rate by Market",
        "Late Delivery Risk Distribution",
        "Late Delivery Risk Rate by Order Region",
        "Late Delivery Risk Rate by Category"
    ]
)

if filtered_df.empty:
    st.warning("No chart available because no data matches the selected filters.")

else:
    if chart_option == "Late Delivery Risk Rate by Shipping Mode":
        chart_data = (
            filtered_df.groupby("Shipping Mode")[target]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        st.bar_chart(chart_data)

    elif chart_option == "Late Delivery Risk Rate by Market":
        chart_data = (
            filtered_df.groupby("Market")[target]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        st.bar_chart(chart_data)

    elif chart_option == "Late Delivery Risk Distribution":
        chart_data = (
            filtered_df[target]
            .map({0: "No Late Risk", 1: "Late Risk"})
            .value_counts()
        )

        st.bar_chart(chart_data)

    elif chart_option == "Late Delivery Risk Rate by Order Region":
        chart_data = (
            filtered_df.groupby("Order Region")[target]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(chart_data)

    elif chart_option == "Late Delivery Risk Rate by Category":
        chart_data = (
            filtered_df.groupby("Category Name")[target]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(chart_data)

# Top 10 high-risk segments
if show_top_segments:
    st.subheader("Top 10 High-Risk Shipping Segments")

    if not filtered_df.empty:
        high_risk_segments = (
            filtered_df.groupby(["Market", "Shipping Mode", "Order Region"])[target]
            .agg(["count", "mean"])
            .reset_index()
        )

        high_risk_segments["Late Delivery Risk Rate (%)"] = (
            high_risk_segments["mean"] * 100
        ).round(2)

        high_risk_segments = high_risk_segments.rename(
            columns={"count": "Number of Orders"}
        )

        high_risk_segments = high_risk_segments[
            [
                "Market",
                "Shipping Mode",
                "Order Region",
                "Number of Orders",
                "Late Delivery Risk Rate (%)"
            ]
        ].sort_values(
            by="Late Delivery Risk Rate (%)",
            ascending=False
        ).head(10)

        st.dataframe(high_risk_segments, use_container_width=True)

    else:
        st.info("No segment analysis available for the selected filters.")

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
    st.dataframe(selected_order[selected_features], use_container_width=True)

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

else:
    st.info("Prediction is not available because no data matches the selected filters.")

# Download filtered data
st.subheader("Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_late_delivery_data.csv",
    mime="text/csv"
)

# Data preview
if show_data_preview:
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)
