import os
import joblib # type: ignore
import pandas as pd
import streamlit as st
import plotly.express as px # type: ignore

# -----------------------------
# BASE DIRECTORY SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(DATA_DIR, "Bank Customer Churn Prediction.csv")
PRED_PATH = os.path.join(ARTIFACTS_DIR, "1_Predictions.csv")
FI_PATH = os.path.join(ARTIFACTS_DIR, "2_Feature_Importance.csv")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "churn_pipeline.pkl")

@st.cache_data
def load_input_schema():
    schema_path = os.path.join(ARTIFACTS_DIR, "input_schema.json")
    if not os.path.exists(schema_path):
        return None

    import json
    with open(schema_path, "r") as f:
        return json.load(f)

# -----------------------------
# CACHED PIPELINE LOADER
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipeline_path = os.path.join(ARTIFACTS_DIR, "churn_pipeline.pkl")
    return joblib.load(pipeline_path)



# -----------------------------
# CACHED DATA LOADER
# -----------------------------
@st.cache_data
def load_main_data():
    return pd.read_csv(DATA_PATH)

# -----------------------------
# CACHED METRICS LOADER
# -----------------------------
@st.cache_data
def load_model_metrics():
    metrics_path = os.path.join(ARTIFACTS_DIR, "model_metrics.json")

    if not os.path.exists(metrics_path):
        return None

    import json
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return pd.DataFrame(metrics)

# -----------------------------
# CACHED FEATURE IMPORTANCE LOADER
# -----------------------------
@st.cache_data
def load_feature_importance():
    if not os.path.exists(FI_PATH):
        return None
    return pd.read_csv(FI_PATH)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    layout="wide"
)

st.title("Bank Customer Churn Dashboard")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Risk Segmentation", "Model Insights", "Upload & Predict"]
)

# -----------------------------
# LOAD DATA (CACHED & PORTABLE)
# -----------------------------
df = load_main_data()

# =====================================================
# ===================== OVERVIEW ======================
# =====================================================
# =====================================================
# ===================== OVERVIEW ======================
# =====================================================
if section == "Overview":

    # -----------------------------
    # KPI SECTION
    # -----------------------------
    total_customers = len(df)
    churned_customers = df["churn"].sum()
    churn_rate = (churned_customers / total_customers) * 100

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("**Total Customers**")
        st.markdown(f"<h2><b>{total_customers}</b></h2>", unsafe_allow_html=True)

    with k2:
        st.markdown("**Churned Customers**")
        st.markdown(f"<h2><b>{churned_customers}</b></h2>", unsafe_allow_html=True)

    with k3:
        st.markdown("**Churn Rate (%)**")
        st.markdown(f"<h2><b>{churn_rate:.2f}</b></h2>", unsafe_allow_html=True)

    st.divider()

    # -----------------------------
    # CHURN DISTRIBUTION
    # -----------------------------
    st.subheader("Customer Churn Distribution")

    churn_dist = df["churn"].value_counts().reset_index()
    churn_dist.columns = ["Churn Status", "Count"]
    churn_dist["Churn Status"] = churn_dist["Churn Status"].map(
        {0: "Retained", 1: "Churned"}
    )

    fig = px.pie(
        churn_dist,
        names="Churn Status",
        values="Count",
        color="Churn Status",
        color_discrete_map={
            "Churned": "#e74c3c",
            "Retained": "#2ecc71"
        },
        title="Overall Customer Churn"
    )

    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # DEMOGRAPHIC ANALYSIS
    # -----------------------------
    st.subheader("Demographic Analysis")

    col1, col2 = st.columns(2)

    # -------- Gender --------
    with col1:
        fig = px.histogram(
            df,
            x="gender",
            color="gender",
            barmode="group",
            color_discrete_map={
                "Male": "#3498db",
                "Female": "#ff69b4"
            },
            title="Churn Distribution by Gender"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- Country (DYNAMIC) --------
    with col2:
        # Take top N countries, rest -> Others
        TOP_N = 5
        top_countries = df["country"].value_counts().nlargest(TOP_N).index

        country_df = df.copy()
        country_df["country_grouped"] = country_df["country"].where(
            country_df["country"].isin(top_countries),
            other="Others"
        )

        fig = px.histogram(
            country_df,
            x="country_grouped",
            color="churn",
            barmode="group",
            color_discrete_map={
                1: "#e74c3c",
                0: "#2ecc71"
            },
            title="Churn Distribution by Country (Top Countries + Others)"
        )

        fig.update_layout(xaxis_title="Country")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # BEHAVIORAL ANALYSIS
    # -----------------------------
    st.subheader("Behavioral Analysis")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.plotly_chart(
            px.box(df, x="churn", y="age", title="Customer Age vs Churn"),
            use_container_width=True
        )

    with b2:
        st.plotly_chart(
            px.box(df, x="churn", y="tenure", title="Customer Tenure vs Churn"),
            use_container_width=True
        )

    with b3:
        st.plotly_chart(
            px.box(df, x="churn", y="balance", title="Account Balance vs Churn"),
            use_container_width=True
        )

# =====================================================
# ================= RISK SEGMENTATION =================
# =====================================================
elif section == "Risk Segmentation":

    st.subheader("Customer Risk Segmentation")

    # ---------------------------------
    # LOAD PIPELINE (CACHED)
    # ---------------------------------
    pipeline = load_pipeline()

    # ---------------------------------
    # PREPARE DATA (FULL DATASET)
    # ---------------------------------
    data_df = df.copy()

    if "churn" in data_df.columns:
        data_df = data_df.drop(columns=["churn"])

    # ---------------------------------
    # GENERATE PREDICTIONS
    # ---------------------------------
    try:
        churn_probs = pipeline.predict_proba(data_df)[:, 1]
    except Exception as e:
        st.error("Error generating churn probabilities.")
        st.code(str(e))
        st.stop()

    data_df["Churn_Probability"] = churn_probs

    # ---------------------------------
    # DYNAMIC COUNTRY GROUPING
    # ---------------------------------
    TOP_N = 5
    top_countries = (
        data_df["country"]
        .value_counts()
        .nlargest(TOP_N)
        .index
    )

    data_df["country_grouped"] = data_df["country"].where(
        data_df["country"].isin(top_countries),
        other="Others"
    )

    # ---------------------------------
    # FILTERS
    # ---------------------------------
    st.markdown("### Filters")

    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox(
            "Country",
            ["All"] + sorted(data_df["country_grouped"].unique().tolist())
        )

    with col2:
        selected_gender = st.selectbox(
            "Gender",
            ["All"] + sorted(data_df["gender"].dropna().unique().tolist())
        )

    filtered_df = data_df.copy()

    if selected_country != "All":
        filtered_df = filtered_df[
            filtered_df["country_grouped"] == selected_country
        ]

    if selected_gender != "All":
        filtered_df = filtered_df[
            filtered_df["gender"] == selected_gender
        ]

    # ---------------------------------
    # RISK THRESHOLD CONFIGURATION
    # ---------------------------------
    st.markdown("### Risk Threshold Configuration")

    low_threshold = st.slider(
        "Low Risk Upper Bound",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

    medium_threshold = st.slider(
        "Medium Risk Upper Bound",
        min_value=low_threshold,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    # ---------------------------------
    # ASSIGN RISK LEVELS
    # ---------------------------------
    filtered_df["Risk_Level"] = filtered_df["Churn_Probability"].apply(
        lambda p:
            "Low Risk" if p < low_threshold
            else "Medium Risk" if p < medium_threshold
            else "High Risk"
    )

    # ---------------------------------
    # RISK KPIs
    # ---------------------------------
    total_customers = len(filtered_df)
    high_risk_count = (filtered_df["Risk_Level"] == "High Risk").sum()
    high_risk_pct = (high_risk_count / total_customers * 100) if total_customers else 0

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("**Customers Analyzed**")
        st.markdown(f"<h3><b>{total_customers}</b></h3>", unsafe_allow_html=True)

    with k2:
        st.markdown("**High-Risk Customers**")
        st.markdown(f"<h3><b>{high_risk_count}</b></h3>", unsafe_allow_html=True)

    with k3:
        st.markdown("**High-Risk Percentage**")
        st.markdown(f"<h3><b>{high_risk_pct:.2f}%</b></h3>", unsafe_allow_html=True)

    st.divider()

    # ---------------------------------
    # RISK DISTRIBUTION
    # ---------------------------------
    fig = px.histogram(
        filtered_df,
        x="Risk_Level",
        color="Risk_Level",
        color_discrete_map={
            "Low Risk": "#2ecc71",
            "Medium Risk": "#f1c40f",
            "High Risk": "#e74c3c"
        },
        title="Customer Churn Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # COUNTRY-WISE HIGH RISK COUNT
    # ---------------------------------
    st.divider()
    st.subheader("Country-wise High-Risk Customers")

    country_risk_count = (
        filtered_df[filtered_df["Risk_Level"] == "High Risk"]
        .groupby("country_grouped")
        .size()
        .reset_index(name="High Risk Customers")
    )

    if not country_risk_count.empty:
        fig = px.bar(
            country_risk_count,
            x="country_grouped",
            y="High Risk Customers",
            color="High Risk Customers",
            color_continuous_scale=["#f1c40f", "#e74c3c"],
            title="High-Risk Customers by Country"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-risk customers found.")

    # ---------------------------------
    # HIGH-RISK CUSTOMER SAMPLE
    # ---------------------------------
    st.divider()
    st.subheader("High-Risk Customer Sample")

    high_risk_df = filtered_df[filtered_df["Risk_Level"] == "High Risk"]

    if not high_risk_df.empty:
        st.dataframe(high_risk_df.head(10), use_container_width=True)
    else:
        st.info("No high-risk customers for selected filters.")

# =====================================================
# =================== MODEL INSIGHTS ===================
# =====================================================
elif section == "Model Insights":

    # =====================================================
    # MODEL COMPARISON SUMMARY
    # =====================================================
    st.subheader("Model Comparison Summary")

    st.markdown("""
    This section presents a comparative evaluation of multiple machine learning models
    trained during experimentation. Performance metrics are calculated on a held-out
    test dataset.
    """)

    metrics_df = load_model_metrics()

    if metrics_df is not None:

        # 🔧 FIX: Ensure proper DataFrame structure
        if isinstance(metrics_df, pd.DataFrame) and "Model" not in metrics_df.columns:
            metrics_df = pd.DataFrame(metrics_df)

        metrics_df = metrics_df[
            ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
        ]

        st.dataframe(metrics_df, use_container_width=True)

        # ✅ Dynamically select best model
        best_model_row = metrics_df.loc[
            metrics_df["F1-Score (%)"].idxmax()
        ]

        st.success(
            f"Selected Model for Deployment: **{best_model_row['Model']}** "
            f"(F1-Score: {best_model_row['F1-Score (%)']}%)"
        )

    else:
        st.warning("Model metrics file not found. Please retrain the models.")

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    st.subheader("Feature Importance")

    fi_df = load_feature_importance()

    if fi_df is not None and {"Feature", "Importance"}.issubset(fi_df.columns):

        fi_df = fi_df.sort_values("Importance", ascending=False).head(10)

        fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            title="Top 10 Features Influencing Customer Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Feature importance file not found or invalid.")

    st.divider()

    # =====================================================
    # FEATURE CORRELATION HEATMAP
    # =====================================================
    st.subheader("Feature Correlation Heatmap")

    st.markdown("""
    This heatmap illustrates correlations among numerical customer attributes
    and the churn target variable.
    """)

    numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(
        columns=["customer_id"], errors="ignore"
    )

    if "churn" in numeric_df.columns:

        heatmap_type = st.radio(
            "Select Correlation View",
            ["Full Feature Correlation", "Churn-only Correlation"],
            horizontal=True
        )

        if heatmap_type == "Full Feature Correlation":
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                title="Correlation Heatmap of Numerical Features"
            )
        else:
            churn_corr = (
                numeric_df.corr()[["churn"]]
                .sort_values(by="churn", ascending=False)
            )
            fig = px.imshow(
                churn_corr,
                color_continuous_scale="RdBu",
                title="Feature Correlation with Churn",
                aspect="auto"
            )

        fig.update_layout(height=800, width=1000)
        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # INSIGHTS
        # =====================================================
        st.markdown("### Key Insights from Correlation Analysis")

        churn_corr_series = numeric_df.corr()["churn"].drop("churn")

        top_positive = churn_corr_series.sort_values(ascending=False).head(3)
        top_negative = churn_corr_series.sort_values().head(3)

        st.markdown("**Attributes associated with higher churn risk:**")
        for feature, value in top_positive.items():
            st.markdown(f"- **{feature}** (correlation: {value:.2f})")

        st.markdown("**Attributes associated with lower churn risk:**")
        for feature, value in top_negative.items():
            st.markdown(f"- **{feature}** (correlation: {value:.2f})")

        st.info(
            "These insights help prioritize customer segments for retention strategies "
            "and guide feature-driven business decisions."
        )

    else:
        st.warning("Churn column not found for correlation analysis.")

# =====================================================
# ================ UPLOAD & PREDICT ===================
# =====================================================
elif section == "Upload & Predict":

    st.subheader("Upload Customer Data for Prediction")

    st.markdown("""
    Upload a CSV file containing customer records.
    The trained churn prediction pipeline automatically preprocesses
    the data and generates churn probabilities and risk levels.
    """)

    # ---------------------------------
    # LOAD PIPELINE (CACHED)
    # ---------------------------------
    try:
        pipeline = load_pipeline()
        st.success("Trained pipeline loaded successfully")
    except Exception as e:
        st.error("Trained pipeline not found or failed to load.")
        st.code(str(e))
        st.stop()

    # ---------------------------------
    # FILE UPLOAD
    # ---------------------------------
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        st.stop()

    # ---------------------------------
    # READ FILE
    # ---------------------------------
    try:
        upload_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(upload_df.head(), use_container_width=True)

    # ---------------------------------
    # HANDLE MISSING OPTIONAL COLUMNS
    # ---------------------------------
    expected_columns = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "estimated_salary",
        "credit_card",
        "active_member",
        "gender",
        "country"
    ]

    for col in expected_columns:
        if col not in upload_df.columns:
            if col in ["products_number", "credit_card", "active_member"]:
                upload_df[col] = 1
            elif col == "estimated_salary":
                upload_df[col] = (
                    upload_df["balance"].median()
                    if "balance" in upload_df.columns else 50000
                )
            else:
                upload_df[col] = 0

    # ---------------------------------
    # PREDICTION
    # ---------------------------------
    st.subheader("Churn Prediction Results")

    try:
        churn_probs = pipeline.predict_proba(upload_df)[:, 1]

        result_df = upload_df.copy()
        result_df["Churn_Probability"] = churn_probs

        # Risk thresholds
        low_threshold = st.slider(
            "Low Risk Upper Bound",
            min_value=0.1,
            max_value=0.6,
            value=0.4,
            step=0.05
        )

        medium_threshold = st.slider(
            "Medium Risk Upper Bound",
            min_value=low_threshold + 0.05,
            max_value=0.9,
            value=0.7,
            step=0.05
        )

        result_df["Risk_Level"] = result_df["Churn_Probability"].apply(
            lambda p:
                "Low Risk" if p < low_threshold
                else "Medium Risk" if p < medium_threshold
                else "High Risk"
        )

        st.success("Prediction completed successfully")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
        st.stop()

    # =====================================================
    # VISUAL ANALYSIS (POST-PREDICTION)
    # =====================================================

    # ---------------------------------
    # DYNAMIC COUNTRY GROUPING
    # ---------------------------------
    TOP_N = 5

    top_countries = (
        result_df["country"]
        .value_counts()
        .nlargest(TOP_N)
        .index
    )

    result_df["country_grouped"] = result_df["country"].where(
        result_df["country"].isin(top_countries),
        other="Others"
    )

    # ---------------------------------
    # PREVIEW RESULTS
    # ---------------------------------
    st.subheader("Prediction Output")
    st.dataframe(result_df, use_container_width=True)

    # ---------------------------------
    # RISK DISTRIBUTION
    # ---------------------------------
    st.subheader("Predicted Risk Distribution")

    risk_dist = (
        result_df["Risk_Level"]
        .value_counts()
        .reset_index()
    )
    risk_dist.columns = ["Risk Level", "Count"]

    fig = px.bar(
        risk_dist,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_map={
            "Low Risk": "#2ecc71",
            "Medium Risk": "#f1c40f",
            "High Risk": "#e74c3c"
        },
        text="Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # CHURN PROBABILITY DISTRIBUTION
    # ---------------------------------
    st.subheader("Churn Probability Distribution")

    fig = px.histogram(
        result_df,
        x="Churn_Probability",
        nbins=30,
        title="Distribution of Churn Probability"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # COUNTRY-WISE RISK (GROUPED)
    # ---------------------------------
    st.subheader("Country-wise Average Churn Risk")

    country_risk = (
        result_df
        .groupby("country_grouped")["Churn_Probability"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        country_risk,
        x="country_grouped",
        y="Churn_Probability",
        color="Churn_Probability",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
        text_auto=".2f",
        title="Average Churn Probability by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # DOWNLOAD RESULTS
    # ---------------------------------
    st.subheader("Download Prediction Results")

    st.download_button(
        "Download Predictions as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )