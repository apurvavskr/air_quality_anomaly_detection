import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Set up path to utils and data folder ---
APP_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(APP_DIR, "..", "data", "air_quality_processed.csv")
PCA_PATH = os.path.join(APP_DIR, "..", "data", "pca_coordinates.csv")
UTILS_PATH = os.path.join(APP_DIR, "..", "utils")
sys.path.append(UTILS_PATH)

st.set_page_config(page_title="Air Quality Anomaly Dashboard", layout="wide")
st.title("🌫️ Air Quality Sensor Anomaly Detection")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    return df

df = load_data()

# --- Sidebar Options ---
st.sidebar.header("Options")
sensor = st.sidebar.selectbox(
    "Select a Sensor",
    ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)", "T", "RH", "AH"]
)
show_anomalies = st.sidebar.checkbox("Highlight Anomalies", value=True)

# --- Data Preview ---
if st.sidebar.checkbox("Show Data Sample"):
    st.dataframe(df.head())

# --- Plotting Functions ---
def plot_time_series(df, sensor, show_anomalies):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=df, x="Datetime", y=sensor, ax=ax, label=sensor)
    if show_anomalies:
        sns.scatterplot(
            data=df[df["anomaly"] == -1],
            x="Datetime", y=sensor, color="red", label="Anomalies", ax=ax
        )
    ax.set_ylabel(sensor)
    ax.set_xlabel("Date")
    ax.legend()
    return fig

def plot_anomaly_count(df, col, title, palette):
    fig, ax = plt.subplots()
    sns.countplot(data=df[df["anomaly"] == -1], x=col, ax=ax, palette=palette)
    ax.set_title(title)
    return fig

def plot_pca_scatter(df):
    try:
        pca_df = pd.read_csv(PCA_PATH)
    except FileNotFoundError:
        # On first run, compute PCA coordinates
        from preprocess import scale_features, run_pca # type: ignore
        scaled_df = scale_features(df)
        X_pca, _ = run_pca(scaled_df)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["anomaly"] = df["anomaly"]
        pca_df.to_csv(PCA_PATH, index=False)
    fig, ax = plt.subplots()
    colors = pca_df["anomaly"].map({1: "blue", -1: "red"})
    ax.scatter(pca_df["PC1"], pca_df["PC2"], c=colors, alpha=0.6)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA of Air Quality Readings (Red = Anomalies)")
    return fig

# --- Display Plots ---
st.subheader(f"📈 Temporal Trend of {sensor}")
st.pyplot(plot_time_series(df, sensor, show_anomalies))

st.subheader("🕒 Anomaly Counts by Hour")
st.pyplot(plot_anomaly_count(df, "Hour", "Anomalies by Hour", "Reds"))

st.subheader("📅 Anomaly Counts by Month")
st.pyplot(plot_anomaly_count(df, "Month", "Anomalies by Month", "Oranges"))

st.subheader("🔍 PCA Anomaly Visualization")
st.pyplot(plot_pca_scatter(df))

# --- Footer ---
st.markdown("---")
st.markdown("**Tip:** Use the sidebar to select sensors and show/hide anomalies.")
