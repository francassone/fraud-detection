import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from src to project root
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load model and artifacts
@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        st.error("Model not found. Please run train.py first.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_example_data():
    data_path = DATA_DIR / "creditcard_2023.csv"
    if not data_path.exists():
        st.error("Example data not found.")
        st.stop()
    return pd.read_csv(data_path)

# Main app
st.title("💳 Credit Card Fraud Detection")
st.write("Upload transaction data or use example data to detect fraudulent transactions")

# Sidebar
st.sidebar.title("Controls")
threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="Probability threshold for classifying a transaction as fraudulent"
)

# Load model
model = load_model()

# Data input section
st.header("Transaction Data")
data_option = st.radio(
    "Choose data source",
    ["Upload CSV", "Use Example Data"]
)

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    df = load_example_data()

# Process and show predictions
if 'df' in locals():
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Preprocess data
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["Amount_log"] = np.log1p(df["Amount"])
    
    # Make predictions
    X = df.drop(columns=["Class"]) if "Class" in df.columns else df
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Results")
        results_df = pd.DataFrame({
            "Transaction ID": range(len(df)),
            "Probability": probas,
            "Prediction": predictions
        })
        st.dataframe(results_df)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("Summary Statistics")
        total = len(predictions)
        fraudulent = predictions.sum()
        
        st.metric("Total Transactions", total)
        st.metric("Fraudulent Transactions", fraudulent)
        st.metric("Fraud Rate", f"{(fraudulent/total)*100:.2f}%")
        
        # Plot probability distribution
        st.subheader("Probability Distribution")
        fig_hist = pd.DataFrame(probas).hist(bins=50)
        st.pyplot(fig_hist[0][0].figure)

# Footer
st.markdown("---")
st.markdown("""
    **Instructions:**
    1. Use the sidebar to adjust the fraud detection threshold
    2. Upload your own CSV or use example data
    3. Review predictions and download results
""")