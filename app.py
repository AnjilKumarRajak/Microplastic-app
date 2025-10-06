import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Microplastic Detection", layout="wide")

# Sidebar
st.sidebar.title("🧪 Microplastic Classifier")
model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "SVM", "Logistic Regression"])

# Load model
model_paths = {
    "KNN": "models/knn_pipeline.joblib",
    "SVM": "models/svm_pipeline.joblib",
    "Logistic Regression": "models/lr_pipeline.joblib"
}
model = joblib.load(model_paths[model_choice])

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your microplastic data (.csv)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Define expected columns
    feature_cols = [
        'Latitude (degree)', 'Longitude(degree)', 'Water Sample Depth (m)',
        'Mesh size (mm)', 'Volunteers Number', 'Collecting Time (min)',
        'Standardized Nurdle  Amount', 'Microplastics measurement',
        'year', 'month', 'day',
        'Ocean', 'Region', 'Country', 'Marine Setting', 'Sampling Method'
    ]
    label_col = "Concentration class text"

    # Check for missing columns
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        st.error(f"❌ Missing columns in uploaded file: {missing}")
        st.stop()

    # Prepare input
    X = data[feature_cols].copy()
    X.fillna(0, inplace=True)

    # Encode true labels
    le = LabelEncoder()
    y_true = le.fit_transform(data[label_col])

    # Predict
    y_pred = model.predict(X)
    y_labels = le.inverse_transform(y_pred)

    # Results
    results = data.copy()
    results["Prediction"] = y_labels
    st.subheader("🔍 Prediction Results")
    st.dataframe(results)

    # Bar chart
    st.subheader("📊 Prediction Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Prediction", data=results, order=le.classes_, ax=ax)
    st.pyplot(fig)

    # Environmental advice
    high_labels = ["High", "Very High"]
    high_count = results["Prediction"].isin(high_labels).sum()
    if high_count > 0:
        st.warning(f"⚠️ {high_count} samples show High or Very High microplastic concentration.")
        st.markdown("""
**🌱 Environmental Advice:**  
- Organize local beach cleanups  
- Reduce single-use plastics  
- Support better waste management policies  
- Educate communities about microplastic pollution
""")

    # Download
    st.download_button(
        label="📥 Download Predictions",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="microplastic_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to begin.")
