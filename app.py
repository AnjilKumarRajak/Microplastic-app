import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastic Detection App", layout="wide")

# Sidebar guide
st.sidebar.title("🧪 Microplastic Classifier")
st.sidebar.markdown("""
Upload your microplastic data and choose a model to classify samples.  
Results are shown instantly and can be downloaded.
""")

# Load models
models = {
    "Logistic Regression": joblib.load("models/lr_pipeline.joblib"),
    "SVM": joblib.load("models/svm_pipeline.joblib"),
    "KNN": joblib.load("models/knn_pipeline.joblib")
}

# Model selector
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

# App title
st.title("🌊 Microplastic Detection App")

# Upload data
uploaded_file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("🔍 Predict"):
        try:
            predictions = model.predict(data)
            results = data.copy()
            results["Prediction"] = predictions

            st.subheader("✅ Predictions")
            st.dataframe(results)

            # Download button
            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="microplastic_predictions.csv",
                mime="text/csv"
            )

            # Visualization
            st.subheader("📊 Prediction Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction", data=results, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a CSV file to begin.")
