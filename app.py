import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastic Detection App", layout="wide")

# Sidebar guide
st.sidebar.title("🧪 Microplastic Classifier")
st.sidebar.markdown("""
Upload your microplastic sampling data and choose a model to classify concentration levels.  
You'll get a summary of predictions, visual insights, and actionable advice.
""")

# Class mapping
class_map = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High"
}

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

# Show label encoding summary
st.subheader("📘 Concentration Labels & Encodings")
label_df = pd.DataFrame({
    "Label": ["Very Low", "Low", "Medium", "High", "Very High"],
    "Encoding": [0, 1, 2, 3, 4]
})
st.table(label_df)

# File upload
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
            results["Prediction Label"] = results["Prediction"].map(class_map)

            # Summary table of prediction counts
            st.subheader("📊 Prediction Summary")
            summary = results["Prediction Label"].value_counts().reindex(["Very Low", "Low", "Medium", "High", "Very High"], fill_value=0)
            summary_df = pd.DataFrame({
                "Concentration Level": summary.index,
                "Count": summary.values
            })
            st.table(summary_df)

            # Download button
            st.download_button(
                label="📥 Download Full Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="microplastic_predictions.csv",
                mime="text/csv"
            )

            # Visualization
            fig, ax = plt.subplots()
            sns.barplot(x="Concentration Level", y="Count", data=summary_df, ax=ax)
            ax.set_xlabel("Concentration Level")
            ax.set_ylabel("Sample Count")
            st.pyplot(fig)

            # Advisory message
            high_count = summary_df.loc[summary_df["Concentration Level"].isin(["High", "Very High"]), "Count"].sum()
            if high_count > 0:
                st.warning(f"⚠️ {high_count} samples show High or Very High microplastic concentration.")
                st.markdown("""
**🌱 Environmental Advice:**  
High microplastic levels can harm marine life and ecosystems. Consider:
- Organizing local beach cleanups  
- Advocating for reduced plastic use and better waste management  
- Supporting policies that regulate industrial plastic discharge  
- Educating communities about microplastic pollution

Every small action helps protect our oceans.
""")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a CSV file to begin.")
