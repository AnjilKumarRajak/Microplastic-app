import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st

# Load models
models = {
    "KNN": joblib.load("models/knn_pipeline.joblib"),
    "SVM": joblib.load("models/svm_pipeline.joblib"),
    "Logistic Regression": joblib.load("models/lr_pipeline.joblib")
}

# Choose model
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

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

            # Show prediction summary
            st.subheader("📊 Prediction Summary")
            summary = results["Prediction"].value_counts().sort_index()
            summary_df = pd.DataFrame({
                "Class": summary.index,
                "Count": summary.values
            })
            st.table(summary_df)

            # Accuracy comparison (if true labels are available)
            if "true_label" in data.columns:
                y_true = data["true_label"]
                st.subheader("📈 Model Accuracy Comparison")
                for name, m in models.items():
                    y_pred = m.predict(data.drop(columns=["true_label"]))
                    acc = accuracy_score(y_true, y_pred)
                    st.write(f"{name}: {acc:.2f}")

            # Confusion matrix (if true labels are available)
            if "true_label" in data.columns:
                st.subheader("🔍 Confusion Matrix")
                cm = confusion_matrix(data["true_label"], predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.subheader("📋 Classification Report")
                report = classification_report(data["true_label"], predictions, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            # Confidence visualization (Logistic Regression only)
            if model_choice == "Logistic Regression":
                st.subheader("📊 Prediction Confidence")
                probs = model.predict_proba(data)
                prob_df = pd.DataFrame(probs, columns=["Very Low", "Low", "Medium", "High", "Very High"])
                fig, ax = plt.subplots()
                sns.boxplot(data=prob_df, ax=ax)
                ax.set_title("Prediction Confidence Distribution")
                st.pyplot(fig)

            # Download button
            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="microplastic_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
