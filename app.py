import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# ------------------------------
# Load models
# ------------------------------
models = {
    "KNN": joblib.load("models/knn_pipeline.joblib"),
    "SVM": joblib.load("models/svm_pipeline.joblib"),
    "Logistic Regression": joblib.load("models/lr_pipeline.joblib")
}

# ------------------------------
# Sidebar controls
# ------------------------------
model_choice = st.sidebar.selectbox("🧠 Choose a model", list(models.keys()))
show_confusion = st.sidebar.checkbox("Show Confusion Matrix")
show_report = st.sidebar.checkbox("Show Classification Report")
compare_models = st.sidebar.checkbox("Compare All Models")
show_advice = st.sidebar.checkbox("Show Environmental Advice", value=True)
show_category_table = st.sidebar.checkbox("Show Breakdown by Region")

model = models[model_choice]

# ------------------------------
# File upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.session_state["data"] = data

# ------------------------------
# Main app
# ------------------------------
if "data" in st.session_state:
    data = st.session_state["data"]
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())
    st.write("📋 Columns in your file:", data.columns.tolist())

    # ------------------------------
    # Normalize column names to avoid mismatch
    # ------------------------------
    data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()

    label_col = "concentration_class"            # categorical labels (target)
    range_col = "concentration_class_range"      # numeric ranges (info only)

    # Check if label exists
    if label_col not in data.columns:
        st.error(f"❌ Could not find '{label_col}' in file. Available columns: {list(data.columns)}")
        st.stop()

    # Filter valid labels
    valid_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    clean_data = data[data[label_col].isin(valid_labels)]
    if clean_data.empty:
        st.error("❌ No valid rows with known concentration class labels.")
        st.stop()

    if st.button("🔍 Predict"):
        try:
            # Encode ground truth labels
            le = LabelEncoder()
            y_true = le.fit_transform(clean_data[label_col])
            label_order = le.classes_

            # Drop label and numeric range columns before prediction
            drop_cols = [label_col, range_col]
            X_input = clean_data.drop(columns=drop_cols, errors="ignore")

            # Predict
            y_pred = model.predict(X_input)

            # Decode predictions
            y_pred_labels = le.inverse_transform(y_pred)

            # Build results dataframe
            results = clean_data.copy()
            results["Prediction"] = y_pred
            results["Prediction Label"] = y_pred_labels

            # ------------------------------
            # Prediction Summary
            # ------------------------------
            st.subheader("📊 Prediction Summary")
            summary = results["Prediction Label"].value_counts().reindex(label_order, fill_value=0)
            summary_df = pd.DataFrame({
                "Concentration Level": summary.index,
                "Count": summary.values
            })
            st.table(summary_df)

            # ------------------------------
            # Region breakdown
            # ------------------------------
            if show_category_table and "region" in results.columns:
                st.subheader("📍 Breakdown by Region")
                region_summary = results.groupby("region")["Prediction Label"].value_counts().unstack().fillna(0).astype(int)
                st.dataframe(region_summary)

            # ------------------------------
            # Advisory
            # ------------------------------
            high_count = summary_df.loc[summary_df["Concentration Level"].isin(["High", "Very High"]), "Count"].sum()
            if show_advice and high_count > 0:
                st.warning(f"⚠️ {high_count} samples show High or Very High microplastic concentration.")
                st.markdown("""
**🌱 Environmental Advice:**  
High microplastic levels can harm marine life and ecosystems. Consider:
- Organizing local beach cleanups  
- Advocating for reduced plastic use and better waste management  
- Supporting policies that regulate industrial plastic discharge  
- Educating communities about microplastic pollution
""")

            # ------------------------------
            # Bar chart
            # ------------------------------
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction Label", data=results, order=label_order, ax=ax)
            ax.set_xlabel("Concentration Level")
            ax.set_ylabel("Sample Count")
            st.pyplot(fig)

            # ------------------------------
            # Confusion matrix
            # ------------------------------
            if show_confusion:
                st.subheader("🔍 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=label_order, yticklabels=label_order)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            # ------------------------------
            # Classification report
            # ------------------------------
            if show_report:
                st.subheader("📋 Classification Report")
                report = classification_report(y_true, y_pred, target_names=label_order, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            # ------------------------------
            # Confidence plot (Logistic Regression only)
            # ------------------------------
            if model_choice == "Logistic Regression":
                st.subheader("📊 Prediction Confidence")
                probs = model.predict_proba(X_input)
                prob_df = pd.DataFrame(probs, columns=label_order)
                fig, ax = plt.subplots()
                sns.boxplot(data=prob_df, ax=ax)
                ax.set_title("Prediction Confidence Distribution")
                st.pyplot(fig)

            # ------------------------------
            # Model comparison
            # ------------------------------
            if compare_models:
                st.subheader("📈 Accuracy Comparison")
                for name, m in models.items():
                    pred = m.predict(X_input)
                    acc = accuracy_score(y_true, pred)
                    st.write(f"✅ {name}: {acc:.2f}")

            # ------------------------------
            # Download predictions
            # ------------------------------
            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="microplastic_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload a CSV file to begin.")
