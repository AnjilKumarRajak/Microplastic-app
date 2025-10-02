import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st

# -------------------------------
# Load models
# -------------------------------
models = {
    "KNN": joblib.load("models/knn_pipeline.joblib"),
    "SVM": joblib.load("models/svm_pipeline.joblib"),
    "Logistic Regression": joblib.load("models/lr_pipeline.joblib")
}

# -------------------------------
# Sidebar controls
# -------------------------------
model_choice = st.sidebar.selectbox("🧠 Choose a model", list(models.keys()))
show_confusion = st.sidebar.checkbox("Show Confusion Matrix")
show_report = st.sidebar.checkbox("Show Classification Report")
compare_models = st.sidebar.checkbox("Compare All Models")
show_advice = st.sidebar.checkbox("Show Environmental Advice", value=True)
show_category_table = st.sidebar.checkbox("Show Breakdown by Region")

model = models[model_choice]

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # ⚡ FIX: Strip spaces and replace spaces with underscores
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(r"\s+", "_", regex=True)

    st.session_state["data"] = data

# -------------------------------
# Use stored data
# -------------------------------
if "data" in st.session_state:
    data = st.session_state["data"]
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())
    st.write("📋 Columns in your file:", data.columns.tolist())

    # Feature columns (match your pipeline)
    numeric_feat = [
        'Mesh_size_(mm)', 'Volunteers_Number', 'Collecting_Time_(min)',
        'year', 'month', 'day',
        'Water_Sample_Depth_(m)', 'Standardized_Nurdle__Amount', 'Microplastics_measurement'
    ]
    categorical_feat = [
        'Ocean', 'Region', 'Country', 'Marine_Setting', 'Sampling_Method'
    ]
    feature_cols = [col for col in numeric_feat + categorical_feat if col in data.columns]
    X_input = data[feature_cols]

    # Check if true labels exist
    label_col = "Concentration_class"
    y_true_available = label_col in data.columns
    if y_true_available:
        y_true = data[label_col].astype(str)

    if st.button("🔍 Predict"):
        try:
            # Predict
            y_pred = model.predict(X_input)
            y_pred_labels = y_pred.astype(str)

            # Add predictions to dataframe
            results = data.copy()
            results["Prediction"] = y_pred
            results["Prediction_Label"] = y_pred_labels

            # Summary table
            st.subheader("📊 Prediction Summary")
            summary = results["Prediction_Label"].value_counts()
            summary_df = pd.DataFrame({
                "Concentration_Level": summary.index,
                "Count": summary.values
            })
            st.table(summary_df)

            # Region breakdown
            if show_category_table and "Region" in results.columns:
                st.subheader("📍 Breakdown by Region")
                region_summary = results.groupby("Region")["Prediction_Label"].value_counts().unstack().fillna(0).astype(int)
                st.dataframe(region_summary)

            # Advisory message
            high_count = summary_df.loc[summary_df["Concentration_Level"].isin(["High", "Very High"]), "Count"].sum()
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

            # Bar chart
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction_Label", data=results, order=summary.index, ax=ax)
            ax.set_xlabel("Concentration Level")
            ax.set_ylabel("Sample Count")
            st.pyplot(fig)

            # Metrics (only if true labels exist)
            if y_true_available:
                st.subheader("📈 Model Metrics")
                st.write("Accuracy:", accuracy_score(y_true, y_pred_labels))

                if show_confusion:
                    st.subheader("🔍 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred_labels)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=sorted(y_true.unique()),
                                yticklabels=sorted(y_true.unique()), ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                if show_report:
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_true, y_pred_labels, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

            # Model comparison
            if compare_models and y_true_available:
                st.subheader("📈 Accuracy Comparison")
                for name, m in models.items():
                    pred = m.predict(X_input)
                    acc = accuracy_score(y_true, pred)
                    st.write(f"✅ {name}: {acc:.2f}")

            # Download button
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
