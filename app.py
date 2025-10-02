import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Load models
models = {
    "KNN": joblib.load("models/knn_pipeline.joblib"),
    "SVM": joblib.load("models/svm_pipeline.joblib"),
    "Logistic Regression": joblib.load("models/lr_pipeline.joblib")
}

# Sidebar controls
model_choice = st.sidebar.selectbox("🧠 Choose a model", list(models.keys()))
show_confusion = st.sidebar.checkbox("Show Confusion Matrix")
show_report = st.sidebar.checkbox("Show Classification Report")
compare_models = st.sidebar.checkbox("Compare All Models")
show_advice = st.sidebar.checkbox("Show Environmental Advice", value=True)
show_category_table = st.sidebar.checkbox("Show Breakdown by Region")

model = models[model_choice]

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.session_state["data"] = data  # Store in session to persist across reruns
st.write("📋 Columns in your file:", data.columns.tolist())    

# Use stored data
if "data" in st.session_state:
    data = st.session_state["data"]
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())

    if "Concentration_class" not in data.columns:
        st.error("❌ 'Concentration_class' column not found in uploaded file.")
    else:
        if st.button("🔍 Predict"):
            try:
                # Encode true labels
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(data["Concentration_class"])
                X_input = data.drop(columns=["Concentration_class"])
                y_pred_encoded = model.predict(X_input)
                y_pred_labels = le.inverse_transform(y_pred_encoded)

                # Add predictions to data
                results = data.copy()
                results["Prediction"] = y_pred_encoded
                results["Prediction Label"] = y_pred_labels

                # Summary table
                st.subheader("📊 Prediction Summary")
                summary = results["Prediction Label"].value_counts().reindex(le.classes_, fill_value=0)
                summary_df = pd.DataFrame({
                    "Concentration Level": summary.index,
                    "Count": summary.values
                })
                st.table(summary_df)

                # Categorical breakdown
                if show_category_table and "Region" in results.columns:
                    st.subheader("📍 Breakdown by Region")
                    region_summary = results.groupby("Region")["Prediction Label"].value_counts().unstack().fillna(0).astype(int)
                    st.dataframe(region_summary)

                # Advisory message
                high_labels = ["High", "Very High"]
                high_count = summary_df.loc[summary_df["Concentration Level"].isin(high_labels), "Count"].sum()
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
                sns.countplot(x="Prediction Label", data=results, order=le.classes_, ax=ax)
                ax.set_xlabel("Concentration Level")
                ax.set_ylabel("Sample Count")
                st.pyplot(fig)

                # Confusion matrix
                if show_confusion:
                    st.subheader("🔍 Confusion Matrix")
                    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=le.classes_, yticklabels=le.classes_)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                # Classification report
                if show_report:
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_true_encoded, y_pred_encoded, target_names=le.classes_, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                # Confidence plot (Logistic Regression only)
                if model_choice == "Logistic Regression":
                    st.subheader("📊 Prediction Confidence")
                    probs = model.predict_proba(X_input)
                    prob_df = pd.DataFrame(probs, columns=le.classes_)
                    fig, ax = plt.subplots()
                    sns.boxplot(data=prob_df, ax=ax)
                    ax.set_title("Prediction Confidence Distribution")
                    st.pyplot(fig)

                # Model comparison
                if compare_models:
                    st.subheader("📈 Accuracy Comparison")
                    for name, m in models.items():
                        y_pred = m.predict(X_input)
                        acc = accuracy_score(y_true_encoded, y_pred)
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
