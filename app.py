"""
Adult Income Classification - Streamlit App
Interactive ML Model Comparison and Prediction Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Page configuration
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_resource
def load_models(version="v2"):  # Change version to bust cache
    """Load all trained models and preprocessing objects"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbor': 'model/k-nearest_neighbor.pkl',
        'Naive Bayes': 'model/naive_bayes_gaussian.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load preprocessor and label encoder
    with open('model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return models, preprocessor, label_encoder

# Load model performance results
@st.cache_data
def load_results():
    """Load pre-computed model performance results from notebook run"""
    # Real metrics from 2025AA05734-ml-assignment-2.ipynb execution
    results = {
        'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 
                  'Naive Bayes', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.8488, 0.8584, 0.8372, 0.6435, 0.8620, 0.8765],
        'AUC Score': [0.9032, 0.9032, 0.8825, 0.8492, 0.9141, 0.9308],
        'Precision': [0.7256, 0.7708, 0.6810, 0.3972, 0.7965, 0.7973],
        'Recall': [0.6049, 0.5914, 0.6170, 0.9107, 0.5784, 0.6577],
        'F1 Score': [0.6597, 0.6693, 0.6474, 0.5532, 0.6702, 0.7208],
        'MCC Score': [0.5674, 0.5894, 0.5430, 0.4042, 0.5977, 0.6474]
    }
    return pd.DataFrame(results)

@st.cache_data
def load_confusion_matrices():
    """Load actual confusion matrix values from notebook run"""
    # Real confusion matrices from notebook: [[TN, FP], [FN, TP]]
    confusion_matrices = {
        'Logistic Regression': np.array([[6689, 528], [912, 1396]]),
        'Decision Tree': np.array([[6811, 406], [943, 1365]]),
        'K-Nearest Neighbor': np.array([[6550, 667], [884, 1424]]),
        'Naive Bayes': np.array([[4027, 3190], [206, 2102]]),
        'Random Forest': np.array([[6876, 341], [973, 1335]]),
        'XGBoost': np.array([[6831, 386], [790, 1518]])
    }
    return confusion_matrices

# Main header
st.markdown('<h1 class="main-header">💰 Adult Income Classification</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

# Sidebar navigation with clickable links
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("---")

pages = ["🏠 Home", "📊 Model Comparison", "🔮 Make Prediction", "📥 Download Test Data", "📖 About"]

for page_name in pages:
    if st.sidebar.button(page_name, key=page_name, use_container_width=True):
        st.session_state.page = page_name

page = st.session_state.page

try:
    models, preprocessor, label_encoder = load_models()
    results_df = load_results()
    confusion_matrices = load_confusion_matrices()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Adult Income Classifier")
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1.5rem;'>
        <p style='margin: 0; color: #555;'><strong>Name:</strong> Tanuj Kapoor | <strong>Student ID:</strong> 2025AA05734</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Models Trained", len(models))
    with col2:
        st.metric("🎯 Best Accuracy", f"{results_df['Accuracy'].max():.2%}")
    with col3:
        st.metric("⭐ Best AUC", f"{results_df['AUC Score'].max():.4f}")
    
    st.markdown("---")
    
    st.subheader("🎯 Objective")
    st.write("""
    This application compares various classification ML models on the Adult Income dataset 
    from UCI ML Repository to predict whether income exceeds $50K/year based on census data.
    """)
    
    st.subheader("🤖 Models Evaluated")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- 📈 Logistic Regression")
        st.markdown("- 🌳 Decision Tree Classifier")
        st.markdown("- 👥 K-Nearest Neighbor Classifier")
    with col2:
        st.markdown("- 🎲 Naive Bayes Classifier (Gaussian)")
        st.markdown("- 🌲 Ensemble Model - Random Forest")
        st.markdown("- 🚀 Ensemble Model - XGBoost")
    
    st.subheader("📊 Evaluation Metrics")
    metrics = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
    st.write(", ".join(metrics))
    
    st.info("👈 Use the sidebar to navigate to Model Comparison or Make Prediction pages")

# MODEL COMPARISON PAGE
elif page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    
    # Display results table
    st.subheader("📋 Performance Metrics Table")
    
    # Format the dataframe for better display
    display_df = results_df.copy()
    numeric_cols = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 Performance Metrics Comparison")
    
    import matplotlib.pyplot as plt
    
    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Across All Metrics', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics_to_plot, colors)):
        values = results_df[metric].values
        models = results_df['Model'].values
        
        bars = ax.barh(models, values, color=color, alpha=0.8)
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Confusion Matrices for all models
    st.subheader("🎯 Confusion Matrices - All Models")
    st.write("Visual representation of prediction accuracy for each model")
    
    # Use actual confusion matrices from notebook run
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 12))
    fig_cm.suptitle('Confusion Matrices - All Classification Models', fontsize=16, fontweight='bold', y=0.995)
    
    from sklearn.metrics import ConfusionMatrixDisplay
    
    model_names = results_df['Model'].tolist()
    cm_colors = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'YlOrBr']
    
    # Display actual confusion matrices from notebook results
    for idx, model_name in enumerate(model_names):
        ax = axes_cm[idx // 3, idx % 3]
        
        # Get actual confusion matrix from notebook
        cm = confusion_matrices[model_name]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        # Get accuracy for display
        accuracy = results_df.loc[idx, 'Accuracy']
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
        disp.plot(ax=ax, cmap=cm_colors[idx], colorbar=False)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(False)
        
        text = f'Acc: {accuracy:.3f}\nTP:{tp} TN:{tn}\nFP:{fp} FN:{fn}'
        ax.text(1.15, 0.5, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    st.pyplot(fig_cm)
    
    st.markdown("---")
    
    st.subheader("🏆 Best Performers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_acc_idx = results_df['Accuracy'].idxmax()
        st.success(f"**Best Accuracy**  \n{results_df.loc[best_acc_idx, 'Model']}  \n{results_df.loc[best_acc_idx, 'Accuracy']:.4f}")
    
    with col2:
        best_auc_idx = results_df['AUC Score'].idxmax()
        st.success(f"**Best AUC Score**  \n{results_df.loc[best_auc_idx, 'Model']}  \n{results_df.loc[best_auc_idx, 'AUC Score']:.4f}")
    
    with col3:
        best_f1_idx = results_df['F1 Score'].idxmax()
        st.success(f"**Best F1 Score**  \n{results_df.loc[best_f1_idx, 'Model']}  \n{results_df.loc[best_f1_idx, 'F1 Score']:.4f}")

# PREDICTION PAGE
elif page == "🔮 Make Prediction":
    st.header("Make Income Prediction")
    
    tab2, tab1 = st.tabs(["📤 Batch Upload (CSV)", "📝 Single Prediction"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.write("Enter the details below to predict if income exceeds $50K/year")
        
        selected_model = st.selectbox("Select Model", list(models.keys()), key="single_model")
        
        st.markdown("---")
        st.subheader("Input Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35)
            fnlwgt = st.number_input("Final Weight (Census)", min_value=10000, max_value=1500000, value=200000, 
                                      help="Census final weight - leave at default if unsure")
            workclass = st.selectbox("Workclass", 
                ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
            education = st.selectbox("Education",
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
            education_num = st.number_input("Education-Num", min_value=1, max_value=16, value=10)
            marital_status = st.selectbox("Marital Status",
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
            occupation = st.selectbox("Occupation",
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                 'Armed-Forces'])
        
        with col2:
            relationship = st.selectbox("Relationship",
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                 'Other-relative', 'Unmarried'])
            race = st.selectbox("Race",
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
            sex = st.selectbox("Sex", ['Male', 'Female'])
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
            native_country = st.selectbox("Native Country",
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
                 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
                 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])
        
        if st.button("🔮 Predict Income", type="primary"):
            input_data = pd.DataFrame({
                'age': [age],
                'fnlwgt': [fnlwgt],
                'workclass': [workclass],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
            
            try:
                # Preprocess and predict
                X_processed = preprocessor.transform(input_data)
                model = models[selected_model]
                prediction = model.predict(X_processed)[0]
                prediction_proba = model.predict_proba(X_processed)[0]
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    income_class = label_encoder.inverse_transform([prediction])[0]
                    if prediction == 1:
                        st.success(f"### Predicted Income: {income_class}")
                    else:
                        st.info(f"### Predicted Income: {income_class}")
                
                with col2:
                    st.metric("Confidence", f"{prediction_proba[prediction]:.2%}")
                
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Income Class': label_encoder.classes_,
                    'Probability': prediction_proba
                })
                st.bar_chart(prob_df.set_index('Income Class'))
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # TAB 2: Batch CSV Upload
    with tab2:
        st.write("""
        Upload a CSV file containing test data to make predictions on multiple records.
        The CSV should have the same format as downloaded from the 'Download Test Data' page.
        """)
        
        batch_model = st.selectbox("Select Model for Batch Prediction", list(models.keys()), key="batch_model")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                          help="Upload a CSV file with the required columns")
        
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                
                st.success(f"✓ File uploaded successfully! {len(upload_df)} rows found.")
                
                # Display uploaded data preview
                st.subheader("Uploaded Data Preview (First 5 Rows)")
                st.dataframe(upload_df.head(), use_container_width=True)
                
                # Required feature columns (must match training data order)
                required_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
                
                missing_cols = [col for col in required_cols if col not in upload_df.columns]
                extra_info_cols = [col for col in upload_df.columns if col not in required_cols and col != 'income']
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info("💡 Please download the test data template to see the required format.")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📊 **Rows:** {len(upload_df)}")
                    with col2:
                        st.info(f"📋 **Features:** {len(required_cols)}")
                    with col3:
                        has_labels = 'income' in upload_df.columns
                        label_icon = "✅" if has_labels else "⚠️"
                        st.info(f"{label_icon} **Labels:** {'Yes' if has_labels else 'No'}")
                    
                    if extra_info_cols:
                        st.caption(f"Note: Extra columns will be preserved in results: {', '.join(extra_info_cols)}")
                    
                    # Extract feature columns in correct order
                    feature_df = upload_df[required_cols].copy()
                    
                    if st.button("🚀 Run Batch Predictions", type="primary"):
                        with st.spinner(f"Making predictions on {len(feature_df)} rows using {batch_model}..."):
                            try:
                                X_batch = preprocessor.transform(feature_df)
                                
                                model = models[batch_model]
                                predictions = model.predict(X_batch)
                                predictions_proba = model.predict_proba(X_batch)
                                
                                predicted_labels = label_encoder.inverse_transform(predictions)
                                confidence_scores = predictions_proba.max(axis=1)
                                proba_high_income = predictions_proba[:, 1]
                                
                                results_df = upload_df.copy()
                                results_df['predicted_income'] = predicted_labels
                                results_df['confidence'] = confidence_scores
                                results_df['probability_>50K'] = proba_high_income
                                results_df['probability_<=50K'] = predictions_proba[:, 0]
                                
                                st.success("✓ Predictions completed successfully!")
                                
                                # Check if ground truth 'income' column exists for evaluation
                                has_ground_truth = 'income' in upload_df.columns
                                
                                if has_ground_truth:
                                    try:
                                        # Clean ground truth labels
                                        y_true_raw = upload_df['income'].astype(str).str.strip().str.rstrip('.')
                                        
                                        # Validate labels
                                        unique_labels = y_true_raw.unique()
                                        expected_labels = set(label_encoder.classes_)
                                        actual_labels = set(unique_labels)
                                        
                                        if not actual_labels.issubset(expected_labels):
                                            unknown_labels = actual_labels - expected_labels
                                            st.warning(f"⚠️ Unknown income labels found: {unknown_labels}. Expected: {expected_labels}")
                                            st.info("Skipping confusion matrix due to label mismatch.")
                                        else:
                                            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
                                            import matplotlib.pyplot as plt
                                            
                                            y_true = label_encoder.transform(y_true_raw)
                                            cm = confusion_matrix(y_true, predictions)
                                            
                                            accuracy = accuracy_score(y_true, predictions)
                                            precision = precision_score(y_true, predictions)
                                            recall = recall_score(y_true, predictions)
                                            f1 = f1_score(y_true, predictions)
                                            
                                            st.markdown("---")
                                            st.subheader("📊 Model Evaluation Metrics")
                                            st.success(f"✓ Ground truth labels detected! Evaluating {batch_model} performance:")
                                            
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Accuracy", f"{accuracy:.4f}")
                                            with col2:
                                                st.metric("Precision", f"{precision:.4f}")
                                            with col3:
                                                st.metric("Recall", f"{recall:.4f}")
                                            with col4:
                                                st.metric("F1 Score", f"{f1:.4f}")
                                    
                                            st.subheader("🎯 Confusion Matrix")
                                            
                                            col1, col2 = st.columns([1, 1])
                                            
                                            with col1:
                                                fig, ax = plt.subplots(figsize=(6, 5))
                                                disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                                                             display_labels=label_encoder.classes_)
                                                disp.plot(ax=ax, cmap='Blues', colorbar=True)
                                                ax.set_title(f'Confusion Matrix - {batch_model}', fontsize=12, fontweight='bold')
                                                st.pyplot(fig)
                                            
                                            with col2:
                                                tn, fp, fn, tp = cm.ravel()
                                                st.markdown("##### Confusion Matrix Breakdown")
                                                st.write(f"**True Negatives (TN):** {tn}")
                                                st.write(f"**False Positives (FP):** {fp}")
                                                st.write(f"**False Negatives (FN):** {fn}")
                                                st.write(f"**True Positives (TP):** {tp}")
                                                
                                                st.markdown("---")
                                                st.markdown("##### Additional Metrics")
                                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                                                st.write(f"**Specificity:** {specificity:.4f}")
                                                st.write(f"**Sensitivity:** {sensitivity:.4f}")
                                            
                                            st.markdown("---")
                                    
                                    except Exception as eval_error:
                                        st.error(f"⚠️ Error evaluating model with ground truth: {str(eval_error)}")
                                        st.info("Proceeding with predictions only (no evaluation metrics).")
                                
                                st.subheader("Prediction Summary")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Predictions", len(results_df))
                                with col2:
                                    high_income_count = (predicted_labels == '>50K').sum()
                                    st.metric("Predicted >50K", high_income_count)
                                with col3:
                                    low_income_count = (predicted_labels == '<=50K').sum()
                                    st.metric("Predicted <=50K", low_income_count)
                                
                                st.subheader("Prediction Results (First 10 Rows)")
                                st.dataframe(results_df.head(10), use_container_width=True)
                                
                                st.markdown("---")
                                st.subheader("Download Results")
                                
                                csv_results = results_df.to_csv(index=False)
                                
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.download_button(
                                        label="📥 Download Results CSV",
                                        data=csv_results,
                                        file_name=f"predictions_{batch_model.replace(' ', '_').lower()}.csv",
                                        mime="text/csv",
                                    )
                                
                                with col2:
                                    st.info(f"Results include: original data + predicted_income + confidence + probabilities")
                                
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                                st.info("Please ensure your CSV has the correct column names and data types.")
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure you're uploading a valid CSV file with the correct format.")
        
        else:
            st.info("👆 Upload a CSV file to get started. Download the test data template from the 'Download Test Data' page if needed.")

# DOWNLOAD TEST DATA PAGE
elif page == "📥 Download Test Data":
    st.header("Download Test Dataset")
    
    st.write("""
    Download **real test data** from the trained model. This dataset contains actual samples
    from the test set with ground truth labels included.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **✅ Use for:**
        - Testing batch predictions
        - Evaluating model performance
        - Viewing confusion matrix
        """)
    with col2:
        st.success("""
        **📊 Includes:**
        - Real test samples
        - All 14 feature columns
        - Ground truth `income` labels
        """)
    
    try:
        test_data = pd.read_csv('model/test_data_sample.csv')
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(test_data))
        with col2:
            st.metric("Features", 14)
        with col3:
            if 'income' in test_data.columns:
                income_dist = test_data['income'].value_counts()
                st.metric("Income ≤50K", f"{income_dist.get('<=50K', 0)}")
        with col4:
            if 'income' in test_data.columns:
                st.metric("Income >50K", f"{income_dist.get('>50K', 0)}")
        
        st.subheader("Sample Preview (First 5 Rows)")
        st.dataframe(test_data.head(5), use_container_width=True)
        
        csv = test_data.to_csv(index=False)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Download Test Data CSV",
                data=csv,
                file_name="adult_income_test_data.csv",
                mime="text/csv",
            )
        
        with col2:
            st.info("💡 **Tip:** Upload this CSV in the 'Batch Upload' tab to see model evaluation metrics and confusion matrix!")
    
    except FileNotFoundError:
        st.error("❌ Test data file not found. Please run the notebook (Cell 12) to generate 'test_data_sample.csv'.")
        st.code("# Run this cell in the notebook:\n# Cell 12: Feature Engineering and Train-Test Split", language="python")
    except Exception as e:
        st.error(f"Error loading test data: {e}")
    
    st.markdown("---")
    st.subheader("📋 Column Descriptions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Numerical Features:**
        - `age`: Age in years
        - `fnlwgt`: Final weight (census weight)
        - `education-num`: Education level (1-16)
        - `capital-gain`: Capital gains
        - `capital-loss`: Capital losses
        - `hours-per-week`: Hours worked per week
        """)
    
    with col2:
        st.write("""
        **Categorical Features:**
        - `workclass`: Employment type
        - `education`: Highest education level
        - `marital-status`: Marital status
        - `occupation`: Job category
        - `relationship`: Family relationship
        - `race`: Race category
        - `sex`: Gender
        - `native-country`: Country of origin
        """)
    
    st.info("💡 **Tip**: You can edit this CSV file and use it for batch predictions by loading it into the prediction page.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Adult Income Classification Experiment | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
