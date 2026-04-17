# -*- coding: utf-8 -*-
"""dashboard_deployment"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from lime.lime_tabular import LimeTabularExplainer

# --- 1. DEFINE CUSTOM FUNCTIONS ---
def enhance_features(df):
    df_copy = df.copy()
    if 'training_hours' in df_copy.columns:
        df_copy['training_intensity'] = (df_copy['training_hours'] > 0.4).astype(int)
    else:
        df_copy['training_intensity'] = 0
    if 'city_development_index' in df_copy.columns:
        df_copy['is_high_risk_city'] = (df_copy['city_development_index'] < 0.6).astype(int)
    else:
        df_copy['is_high_risk_city'] = 0
    return df_copy

def get_lime_explanation(processed_df, explainer, model, feature_names):
    data_row = processed_df.iloc[0].values
    def predict_proba_wrapper(numpy_data):
        df_input = pd.DataFrame(numpy_data, columns=feature_names)
        return model.predict_proba(df_input)

    explanation = explainer.explain_instance(
        data_row=data_row,
        predict_fn=predict_proba_wrapper,
        num_features=len(feature_names)
    )
    return explanation.as_list()

# --- FUNGSI RISK DETAILS SESUAI PERMINTAAN ---
def get_risk_details(p, threshold):
    if p > 0.80:
        return 'High Risk (Critical)', '#8B1A1A', 'Immediate Counter-offer'
    elif p >= threshold:
        return 'Medium Risk (Warning)', '#E3BC55', 'Stay Interview / Development'
    else:
        return 'Low Risk (Stable)', '#3D7D44', 'Routine Monitoring'

# --- 2. CONFIGURATION & STYLING ---
MODEL_PATH = 'model_employee_retention_final_v2.pkl'

st.set_page_config(page_title="Job Prediction", page_icon="👥", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .stApp { background-color: #0F172A; font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border-bottom: 4px solid #3B82F6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        width: 100%;
    }
    .main-header h1 {
        background: linear-gradient(to right, #60A5FA, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem !important;
        margin: 0;
        text-align: center;
    }
    .footer {
        width: 100%;
        color: #94A3B8;
        text-align: center;
        padding: 20px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .stSlider label, .stSelectbox label, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    .result-container {
        padding: 25px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_model_package():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, MODEL_PATH)
    if not os.path.exists(full_path):
        full_path = MODEL_PATH 
    try:
        if os.path.exists(full_path):
            package = joblib.load(full_path)
            return package
        else:
            return "FILE_NOT_FOUND"
    except Exception as e:
        return str(e)

package_result = load_model_package()

if isinstance(package_result, str):
    if package_result == "FILE_NOT_FOUND":
        st.error(f"❌ **File model tidak ditemukan.**")
    else:
        st.error(f"❌ **Gagal memuat model.**")
    st.stop()
else:
    package = package_result

# --- 4. HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>HR PREDICTION SYSTEM</h1>
        <p style="color: #60A5FA; margin-top:5px; font-weight: 500; text-align: center;">Job Change Analysis for Data Scientists</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. LOGIC & TABS ---
if package is None:
    st.error("Model tidak ditemukan. Pastikan file .pkl tersedia.")
else:
    model = package['model']
    final_features = package['features']
    optimal_threshold = package['threshold']
    lime_config = package.get('lime_config', {})
    lime_training_data = lime_config.get('training_data_for_explainer')
    lime_feature_names = lime_config.get('feature_names_for_explainer', final_features)

    feature_map = {
        'relevent_experience': 'Relevant Experience',
        'is_high_risk_city': 'High Risk City',
        'training_intensity': 'Training Intensity',
        'city_development_index': 'City Development Index',
        'experience': 'Experience',
        'training_hours': 'Training Hours',
        'last_new_job': 'Last New Job'
    }

    tab_about, tab_manual, tab_batch = st.tabs(["ℹ️ About System", "📝 Manual Input", "📁 Upload Employee Data (CSV)"])

    with tab_about:
        st.header("About HR Predicting Job Transition")
        st.write("This dashboard is specifically designed to predict the probability of employee job transition.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Impact of Use")
            st.write("* Reduce recruitment costs.\n* Allows HR to intervene.\n* Replacing intuition with data.")
        with col2:
            st.subheader("Success Metrics")
            st.write("* **Accuracy & Recall**\n* **F2-Score**")

    with tab_manual:
        col_in, col_out = st.columns([1, 2], gap="large")
        with col_in:
            st.subheader("Employee data")
            with st.form("manual_form"):
                cdi = st.slider("City Development Index", 0.0, 1.0, 0.7)
                rel_exp = st.selectbox("Relevant Experience", [1.0, 0.0], format_func=lambda x: "Yes" if x==1.0 else "No")
                exp = st.slider("Experience", 0.0, 1.0, 0.5)
                last_job = st.slider("Last New Job", 0.0, 1.0, 0.2)
                hrs = st.slider("Training Hours", 0.0, 1.0, 0.3)
                btn = st.form_submit_button("Risk Analysis")

        with col_out:
            if btn:
                input_df = pd.DataFrame([{'city_development_index': cdi, 'relevent_experience': rel_exp, 'experience': exp, 'last_new_job': last_job, 'training_hours': hrs}])
                proc_df = enhance_features(input_df)[final_features]
                prob = model.predict_proba(proc_df)[:, 1][0]

                # Penerapan Logika get_risk_details
                risk_label, risk_color, recommendation = get_risk_details(prob, optimal_threshold)
                label = "Resign" if prob >= optimal_threshold else "Stay"

                st.markdown(f"""
                    <div class="result-container" style="border-top: 5px solid {risk_color};">
                        <h2 style="color: {risk_color};">{label.upper()}</h2>
                        <p>Probability: <b>{prob:.2%}</b></p>
                        <p>Risk Level: <span style="color: {risk_color}; font-weight: bold;">{risk_label}</span></p>
                        <p style="font-style: italic; color: #CBD5E1;">Rec: {recommendation}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.write("### Key Driving Factors")
                explainer = LimeTabularExplainer(training_data=lime_training_data if lime_training_data is not None else np.zeros((1,5)), feature_names=lime_feature_names, mode='classification')
                exp_list = get_lime_explanation(proc_df, explainer, model, lime_feature_names)

                lime_df = pd.DataFrame(exp_list, columns=['Raw_Feature', 'Contribution Score'])
                lime_df['Feature Name'] = lime_df['Raw_Feature'].apply(lambda x: next((feature_map[k] for k in feature_map if k in x), x))
                lime_df['Information'] = lime_df['Contribution Score'].apply(lambda v: "⚠️ Driving factors for resignation" if v > 0 else "✅ Stay holding factor")
                st.table(lime_df[['Feature Name', 'Contribution Score', 'Information']].sort_values(by='Contribution Score', ascending=False))

    with tab_batch:
        st.subheader("Upload Employee Data (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(data)} data rows.")
            proc_batch = enhance_features(data)
            for col in final_features:
                if col not in proc_batch.columns: proc_batch[col] = 0
            
            probs = model.predict_proba(proc_batch[final_features])[:, 1]
            data['Prob_Resign'] = probs
            
            # Penerapan Logika get_risk_details untuk Batch
            data['Risk_Details'] = data['Prob_Resign'].apply(lambda x: get_risk_details(x, optimal_threshold))
            data['Risk_Level'] = data['Risk_Details'].apply(lambda x: x[0])
            data['Recommendation'] = data['Risk_Details'].apply(lambda x: x[2])
            data['Prediction Status'] = data['Prob_Resign'].apply(lambda x: "Resign" if x >= optimal_threshold else "Stay")
            
            st.write("### Preview Prediction Results Data")
            # Menampilkan kolom-kolom penting hasil olahan fungsi risk
            display_cols = [col for col in data.columns if col not in ['Risk_Details']]
            st.dataframe(data[display_cols].style.background_gradient(subset=['Prob_Resign'], cmap='RdYlGn_r'), use_container_width=True)

# --- 6. FOOTER ---
st.markdown("---")
st.markdown('<div class="footer">TEAM 1 BOOLEAN | HR Job Prediction v2.0 © 2026</div>', unsafe_allow_html=True)
