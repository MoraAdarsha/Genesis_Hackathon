import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Network Congestion Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- File Loading Functions ---
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost models reliably."""
    models = {}
    try:
        # Assumes models are in a 'models' subdirectory relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        
        for router in ['A', 'B', 'C']:
            model_path = os.path.join(models_dir, f"model{router}_p.pkl")
            # Fallback for environments where the file might be in the root
            if not os.path.exists(model_path):
                model_path = f"model{router}_p.pkl"
            
            if not os.path.exists(model_path):
                st.error(f"FATAL: Model file for Router {router} not found. Please ensure 'model{router}_p.pkl' is in the 'models' folder or the root directory.")
                return None
            models[router] = joblib.load(model_path)
        return models
    except Exception as e:
        st.error(f"A critical error occurred while loading the predictive models: {e}")
        return None

@st.cache_resource
def load_label_encoder():
    """Load the label encoder or create a default one if it's missing."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        encoder_path = os.path.join(script_dir, "label_encoder_impact.pkl")
        if os.path.exists(encoder_path):
            return joblib.load(encoder_path)
        else:
            st.warning("Info: `label_encoder_impact.pkl` not found. A default encoder is being used for this session.")
            le = LabelEncoder()
            le.fit(['None', 'Low', 'Medium', 'High'])
            return le
    except Exception as e:
        st.error(f"Error loading the label encoder: {e}")
        return None

# --- Core Data Processing and Prediction Logic ---
def preprocess_data(df, le_impact):
    """
    This function takes the raw DataFrame and cleans it completely.
    It handles type errors, date parsing, and encoding in one place.
    """
    # 1. Correctly parse timestamps, interpreting DD/MM/YY format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    if df.empty:
        return df # Return empty df if no valid dates

    # 2. Define all columns that should be numeric
    numeric_cols = [
        'Traffic Volume (MB/s)', 'Latency (ms)', 'Bandwidth Allocated (MB/s)',
        'Bandwidth Used (MB/s)', 'total_avg_app_traffic', 'total_peak_app_traffic',
        'total_logins', 'total_peak_user_usage', 'Num_Config_Changes', 'congestion Flag'
    ]
    
    # 3. Force columns to numeric type. This is the main fix for the TypeError.
    # It converts non-numeric values (like "None") to NaN, then fills them with 0.
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Process the 'Impact' column for the model
    if 'Impact' in df.columns:
        df['Impact'] = df['Impact'].fillna('None')
        # Handle categories that the encoder hasn't seen before by defaulting them to 0
        known_labels = set(le_impact.classes_)
        df['Impact_encoded'] = df['Impact'].apply(
            lambda x: le_impact.transform([x])[0] if x in known_labels else 0
        )
    else:
        df['Impact_encoded'] = 0 # Create a default column if 'Impact' is missing

    # 5. Ensure data is in chronological order
    df.sort_values('Timestamp', inplace=True)
    return df

def create_training_samples_single(data, target_timestamp, hours=12):
    """Create features for a single prediction using the last 12 hours of data."""
    window_start = target_timestamp - timedelta(hours=hours)
    window_data = data[(data['Timestamp'] >= window_start) & (data['Timestamp'] < target_timestamp)]
    
    if window_data.empty:
        return None
    
    features = []
    time_indexed = sorted(window_data['Timestamp'].unique())
    
    for ts in time_indexed:
        ts_data = window_data[window_data['Timestamp'] == ts]
        for router in ['Router_A', 'Router_B', 'Router_C']:
            router_data = ts_data[ts_data['Device Name'] == router]
            if not router_data.empty:
                row = router_data.iloc[0]
                features.extend([
                    row['Traffic Volume (MB/s)'], row['Latency (ms)'],
                    row['Bandwidth Used (MB/s)'], row['Bandwidth Allocated (MB/s)'],
                    row['total_avg_app_traffic'], row['total_peak_app_traffic'],
                    row['Impact_encoded'], row['total_peak_user_usage'], row['total_logins']
                ])
            else:
                features.extend([0] * 9)
    
    return np.array(features).reshape(1, -1)

def predict_congestion_proba(df, models, target_timestamp):
    """Predict congestion probabilities for all routers at a target timestamp."""
    features = create_training_samples_single(df, target_timestamp)
    if features is None:
        return None
    
    proba_results = {}
    for router in ['Router_A', 'Router_B', 'Router_C']:
        model_key = router.split('_')[1]
        if models and model_key in models:
            proba = models[model_key].predict_proba(features)[0, 1]
            proba_results[router] = proba
        else:
            proba_results[router] = 0.0
    return proba_results

def bandwidth_recommendation(window_data, congestion_probs):
    """Generate bandwidth recommendations based on predictions and current usage."""
    recommendations = {}
    for router in ['Router_A', 'Router_B', 'Router_C']:
        router_data = window_data[window_data['Device Name'] == router]
        if router_data.empty:
            recommendations[router] = {'action': 'monitor', 'amount': 0, 'reason': 'No data in window', 'utilization': 0, 'congestion_prob': 0, 'current_allocated': 0, 'current_used': 0}
            continue

        prob = congestion_probs.get(router, 0.0)
        allocated = router_data['Bandwidth Allocated (MB/s)'].mean()
        used = router_data['Bandwidth Used (MB/s)'].mean()
        utilization = (used / allocated) if allocated > 0 else 0
        
        # Simplified and robust recommendation logic
        action, amount, reason = 'maintain', 0, f'NORMAL: Risk {prob:.1%} and utilization {utilization:.1%} are stable.'
        if prob >= 0.8:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.3, 1), f'CRITICAL: High congestion risk ({prob:.1%})'
        elif prob >= 0.6:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.2, 1), f'HIGH RISK: Proactive increase needed for risk {prob:.1%}'
        elif prob <= 0.2 and utilization < 0.4:
            action, amount, reason = 'decrease_bandwidth', round(-allocated * 0.1, 1), f'OPTIMIZE: Low risk ({prob:.1%}) and low utilization ({utilization:.1%})'
        elif utilization > 0.85:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.1, 1), f'PREVENTIVE: High utilization ({utilization:.1%})'

        recommendations[router] = {'action': action, 'amount': amount, 'reason': reason, 'utilization': utilization, 'congestion_prob': prob, 'current_allocated': allocated, 'current_used': used}
    return recommendations


# --- Visualization Functions ---
def create_visualizations(df, target_time, congestion_probs, recommendations):
    """Create all Plotly charts for the dashboard."""
    fig_traffic = px.line(df, x='Timestamp', y='Traffic Volume (MB/s)', color='Device Name', title='Traffic Volume Over Time')
    fig_traffic.add_vline(x=target_time, line_dash="dash", line_color="red", annotation_text="Prediction Point")

    prob_df = pd.DataFrame(list(congestion_probs.items()), columns=['Router', 'Probability'])
    fig_prob = px.bar(prob_df, x='Router', y='Probability', title='Congestion Probability', color='Router', range_y=[0,1])
    
    util_df = pd.DataFrame([{'Router': r, 'Utilization': rec['utilization']} for r, rec in recommendations.items()])
    fig_util = px.bar(util_df, x='Router', y='Utilization', title='Bandwidth Utilization', color='Router', range_y=[0,1])
    
    return fig_traffic, fig_prob, fig_util

# --- Main Streamlit App ---
def main():
    st.title("📊 Network Congestion Prediction Dashboard")
    st.markdown("Upload your network traffic data (CSV) to get congestion predictions and bandwidth recommendations.")

    models = load_models()
    le_impact = load_label_encoder()
    if not models or not le_impact:
        st.error("Application cannot start because a model or encoder failed to load.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # --- CRITICAL STEP: Clean and process the data ---
            df = preprocess_data(raw_df, le_impact)
            
            if df.empty:
                st.error("❌ No valid data found in the file after cleaning. Please check timestamp formats and data integrity.")
                st.stop()

            latest_time = df['Timestamp'].max()
            prediction_time = latest_time + timedelta(hours=1)
            
            st.success(f"✅ Data loaded and cleaned successfully! {len(df)} records are ready.")
            st.info(f"📅 Latest Data: **{latest_time.strftime('%Y-%m-%d %H:%M')}** | 🔮 Predicting for: **{prediction_time.strftime('%Y-%m-%d %H:%M')}**")
            
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Analyzing data and running predictions..."):
                    congestion_probs = predict_congestion_proba(df, models, prediction_time)
                    if congestion_probs is None:
                        st.error("❌ Not enough historical data to make a prediction. At least 12 hours of data prior to the prediction time is required.")
                    else:
                        window_start = latest_time - timedelta(hours=1)
                        window_data = df[df['Timestamp'] >= window_start]
                        recommendations = bandwidth_recommendation(window_data, congestion_probs)
                        
                        st.subheader("🎯 Congestion Prediction Results")
                        cols = st.columns(3)
                        for i, (router, prob) in enumerate(congestion_probs.items()):
                            color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
                            cols[i].metric(label=f"{color} {router}", value=f"{prob:.1%}")

                        st.subheader("💡 Bandwidth Recommendations")
                        for router, rec in recommendations.items():
                            with st.expander(f"**{router}** - Recommendation: **{rec['action'].replace('_', ' ').title()}**"):
                                st.metric("Recommended Change", f"{rec['amount']} MB/s")
                                st.write(f"**Reason:** {rec['reason']}")
                                st.progress(rec['utilization'], text=f"Current Utilization: {rec['utilization']:.1%}")
                        
                        st.subheader("📈 Visual Analysis")
                        fig_traffic, fig_prob, fig_util = create_visualizations(df, latest_time, congestion_probs, recommendations)
                        st.plotly_chart(fig_traffic, use_container_width=True)
                        col1, col2 = st.columns(2)
                        col1.plotly_chart(fig_prob, use_container_width=True)
                        col2.plotly_chart(fig_util, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred during file processing: {str(e)}")
            st.warning("Please ensure the CSV file is correctly formatted and contains the required columns.")

if __name__ == "__main__":
    main()
