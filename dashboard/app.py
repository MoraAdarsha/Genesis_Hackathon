import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import os

# Page config
st.set_page_config(
    page_title="Network Congestion Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- HELPER FUNCTION to find file path reliably ---
def get_file_path(file_name):
    """Constructs a reliable path to a file, checking the script's directory first."""
    # Path relative to the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    primary_path = os.path.join(script_dir, file_name)
    
    if os.path.exists(primary_path):
        return primary_path
    
    # Fallback for environments where the file might be in the root
    if os.path.exists(file_name):
        return file_name
        
    return None # Return None if not found

# Load trained models
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost models for each router"""
    try:
        models = {}
        # Assumes models are in a 'models' subdirectory relative to the script
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        for router in ['A', 'B', 'C']:
            model_path = os.path.join(models_dir, f"model{router}_p.pkl")
            if not os.path.exists(model_path):
                 # If not in 'models' folder, check the root directory
                 model_path = f"model{router}_p.pkl"
                 if not os.path.exists(model_path):
                     st.error(f"FATAL: Model file 'model{router}_p.pkl' not found in any expected location.")
                     return None
            models[router] = joblib.load(model_path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load label encoder for Impact column
@st.cache_resource
def load_label_encoder():
    """Load label encoder from file or create a default one if not found."""
    encoder_path = get_file_path("label_encoder_impact.pkl")
    
    if encoder_path:
        return joblib.load(encoder_path)
    else:
        # This is the fallback logic that triggers your warning
        st.warning("Info: 'label_encoder_impact.pkl' not found. Creating a temporary encoder for this session.")
        le = LabelEncoder()
        # Fit with the expected categories from your data
        le.fit(['None', 'Low', 'Medium', 'High'])
        return le

def create_training_samples_single(data, target_timestamp, hours=12):
    """
    Create features for a single prediction using last 12 hours of data
    """
    if isinstance(target_timestamp, str):
        target_dt = pd.to_datetime(target_timestamp)
    else:
        target_dt = target_timestamp
    
    window_start = target_dt - timedelta(hours=hours)
    window_end = target_dt
    
    window_data = data[(data['Timestamp'] >= window_start) & (data['Timestamp'] < window_end)]
    
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
                features.extend([0] * 9) # Add 9 zero features if router data is missing
    
    return np.array(features).reshape(1, -1)

def predict_congestion_proba(df, models, target_timestamp):
    """
    Predict congestion probabilities for all routers at target timestamp
    """
    features = create_training_samples_single(df, target_timestamp)
    
    if features is None:
        return None
    
    proba_results = {}
    for router in ['Router_A', 'Router_B', 'Router_C']:
        model_key = router.split('_')[1]
        if model_key in models:
            try:
                proba = models[model_key].predict_proba(features)[0, 1]
                proba_results[router] = proba
            except Exception as e:
                st.warning(f"Could not predict for {router}: {e}")
                proba_results[router] = 0.0
        else:
            proba_results[router] = 0.0
    
    return proba_results

def bandwidth_recommendation(window_data, congestion_probs):
    """
    Generate bandwidth recommendations based on congestion probabilities and current usage
    """
    recommendations = {}
    
    for router in ['Router_A', 'Router_B', 'Router_C']:
        router_data = window_data[window_data['Device Name'] == router]
        
        if router_data.empty:
            recommendations[router] = {
                'action': 'monitor', 'amount': 0, 'reason': 'No data available in window',
                'utilization': 0, 'congestion_prob': 0, 'current_allocated': 0, 'current_used': 0
            }
            continue
        
        congestion_prob = congestion_probs.get(router, 0.0)
        
        current_allocated = router_data['Bandwidth Allocated (MB/s)'].mean()
        current_used = router_data['Bandwidth Used (MB/s)'].mean()
        avg_latency = router_data['Latency (ms)'].mean()
        
        utilization = (current_used / current_allocated) * 100 if current_allocated > 0 else 0
        
        action, amount, reason = 'maintain', 0, 'Default state'
        
        if congestion_prob >= 0.8:
            amount = min(current_allocated * 0.4, 50) if utilization >= 90 else current_allocated * 0.25
            action = 'increase_bandwidth'
            reason = f'CRITICAL: Congestion prob ({congestion_prob:.1%}) & utilization ({utilization:.0f}%)'
        elif congestion_prob >= 0.6:
            if utilization >= 80:
                amount, action, reason = current_allocated * 0.2, 'increase_bandwidth', f'HIGH RISK: Congestion ({congestion_prob:.1%}) & high utilization ({utilization:.0f}%)'
            elif avg_latency > 60:
                amount, action, reason = current_allocated * 0.15, 'increase_bandwidth', f'LATENCY: High latency ({avg_latency:.1f}ms) & risk ({congestion_prob:.1%})'
            else:
                action, reason = 'monitor_closely', f'WATCH: Medium congestion risk ({congestion_prob:.1%})'
        elif congestion_prob >= 0.4:
            if utilization >= 85:
                amount, action, reason = current_allocated * 0.1, 'increase_bandwidth', f'PREVENTIVE: High utilization ({utilization:.0f}%) with moderate risk ({congestion_prob:.1%})'
            else:
                action, reason = 'monitor', f'NORMAL: Moderate risk ({congestion_prob:.1%})'
        elif congestion_prob <= 0.2 and utilization <= 40:
                amount, action, reason = -min(current_allocated * 0.15, 20), 'decrease_bandwidth', f'OPTIMIZE: Low risk ({congestion_prob:.1%}) & utilization ({utilization:.0f}%)'
        else:
            action, reason = 'maintain', f'STABLE: System operating within normal parameters.'

        recommendations[router] = {
            'action': action, 'amount': round(amount, 1), 'reason': reason,
            'utilization': utilization / 100, 'congestion_prob': congestion_prob,
            'current_allocated': current_allocated, 'current_used': current_used
        }
    return recommendations

def create_visualizations(df, target_time, congestion_probs, recommendations):
    """Create visualizations for the dashboard"""
    target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')

    fig_traffic = px.line(df, x='Timestamp', y='Traffic Volume (MB/s)', 
                          color='Device Name', title='Traffic Volume Over Time')
    fig_traffic.add_vline(x=target_time_str, line_dash="dash", line_color="red", 
                          annotation_text="Prediction Point")

    prob_df = pd.DataFrame({
        'Router': list(congestion_probs.keys()),
        'Probability (%)': [p * 100 for p in congestion_probs.values()]
    })
    prob_df['Color'] = prob_df['Probability (%)'].apply(lambda p: 'red' if p > 70 else 'orange' if p > 40 else 'green')
    fig_prob = px.bar(prob_df, x='Router', y='Probability (%)', title='Congestion Probability', color='Color',
                      color_discrete_map={'red': '#FF4136', 'orange': '#FF851B', 'green': '#2ECC40'})

    util_df = pd.DataFrame([{'Router': r, 'Utilization (%)': rec['utilization'] * 100} for r, rec in recommendations.items()])
    fig_util = px.bar(util_df, x='Router', y='Utilization (%)', title='Current Bandwidth Utilization',
                      color='Utilization (%)', color_continuous_scale='RdYlGn_r', range_color=[0,100])
    
    return fig_traffic, fig_prob, fig_util

# Main Dashboard
def main():
    st.title("📊 Network Congestion Prediction Dashboard")
    st.markdown("Upload network traffic data to get congestion predictions and bandwidth recommendations.")
    
    models = load_models()
    if models is None:
        st.error("Models could not be loaded. The application cannot proceed.")
        st.stop()
    
    le_impact = load_label_encoder()
    
    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            numeric_cols = [
                'Traffic Volume (MB/s)', 'Latency (ms)', 'Bandwidth Used (MB/s)', 
                'Bandwidth Allocated (MB/s)', 'total_avg_app_traffic', 
                'total_peak_app_traffic', 'total_logins', 'total_peak_user_usage'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            df.sort_values('Timestamp', inplace=True)
            
            if df.empty:
                st.error("❌ No valid data with parseable timestamps found.")
                st.stop()
            
            if 'Impact' in df.columns:
                df['Impact'] = df['Impact'].fillna('None')
                known_labels = set(le_impact.classes_)
                df['Impact_encoded'] = df['Impact'].apply(lambda x: le_impact.transform([x])[0] if x in known_labels else 0)
            else:
                df['Impact_encoded'] = 0

            latest_time = df['Timestamp'].max()
            prediction_time = latest_time + timedelta(hours=1)
            
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            st.info(f"📅 Latest data: {latest_time:%Y-%m-%d %H:%M} | 🔮 Predicting for: {prediction_time:%Y-%m-%d %H:%M}")
            
            if st.button("🔄 Run Prediction", type="primary"):
                # All prediction logic happens inside this block now
                ...

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.warning("Please ensure your CSV file is correctly formatted.")

if __name__ == "__main__":
    main()
