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

# Load trained models
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost models for each router"""
    try:
        models = {}
        # This assumes the script is run from a location where `../models` is valid.
        # For more robust pathing, consider absolute paths or environment variables.
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        
        for router in ['A', 'B', 'C']:
            model_path = os.path.join(models_dir, f"model{router}_p.pkl")
            if os.path.exists(model_path):
                models[router] = joblib.load(model_path)
            else:
                 # Fallback for Streamlit Cloud or environments where pathing is tricky
                st.warning(f"Model file not found at {model_path}. Trying alternative path.")
                alt_path = f"model{router}_p.pkl"
                try:
                    models[router] = joblib.load(alt_path)
                except FileNotFoundError:
                     st.error(f"Could not find model file at primary or alternative path: {alt_path}")
                     return None
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load label encoder for Impact column
@st.cache_resource
def load_label_encoder():
    """Load or create label encoder for Impact column"""
    try:
        # Assuming the encoder is in the same directory as the script
        encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder_impact.pkl")
        if os.path.exists(encoder_path):
            return joblib.load(encoder_path)
        else:
            # Fallback for Streamlit Cloud
            alt_path = "label_encoder_impact.pkl"
            return joblib.load(alt_path)
    except FileNotFoundError:
        st.warning("Label encoder not found. Creating a default one.")
        le = LabelEncoder()
        le.fit(['None', 'Low', 'Medium', 'High'])  # Default categories
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
                    row['Traffic Volume (MB/s)'],
                    row['Latency (ms)'],
                    row['Bandwidth Used (MB/s)'],
                    row['Bandwidth Allocated (MB/s)'],
                    row['total_avg_app_traffic'],
                    row['total_peak_app_traffic'],
                    row['Impact_encoded'],
                    row['total_peak_user_usage'],
                    row['total_logins']
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
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
        model_key = router.split('_')[1]  # Convert Router_A to A
        if model_key in models:
            try:
                proba = models[model_key].predict_proba(features)[0, 1]
                proba_results[router] = proba
            except Exception as e:
                st.warning(f"Error predicting for {router}: {e}")
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
        
        utilization = (current_used / current_allocated) if current_allocated > 0 else 0
        
        action, amount, reason = 'maintain', 0, 'Default state'
        
        if congestion_prob >= 0.8:
            amount = min(current_allocated * 0.4, 50) if utilization >= 0.9 else current_allocated * 0.25
            action = 'increase_bandwidth'
            reason = f'CRITICAL: High congestion probability ({congestion_prob:.2f}) with {utilization:.1%} utilization'
        elif congestion_prob >= 0.6:
            if utilization >= 0.8:
                amount, action, reason = current_allocated * 0.2, 'increase_bandwidth', f'MODERATE RISK: Congestion ({congestion_prob:.2f}) with high utilization ({utilization:.1%})'
            elif avg_latency > 60:
                amount, action, reason = current_allocated * 0.15, 'increase_bandwidth', f'LATENCY CONCERN: High latency ({avg_latency:.1f}ms) with congestion risk ({congestion_prob:.2f})'
            else:
                action, reason = 'monitor_closely', f'WATCH: Medium congestion risk ({congestion_prob:.2f})'
        elif congestion_prob >= 0.4:
            if utilization >= 0.85:
                amount, action, reason = current_allocated * 0.1, 'increase_bandwidth', f'PREVENTIVE: High utilization ({utilization:.1%}) with moderate risk ({congestion_prob:.2f})'
            else:
                action, reason = 'monitor', f'NORMAL: Moderate risk ({congestion_prob:.2f})'
        elif congestion_prob <= 0.2:
            if utilization <= 0.4:
                amount, action, reason = -min(current_allocated * 0.15, 20), 'decrease_bandwidth', f'OPTIMIZE: Low utilization ({utilization:.1%}) and risk ({congestion_prob:.2f})'
            elif utilization <= 0.6:
                action, reason = 'maintain', f'EFFICIENT: Good utilization ({utilization:.1%}) with low risk ({congestion_prob:.2f})'
            else:
                action, reason = 'monitor', f'STABLE: Acceptable utilization ({utilization:.1%})'
        else:
            if utilization >= 0.8:
                amount, action, reason = current_allocated * 0.1, 'increase_bandwidth', f'PREVENTIVE: High utilization ({utilization:.1%})'
            else:
                action, reason = 'maintain', f'NORMAL: Balanced operation ({congestion_prob:.2f} risk)'

        recommendations[router] = {
            'action': action, 'amount': round(amount, 1), 'reason': reason,
            'utilization': utilization, 'congestion_prob': congestion_prob,
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

    router_names = list(congestion_probs.keys())
    prob_values = [p * 100 for p in congestion_probs.values()]
    colors = ['red' if p > 70 else 'orange' if p > 40 else 'green' for p in prob_values]
    fig_prob = go.Figure(data=[go.Bar(x=router_names, y=prob_values, marker_color=colors)])
    fig_prob.update_layout(title='Congestion Probability (%)', yaxis_title='Probability (%)')

    util_data = [{'Router': r, 'Utilization (%)': rec['utilization'] * 100} for r, rec in recommendations.items()]
    util_df = pd.DataFrame(util_data)
    fig_util = px.bar(util_df, x='Router', y='Utilization (%)', title='Current Bandwidth Utilization',
                      color='Utilization (%)', color_continuous_scale=['green', 'yellow', 'red'], range_color=[0,100])
    
    return fig_traffic, fig_prob, fig_util

# Main Dashboard
def main():
    st.title("📊 Network Congestion Prediction Dashboard")
    st.markdown("Upload your network traffic data to get congestion predictions and bandwidth recommendations.")
    
    models = load_models()
    if models is None:
        st.error("Models could not be loaded. The application cannot proceed.")
        st.stop()
    
    le_impact = load_label_encoder()
    
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV", type=['csv'],
        help="Upload CSV with required columns for traffic, latency, bandwidth, etc."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # --- FIX STARTS HERE ---
            # Define columns that must be numeric
            numeric_cols = [
                'Traffic Volume (MB/s)', 'Latency (ms)', 'Bandwidth Allocated (MB/s)',
                'Bandwidth Used (MB/s)', 'total_avg_app_traffic', 'total_peak_app_traffic',
                'total_logins', 'total_peak_user_usage', 'Num_Config_Changes', 'congestion Flag'
            ]
            
            # Coerce columns to numeric, turning errors into NaN, then fill NaN with 0
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
            # --- FIX ENDS HERE ---

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
            
            if df.empty:
                st.error("❌ No valid data with parseable timestamps found.")
                st.stop()
            
            if 'Impact' in df.columns:
                df['Impact'] = df['Impact'].fillna('None')
                # Use a mask to handle unseen labels gracefully
                known_labels = set(le_impact.classes_)
                df['Impact_encoded'] = df['Impact'].apply(lambda x: le_impact.transform([x])[0] if x in known_labels else 0)
            else:
                df['Impact_encoded'] = 0

            latest_time = df['Timestamp'].max()
            prediction_time = latest_time + timedelta(hours=1)
            
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            st.info(f"📅 Latest data timestamp: {latest_time.strftime('%Y-%m-%d %H:%M')}")
            st.info(f"🔮 Predicting congestion for: {prediction_time.strftime('%Y-%m-%d %H:%M')}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 Data Overview")
                st.dataframe(df.head())
            with col2:
                st.subheader("📊 Data Summary")
                summary_stats = df.groupby('Device Name').agg({
                    'Traffic Volume (MB/s)': ['mean', 'max'],
                    'Latency (ms)': ['mean', 'max'],
                    'Bandwidth Used (MB/s)': 'mean'
                }).round(2)
                st.dataframe(summary_stats)
            
            if st.button("🔄 Run Prediction", type="primary"):
                with st.spinner("Running predictions..."):
                    congestion_probs = predict_congestion_proba(df, models, prediction_time)

                    if congestion_probs is None:
                        st.error("❌ Not enough data for prediction. Need at least 12 hours of historical data.")
                    else:
                        window_start = latest_time - timedelta(hours=12)
                        window_data = df[(df['Timestamp'] >= window_start) & (df['Timestamp'] <= latest_time)]
                        recommendations = bandwidth_recommendation(window_data, congestion_probs)
                        
                        st.subheader(f"🎯 Congestion Predictions for {prediction_time.strftime('%Y-%m-%d %H:%M')}")
                        prob_cols = st.columns(3)
                        for i, (router, prob) in enumerate(congestion_probs.items()):
                            with prob_cols[i]:
                                color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
                                st.metric(label=f"{color} {router}", value=f"{prob:.1%}")
                        
                        st.subheader("💡 Bandwidth Recommendations")
                        for router, rec in recommendations.items():
                            with st.expander(f"{router} - **{rec['action'].replace('_', ' ').title()}**"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if rec['amount'] > 0:
                                        st.success(f"📈 Increase by {rec['amount']} MB/s")
                                    elif rec['amount'] < 0:
                                        st.info(f"📉 Decrease by {abs(rec['amount'])} MB/s")
                                    else:
                                        st.info("➡️ Maintain current allocation")
                                    st.write(f"**Utilization:** {rec['utilization']:.1%}")
                                    st.write(f"**Congestion Risk:** {rec['congestion_prob']:.1%}")
                                with col_b:
                                    st.write(f"**Current Allocated:** {rec['current_allocated']:.1f} MB/s")
                                    st.write(f"**Current Used:** {rec['current_used']:.1f} MB/s")
                                    st.write(f"**Reason:** {rec['reason']}")
                        
                        st.subheader("📊 Visualizations")
                        fig_traffic, fig_prob, fig_util = create_visualizations(df, latest_time, congestion_probs, recommendations)
                        st.plotly_chart(fig_traffic, use_container_width=True)
                        
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            st.plotly_chart(fig_prob, use_container_width=True)
                        with viz_col2:
                            st.plotly_chart(fig_util, use_container_width=True)
                        
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(), 'prediction_for': prediction_time,
                            'congestion_probs': congestion_probs, 'recommendations': recommendations
                        })
            
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                st.subheader("📜 Prediction History")
                history_data = []
                for entry in reversed(st.session_state.prediction_history[-5:]): # Last 5 predictions
                    for router, prob in entry['congestion_probs'].items():
                        history_data.append({
                            'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
                            'Prediction For': entry['prediction_for'].strftime('%Y-%m-%d %H:%M'),
                            'Router': router,
                            'Congestion Prob.': f"{prob:.1%}",
                            'Recommendation': entry['recommendations'][router]['action'].replace('_', ' ').title()
                        })
                if history_data:
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True)
                    if st.button("🗑️ Clear History"):
                        st.session_state.prediction_history = []
                        st.rerun()

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.warning("Please ensure your CSV file is correctly formatted with the required columns.")

if __name__ == "__main__":
    main()
