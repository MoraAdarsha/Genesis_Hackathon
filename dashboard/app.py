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
        current_dir = os.path.dirname(os.path.abspath(__file__))  # dashboard foldergi
        parent_dir = os.path.dirname(current_dir)  # project_folder
        models_dir = os.path.join(parent_dir, "models")
        
        for router in ['A', 'B', 'C']:
            model_path = os.path.join(models_dir, f"model{router}_p.pkl")
            models[router] = joblib.load(model_path)
        return models
    except FileNotFoundError as e:
        st.error(f"Model files not found at {models_dir}: {e}")
        return None

# Load label encoder for Impact column
@st.cache_resource
def load_label_encoder():
    """Load or create label encoder for Impact column"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        encoder_path = os.path.join(current_dir, "label_encoder_impact.pkl")
        return joblib.load(encoder_path)
    except FileNotFoundError:
        # Create a default encoder if file doesn't exist
        le = LabelEncoder()
        le.fit(['None', 'Low', 'Medium', 'High'])  # Default categories
        return le

def create_training_samples_single(data, target_timestamp, hours=12):
    """
    Create features for a single prediction using last 12 hours of data
    """
    # Convert target_timestamp to datetime for arithmetic
    if isinstance(target_timestamp, str):
        target_dt = pd.to_datetime(target_timestamp).to_pydatetime()
    elif isinstance(target_timestamp, pd.Timestamp):
        target_dt = target_timestamp.to_pydatetime()
    else:
        target_dt = target_timestamp
    
    # Use Python datetime arithmetic
    window_start_dt = target_dt - timedelta(hours=hours)
    window_end_dt = target_dt
    
    # Convert back to pandas timestamps for filtering
    window_start = pd.to_datetime(window_start_dt)
    window_end = pd.to_datetime(window_end_dt)
    
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
                'action': 'monitor',
                'amount': 0,
                'reason': 'No data available in window',
                'utilization': 0,
                'congestion_prob': 0,
                'current_allocated': 0,
                'current_used': 0
            }
            continue
        
        congestion_prob = congestion_probs.get(router, 0.0)
        
        # Calculate key metrics
        current_allocated = router_data['Bandwidth Allocated (MB/s)'].mean()
        current_used = router_data['Bandwidth Used (MB/s)'].mean()
        avg_latency = router_data['Latency (ms)'].mean()
        
        # Calculate utilization percentage
        utilization = (current_used / current_allocated) if current_allocated > 0 else 0
        
        # Decision logic
        if congestion_prob >= 0.8:
            if utilization >= 0.9:
                amount = min(current_allocated * 0.4, 50)
                action = 'increase_bandwidth'
                reason = f'CRITICAL: High congestion probability ({congestion_prob:.2f}) with {utilization:.1%} utilization'
            else:
                amount = current_allocated * 0.25
                action = 'increase_bandwidth'
                reason = f'HIGH RISK: Congestion probability ({congestion_prob:.2f}) requires bandwidth increase'
        elif congestion_prob >= 0.6:
            if utilization >= 0.8:
                amount = current_allocated * 0.2
                action = 'increase_bandwidth'
                reason = f'MODERATE RISK: Congestion probability ({congestion_prob:.2f}) with high utilization ({utilization:.1%})'
            elif avg_latency > 60:
                amount = current_allocated * 0.15
                action = 'increase_bandwidth'
                reason = f'LATENCY CONCERN: High latency ({avg_latency:.1f}ms) with congestion risk ({congestion_prob:.2f})'
            else:
                amount = 0
                action = 'monitor_closely'
                reason = f'WATCH: Medium congestion risk ({congestion_prob:.2f}) - monitor for changes'
        elif congestion_prob >= 0.4:
            if utilization >= 0.85:
                amount = current_allocated * 0.1
                action = 'increase_bandwidth'
                reason = f'PREVENTIVE: High utilization ({utilization:.1%}) with moderate risk ({congestion_prob:.2f})'
            else:
                amount = 0
                action = 'monitor'
                reason = f'NORMAL: Moderate risk ({congestion_prob:.2f}) within acceptable range'
        elif congestion_prob <= 0.2:
            if utilization <= 0.4:
                amount = -min(current_allocated * 0.15, 20)
                action = 'decrease_bandwidth'
                reason = f'OPTIMIZE: Low utilization ({utilization:.1%}) and low risk ({congestion_prob:.2f})'
            elif utilization <= 0.6:
                amount = 0
                action = 'maintain'
                reason = f'EFFICIENT: Good utilization ({utilization:.1%}) with low risk ({congestion_prob:.2f})'
            else:
                amount = 0
                action = 'monitor'
                reason = f'STABLE: Acceptable utilization ({utilization:.1%}) with low risk'
        else:
            if utilization >= 0.8:
                amount = current_allocated * 0.1
                action = 'increase_bandwidth'
                reason = f'PREVENTIVE: High utilization ({utilization:.1%}) requires small increase'
            else:
                amount = 0
                action = 'maintain'
                reason = f'NORMAL: Balanced operation with {congestion_prob:.2f} risk and {utilization:.1%} utilization'
        
        amount = round(amount, 1)
        
        recommendations[router] = {
            'action': action,
            'amount': amount,
            'reason': reason,
            'utilization': utilization,
            'congestion_prob': congestion_prob,
            'current_allocated': current_allocated,
            'current_used': current_used
        }
    
    return recommendations

def create_visualizations(df, target_time, congestion_probs, recommendations):
    """Create visualizations for the dashboard"""
    # Convert target_time to string for Plotly add_vline
    if hasattr(target_time, "strftime"):
        target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        target_time_str = str(target_time)
    fig_traffic = px.line(df, x='Timestamp', y='Traffic Volume (MB/s)', 
                         color='Device Name', title='Traffic Volume Over Time')
    fig_traffic.add_vline(x=target_time_str, line_dash="dash", line_color="red", 
                         annotation_text="Prediction Point")
    
    
    # 2. Congestion Probabilities
    router_names = list(congestion_probs.keys())
    prob_values = [congestion_probs[router] * 100 for router in router_names]
    
    fig_prob = go.Figure(data=[
        go.Bar(x=router_names, y=prob_values,
               marker_color=['red' if p > 70 else 'orange' if p > 40 else 'green' 
                           for p in prob_values])
    ])
    fig_prob.update_layout(title='Congestion Probability (%)', 
                          yaxis_title='Probability (%)')
    
    # 3. Bandwidth Utilization
    utilization_data = []
    for router, rec in recommendations.items():
        utilization_data.append({
            'Router': router,
            'Utilization (%)': rec['utilization'] * 100,
            'Allocated (MB/s)': rec['current_allocated'],
            'Used (MB/s)': rec['current_used']
        })
    
    util_df = pd.DataFrame(utilization_data)
    fig_util = px.bar(util_df, x='Router', y='Utilization (%)', 
                     title='Current Bandwidth Utilization',
                     color='Utilization (%)',
                     color_continuous_scale=['green', 'yellow', 'red'])
    
    return fig_traffic, fig_prob, fig_util

# Main Dashboard
def main():
    st.title("📊 Network Congestion Prediction Dashboard")
    st.markdown("Upload your network traffic data to get congestion predictions and bandwidth recommendations.")
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    le_impact = load_label_encoder()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=['csv'],
        help="Upload CSV with columns: Timestamp, Device Name, Traffic Volume (MB/s), Latency (ms), Bandwidth Used (MB/s), Bandwidth Allocated (MB/s), total_avg_app_traffic, total_peak_app_traffic, Impact, total_peak_user_usage, total_logins"
    )
    
# Load and process data
            df = pd.read_csv(uploaded_file)
            
            # 1. Convert Timestamp column
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # 2. Define and convert all expected numeric columns
            numeric_cols = [
                'Traffic Volume (MB/s)', 'Latency (ms)', 
                'Bandwidth Used (MB/s)', 'Bandwidth Allocated (MB/s)',
                'total_avg_app_traffic', 'total_peak_app_traffic',
                'total_peak_user_usage', 'total_logins'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 3. Drop rows where essential columns failed to parse
            df = df.dropna(subset=['Timestamp'] + numeric_cols)
            
            if df.empty:
                st.error("❌ No valid data found. Please check your timestamp format.")
                st.stop()
            
            # Handle Impact column encoding
            if 'Impact' in df.columns:
                df['Impact'] = df['Impact'].fillna('None')
                try:
                    df['Impact_encoded'] = le_impact.transform(df['Impact'])
                except ValueError:
                    # Handle unseen categories
                    df['Impact_encoded'] = 0
            else:
                df['Impact_encoded'] = 0
            
            # Get latest timestamp and create prediction window using datetime arithmetic
            latest_time = df['Timestamp'].max()
            
            # Convert to datetime for arithmetic
            if isinstance(latest_time, pd.Timestamp):
                latest_dt = latest_time.to_pydatetime()
            else:
                latest_dt = pd.to_datetime(latest_time).to_pydatetime()
            
            # Calculate prediction time
            prediction_dt = latest_dt + timedelta(hours=1)
            prediction_time = pd.to_datetime(prediction_dt)
            
            st.success(f"✅ Data loaded successfully! {len(df)} records")
            st.info(f"📅 Latest  {latest_time}")
            st.info(f"🔮 Predicting congestion for: {prediction_time}")
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Data Overview")
                st.dataframe(df.head())
                
                # Show data summary
                st.subheader("📊 Data Summary")
                summary_stats = df.groupby('Device Name').agg({
                    'Traffic Volume (MB/s)': ['mean', 'max'],
                    'Latency (ms)': ['mean', 'max'],
                    'Bandwidth Used (MB/s)': 'mean'
                }).round(2)
                st.dataframe(summary_stats)
            
            with col2:
                st.subheader("⚙️ Prediction Settings")
                st.write(f"**Prediction Window:** Next 1 hour")
                st.write(f"**Historical Window:** Last 12 hours")
                st.write(f"**Routers:** Router_A, Router_B, Router_C")
            
            if st.button("🔄 Run Prediction", type="primary"):
                with st.spinner("Running predictions..."):
                    # Make predictions
                    congestion_probs = predict_congestion_proba(df, models, prediction_time)

                    if congestion_probs is None:
                        st.error("❌ Not enough data for prediction. Need at least 12 hours of historical data.")
                    else:
                        # Get window data for recommendations using datetime arithmetic
                        window_start_dt = latest_dt - timedelta(hours=12)
                        window_start = pd.to_datetime(window_start_dt)
                        
                        window_data = df[(df['Timestamp'] >= window_start) & 
                                       (df['Timestamp'] <= latest_time)]
                        
                        # Get recommendations
                        recommendations = bandwidth_recommendation(window_data, congestion_probs)
                        
                        # Display results
                        st.subheader(f"🎯 Congestion Predictions for {prediction_time}")
                        
                        # Create metrics
                        prob_cols = st.columns(3)
                        for i, (router, prob) in enumerate(congestion_probs.items()):
                            with prob_cols[i]:
                                color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
                                st.metric(
                                    label=f"{color} {router}",
                                    value=f"{prob:.1%}",
                                    delta=None
                                )
                        
                        # Recommendations
                        st.subheader("💡 Bandwidth Recommendations")
                        for router, rec in recommendations.items():
                            with st.expander(f"{router} - {rec['action'].replace('_', ' ').title()}"):
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
                        
                        # Visualizations
                        st.subheader("📊 Visualizations")
                        fig_traffic, fig_prob, fig_util = create_visualizations(
                            df, prediction_time, congestion_probs, recommendations # <-- Use prediction_time here
                        )
                        
                        st.plotly_chart(fig_traffic, use_container_width=True)
                        
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            st.plotly_chart(fig_prob, use_container_width=True)
                        with viz_col2:
                            st.plotly_chart(fig_util, use_container_width=True)
                        
                        # Store in session state for history
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'prediction_for': prediction_time,
                            'congestion_probs': congestion_probs,
                            'recommendations': recommendations
                        })
            
            # Prediction History
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                st.subheader("📜 Prediction History")
                history_data = []
                
                for entry in st.session_state.prediction_history[-10:]:  # Last 10 predictions
                    for router, prob in entry['congestion_probs'].items():
                        history_data.append({
                            'Timestamp': entry['timestamp'],
                            'Prediction For': entry['prediction_for'],
                            'Router': router,
                            'Congestion Probability': f"{prob:.1%}",
                            'Recommendation': entry['recommendations'][router]['action']
                        })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please check that your CSV has the required columns:")
            st.code("""
Required columns:
- Timestamp
- Device Name (Router_A, Router_B, Router_C)
- Traffic Volume (MB/s)
- Latency (ms)  
- Bandwidth Used (MB/s)
- Bandwidth Allocated (MB/s)
- total_avg_app_traffic
- total_peak_app_traffic
- Impact
- total_peak_user_usage
- total_logins
""")

if __name__ == "__main__":
    main()
