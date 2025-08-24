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
    page_icon="🧠",
    layout="wide"
)

# --- MOCK MODEL (PLACEHOLDER) ---
# This dummy model is used if real .pkl files are not found.
class MockModel:
    """A dummy model that mimics a real Scikit-learn model for demonstration purposes."""
    def predict_proba(self, features):
        # Generates a consistent but random-looking probability for demonstration.
        # This makes the mock predictions look plausible across runs.
        seed = int(np.sum(features) % 100)
        np.random.seed(seed)
        random_prob = np.random.rand()
        return np.array([[1 - random_prob, random_prob]])

# --- File Loading Functions ---
@st.cache_resource
def load_models():
    """
    Loads pre-trained models. If not found, it returns mock models and a warning flag.
    """
    models = {}
    are_models_mocked = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    for router in ['A', 'B', 'C']:
        model_path = os.path.join(models_dir, f"model{router}_p.pkl")
        if not os.path.exists(model_path):
            model_path = f"model{router}_p.pkl"  # Check root directory
        
        if not os.path.exists(model_path):
            st.warning(f"Warning: Model file 'model{router}_p.pkl' not found. Switching to Mock Prediction Mode.")
            are_models_mocked = True
            break
            
        try:
            models[router] = joblib.load(model_path)
        except Exception:
            are_models_mocked = True
            break
            
    if are_models_mocked:
        mock_model = MockModel()
        models = {'A': mock_model, 'B': mock_model, 'C': mock_model}

    return models, are_models_mocked

@st.cache_resource
def load_label_encoder():
    """Loads the label encoder or creates a default one if it's missing."""
    le = LabelEncoder()
    le.fit(['None', 'Low', 'Medium', 'High'])
    return le

# --- Core Data Processing and Prediction Logic ---
def preprocess_data(df):
    """Takes the raw DataFrame and cleans it completely."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    if df.empty: return df, None

    le_impact = load_label_encoder()
    numeric_cols = [
        'Traffic Volume (MB/s)', 'Latency (ms)', 'Bandwidth Allocated (MB/s)',
        'Bandwidth Used (MB/s)', 'total_avg_app_traffic', 'total_peak_app_traffic',
        'total_logins', 'total_peak_user_usage'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Impact' in df.columns:
        df['Impact'] = df['Impact'].fillna('None')
        known_labels = set(le_impact.classes_)
        df['Impact_encoded'] = df['Impact'].apply(lambda x: le_impact.transform([x])[0] if x in known_labels else 0)
    else:
        df['Impact_encoded'] = 0

    df.sort_values('Timestamp', inplace=True)
    return df

def create_training_samples_single(data, target_timestamp, hours=12):
    """Creates features for a single prediction."""
    window_start = target_timestamp - timedelta(hours=hours)
    window_data = data[(data['Timestamp'] >= window_start) & (data['Timestamp'] < target_timestamp)]
    if window_data.empty: return None
    
    features = []
    required_cols = [
        'Traffic Volume (MB/s)', 'Latency (ms)', 'Bandwidth Used (MB/s)',
        'Bandwidth Allocated (MB/s)', 'total_avg_app_traffic',
        'total_peak_app_traffic', 'Impact_encoded',
        'total_peak_user_usage', 'total_logins'
    ]
    
    time_indexed = sorted(window_data['Timestamp'].unique())
    for ts in time_indexed:
        ts_data = window_data[window_data['Timestamp'] == ts]
        for router in ['Router_A', 'Router_B', 'Router_C']:
            router_data = ts_data[ts_data['Device Name'] == router]
            features.extend(router_data.iloc[0][required_cols].values if not router_data.empty else [0] * len(required_cols))
    return np.array(features).reshape(1, -1)

def predict_congestion_proba(df, models, target_timestamp):
    """Predicts congestion probabilities for all routers."""
    features = create_training_samples_single(df, target_timestamp)
    if features is None: return None
    
    return {
        router: models[router.split('_')[1]].predict_proba(features)[0, 1]
        for router in ['Router_A', 'Router_B', 'Router_C']
    }

def bandwidth_recommendation(window_data, congestion_probs):
    """Generates bandwidth recommendations."""
    recommendations = {}
    for router in ['Router_A', 'Router_B', 'Router_C']:
        router_data = window_data[window_data['Device Name'] == router]
        if router_data.empty:
            recommendations[router] = {'action': 'monitor', 'amount': 0, 'reason': 'No data in window', 'utilization': 0}
            continue

        prob = congestion_probs.get(router, 0.0)
        allocated = router_data['Bandwidth Allocated (MB/s)'].mean()
        used = router_data['Bandwidth Used (MB/s)'].mean()
        utilization = (used / allocated) if allocated > 0 else 0
        
        action, amount, reason = 'maintain', 0, f'NORMAL: Risk {prob:.1%} and utilization {utilization:.1%} are stable.'
        if prob >= 0.8:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.3, 1), f'CRITICAL: High congestion risk ({prob:.1%})'
        elif prob >= 0.6:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.2, 1), f'HIGH RISK: Proactive increase for risk {prob:.1%}'
        elif prob <= 0.2 and utilization < 0.4:
            action, amount, reason = 'decrease_bandwidth', round(-allocated * 0.1, 1), f'OPTIMIZE: Low risk ({prob:.1%}) & low utilization ({utilization:.1%})'
        elif utilization > 0.85:
            action, amount, reason = 'increase_bandwidth', round(allocated * 0.1, 1), f'PREVENTIVE: High utilization ({utilization:.1%})'

        recommendations[router] = {'action': action, 'amount': amount, 'reason': reason, 'utilization': utilization}
    return recommendations

# --- NEW VISUALIZATION FUNCTION ---
def create_historical_chart(df, history):
    """Creates a chart comparing historical predictions to actual traffic."""
    if not history:
        return go.Figure()

    fig = px.line(df, x='Timestamp', y='Traffic Volume (MB/s)', color='Device Name',
                  title='Historical Performance: Actual Traffic vs. Predicted Risk')

    # Prepare prediction data for plotting
    pred_data = []
    for entry in history:
        for router, prob in entry['congestion_probs'].items():
            pred_data.append({
                'Timestamp': entry['prediction_for'],
                'Router': router,
                'Risk': prob
            })
    pred_df = pd.DataFrame(pred_data)

    # Add prediction markers to the plot
    for router_name in pred_df['Router'].unique():
        router_preds = pred_df[pred_df['Router'] == router_name]
        fig.add_trace(go.Scatter(
            x=router_preds['Timestamp'],
            y=np.zeros(len(router_preds)), # Plot on the zero-line for visibility
            mode='markers',
            marker=dict(
                size=12,
                color=router_preds['Risk'],
                colorscale='RdYlGn_r', # Red-Yellow-Green (reversed)
                cmin=0, cmax=1,
                colorbar=dict(title="Congestion Risk"),
                symbol='diamond'
            ),
            name=f'{router_name} Prediction'
        ))
    return fig

# --- Main Streamlit App ---
def main():
    st.title("🧠 Network Congestion Intelligence")

    models, are_models_mocked = load_models()
    if are_models_mocked:
        st.error("⚠️ **DEMO MODE**: Real predictive models (`.pkl` files) not found. Displaying random predictions.")

    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = preprocess_data(raw_df)
            
            if df.empty:
                st.error("❌ No valid data found after cleaning. Check timestamp formats.")
                st.stop()

            st.session_state.df = df # Store full dataframe for later use
            latest_time = df['Timestamp'].max()
            prediction_time = latest_time + timedelta(hours=1)
            
            st.success(f"✅ Data loaded! {len(df)} records ready. Predicting for **{prediction_time.strftime('%Y-%m-%d %H:%M')}**")
            
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Analyzing data..."):
                    congestion_probs = predict_congestion_proba(df, models, prediction_time)
                    if congestion_probs is None:
                        st.error("❌ Not enough historical data (need 12 hours).")
                    else:
                        st.session_state.last_prediction = {
                            'congestion_probs': congestion_probs,
                            'prediction_for': prediction_time,
                            'timestamp': datetime.now()
                        }
                        # Add to history
                        st.session_state.prediction_history.append(st.session_state.last_prediction)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Display results if a prediction has been made
    if 'last_prediction' in st.session_state:
        last_pred = st.session_state.last_prediction
        st.header(f"Results for {last_pred['prediction_for'].strftime('%Y-%m-%d %H:%M')}", divider='rainbow')

        st.subheader("🎯 Congestion Prediction")
        cols = st.columns(3)
        for i, (router, prob) in enumerate(last_pred['congestion_probs'].items()):
            color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
            cols[i].metric(label=f"{color} {router}", value=f"{prob:.1%}")

        st.subheader("💡 Bandwidth Recommendations")
        window_start = last_pred['prediction_for'] - timedelta(hours=2)
        window_end = last_pred['prediction_for'] - timedelta(hours=1)
        window_data = st.session_state.df[(st.session_state.df['Timestamp'] >= window_start) & (st.session_state.df['Timestamp'] <= window_end)]
        recommendations = bandwidth_recommendation(window_data, last_pred['congestion_probs'])
        
        for router, rec in recommendations.items():
            with st.expander(f"**{router}** - Action: **{rec['action'].replace('_', ' ').title()}**"):
                st.metric("Recommended Change", f"{rec['amount']} MB/s")
                st.write(f"**Reason:** {rec['reason']}")
                st.progress(rec['utilization'], text=f"Current Utilization: {rec['utilization']:.1%}")
    
    # Display historical chart if history exists
    if st.session_state.prediction_history:
        st.header("📈 Historical Performance", divider='rainbow')
        st.info("This chart shows actual traffic volume over time. The diamonds on the bottom represent the predicted congestion risk for the following hour.")
        fig_hist = create_historical_chart(st.session_state.df, st.session_state.prediction_history)
        st.plotly_chart(fig_hist, use_container_width=True)


if __name__ == "__main__":
    main()
