# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import StringIO
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Walk-Forward Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #00ff88;
    }
    .stApp {
        background-color: #0a0a0a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #00ff88;
    }
    .stButton>button {
        background-color: #00ff88;
        color: #0a0a0a;
        font-weight: bold;
    }
    .stSelectbox>div>div {
        background-color: #1a1a2e;
        color: #00ff88;
    }
    h1, h2, h3 {
        color: #00ff88 !important;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🚀 Crypto Walk-Forward Strategy")
    st.caption("Meta-Labelling AI Trading System | Built at 4 AM during a hackathon")
with col2:
    st.metric("Status", "🟢 Live")
with col3:
    st.metric("Data", "BTC/USDT")

st.markdown("---")

# Load data function
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/harjasbb07-eng/BTC-crypto/refs/heads/main/BTC-Hourly.csv"
    response = requests.get(csv_url)
    df = pd.read_csv(StringIO(response.content.decode('utf-8')))
    df['timestamp'] = pd.to_datetime(df['unix'], unit='s')
    df = df.set_index('timestamp').sort_index()
    df = df.rename(columns={'Volume BTC': 'volume_btc', 'Volume USD': 'volume_usd'})
    return df

def compute_features(df):
    data = df.copy()
    data['returns'] = data['close'].pct_change()
    
    data['ema_fast'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=50, adjust=False).mean()
    data['primary_signal'] = np.where(data['ema_fast'] > data['ema_slow'], 1, -1)
    
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi_14'] = 100 - (100 / (1 + gain/loss))
    
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(14).mean()
    data['atr_ratio'] = data['atr_14'] / data['close']
    data['volatility_30d'] = data['returns'].rolling(24*30).std()
    
    data['volume_ratio'] = data['volume_btc'] / data['volume_btc'].rolling(24).mean()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    return data

class RobustMetaStrategy:
    def __init__(self):
        self.results = []
        
    def run_fold(self, train_years, test_year, df_full, feature_cols, progress_bar=None):
        df_train = df_full[df_full.index.year.isin(train_years)].copy()
        df_test = df_full[df_full.index.year == test_year].copy()
        
        if len(df_train) < 1000 or len(df_test) < 100:
            return None
        
        holding = 24
        future_ret = df_train['close'].shift(-holding) / df_train['close'] - 1
        df_train['meta_label'] = 0
        df_train.loc[(df_train['primary_signal'] == 1) & (future_ret > 0.002), 'meta_label'] = 1
        df_train.loc[(df_train['primary_signal'] == -1) & (future_ret < -0.002), 'meta_label'] = 1
        
        future_ret_test = df_test['close'].shift(-holding) / df_test['close'] - 1
        df_test['meta_label'] = 0
        df_test.loc[(df_test['primary_signal'] == 1) & (future_ret_test > 0.002), 'meta_label'] = 1
        df_test.loc[(df_test['primary_signal'] == -1) & (future_ret_test < -0.002), 'meta_label'] = 1
        
        train_sig = df_train[df_train['primary_signal'] != 0].copy()
        test_sig = df_test[df_test['primary_signal'] != 0].copy()
        
        if len(train_sig) < 100:
            return None
        
        X_train = train_sig[feature_cols]
        y_train = train_sig['meta_label']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBClassifier(
            tree_method='hist',
            n_estimators=100, learning_rate=0.05, max_depth=3,
            scale_pos_weight=10,
            random_state=42
        )
        
        split = int(len(X_train_scaled) * 0.8)
        model.fit(X_train_scaled[:split], y_train[:split], verbose=False)
        
        X_val = X_train_scaled[split:]
        y_val = y_train.iloc[split:]
        proba_val = model.predict_proba(X_val)[:, 1]
        
        best_thresh = 0.5
        best_score = -999
        for thresh in [0.4, 0.5, 0.6, 0.7]:
            preds = (proba_val > thresh).astype(int)
            if preds.sum() > 0:
                score = (preds == y_val.iloc[:len(preds)]).mean()
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        X_test = scaler.transform(test_sig[feature_cols])
        test_sig['meta_proba'] = model.predict_proba(X_test)[:, 1]
        
        metrics, equity = self.backtest(test_sig, best_thresh)
        bh = (df_test['close'].iloc[-1] / df_test['close'].iloc[0] - 1) * 100
        
        result = {
            'train_years': train_years, 'test_year': test_year,
            'return': metrics['return'], 'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'], 'trades': metrics['trades'],
            'buy_hold': bh, 'equity_df': equity, 'threshold': best_thresh,
            'model': model, 'scaler': scaler, 'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
        return result
    
    def backtest(self, df, threshold, cash=10000):
        position = 0
        entry_price = 0
        trades = 0
        equity = []
        
        for idx, row in df.iterrows():
            price = row['close']
            
            if position == 0 and row['meta_proba'] > threshold:
                if row['primary_signal'] == 1:
                    position = 1
                    entry_price = price
                    trades += 1
                elif row['primary_signal'] == -1:
                    position = -1
                    entry_price = price
                    trades += 1
            
            elif position != 0:
                exit_signal = (position == 1 and row['primary_signal'] == -1) or \
                             (position == -1 and row['primary_signal'] == 1)
                
                if exit_signal or row['meta_proba'] < threshold * 0.8:
                    pnl = (price - entry_price) / entry_price * position
                    cash *= (1 + pnl * 0.998)
                    position = 0
            
            val = cash if position == 0 else cash * (1 + (price - entry_price) / entry_price * position)
            equity.append({'time': idx, 'value': val})
        
        equity_df = pd.DataFrame(equity).set_index('time')
        returns = equity_df['value'].pct_change().dropna()
        
        total_ret = (equity_df['value'].iloc[-1] / 10000 - 1) * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(24*365) if returns.std() > 0 else 0
        peak = equity_df['value'].cummax()
        max_dd = ((peak - equity_df['value']) / peak).max() * 100
        
        return {'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd, 'trades': trades}, equity_df

# Sidebar controls
st.sidebar.header("⚙️ Strategy Parameters")

st.sidebar.markdown("### Walk-Forward Folds")
fold_options = st.sidebar.multiselect(
    "Select folds to run",
    ["2018→2019", "2018-19→2020", "2018-20→2021", "2018-21→2022"],
    default=["2018→2019", "2018-19→2020", "2018-20→2021", "2018-21→2022"]
)

st.sidebar.markdown("### Model Settings")
n_estimators = st.sidebar.slider("XGBoost Estimators", 50, 200, 100)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.1, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.info("💡 This strategy uses meta-labelling to filter EMA crossover signals")

# Main content
if st.button("🚀 Run Strategy", type="primary"):
    with st.spinner("Loading data and computing features..."):
        df = load_data()
        df = compute_features(df)
        feature_cols = ['rsi_14', 'atr_ratio', 'volume_ratio', 'hour', 'day_of_week', 'volatility_30d']
        df = df.dropna()
    
    st.success(f"✅ Data loaded: {len(df):,} samples from {df.index.min().date()} to {df.index.max().date()}")
    
    # Run strategy
    strategy = RobustMetaStrategy()
    all_results = []
    
    progress_bar = st.progress(0)
    fold_mapping = {
        "2018→2019": ([2018], 2019),
        "2018-19→2020": ([2018, 2019], 2020),
        "2018-20→2021": ([2018, 2019, 2020], 2021),
        "2018-21→2022": ([2018, 2019, 2020, 2021], 2022)
    }
    
    for i, fold_name in enumerate(fold_options):
        train_years, test_year = fold_mapping[fold_name]
        with st.spinner(f"Running fold: {fold_name}..."):
            res = strategy.run_fold(train_years, test_year, df, feature_cols)
            if res:
                all_results.append(res)
        progress_bar.progress((i + 1) / len(fold_options))
    
    if not all_results:
        st.error("❌ No results generated. Check data and parameters.")
        st.stop()
    
    # Charts
    st.markdown("---")
    st.subheader("📈 Equity Curves")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Walk-Forward Results", "2022 Out-of-Sample", "Feature Importance"])
    
    with tab1:
        fig = go.Figure()
        for r in all_results:
            if r['test_year'] != 2022:
                eq = r['equity_df']
                norm = (eq['value'] / 10000 - 1) * 100
                fig.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=f"{r['test_year']}: {r['return']:+.1f}% (Sharpe {r['sharpe']:.2f})",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Walk-Forward Performance (2019-2021)",
            xaxis_title="Date",
            yaxis_title="Return %",
            template="plotly_dark",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if any(r['test_year'] == 2022 for r in all_results):
            r_2022 = [r for r in all_results if r['test_year'] == 2022][0]
            eq = r_2022['equity_df']
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, subplot_titles=("Equity Curve", "Drawdown"))
            
            # Equity
            norm = (eq['value'] / 10000 - 1) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, 
                                    name="Strategy", line=dict(color="#00ff88", width=2)), row=1, col=1)
            
            # Buy and hold
            test_data = df[df.index.year == 2022]
            bh_norm = (test_data['close'] / test_data['close'].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=bh_norm.index, y=bh_norm.values, 
                                    name="Buy & Hold", line=dict(color="#ff6b6b", width=2)), row=1, col=1)
            
            # Drawdown
            peak = eq['value'].cummax()
            dd = (peak - eq['value']) / peak * 100
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, 
                                    name="Drawdown", fill='tozeroy', 
                                    fillcolor="rgba(255,0,0,0.3)", line=dict(color="red")),
                         row=2, col=1)
            
            fig.update_layout(
                title=f"2022 Out-of-Sample | Return: {r_2022['return']:+.2f}% | Sharpe: {r_2022['sharpe']:.2f}",
                template="plotly_dark",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select 2018-21→2022 fold to see 2022 results")
    
    with tab3:
        if all_results:
            # Get feature importance from last model
            last_result = all_results[-1]
            if 'feature_importance' in last_result:
                imp_df = pd.DataFrame({
                    'Feature': list(last_result['feature_importance'].keys()),
                    'Importance': list(last_result['feature_importance'].values())
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='viridis')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Results Table
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    
    results_df = pd.DataFrame([
        {
            'Year': r['test_year'],
            'Strategy Return': f"{r['return']:+.2f}%",
            'Buy & Hold': f"{r['buy_hold']:+.2f}%",
            'Alpha': f"{r['return'] - r['buy_hold']:+.2f}%",
            'Sharpe': f"{r['sharpe']:.2f}",
            'Max DD': f"{r['max_dd']:.1f}%",
            'Trades': r['trades']
        } for r in all_results
    ])
    st.dataframe(results_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ at 4 AM during a hackathon | 
        <a href='https://github.com/harjasbb07-eng/BTC-crypto'>GitHub Repo</a>
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial state
    st.info("👆 Click 'Run Strategy' to start the backtest")
    
    # Show sample data preview
    with st.expander("Preview Data"):
        df = load_data()
        st.write(f"Data shape: {df.shape}")
        st.write(df.head())