import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings("ignore")

# Prophet import with error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet not installed. Install with: pip install prophet")

# ==================== DATA FETCHING ====================
def load_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ==================== DATA PREPROCESSING ====================
def prepare_data(data):
    """Prepare data with time-based and seasonal features"""
    df = data.copy()
    df.rename(columns={"Close": "Price"}, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Price", "Volume"]]
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    # Time features
    df["t"] = np.arange(1, len(df) + 1)
    df["t_square"] = df["t"] ** 2
    df["log_Price"] = np.log(df["Price"])
    
    # Seasonal features
    df["Sin"] = np.sin((2 * np.pi * df["t"]) / 365.25)
    df["Cos"] = np.cos((2 * np.pi * df["t"]) / 365.25)
    
    return df

# ==================== MODEL TRAINING ====================
def train_all_models(train_data, test_data):
    """Train all trend and seasonality models and return best model"""
    models = {}
    rmse_scores = {}
    
    # Linear Trend
    models['linear'] = smf.ols("Price ~ t", data=train_data).fit()
    pred_linear = models['linear'].predict(test_data["t"])
    rmse_scores['rmse_linear'] = root_mean_squared_error(test_data["Price"], pred_linear)
    
    # Exponential Trend
    models['exp'] = smf.ols("log_Price ~ t", data=train_data).fit()
    pred_exp = models['exp'].predict(test_data["t"])
    rmse_scores['rmse_Exp'] = root_mean_squared_error(test_data["Price"], np.exp(pred_exp))
    
    # Quadratic Trend
    models['quad'] = smf.ols("Price ~ t + t_square", data=train_data).fit()
    pred_quad = models['quad'].predict(test_data[["t", "t_square"]])
    rmse_scores['rmse_Quad'] = root_mean_squared_error(test_data["Price"], pred_quad)
    
    # Additive Seasonality
    models['add_sea'] = smf.ols('Price ~ Sin + Cos', data=train_data).fit()
    pred_add_sea = models['add_sea'].predict(test_data)
    rmse_scores['rmse_add_sea'] = root_mean_squared_error(test_data['Price'], pred_add_sea)
    
    # Multiplicative Seasonality
    models['mul_sea'] = smf.ols('log_Price ~ Sin + Cos', data=train_data).fit()
    pred_mul_sea = models['mul_sea'].predict(test_data)
    rmse_scores['rmse_Mult_sea'] = root_mean_squared_error(test_data['Price'], np.exp(pred_mul_sea))
    
    # Additive Seasonality + Quadratic Trend
    models['add_sea_quad'] = smf.ols('Price ~ t + t_square + Sin + Cos', data=train_data).fit()
    pred_add_sea_quad = models['add_sea_quad'].predict(test_data)
    rmse_scores['rmse_add_sea_quad'] = root_mean_squared_error(test_data['Price'], pred_add_sea_quad)
    
    # Multiplicative Seasonality + Linear Trend
    models['mul_add_sea'] = smf.ols('log_Price ~ t + Sin + Cos', data=train_data).fit()
    pred_mul_add_sea = models['mul_add_sea'].predict(test_data)
    rmse_scores['rmse_Mult_add_sea'] = root_mean_squared_error(test_data['Price'], np.exp(pred_mul_add_sea))
    
    # Find best model
    best_model_name = min(rmse_scores, key=rmse_scores.get)
    model_mapping = {
        'rmse_linear': 'linear',
        'rmse_Exp': 'exp',
        'rmse_Quad': 'quad',
        'rmse_add_sea': 'add_sea',
        'rmse_Mult_sea': 'mul_sea',
        'rmse_add_sea_quad': 'add_sea_quad',
        'rmse_Mult_add_sea': 'mul_add_sea'
    }
    
    best_model = models[model_mapping[best_model_name]]
    
    return best_model, rmse_scores, best_model_name, models

# ==================== RESIDUAL ANALYSIS ====================
def calculate_residuals(data, model):
    """Calculate residuals from the best model"""
    residuals = data["Price"] - model.predict(data)
    return residuals

def create_acf_pacf_plots(residuals):
    """Create ACF and PACF plots"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ACF (Autocorrelation)', 'PACF (Partial Autocorrelation)')
    )
    
    # Note: This is a simplified version - in production, use statsmodels plots
    lags = 12
    acf_values = [residuals.autocorr(lag=i) for i in range(lags + 1)]
    
    fig.add_trace(
        go.Bar(x=list(range(lags + 1)), y=acf_values, name='ACF'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=list(range(lags + 1)), y=acf_values, name='PACF'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, template='plotly_dark')
    return fig

# ==================== FORECASTING ====================
def create_future_dataframe(data, forecast_days=365):
    """Create future dates and features"""
    last_date = pd.to_datetime(data["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    future_df = pd.DataFrame({'Date': future_dates})
    future_df["t"] = np.arange(data.shape[0] + 1, data.shape[0] + forecast_days + 1)
    future_df["t_square"] = future_df["t"] ** 2
    future_df["Sin"] = np.sin((2 * np.pi * future_df["t"]) / 365.25)
    future_df["Cos"] = np.cos((2 * np.pi * future_df["t"]) / 365.25)
    
    return future_df

def forecast_with_autoreg(best_model, data, future_df, residuals):
    """Forecast using AutoRegression"""
    pred_trend = pd.Series(best_model.predict(future_df))
    
    # AutoReg model
    model_ar = AutoReg(residuals, lags=[1])
    model_fit = model_ar.fit()
    
    pred_res = model_fit.predict(
        start=len(residuals),
        end=len(residuals) + len(future_df) - 1,
        dynamic=False
    )
    pred_res.reset_index(drop=True, inplace=True)
    
    final_pred = pred_trend + pred_res
    future_df["AutoReg_Price"] = final_pred.values
    
    return future_df, model_fit

def forecast_with_arima(best_model, data, future_df, residuals, order=(1,1,1)):
    """Forecast using ARIMA"""
    pred_trend = pd.Series(best_model.predict(future_df))
    
    # ARIMA model
    model = ARIMA(residuals, order=order)
    results = model.fit()
    
    forecast = results.predict(
        start=len(residuals),
        end=len(residuals) + len(future_df) - 1,
        typ='levels'
    )
    forecast = pd.Series(forecast.values)
    forecast.reset_index(drop=True, inplace=True)
    
    final_pred = pred_trend + forecast
    future_df["ARIMA_Price"] = final_pred.values
    
    return future_df, results

def forecast_with_sarimax(best_model, data, future_df, residuals, order=(1,1,1)):
    """Forecast using SARIMAX"""
    pred_trend = pd.Series(best_model.predict(future_df))
    
    # SARIMAX model
    model = SARIMAX(residuals, order=order)
    results = model.fit(disp=False)
    
    forecast = results.predict(
        start=len(residuals),
        end=len(residuals) + len(future_df) - 1,
        typ='levels'
    )
    forecast = pd.Series(forecast.values)
    forecast.reset_index(drop=True, inplace=True)
    
    final_pred = pred_trend + forecast
    future_df["SARIMAX_Price"] = final_pred.values
    
    return future_df, results

def forecast_with_prophet(data, forecast_days=365):
    """Forecast using Facebook Prophet"""
    if not PROPHET_AVAILABLE:
        st.error("Prophet not available. Please install: pip install prophet")
        return None, None
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = data[['Date', 'Price']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Initialize and fit model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    
    with st.spinner("Training Prophet model..."):
        model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract only future predictions
    future_forecast = forecast[len(data):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_forecast.columns = ['Date', 'Prophet_Price', 'Prophet_Lower', 'Prophet_Upper']
    future_forecast.reset_index(drop=True, inplace=True)
    
    return future_forecast, model, forecast

# ==================== VISUALIZATION ====================
def plot_candlestick(data, ticker):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Price'],
        name='OHLC',
        increasing_line_color='#00ff41',
        decreasing_line_color='#ff0266'
    )])
    
    fig.update_layout(
        title=f'{ticker} - Candlestick Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def plot_prophet_components(prophet_model, forecast):
    """Plot Prophet model components (trend, seasonality)"""
    if not PROPHET_AVAILABLE or prophet_model is None:
        return None
    
    from prophet.plot import plot_components_plotly
    
    fig = plot_components_plotly(prophet_model, forecast)
    fig.update_layout(
        template='plotly_dark',
        height=800
    )
    
    return fig

def plot_price_trends(data, ticker):
    """Plot historical price with trend"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='cyan', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{ticker} - Historical Price Trend',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_model_comparison(rmse_scores):
    """Plot RMSE comparison of all models"""
    models = list(rmse_scores.keys())
    scores = list(rmse_scores.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            marker_color=['#00ff41' if s == min(scores) else '#ff0266' for s in scores],
            text=[f'{s:.2f}' for s in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Model Performance Comparison (RMSE)',
        xaxis_title='Model',
        yaxis_title='RMSE',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_forecast_comparison(data, forecast_df, ticker, prophet_forecast=None):
    """Plot all forecasting methods comparison"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='cyan', width=2)
    ))
    
    # AutoReg forecast
    if 'AutoReg_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['AutoReg_Price'],
            mode='lines',
            name='AutoReg Forecast',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    # ARIMA forecast
    if 'ARIMA_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['ARIMA_Price'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='magenta', width=2, dash='dot')
        ))
    
    # SARIMAX forecast
    if 'SARIMAX_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['SARIMAX_Price'],
            mode='lines',
            name='SARIMAX Forecast',
            line=dict(color='yellow', width=2, dash='dashdot')
        ))
    
    # Prophet forecast
    if prophet_forecast is not None and 'Prophet_Price' in prophet_forecast.columns:
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=prophet_forecast['Date'],
            y=prophet_forecast['Prophet_Upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=prophet_forecast['Date'],
            y=prophet_forecast['Prophet_Lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 255, 100, 0.2)',
            fill='tonexty',
            name='Prophet Confidence',
            showlegend=True
        ))
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=prophet_forecast['Date'],
            y=prophet_forecast['Prophet_Price'],
            mode='lines',
            name='Prophet Forecast',
            line=dict(color='lime', width=2.5)
        ))
    
    fig.update_layout(
        title=f'{ticker} - Forecast Comparison (Next {len(forecast_df)} Days)',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_residuals_analysis(residuals, data):
    """Plot residual analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals Over Time',
            'Residual Distribution',
            'Q-Q Plot',
            'Residuals vs Fitted'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(x=data['Date'], y=residuals, mode='lines',
                   name='Residuals', line=dict(color='yellow', width=1)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=residuals, name='Distribution',
                     marker=dict(color='orange'), nbinsx=30),
        row=1, col=2
    )
    
    # Q-Q plot approximation
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                   mode='markers', name='Q-Q',
                   marker=dict(color='cyan', size=4)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                   mode='lines', name='Reference',
                   line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # Residuals vs Fitted
    fitted = data['Price'] - residuals
    fig.add_trace(
        go.Scatter(x=fitted, y=residuals, mode='markers',
                   name='Residuals vs Fitted',
                   marker=dict(color='magenta', size=4)),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

# ==================== DATA EXPORT ====================
def convert_to_csv(df):
    """Convert dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def create_excel_download(data, forecast_df, rmse_scores):
    """Create Excel file with multiple sheets"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Historical_Data', index=False)
        forecast_df.to_excel(writer, sheet_name='Forecast_Data', index=False)
        
        rmse_df = pd.DataFrame({
            'Model': list(rmse_scores.keys()),
            'RMSE': list(rmse_scores.values())
        })
        rmse_df.to_excel(writer, sheet_name='Model_Performance', index=False)
    
    return output.getvalue()

# ==================== MAIN APP ====================
def main():
    st.set_page_config(
        page_title="Stock Forecast Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00ff41, #00b8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📈 Advanced Stock Forecasting Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Professional Time Series Analysis with Multiple Forecasting Models**")
    
    # Sidebar
    st.sidebar.image("sl_022321_41020_26.jpg", width=100)
    st.sidebar.title("⚙️ Configuration")
    
    # Popular tickers
    popular_tickers = {
        "Gold ETF (India)": "SETFGOLD.NS",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Reliance (India)": "RELIANCE.NS",
        "TCS (India)": "TCS.NS",
        "Custom": "CUSTOM"
    }
    
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        list(popular_tickers.keys()),
        index=0
    )
    
    if popular_tickers[selected_ticker] == "CUSTOM":
        ticker = st.sidebar.text_input("Enter Custom Ticker:", "AAPL")
    else:
        ticker = popular_tickers[selected_ticker]
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = col2.date_input("End Date", pd.to_datetime("today"))
    
    forecast_days = st.sidebar.slider("Forecast Days", 30, 730, 365)
    test_size = st.sidebar.slider("Test Split (%)", 10, 30, 20) / 100
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 ARIMA Configuration")
    arima_p = st.sidebar.selectbox("AR Order (p)", [0, 1, 2, 3], index=1)
    arima_d = st.sidebar.selectbox("Differencing (d)", [0, 1, 2], index=1)
    arima_q = st.sidebar.selectbox("MA Order (q)", [0, 1, 2, 3], index=1)
    arima_order = (arima_p, arima_d, arima_q)
    
    forecast_methods = st.sidebar.multiselect(
        "Select Forecast Methods",
        ["AutoReg", "ARIMA", "SARIMAX", "Prophet"],
        default=["SARIMAX", "Prophet"] if PROPHET_AVAILABLE else ["SARIMAX"]
    )
    
    if st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True):
        
        # Fetch data
        with st.spinner(f"📡 Fetching data for {ticker}..."):
            raw_data = load_stock_data(ticker, start_date, end_date)
        
        if raw_data is None or raw_data.empty:
            st.error("❌ No data available. Please check the ticker symbol.")
            return
        
        # Prepare data
        with st.spinner("⚙️ Preparing data..."):
            data = prepare_data(raw_data)
        
        st.success(f"✅ Successfully loaded {len(data)} days of data")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "🕯️ Candlestick",
            "📈 Models",
            "🔮 Forecast",
            "📉 Diagnostics",
            "💾 Export"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.subheader(f"📊 Data Overview - {ticker}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Latest Price", f"₹{data['Price'].iloc[-1]:.2f}")
            col2.metric("Highest", f"₹{data['High'].max():.2f}")
            col3.metric("Lowest", f"₹{data['Low'].min():.2f}")
            col4.metric("Avg Volume", f"{data['Volume'].mean()/1e6:.2f}M")
            col5.metric("Data Points", f"{len(data)}")
            
            st.plotly_chart(plot_price_trends(data, ticker), use_container_width=True)
            
            st.subheader("📋 Recent Data")
            st.dataframe(
                data[['Date', 'Open', 'High', 'Low', 'Price', 'Volume']].tail(10),
                use_container_width=True
            )
        
        # Tab 2: Candlestick
        with tab2:
            st.subheader("🕯️ Candlestick Analysis")
            st.plotly_chart(plot_candlestick(data, ticker), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price Range", f"₹{data['High'].max() - data['Low'].min():.2f}")
            with col2:
                avg_daily_change = ((data['Price'].iloc[-1] / data['Price'].iloc[0]) - 1) * 100
                st.metric("Total Change", f"{avg_daily_change:.2f}%")
        
        # Tab 3: Models
        with tab3:
            st.subheader("🤖 Model Training & Evaluation")
            
            with st.spinner("Training models..."):
                train_data, test_data = train_test_split(
                    data, test_size=test_size, random_state=42, shuffle=False
                )
                
                best_model, rmse_scores, best_model_name, all_models = train_all_models(
                    train_data, test_data
                )
            
            st.success(f"🏆 Best Model: **{best_model_name}** (RMSE: {rmse_scores[best_model_name]:.2f})")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(plot_model_comparison(rmse_scores), use_container_width=True)
            
            with col2:
                st.subheader("📊 Model Scores")
                rmse_df = pd.DataFrame({
                    'Model': list(rmse_scores.keys()),
                    'RMSE': [f"{v:.2f}" for v in rmse_scores.values()]
                })
                st.dataframe(rmse_df, use_container_width=True, hide_index=True)
        
        # Tab 4: Forecast
        with tab4:
            st.subheader(f"🔮 Price Forecast - Next {forecast_days} Days")
            
            with st.spinner("Generating forecasts..."):
                # Calculate residuals
                residuals = calculate_residuals(data, best_model)
                
                # Create future dataframe
                future_df = create_future_dataframe(data, forecast_days)
                
                # Initialize variables
                prophet_forecast = None
                prophet_model = None
                full_prophet_forecast = None
                
                # Run selected forecast methods
                if "AutoReg" in forecast_methods:
                    future_df, ar_model = forecast_with_autoreg(
                        best_model, data, future_df, residuals
                    )
                
                if "ARIMA" in forecast_methods:
                    future_df, arima_model = forecast_with_arima(
                        best_model, data, future_df, residuals, arima_order
                    )
                
                if "SARIMAX" in forecast_methods:
                    future_df, sarimax_model = forecast_with_sarimax(
                        best_model, data, future_df, residuals, arima_order
                    )
                
                if "Prophet" in forecast_methods and PROPHET_AVAILABLE:
                    prophet_forecast, prophet_model, full_prophet_forecast = forecast_with_prophet(
                        data, forecast_days
                    )
                    if prophet_forecast is not None:
                        # Merge Prophet results with other forecasts
                        future_df = future_df.merge(
                            prophet_forecast,
                            on='Date',
                            how='left'
                        )
            
            st.plotly_chart(
                plot_forecast_comparison(data, future_df, ticker, prophet_forecast),
                use_container_width=True
            )
            
            # Prophet components
            if "Prophet" in forecast_methods and prophet_model is not None:
                with st.expander("📊 Prophet Model Components (Trend & Seasonality)", expanded=False):
                    st.plotly_chart(
                        plot_prophet_components(prophet_model, full_prophet_forecast),
                        use_container_width=True
                    )
            
            # Forecast statistics
            st.subheader("📊 Forecast Summary")
            cols = st.columns(len(forecast_methods))
            
            for idx, method in enumerate(forecast_methods):
                col_name = f"{method}_Price"
                if col_name in future_df.columns:
                    with cols[idx]:
                        current_price = data['Price'].iloc[-1]
                        forecast_price = future_df[col_name].iloc[-1]
                        change_pct = ((forecast_price / current_price) - 1) * 100
                        
                        st.metric(
                            f"{method} Prediction",
                            f"₹{forecast_price:.2f}",
                            f"{change_pct:+.2f}%"
                        )
            
            st.subheader("📅 Forecast Data Preview")
            display_cols = ['Date'] + [col for col in future_df.columns if 'Price' in col or 'Lower' in col or 'Upper' in col]
            st.dataframe(future_df[display_cols].tail(10), use_container_width=True)
        
        # Tab 5: Diagnostics
        with tab5:
            st.subheader("🔍 Model Diagnostics")
            
            residuals = calculate_residuals(data, best_model)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Residual", f"{residuals.mean():.4f}")
            col2.metric("Std Dev", f"{residuals.std():.4f}")
            col3.metric("Max Abs Error", f"{abs(residuals).max():.2f}")
            
            st.plotly_chart(plot_residuals_analysis(residuals, data), use_container_width=True)
            
            # ACF/PACF plot
            st.subheader("Autocorrelation Analysis")
            acf_pacf_fig = create_acf_pacf_plots(residuals)
            st.plotly_chart(acf_pacf_fig, use_container_width=True)
        
        # Tab 6: Export
        with tab6:
            st.subheader("💾 Download Data & Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📊 Historical Data")
                historical_csv = convert_to_csv(data)
                st.download_button(
                    label="📥 Download CSV",
                    data=historical_csv,
                    file_name=f"{ticker}_historical_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🔮 Forecast Data")
                forecast_csv = convert_to_csv(future_df)
                st.download_button(
                    label="📥 Download CSV",
                    data=forecast_csv,
                    file_name=f"{ticker}_forecast_{forecast_days}days.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("#### 📈 Complete Report")
                excel_data = create_excel_download(data, future_df, rmse_scores)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{ticker}_complete_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Combined dataset
            st.subheader("📦 Combined Dataset")
            combined_df = pd.concat([
                data[['Date', 'Price']].assign(Type='Historical'),
                future_df[['Date'] + [col for col in future_df.columns if 'Price' in col]].assign(Type='Forecast')
            ])
            
            st.dataframe(combined_df.head(10), use_container_width=True)
            
            combined_csv = convert_to_csv(combined_df)
            st.download_button(
                label="📥 Download Combined Dataset (CSV)",
                data=combined_csv,
                file_name=f"{ticker}_combined_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Model summary
            st.markdown("---")
            st.subheader("📄 Analysis Summary")
            
            summary_text = f"""
### Stock Analysis Report - {ticker}

**Analysis Period:** {start_date} to {end_date}  
**Total Days Analyzed:** {len(data)}  
**Forecast Horizon:** {forecast_days} days  

#### Best Performing Model
- **Model:** {best_model_name}  
- **RMSE:** {rmse_scores[best_model_name]:.2f}  
- **Test Split:** {test_size * 100:.0f}%  

#### Current Metrics
- **Latest Price:** ₹{data['Price'].iloc[-1]:.2f}  
- **52-Week High:** ₹{data['High'].max():.2f}  
- **52-Week Low:** ₹{data['Low'].min():.2f}  
- **Average Volume:** {data['Volume'].mean():,.0f}  

#### Forecast Summary
"""
            
            for method in forecast_methods:
                col_name = f"{method}_Price"
                if col_name in future_df.columns:
                    forecast_price = future_df[col_name].iloc[-1]
                    change_pct = ((forecast_price / data['Price'].iloc[-1]) - 1) * 100
                    summary_text += f"- **{method}:** ₹{forecast_price:.2f} ({change_pct:+.2f}%)\n"
            
            st.markdown(summary_text)
            
            st.download_button(
                label="📥 Download Summary Report (TXT)",
                data=summary_text,
                file_name=f"{ticker}_analysis_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📚 How to Use:**
    1. Select a stock ticker
    2. Choose date range
    3. Configure forecast parameters
    4. Click 'Run Forecast'
    5. Explore tabs for insights
    6. Download reports
    
    **🔧 Forecast Methods:**
    - **AutoReg:** Auto-Regression model
    - **ARIMA:** Integrated Moving Average
    - **SARIMAX:** Seasonal ARIMA with exogenous variables
    - **Prophet:** Facebook's time series forecasting (with trend & seasonality decomposition)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with ❤️ using Streamlit | Data: Yahoo Finance")

if __name__ == "__main__":
    main()
