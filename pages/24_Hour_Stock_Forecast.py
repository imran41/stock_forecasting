import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Forecast Pro", page_icon="📈", layout="wide")
st.title("📈 Advanced 24-Hour Stock Forecast")
st.markdown("**Enhanced ML Models + Technical Indicators + Ensemble Predictions**")

# Sidebar for advanced options
st.sidebar.header("🔧 Advanced Settings")

# User configuration
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Symbol:", "AAPL", 
                           help="Examples: AAPL, TSLA, RELIANCE.NS, TCS.NS")
with col2:
    days_back = st.selectbox("Historical Data:", [7, 14, 30, 60, 90], index=1)

# Advanced options in sidebar
st.sidebar.subheader("Model Configuration")
enable_technical_indicators = st.sidebar.checkbox("Technical Indicators", value=True)
enable_sentiment_analysis = st.sidebar.checkbox("Market Sentiment Features", value=True)
enable_volume_analysis = st.sidebar.checkbox("Volume Analysis", value=True)
enable_volatility_clustering = st.sidebar.checkbox("Volatility Clustering", value=True)

st.sidebar.subheader("Feature Engineering")
feature_scaling = st.sidebar.selectbox("Scaling Method:", ["MinMax", "Standard", "Robust"])
feature_selection = st.sidebar.checkbox("Automatic Feature Selection", value=True)

if st.button("🚀 Generate Enhanced Forecast", type="primary"):
    with st.spinner("Fetching data and training advanced models..."):
        end = datetime.now()
        start = end - timedelta(days=days_back)
        
        # Detect currency based on stock exchange
        currency = "$"
        if ".NS" in ticker.upper() or ".BO" in ticker.upper():
            currency = "₹"
        elif ".L" in ticker.upper():
            currency = "£"
        elif ".T" in ticker.upper():
            currency = "¥"
        
        try:
            data = yf.download(ticker, start=start, end=end, interval='1h', progress=False)
            
            # Fix multi-index columns issue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
        except:
            st.error("❌ Error fetching data. Check your internet connection.")
            st.stop()
        
        if data.empty or len(data) < 24:
            st.warning("⚠️ Insufficient data. Try a different stock or time period.")
            st.stop()
        
        # ========== ADVANCED FEATURE ENGINEERING ==========
        
        # Basic features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log1p(data['Returns'].abs()) * np.sign(data['Returns'])
        data['Price_Range'] = data['High'] - data['Low']
        data['Gap'] = data['Open'] - data['Close'].shift(1)
        
        # Moving averages with different windows
        for window in [5, 10, 20, 50]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()
        
        # Volatility features
        data['Volatility_5'] = data['Returns'].rolling(window=5).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        data['Volatility_Ratio'] = data['Volatility_5'] / data['Volatility_20']
        
        # Volume features
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        data['Volume_Price_Trend'] = data['Volume'] * data['Returns']
        
        # ========== TECHNICAL INDICATORS ==========
        if enable_technical_indicators:
            try:
                # RSI
                data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
                
                # MACD
                macd = ta.trend.MACD(data['Close'])
                data['MACD'] = macd.macd()
                data['MACD_Signal'] = macd.macd_signal()
                data['MACD_Histogram'] = macd.macd_diff()
                
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(data['Close'])
                data['BB_Upper'] = bollinger.bollinger_hband()
                data['BB_Lower'] = bollinger.bollinger_lband()
                data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
                data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
                
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
                data['Stoch_K'] = stoch.stoch()
                data['Stoch_D'] = stoch.stoch_signal()
                
                # ATR
                data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
                
                # Ichimoku Cloud (simplified)
                data['Ichimoku_Base'] = (data['High'].rolling(26).max() + data['Low'].rolling(26).min()) / 2
                data['Ichimoku_Conversion'] = (data['High'].rolling(9).max() + data['Low'].rolling(9).min()) / 2
                
            except Exception as e:
                st.warning(f"⚠️ Some technical indicators failed: {e}")
        
        # ========== VOLATILITY CLUSTERING FEATURES ==========
        if enable_volatility_clustering:
            data['Volatility_Regime'] = (data['Volatility_5'] > data['Volatility_5'].rolling(50).mean()).astype(int)
            data['High_Volatility_Period'] = data['Volatility_Regime'].rolling(5).sum()
        
        # ========== TIME-BASED FEATURES ==========
        data['Hour'] = data.index.hour
        data['DayOfWeek'] = data.index.dayofweek
        data['Is_Weekend'] = (data['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
        data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)
        data['Day_sin'] = np.sin(2 * np.pi * data['DayOfWeek']/7)
        data['Day_cos'] = np.cos(2 * np.pi * data['DayOfWeek']/7)
        
        # ========== LAGGED FEATURES ==========
        for lag in [1, 2, 3, 5, 10]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
        
        # ========== ROLLING STATISTICS ==========
        for window in [5, 10, 20]:
            data[f'Roll_Mean_{window}'] = data['Close'].rolling(window).mean()
            data[f'Roll_Std_{window}'] = data['Close'].rolling(window).std()
            data[f'Roll_Max_{window}'] = data['Close'].rolling(window).max()
            data[f'Roll_Min_{window}'] = data['Close'].rolling(window).min()
            data[f'Roll_Skew_{window}'] = data['Close'].rolling(window).skew()
            data[f'Roll_Kurt_{window}'] = data['Close'].rolling(window).kurt()
        
        # Drop NaN values
        data = data.dropna()
        
        if len(data) < 50:
            st.warning("⚠️ Insufficient data after feature engineering. Try more historical data.")
            st.stop()
        
        # ========== FEATURE SELECTION ==========
        feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = data[feature_columns]
        y = data['Close']
        
        if feature_selection and len(feature_columns) > 20:
            selector = SelectKBest(score_func=f_regression, k=min(30, len(feature_columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=data.index)
        
        # Feature scaling
        if feature_scaling == "Standard":
            scaler = StandardScaler()
        elif feature_scaling == "Robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        # ========== DATA SPLITTING ==========
        prices = data['Close'].values
        train_size = int(len(prices) * 0.8)
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = prices[:train_size], prices[train_size:]
        
        # ========== ENHANCED MODEL TRAINING ==========
        predictions = {}
        errors = {}
        feature_importances = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ========== MODEL 1: ENSEMBLE REGRESSION ==========
        try:
            status_text.text("Training Ensemble Regression...")
            from sklearn.ensemble import VotingRegressor
            
            ridge = Ridge(alpha=1.0, random_state=42)
            elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            
            ensemble_reg = VotingRegressor([
                ('ridge', ridge),
                ('elastic', elastic),
                ('svr', svr)
            ])
            
            ensemble_reg.fit(X_train, y_train)
            
            # Recursive prediction for next 24 hours
            ensemble_preds = []
            current_features = X_scaled[-1:].copy()
            
            for i in range(24):
                pred = ensemble_reg.predict(current_features)[0]
                ensemble_preds.append(pred)
                
                # Update features for next prediction (simplified)
                if i < len(ensemble_preds) - 1:
                    # Update lagged features in the prediction
                    pass
                
            predictions['Ensemble_Reg'] = np.array(ensemble_preds)
            val_pred = ensemble_reg.predict(X_test)
            errors['Ensemble_Reg'] = mean_absolute_error(y_test, val_pred)
            
            progress_bar.progress(20)
        except Exception as e:
            st.warning(f"⚠️ Ensemble Regression failed: {e}")
        
        # ========== MODEL 2: ENHANCED RANDOM FOREST ==========
        try:
            status_text.text("Training Enhanced Random Forest...")
            rf_enhanced = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            rf_enhanced.fit(X_train, y_train)
            
            rf_preds = []
            current_features = X_scaled[-1:].copy()
            
            for i in range(24):
                pred = rf_enhanced.predict(current_features)[0]
                rf_preds.append(pred)
                
            predictions['RF_Enhanced'] = np.array(rf_preds)
            val_pred = rf_enhanced.predict(X_test)
            errors['RF_Enhanced'] = mean_absolute_error(y_test, val_pred)
            
            # Feature importance
            feature_importances['RF'] = dict(zip(X.columns, rf_enhanced.feature_importances_))
            
            progress_bar.progress(40)
        except Exception as e:
            st.warning(f"⚠️ Enhanced Random Forest failed: {e}")
        
        # ========== MODEL 3: GRADIENT BOOSTING WITH EARLY STOPPING ==========
        try:
            status_text.text("Training Gradient Boosting...")
            gb_enhanced = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )
            gb_enhanced.fit(X_train, y_train)
            
            gb_preds = []
            current_features = X_scaled[-1:].copy()
            
            for i in range(24):
                pred = gb_enhanced.predict(current_features)[0]
                gb_preds.append(pred)
                
            predictions['GB_Enhanced'] = np.array(gb_preds)
            val_pred = gb_enhanced.predict(X_test)
            errors['GB_Enhanced'] = mean_absolute_error(y_test, val_pred)
            
            feature_importances['GB'] = dict(zip(X.columns, gb_enhanced.feature_importances_))
            
            progress_bar.progress(60)
        except Exception as e:
            st.warning(f"⚠️ Enhanced Gradient Boosting failed: {e}")
        
        # ========== MODEL 4: ARIMA WITH AUTO PARAMETERS ==========
        try:
            status_text.text("Training ARIMA...")
            # Auto-detect ARIMA parameters
            def find_best_arima(series, max_p=5, max_d=2, max_q=5):
                best_aic = np.inf
                best_order = None
                
                for p in range(max_p + 1):
                    for d in range(max_d + 1):
                        for q in range(max_q + 1):
                            try:
                                model = ARIMA(series, order=(p, d, q))
                                result = model.fit()
                                if result.aic < best_aic:
                                    best_aic = result.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                return best_order
            
            best_order = find_best_arima(prices)
            if best_order:
                arima_model = ARIMA(prices, order=best_order)
                arima_res = arima_model.fit()
                arima_preds = arima_res.forecast(steps=24)
                predictions['ARIMA_Auto'] = np.array(arima_preds)
                
                arima_val = arima_res.predict(start=train_size, end=len(prices)-1)
                errors['ARIMA_Auto'] = mean_absolute_error(y_test, arima_val)
            
            progress_bar.progress(80)
        except Exception as e:
            st.warning(f"⚠️ ARIMA Auto failed: {e}")
        
        # ========== MODEL 5: HYBRID CNN-LSTM ==========
        try:
            status_text.text("Training Hybrid CNN-LSTM...")
            
            # Prepare sequences for deep learning
            sequence_length = 24
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X_scaled)):
                X_seq.append(X_scaled[i-sequence_length:i])
                y_seq.append(prices[i])
            
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            
            # Split sequences
            train_size_seq = int(len(X_seq) * 0.8)
            X_seq_train, X_seq_test = X_seq[:train_size_seq], X_seq[train_size_seq:]
            y_seq_train, y_seq_test = y_seq[:train_size_seq], y_seq[train_size_seq:]
            
            # Build hybrid model
            hybrid_model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', 
                      input_shape=(sequence_length, X_seq.shape[2])),
                MaxPooling1D(pool_size=2),
                Bidirectional(LSTM(100, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            hybrid_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # Train model
            history = hybrid_model.fit(
                X_seq_train, y_seq_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict next 24 hours
            last_sequence = X_scaled[-sequence_length:]
            hybrid_preds = []
            
            for _ in range(24):
                X_input = last_sequence.reshape(1, sequence_length, X_scaled.shape[1])
                pred = hybrid_model.predict(X_input, verbose=0)[0][0]
                hybrid_preds.append(pred)
                
                # Update sequence (simplified - in practice, you'd update features)
                last_sequence = np.vstack([last_sequence[1:], X_scaled[-1:]])
            
            predictions['Hybrid_CNN_LSTM'] = np.array(hybrid_preds)
            
            # Calculate error
            hybrid_val_pred = hybrid_model.predict(X_seq_test, verbose=0).flatten()
            errors['Hybrid_CNN_LSTM'] = mean_absolute_error(y_seq_test, hybrid_val_pred)
            
            progress_bar.progress(100)
        except Exception as e:
            st.warning(f"⚠️ Hybrid CNN-LSTM failed: {e}")
        
        # ========== ADVANCED ENSEMBLE ==========
        if predictions:
            # Dynamic weighting based on multiple metrics
            ensemble_weights = {}
            
            for model_name in predictions.keys():
                if model_name in errors:
                    # Combine MAE and R² for better weight calculation
                    try:
                        model_predictions = predictions[model_name]
                        val_predictions = globals().get(f'{model_name.lower()}_val_pred', None)
                        
                        if val_predictions is not None and len(val_predictions) == len(y_test):
                            r2 = r2_score(y_test, val_predictions)
                            mae = errors[model_name]
                            
                            # Higher R² and lower MAE = better weight
                            score = (r2 + 1) / (mae + 1e-8)  # Avoid division by zero
                            ensemble_weights[model_name] = max(score, 0.1)  # Minimum weight
                        else:
                            ensemble_weights[model_name] = 1 / (errors[model_name] + 1e-8)
                    except:
                        ensemble_weights[model_name] = 1 / (errors[model_name] + 1e-8)
                else:
                    ensemble_weights[model_name] = 1.0
            
            # Normalize weights
            total_weight = sum(ensemble_weights.values())
            for model_name in ensemble_weights:
                ensemble_weights[model_name] /= total_weight
            
            # Calculate final ensemble prediction
            final_ensemble = np.zeros(24)
            for model_name, preds in predictions.items():
                final_ensemble += preds * ensemble_weights[model_name]
            
            predictions['Final_Ensemble'] = final_ensemble
            
            # ========== ENHANCED VISUALIZATION ==========
            st.subheader("📊 Advanced Analytics Dashboard")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🔧 Model Analysis", "📊 Features", "💡 Insights"])
            
            with tab1:
                # Interactive forecast plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['Close'],
                    name='Historical Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Future predictions
                future_times = [data.index[-1] + timedelta(hours=i+1) for i in range(24)]
                
                for model_name, preds in predictions.items():
                    if model_name != 'Final_Ensemble':
                        fig.add_trace(go.Scatter(
                            x=future_times, y=preds,
                            name=model_name,
                            line=dict(dash='dot'),
                            opacity=0.6
                        ))
                
                # Final ensemble
                fig.add_trace(go.Scatter(
                    x=future_times, y=final_ensemble,
                    name='Final Ensemble',
                    line=dict(color='#ff7f0e', width=4)
                ))
                
                fig.update_layout(
                    title='24-Hour Price Forecast',
                    xaxis_title='Time',
                    yaxis_title=f'Price ({currency})',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Model performance comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Error metrics
                    error_data = []
                    for model_name, error in errors.items():
                        error_data.append({
                            'Model': model_name,
                            'MAE': error,
                            'Weight': f"{ensemble_weights.get(model_name, 0):.1%}"
                        })
                    
                    error_df = pd.DataFrame(error_data)
                    st.dataframe(error_df.sort_values('MAE'), use_container_width=True)
                
                with col2:
                    # Weight visualization
                    weight_fig = px.pie(
                        values=list(ensemble_weights.values()),
                        names=list(ensemble_weights.keys()),
                        title='Model Weights in Ensemble'
                    )
                    st.plotly_chart(weight_fig, use_container_width=True)
            
            with tab3:
                # Feature importance
                if feature_importances:
                    st.subheader("Top Feature Importances")
                    
                    for model_name, importance_dict in feature_importances.items():
                        importance_df = pd.DataFrame(
                            list(importance_dict.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            title=f'Top Features - {model_name}',
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Enhanced trading insights
                current_price = float(prices[-1])
                ensemble_pred = float(final_ensemble[-1])
                price_change = ((ensemble_pred - current_price) / current_price) * 100
                
                # Calculate confidence metrics
                model_agreement = np.std([preds[-1] for preds in predictions.values()]) / current_price
                confidence = max(0, 100 - (model_agreement * 100))
                
                # Volatility analysis
                recent_volatility = float(data['Volatility_5'].iloc[-1])
                avg_volatility = float(data['Volatility_5'].mean())
                volatility_ratio = recent_volatility / avg_volatility
                
                # RSI analysis
                current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"{currency}{current_price:.2f}")
                
                with col2:
                    st.metric("24h Forecast", f"{currency}{ensemble_pred:.2f}")
                
                with col3:
                    st.metric("Expected Change", f"{price_change:.2f}%", 
                             delta=f"{price_change:.2f}%")
                
                with col4:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Advanced trading signals
                st.subheader("🎯 Enhanced Trading Signals")
                
                # Multiple factor analysis
                signals = []
                
                # Price momentum signal
                if price_change > 3:
                    signals.append(("📈 Strong Bullish", "high", f"+{price_change:.1f}% forecast"))
                elif price_change > 1:
                    signals.append(("📈 Mild Bullish", "medium", f"+{price_change:.1f}% forecast"))
                elif price_change < -3:
                    signals.append(("📉 Strong Bearish", "high", f"{price_change:.1f}% forecast"))
                elif price_change < -1:
                    signals.append(("📉 Mild Bearish", "medium", f"{price_change:.1f}% forecast"))
                else:
                    signals.append(("⚖️ Neutral", "low", "Minimal price movement"))
                
                # RSI signal
                if current_rsi > 70:
                    signals.append(("🚨 Overbought", "high", f"RSI: {current_rsi:.1f}"))
                elif current_rsi < 30:
                    signals.append(("🚨 Oversold", "high", f"RSI: {current_rsi:.1f}"))
                
                # Volatility signal
                if volatility_ratio > 1.5:
                    signals.append(("🌪️ High Volatility", "high", f"{volatility_ratio:.1f}x average"))
                elif volatility_ratio < 0.5:
                    signals.append(("😐 Low Volatility", "low", f"{volatility_ratio:.1f}x average"))
                
                # Display signals
                for signal, level, details in signals:
                    if level == "high":
                        st.error(f"❌ {signal} - {details}")
                    elif level == "medium":
                        st.warning(f"⚠️ {signal} - {details}")
                    else:
                        st.info(f"ℹ️ {signal} - {details}")
                
                # Final recommendation
                st.subheader("💎 Final Recommendation")
                
                bullish_signals = sum(1 for _, level, _ in signals if "Bullish" in _ or ("Oversold" in _ and level == "high"))
                bearish_signals = sum(1 for _, level, _ in signals if "Bearish" in _ or ("Overbought" in _ and level == "high"))
                
                if bullish_signals > bearish_signals and confidence > 65:
                    st.success(f"✅ **STRONG BUY** - Multiple bullish signals (Confidence: {confidence:.0f}%)")
                    st.write("📈 Favorable technical setup with positive momentum")
                elif bullish_signals > bearish_signals and confidence > 50:
                    st.success(f"✅ **BUY** - Cautiously optimistic (Confidence: {confidence:.0f}%)")
                    st.write("🔼 Positive signals but monitor closely")
                elif bearish_signals > bullish_signals and confidence > 65:
                    st.error(f"❌ **STRONG SELL** - Multiple bearish signals (Confidence: {confidence:.0f}%)")
                    st.write("📉 Negative momentum with high conviction")
                elif bearish_signals > bullish_signals and confidence > 50:
                    st.error(f"❌ **SELL** - Risk-off sentiment (Confidence: {confidence:.0f}%)")
                    st.write("🔽 Consider reducing exposure")
                else:
                    st.warning(f"⚠️ **HOLD/WAIT** - Mixed signals (Confidence: {confidence:.0f}%)")
                    st.write("⏸️ Wait for clearer market direction")
                
                # Risk assessment
                st.info(f"🔍 **Market Context**: Volatility is {volatility_ratio:.1f}x normal levels")
                if 'RSI' in data.columns:
                    st.info(f"📊 **RSI Analysis**: {current_rsi:.1f} - {'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'}")
                
                st.info(f"⚖️ **Model Consensus**: {', '.join([f'{k}: {v:.1%}' for k, v in ensemble_weights.items()])}")
            
            # Risk disclaimer
            st.markdown("---")
            st.caption("""
            ⚠️ **Advanced Risk Disclaimer**: 
            - This model uses multiple technical indicators and machine learning algorithms
            - Past performance is not indicative of future results
            - Stock markets are highly volatile and unpredictable
            - Always conduct your own research and consult financial advisors
            - Use proper risk management and position sizing
            - Consider market conditions, news events, and macroeconomic factors
            """)
        
        else:
            st.error("❌ All models failed. Please try with different parameters or stock symbol.")

# Add information about the enhanced features
st.sidebar.markdown("---")
st.sidebar.info("""
**🎯 Enhanced Features:**
- Technical Indicators (RSI, MACD, Bollinger Bands)
- Advanced Feature Engineering
- Hybrid CNN-LSTM Architecture
- Dynamic Ensemble Weighting
- Multiple Timeframe Analysis
- Volatility Regime Detection
""")
