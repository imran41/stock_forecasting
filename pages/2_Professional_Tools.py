import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import root_mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from scipy import stats
import io
import warnings
warnings.filterwarnings("ignore")

# Prophet import with error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ==================== RISK MANAGEMENT CLASS ====================
class RiskManager:
    """Professional Risk Management System"""
    
    def __init__(self, capital, max_position_pct=0.20, max_daily_loss_pct=0.02, 
                 stop_loss_pct=0.05, take_profit_pct=0.10):
        self.initial_capital = capital
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.daily_loss = 0
        self.daily_profit = 0
        self.open_positions = {}
        self.trade_history = []
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_limits(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_loss = 0
            self.daily_profit = 0
            self.last_reset_date = current_date
    
    def can_trade(self, signal_confidence, current_price, prediction_price):
        self.reset_daily_limits()
        
        if self.daily_loss >= self.max_daily_loss_pct * self.initial_capital:
            return False, f"Daily loss limit reached (₹{self.daily_loss:.2f})", 0.0
        
        if signal_confidence < 0.60:
            return False, f"Low confidence ({signal_confidence*100:.1f}% < 60%)", 0.0
        
        if self.capital < current_price:
            return False, "Insufficient capital", 0.0
        
        expected_return = (prediction_price / current_price) - 1
        if abs(expected_return) < 0.01:
            return False, f"Expected return too low ({expected_return*100:.1f}%)", 0.0
        
        if expected_return > 0.50:
            return False, "Prediction unrealistically high (>50% gain)", 0.0
        if expected_return < -0.30:
            return False, "Prediction unrealistically low (>30% loss)", 0.0
        
        risk_score = min(
            signal_confidence * (abs(expected_return) / 0.10) * 
            (1 - self.daily_loss / (self.max_daily_loss_pct * self.initial_capital)),
            1.0
        )
        
        return True, "Trade approved", risk_score
    
    def calculate_position_size(self, confidence, current_price, risk_score):
        base_position = self.capital * self.max_position_pct
        
        if confidence >= 0.85:
            confidence_multiplier = 1.0
        elif confidence >= 0.75:
            confidence_multiplier = 0.6
        else:
            confidence_multiplier = 0.3
        
        risk_multiplier = risk_score ** 0.5
        position_value = base_position * confidence_multiplier * risk_multiplier
        shares = int(position_value / current_price)
        actual_position_value = shares * current_price
        position_pct = (actual_position_value / self.capital) * 100
        
        return shares, actual_position_value, position_pct
    
    def calculate_stop_loss(self, entry_price, volatility=None):
        if volatility and volatility > 0.03:
            stop_pct = min(self.stop_loss_pct * 1.5, 0.10)
        else:
            stop_pct = self.stop_loss_pct
        
        return entry_price * (1 - stop_pct)
    
    def calculate_take_profit(self, entry_price, prediction_price):
        predicted_gain = prediction_price - entry_price
        conservative_target = entry_price + (predicted_gain * 0.70)
        min_target = entry_price * (1 + self.take_profit_pct)
        take_profit = max(conservative_target, min_target)
        take_profit_pct = ((take_profit / entry_price) - 1) * 100
        
        return take_profit, take_profit_pct
    
    def calculate_risk_reward_ratio(self, entry_price, stop_loss, take_profit):
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        
        if risk <= 0:
            return 0
        
        return reward / risk
    
    def open_position(self, ticker, shares, entry_price, stop_loss, take_profit):
        position = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'entry_value': shares * entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_date': datetime.now(),
            'status': 'OPEN'
        }
        
        self.open_positions[ticker] = position
        self.capital -= position['entry_value']
        
        return position
    
    def close_position(self, ticker, exit_price):
        if ticker not in self.open_positions:
            return None
        
        position = self.open_positions[ticker]
        exit_value = position['shares'] * exit_price
        pnl = exit_value - position['entry_value']
        pnl_pct = (pnl / position['entry_value']) * 100
        
        self.capital += exit_value
        
        if pnl > 0:
            self.daily_profit += pnl
        else:
            self.daily_loss += abs(pnl)
        
        trade_record = {
            **position,
            'exit_price': exit_price,
            'exit_value': exit_value,
            'exit_date': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'status': 'CLOSED'
        }
        
        self.trade_history.append(trade_record)
        del self.open_positions[ticker]
        
        return trade_record
    
    def check_exit_conditions(self, ticker, current_price):
        if ticker not in self.open_positions:
            return False, "No open position"
        
        position = self.open_positions[ticker]
        
        if current_price <= position['stop_loss']:
            return True, f"STOP LOSS HIT (₹{position['stop_loss']:.2f})"
        
        if current_price >= position['take_profit']:
            return True, f"TAKE PROFIT HIT (₹{position['take_profit']:.2f})"
        
        return False, "Conditions not met"
    
    def get_performance_summary(self):
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'current_capital': self.capital,
                'daily_profit': self.daily_profit,
                'daily_loss': self.daily_loss
            }
        
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]  # Fixed: only trades with negative pnl
        
        total_profit = sum([t['pnl'] for t in winning_trades])
        total_loss = abs(sum([t['pnl'] for t in losing_trades]))
        
        # FIXED: Proper profit factor calculation
        if total_loss == 0:
            if total_profit > 0:
                profit_factor = float('inf')  # Perfect - no losses, only profits
            else:
                profit_factor = 0.0  # No profitable trades
        else:
            profit_factor = total_profit / total_loss
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
            'avg_profit': total_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss': total_loss / len(losing_trades) if losing_trades else 0,
            'profit_factor': profit_factor,
            'total_pnl': sum([t['pnl'] for t in closed_trades]),
            'total_return_pct': ((self.capital / self.initial_capital) - 1) * 100,
            'current_capital': self.capital,
            'daily_profit': self.daily_profit,
            'daily_loss': self.daily_loss
        }


# ==================== CONFIDENCE CALCULATOR ====================
class ConfidenceCalculator:
    """Calculate prediction confidence using multiple methods"""
    
    @staticmethod
    def calculate_model_agreement(predictions_dict, current_price):
        if not predictions_dict:
            return 0.0, 0.0, "UNKNOWN"
        
        signals = []
        changes = []
        
        for model_name, pred_price in predictions_dict.items():
            change_pct = ((pred_price / current_price) - 1) * 100
            changes.append(change_pct)
            
            if change_pct > 2:
                signals.append(1)
            elif change_pct < -2:
                signals.append(-1)
            else:
                signals.append(0)
        
        avg_signal = np.mean(signals)
        if avg_signal > 0:
            direction = "BULLISH"
            agreement_count = sum([1 for s in signals if s > 0])
        elif avg_signal < 0:
            direction = "BEARISH"
            agreement_count = sum([1 for s in signals if s < 0])
        else:
            direction = "NEUTRAL"
            agreement_count = sum([1 for s in signals if s == 0])
        
        agreement_pct = (agreement_count / len(signals)) * 100
        
        avg_change = abs(np.mean(changes))
        change_std = np.std(changes)
        
        consistency_score = 1 - min(change_std / max(avg_change, 0.01), 1)
        agreement_score = agreement_pct / 100
        strength_score = min(avg_change / 10, 1)
        
        confidence = (agreement_score * 0.5 + consistency_score * 0.3 + strength_score * 0.2)
        
        return confidence, agreement_pct, direction
    
    @staticmethod
    def calculate_statistical_confidence(residuals, prediction, current_price):
        try:
            residual_std = residuals.std()
            z_score = 1.96
            margin = z_score * residual_std
            
            lower_bound = prediction - margin
            upper_bound = prediction + margin
            
            uncertainty_ratio = margin / prediction if prediction != 0 else 1
            confidence = max(0, 1 - uncertainty_ratio)
            
            return confidence, lower_bound, upper_bound
            
        except Exception as e:
            return 0.5, prediction * 0.85, prediction * 1.15


# ==================== MARKET REGIME DETECTOR ====================
class MarketRegimeDetector:
    """Detect current market regime (Bull/Bear/Sideways/Volatile)"""
    
    @staticmethod
    def detect_regime(data, short_window=20, long_window=50):
        returns = data['Price'].pct_change()
        
        short_ma = data['Price'].rolling(short_window).mean()
        long_ma = data['Price'].rolling(long_window).mean()
        current_price = data['Price'].iloc[-1]
        
        short_trend = (current_price / short_ma.iloc[-1]) - 1
        long_trend = (current_price / long_ma.iloc[-1]) - 1
        
        volatility = returns.rolling(short_window).std().iloc[-1]
        avg_volatility = returns.std()
        volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        
        momentum = returns.rolling(short_window).mean().iloc[-1]
        
        metrics = {
            'short_trend': short_trend * 100,
            'long_trend': long_trend * 100,
            'volatility': volatility * 100,
            'volatility_ratio': volatility_ratio,
            'momentum': momentum * 100
        }
        
        if long_trend > 0.02 and volatility_ratio < 1.2:
            regime = "BULL_STABLE"
            confidence = 0.85
            recommendation = "🟢 EXCELLENT - Best conditions for trading"
            
        elif long_trend > 0.02 and volatility_ratio >= 1.2:
            regime = "BULL_VOLATILE"
            confidence = 0.60
            recommendation = "🟡 CAUTION - Reduce position sizes due to volatility"
            
        elif long_trend < -0.02:
            regime = "BEAR"
            confidence = 0.75
            recommendation = "🔴 AVOID - Consider only short positions or stay out"
            
        elif abs(long_trend) <= 0.02 and volatility_ratio < 1.2:
            regime = "SIDEWAYS"
            confidence = 0.50
            recommendation = "🟡 NEUTRAL - Trade only high-confidence signals"
            
        else:
            regime = "CHOPPY"
            confidence = 0.30
            recommendation = "🔴 DANGEROUS - Avoid trading, wait for clarity"
        
        return regime, confidence, recommendation, metrics
    
    @staticmethod
    def get_regime_appropriate_strategy(regime):
        strategies = {
            "BULL_STABLE": {
                "position_size_multiplier": 1.0,
                "confidence_threshold": 0.70,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.12,
                "preferred_direction": "LONG"
            },
            "BULL_VOLATILE": {
                "position_size_multiplier": 0.6,
                "confidence_threshold": 0.75,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.10,
                "preferred_direction": "LONG"
            },
            "BEAR": {
                "position_size_multiplier": 0.3,
                "confidence_threshold": 0.80,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.08,
                "preferred_direction": "SHORT"
            },
            "SIDEWAYS": {
                "position_size_multiplier": 0.5,
                "confidence_threshold": 0.80,
                "stop_loss_pct": 0.04,
                "take_profit_pct": 0.08,
                "preferred_direction": "RANGE"
            },
            "CHOPPY": {
                "position_size_multiplier": 0.0,
                "confidence_threshold": 0.90,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
                "preferred_direction": "AVOID"
            }
        }
        
        return strategies.get(regime, strategies["SIDEWAYS"])


# ==================== UTILITY FUNCTIONS ====================
def get_currency_symbol(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'
    elif ticker.endswith('.L'):
        return '£'
    elif ticker.endswith('.T'):
        return '¥'
    else:
        return '$'

def validate_dates(start_date, end_date):
    if start_date >= end_date:
        return False, "Start date must be before end date"
    
    days_diff = (end_date - start_date).days
    if days_diff < 30:
        return False, "Please select at least 30 days of data"
    
    return True, "Valid"

def validate_forecast_horizon(data_length, forecast_days):
    ratio = forecast_days / data_length
    
    if ratio > 1.0:
        return False, "⚠️ Forecast horizon is longer than training data. Results may be unreliable."
    elif ratio > 0.5:
        return True, "⚠️ Forecast horizon is quite long. Consider shorter horizons for better accuracy."
    
    return True, None

def display_profit_factor(profit_factor):
    """Properly display profit factor in Streamlit"""
    if profit_factor == float('inf'):
        return "∞ (Perfect)"
    elif profit_factor > 1000:
        return "1000+ (Excellent)"
    elif profit_factor == 0:
        return "0.00 (No Trades)"
    else:
        return f"{profit_factor:.2f}x"


# ==================== DATA FETCHING ====================
def load_stock_data(ticker, start_date, end_date):
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
    df = data.copy()
    df.rename(columns={"Close": "Price"}, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Price", "Volume"]]
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    df["t"] = np.arange(1, len(df) + 1)
    df["t_square"] = df["t"] ** 2
    df["log_Price"] = np.log(df["Price"])
    
    df["Sin"] = np.sin((2 * np.pi * df["t"]) / 365.25)
    df["Cos"] = np.cos((2 * np.pi * df["t"]) / 365.25)
    
    return df

def proper_time_series_split(data, test_size):
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    return train_data, test_data


# ==================== MODEL TRAINING ====================
def get_model_prediction(model, model_type, data):
    try:
        if model_type in ['exp', 'mul_sea', 'mul_add_sea']:
            pred = model.predict(data)
            return np.exp(pred)
        else:
            pred = model.predict(data)
            return pred
    except Exception as e:
        st.error(f"Prediction error for {model_type}: {str(e)}")
        return None

def train_all_models(train_data, test_data):
    models = {}
    rmse_scores = {}
    
    try:
        models['linear'] = smf.ols("Price ~ t", data=train_data).fit()
        pred_linear = models['linear'].predict(test_data[["t"]])
        rmse_scores['rmse_linear'] = root_mean_squared_error(test_data["Price"], pred_linear)
    except Exception as e:
        st.warning(f"Linear model failed: {str(e)}")
    
    try:
        models['exp'] = smf.ols("log_Price ~ t", data=train_data).fit()
        pred_exp = models['exp'].predict(test_data[["t"]])
        rmse_scores['rmse_Exp'] = root_mean_squared_error(test_data["Price"], np.exp(pred_exp))
    except Exception as e:
        st.warning(f"Exponential model failed: {str(e)}")
    
    try:
        models['quad'] = smf.ols("Price ~ t + t_square", data=train_data).fit()
        pred_quad = models['quad'].predict(test_data[["t", "t_square"]])
        rmse_scores['rmse_Quad'] = root_mean_squared_error(test_data["Price"], pred_quad)
    except Exception as e:
        st.warning(f"Quadratic model failed: {str(e)}")
    
    try:
        models['add_sea'] = smf.ols('Price ~ Sin + Cos', data=train_data).fit()
        pred_add_sea = models['add_sea'].predict(test_data[["Sin", "Cos"]])
        rmse_scores['rmse_add_sea'] = root_mean_squared_error(test_data['Price'], pred_add_sea)
    except Exception as e:
        st.warning(f"Additive seasonality model failed: {str(e)}")
    
    try:
        models['mul_sea'] = smf.ols('log_Price ~ Sin + Cos', data=train_data).fit()
        pred_mul_sea = models['mul_sea'].predict(test_data[["Sin", "Cos"]])
        rmse_scores['rmse_Mult_sea'] = root_mean_squared_error(test_data['Price'], np.exp(pred_mul_sea))
    except Exception as e:
        st.warning(f"Multiplicative seasonality model failed: {str(e)}")
    
    try:
        models['add_sea_quad'] = smf.ols('Price ~ t + t_square + Sin + Cos', data=train_data).fit()
        pred_add_sea_quad = models['add_sea_quad'].predict(test_data[["t", "t_square", "Sin", "Cos"]])
        rmse_scores['rmse_add_sea_quad'] = root_mean_squared_error(test_data['Price'], pred_add_sea_quad)
    except Exception as e:
        st.warning(f"Additive seasonality + quadratic model failed: {str(e)}")
    
    try:
        models['mul_add_sea'] = smf.ols('log_Price ~ t + Sin + Cos', data=train_data).fit()
        pred_mul_add_sea = models['mul_add_sea'].predict(test_data[["t", "Sin", "Cos"]])
        rmse_scores['rmse_Mult_add_sea'] = root_mean_squared_error(test_data['Price'], np.exp(pred_mul_add_sea))
    except Exception as e:
        st.warning(f"Multiplicative seasonality + linear model failed: {str(e)}")
    
    if not rmse_scores:
        st.error("All models failed to train!")
        return None, None, None, None, None
    
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
    best_model_type = model_mapping[best_model_name]
    
    return best_model, rmse_scores, best_model_name, models, best_model_type


# ==================== RESIDUAL ANALYSIS ====================
def calculate_residuals(train_data, model, model_type):
    predictions = get_model_prediction(model, model_type, train_data)
    if predictions is None:
        return None
    residuals = train_data["Price"].values - predictions.values
    return pd.Series(residuals, index=train_data.index)

def create_acf_pacf_plots(residuals):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ACF (Autocorrelation)', 'PACF (Partial Autocorrelation)')
    )
    
    nlags = min(40, len(residuals) // 2)
    
    try:
        acf_values = acf(residuals, nlags=nlags, alpha=0.05)
        acf_vals = acf_values[0] if isinstance(acf_values, tuple) else acf_values
        
        pacf_values = pacf(residuals, nlags=nlags, alpha=0.05)
        pacf_vals = pacf_values[0] if isinstance(pacf_values, tuple) else pacf_values
        
        fig.add_trace(
            go.Bar(x=list(range(len(acf_vals))), y=acf_vals, 
                   name='ACF', marker_color='cyan'),
            row=1, col=1
        )
        
        conf_int = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=conf_int, line_dash="dash", line_color="red", 
                      opacity=0.5, row=1, col=1)
        fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", 
                      opacity=0.5, row=1, col=1)
        
        fig.add_trace(
            go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, 
                   name='PACF', marker_color='orange'),
            row=1, col=2
        )
        
        fig.add_hline(y=conf_int, line_dash="dash", line_color="red", 
                      opacity=0.5, row=1, col=2)
        fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", 
                      opacity=0.5, row=1, col=2)
        
    except Exception as e:
        st.warning(f"Error creating ACF/PACF plots: {str(e)}")
    
    fig.update_layout(height=400, showlegend=False, template='plotly_dark')
    return fig


# ==================== FORECASTING ====================
def create_future_dataframe(data, forecast_days=365):
    last_date = pd.to_datetime(data["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    future_df = pd.DataFrame({'Date': future_dates})
    future_df["t"] = np.arange(data.shape[0] + 1, data.shape[0] + forecast_days + 1)
    future_df["t_square"] = future_df["t"] ** 2
    future_df["Sin"] = np.sin((2 * np.pi * future_df["t"]) / 365.25)
    future_df["Cos"] = np.cos((2 * np.pi * future_df["t"]) / 365.25)
    
    return future_df

def forecast_with_autoreg(best_model, best_model_type, data, train_data, future_df, residuals):
    try:
        pred_trend = get_model_prediction(best_model, best_model_type, future_df)
        if pred_trend is None:
            return future_df, None
        
        pred_trend = pd.Series(pred_trend.values)
        
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
    except Exception as e:
        st.error(f"AutoReg forecasting failed: {str(e)}")
        return future_df, None

def forecast_with_arima(best_model, best_model_type, data, train_data, future_df, residuals, order=(1,1,1)):
    try:
        pred_trend = get_model_prediction(best_model, best_model_type, future_df)
        if pred_trend is None:
            return future_df, None
        
        pred_trend = pd.Series(pred_trend.values)
        
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
    except Exception as e:
        st.error(f"ARIMA forecasting failed: {str(e)}")
        return future_df, None

def forecast_with_sarimax(best_model, best_model_type, data, train_data, future_df, residuals, order=(1,1,1)):
    try:
        pred_trend = get_model_prediction(best_model, best_model_type, future_df)
        if pred_trend is None:
            return future_df, None
        
        pred_trend = pd.Series(pred_trend.values)
        
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
    except Exception as e:
        st.error(f"SARIMAX forecasting failed: {str(e)}")
        return future_df, None

def forecast_with_prophet(data, forecast_days=365):
    if not PROPHET_AVAILABLE:
        st.error("Prophet not available. Please install: pip install prophet")
        return None, None, None
    
    try:
        prophet_df = data[['Date', 'Price']].copy()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        with st.spinner("Training Prophet model..."):
            model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)
    
        future_forecast = forecast[len(data):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_forecast.columns = ['Date', 'Prophet_Price', 'Prophet_Lower', 'Prophet_Upper']
        future_forecast.reset_index(drop=True, inplace=True)
        
        return future_forecast, model, forecast
    except Exception as e:
        st.error(f"Prophet forecasting failed: {str(e)}")
        return None, None, None
    
# ==================== WALK-FORWARD BACKTESTING ====================
def walk_forward_backtest(data, model_type, window=180, step=7, initial_capital=100000):
    risk_mgr = RiskManager(initial_capital)
    confidence_calc = ConfidenceCalculator()

    results = {
        'dates': [],
        'prices': [],
        'predictions': [],
        'signals': [],
        'positions': [],
        'portfolio_values': [],
        'trades': []
    }

    for i in range(window, len(data), step):
        train_data = data.iloc[i-window:i].copy()
        
        if i + step < len(data):
            test_data = data.iloc[i:i+step].copy()
        else:
            test_data = data.iloc[i:].copy()
        
        if len(test_data) == 0:
            break
        
        try:
            if model_type == 'linear':
                model = smf.ols("Price ~ t", data=data).fit()
            elif model_type == 'exp':
                model = smf.ols("log_Price ~ t", data=data).fit()
            elif model_type == 'quad':
                model = smf.ols("Price ~ t + t_square", data=data).fit()
            elif model_type == 'add_sea':
                model = smf.ols('Price ~ Sin + Cos', data=data).fit()
            elif model_type == 'mul_sea':
                model = smf.ols('log_Price ~ Sin + Cos', data=data).fit()
            elif model_type == 'add_sea_quad':
                model = smf.ols('Price ~ t + t_square + Sin + Cos', data=data).fit()
            elif model_type == 'mul_add_sea':
                model = smf.ols('log_Price ~ t + Sin + Cos', data=data).fit()
            else:
                model = smf.ols('Price ~ t + t_square + Sin + Cos', data=data).fit()
            
            prediction = get_model_prediction(model, model_type, test_data.iloc[[0]])
            pred_price = float(prediction.iloc[0])
            
            current_price = test_data['Price'].iloc[0]
            
            train_predictions = get_model_prediction(model, model_type, data)
            residuals = data['Price'].values - train_predictions.values
            residuals_series = pd.Series(residuals)
            
            stat_confidence, lower, upper = confidence_calc.calculate_statistical_confidence(
                residuals_series, pred_price, current_price
            )
            
            if "TEST" in risk_mgr.open_positions:
                for j in range(len(test_data)):
                    current_test_price = test_data['Price'].iloc[j]
                    should_exit, exit_reason = risk_mgr.check_exit_conditions("TEST", current_test_price)
                    
                    if should_exit:
                        trade = risk_mgr.close_position("TEST", current_test_price)
                        results['trades'].append({
                            'date': test_data['Date'].iloc[j],
                            'action': 'CLOSE',
                            'price': current_test_price,
                            'pnl': trade['pnl'] if trade else 0,
                            'pnl_pct': trade['pnl_pct'] if trade else 0,
                            'reason': exit_reason
                        })
                        break
            
            if "TEST" not in risk_mgr.open_positions:
                can_trade, reason, risk_score = risk_mgr.can_trade(
                    stat_confidence, current_price, pred_price
                )
                
                if can_trade and pred_price > current_price * 1.01:
                    signal = "BUY"
                    
                    shares, position_value, position_pct = risk_mgr.calculate_position_size(
                        stat_confidence, current_price, risk_score
                    )
                    
                    if shares > 0:
                        stop_loss = risk_mgr.calculate_stop_loss(current_price)
                        take_profit, _ = risk_mgr.calculate_take_profit(current_price, pred_price)
                        
                        risk_mgr.open_position("TEST", shares, current_price, stop_loss, take_profit)
                        
                        results['trades'].append({
                            'date': test_data['Date'].iloc[0],
                            'action': 'OPEN',
                            'price': current_price,
                            'shares': shares,
                            'confidence': stat_confidence,
                            'pred_price': pred_price
                        })
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"
            
            if "TEST" in risk_mgr.open_positions and (i + step >= len(data)):
                exit_price = test_data['Price'].iloc[-1]
                trade = risk_mgr.close_position("TEST", exit_price)
                
                results['trades'].append({
                    'date': test_data['Date'].iloc[-1],
                    'action': 'CLOSE',
                    'price': exit_price,
                    'pnl': trade['pnl'] if trade else 0,
                    'pnl_pct': trade['pnl_pct'] if trade else 0,
                    'reason': 'End of backtest'
                })
            
            results['dates'].append(test_data['Date'].iloc[-1])
            results['prices'].append(test_data['Price'].iloc[-1])
            results['predictions'].append(pred_price)
            results['signals'].append(signal)
            results['positions'].append(1 if "TEST" in risk_mgr.open_positions else 0)
            
            position_value = sum([p['shares'] * test_data['Price'].iloc[-1] 
                                 for p in risk_mgr.open_positions.values()])
            results['portfolio_values'].append(risk_mgr.capital + position_value)
            
        except Exception as e:
            st.warning(f"Backtest error at iteration {i}: {str(e)}")
            continue

    performance = risk_mgr.get_performance_summary()

    return {
        'results': results,
        'performance': performance,
        'risk_manager': risk_mgr
    }

# ==================== SIGNAL GENERATION ====================
def generate_trading_signal(predictions_dict, current_price, data, regime_info):
    conf_calc = ConfidenceCalculator()
    agreement_conf, agreement_pct, direction = conf_calc.calculate_model_agreement(
        predictions_dict, current_price)

    avg_prediction = np.mean(list(predictions_dict.values()))

    stat_conf = agreement_conf

    overall_confidence = (agreement_conf * 0.6 + stat_conf * 0.4)

    regime, regime_conf, recommendation, metrics = regime_info
    strategy_params = MarketRegimeDetector.get_regime_appropriate_strategy(regime)

    regime_adjusted_confidence = overall_confidence * regime_conf

    expected_return = ((avg_prediction / current_price) - 1) * 100

    if regime_adjusted_confidence >= strategy_params['confidence_threshold']:
        if expected_return > 2 and direction == "BULLISH":
            signal = "STRONG BUY"
            signal_strength = min(regime_adjusted_confidence * abs(expected_return) / 10, 1.0)
        elif expected_return < -2 and direction == "BEARISH":
            signal = "STRONG SELL"
            signal_strength = min(regime_adjusted_confidence * abs(expected_return) / 10, 1.0)
        elif expected_return > 1:
            signal = "WEAK BUY"
            signal_strength = regime_adjusted_confidence * 0.6
        elif expected_return < -1:
            signal = "WEAK SELL"
            signal_strength = regime_adjusted_confidence * 0.6
        else:
            signal = "HOLD"
            signal_strength = 0.3
    else:
        signal = "NO SIGNAL"
        signal_strength = regime_adjusted_confidence * 0.5

    return {
        'signal': signal,
        'signal_strength': signal_strength,
        'direction': direction,
        'confidence': regime_adjusted_confidence,
        'model_agreement': agreement_pct,
        'expected_return': expected_return,
        'avg_prediction': avg_prediction,
        'regime': regime,
        'regime_confidence': regime_conf,
        'recommendation': recommendation,
        'strategy_params': strategy_params,
        'all_predictions': predictions_dict
    }

# ==================== VISUALIZATION ====================
def plot_candlestick(data, ticker):
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
    if not PROPHET_AVAILABLE or prophet_model is None:
        return None
    try:
        from prophet.plot import plot_components_plotly
        
        fig = plot_components_plotly(prophet_model, forecast)
        fig.update_layout(
            template='plotly_dark',
            height=800
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create Prophet components plot: {str(e)}")
        return None
    
def plot_price_trends(data, ticker):
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
    models = list(rmse_scores.keys())
    scores = list(rmse_scores.values())
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            marker_color=['#00ff41' if s == min(scores) else '#ff0266' for s in scores],
            text=[f'{s:.2f}' for s in scores],
            textposition='auto'
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
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='cyan', width=2)
    ))

    if 'AutoReg_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['AutoReg_Price'],
            mode='lines',
            name='AutoReg Forecast',
            line=dict(color='orange', width=2, dash='dash')
        ))

    if 'ARIMA_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['ARIMA_Price'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='magenta', width=2, dash='dot')
        ))

    if 'SARIMAX_Price' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['SARIMAX_Price'],
            mode='lines',
            name='SARIMAX Forecast',
            line=dict(color='yellow', width=2, dash='dashdot')
        ))

    if prophet_forecast is not None and 'Prophet_Price' in prophet_forecast.columns:
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
    
    fig.add_trace(
        go.Scatter(x=data['Date'], y=residuals, mode='lines',
                   name='Residuals', line=dict(color='yellow', width=1)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(
        go.Histogram(x=residuals, name='Distribution',
                    marker=dict(color='orange'), nbinsx=30),
        row=1, col=2
    )

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

    fitted = data['Price'].values[:len(residuals)] - residuals.values
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
    return df.to_csv(index=False).encode('utf-8')

def create_excel_download(data, forecast_df, rmse_scores):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Historical_Data', index=False)
            forecast_df.to_excel(writer, sheet_name='Forecast_Data', index=False)
            rmse_df = pd.DataFrame({
                'Model': list(rmse_scores.keys()),
                'RMSE': list(rmse_scores.values())
            })
            rmse_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

# ==================== SAFETY CHECK CLASSES ====================
class OverfittingDetector:
    @staticmethod
    def check_overfitting(train_rmse, test_rmse, complexity_penalty=1.1):
        ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
        if ratio > 2.0:
            return "HIGH", f"Severe overfitting (test/train ratio: {ratio:.2f})"
        elif ratio > 1.5:
            return "MEDIUM", f"Possible overfitting (test/train ratio: {ratio:.2f})"
        else:
            return "LOW", "Good generalization"

class LiquidityAnalyzer:
    def __init__(self):
        self.min_daily_volume = 100000
        self.min_dollar_volume = 1000000
    
    def check_liquidity(self, data, recommended_shares):
        avg_volume = data['Volume'].tail(30).mean()
        position_to_volume = recommended_shares / avg_volume if avg_volume > 0 else float('inf')
        
        return {
            'sufficient_volume': avg_volume > self.min_daily_volume,
            'position_size_reasonable': position_to_volume < 0.01,
            'avg_daily_volume': avg_volume,
            'position_volume_ratio': position_to_volume
        }

class CircuitBreaker:
    def __init__(self):
        self.consecutive_losses_limit = 3
    
    def should_halt_trading(self, performance, market_conditions):
        if performance.get('consecutive_losing_trades', 0) >= self.consecutive_losses_limit:
            return True, f"Too many consecutive losses: {performance['consecutive_losing_trades']}"
        if market_conditions.get('volatility_ratio', 1) > 4:
            return True, f"Extreme volatility: {market_conditions['volatility_ratio']:.1f}x"
        return False, "OK"

def enhanced_data_validation(data, ticker):
    """Comprehensive data quality checks"""
    checks = {
        'has_data': len(data) > 0,
        'no_nan_prices': not data['Price'].isna().any(),
        'sufficient_variation': data['Price'].std() > 0,
        'no_zeros': (data['Price'] > 0).all(),
    }
    
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Data quality issues: {failed}")

def detect_anomalies_and_black_swans(data):
    """Detect unusual market conditions"""
    returns = data['Price'].pct_change().dropna()
    if len(returns) < 20:
        return {'alerts': [], 'volatility_ratio': 1, 'current_environment': "INSUFFICIENT_DATA"}
        
    current_volatility = returns.rolling(20).std().iloc[-1]
    historical_volatility = returns.std()
    volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
    
    alerts = []
    if volatility_ratio > 3:
        alerts.append(f"🚨 EXTREME VOLATILITY: {volatility_ratio:.1f}x normal")
    
    return {
        'alerts': alerts,
        'volatility_ratio': volatility_ratio,
        'current_environment': "NORMAL" if volatility_ratio < 2 else "STRESSED"
    }

# ==================== MAIN APP ====================
def main():
    st.set_page_config(
        page_title="Stock Forecast Pro - Enhanced",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        .warning-box {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .success-box {
            background: linear-gradient(135deg, #06df6f 0%, #06aa56 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📈 Professional Stock Forecasting Platform</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Advanced Time Series Analysis with Risk Management & Market Regime Detection**")

    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Risk Management")

    use_risk_mgmt = st.sidebar.checkbox("Enable Risk Management", value=True)

    if use_risk_mgmt:
        initial_capital = st.sidebar.number_input(
            "Trading Capital (₹)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        max_position_pct = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum % of capital to risk per trade"
        ) / 100
        
        stop_loss_pct = st.sidebar.slider(
            "Stop Loss (%)",
            min_value=2,
            max_value=15,
            value=5,
            help="Automatic exit if loss exceeds this %"
        ) / 100
        
        take_profit_pct = st.sidebar.slider(
            "Take Profit (%)",
            min_value=5,
            max_value=30,
            value=10,
            help="Target profit %"
        ) / 100
        
        confidence_threshold = st.sidebar.slider(
            "Min Confidence (%)",
            min_value=50,
            max_value=95,
            value=70,
            help="Only trade if confidence > this threshold"
        ) / 100

    popular_tickers = {
        "Reliance Industries": "RELIANCE.NS",
        "SBI ETF (GOLD)":"SETFGOLD.NS",
        "Idea Vodafone": "IDEA.NS",
        "TATA Gold": "TATAGOLD.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ITC": "ITC.NS",
        "State Bank": "SBIN.NS",
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Custom": "CUSTOM"
    }

    selected_ticker = st.sidebar.selectbox(
        "Select Stock",
        list(popular_tickers.keys()),
        index=0
    )

    if popular_tickers[selected_ticker] == "CUSTOM":
        ticker = st.sidebar.text_input("Enter Ticker:", "AAPL")
    else:
        ticker = popular_tickers[selected_ticker]

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = col2.date_input("End Date", pd.to_datetime("today"))

    is_valid, message = validate_dates(start_date, end_date)
    if not is_valid:
        st.sidebar.error(message)

    data_length = (end_date - start_date).days
    max_safe_forecast = min(int(data_length * 0.10), 60)

    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=7,
        max_value=90,
        value=min(30, max_safe_forecast),
        help=f"⚠️ Recommended max: {max_safe_forecast} days (10% of data length)"
    )

    if forecast_days > 60:
        st.sidebar.warning("⚠️ Long-term forecasts (>60 days) are unreliable!")

    test_size = st.sidebar.slider("Test Split (%)", 10, 30, 20) / 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 ARIMA Configuration")
    arima_p = st.sidebar.selectbox("AR Order (p)", [0, 1, 2, 3], index=1)
    arima_d = st.sidebar.selectbox("Differencing (d)", [0, 1, 2], index=1)
    arima_q = st.sidebar.selectbox("MA Order (q)", [0, 1, 2, 3], index=1)
    arima_order = (arima_p, arima_d, arima_q)

    available_methods = ["AutoReg", "ARIMA", "SARIMAX"]
    if PROPHET_AVAILABLE:
        available_methods.append("Prophet")
    else:
        st.sidebar.warning("⚠️ Prophet not installed")

    forecast_methods = st.sidebar.multiselect(
        "Select Forecast Methods",
        available_methods,
        default=["SARIMAX", "Prophet"] if PROPHET_AVAILABLE else ["SARIMAX"]
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("🔬 Advanced Options"):
        enable_regime_detection = st.checkbox("Market Regime Detection", value=True)
        enable_walk_forward = st.checkbox("Walk-Forward Validation", value=True)
        show_diagnostics = st.checkbox("Show Full Diagnostics", value=False)

    st.sidebar.markdown("---")
    with st.sidebar.expander("📖 Quick Guide"):
        st.markdown("""
        **How to Use:**
        1. Select stock & date range
        2. Check market regime
        3. Review model performance
        4. Verify walk-forward results
        5. Analyze trading signal
        6. Paper trade first!
        
        **⚠️ ALWAYS:**
        - Set stop loss
        - Start small
        - Never risk >2%/trade
        - This is NOT financial advice
        """)

    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        
        if not is_valid:
            st.error(message)
            return
        
        if use_risk_mgmt:
            risk_mgr = RiskManager(
                capital=initial_capital,
                max_position_pct=max_position_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
        else:
            risk_mgr = None
        
        # ==================== DATA LOADING & VALIDATION ====================
        
        with st.spinner(f"📡 Fetching data for {ticker}..."):
            raw_data = load_stock_data(ticker, start_date, end_date)
        
        if raw_data is None or raw_data.empty:
            st.error("❌ No data available. Please check the ticker symbol.")
            return
        
        with st.spinner("⚙️ Preparing data..."):
            data = prepare_data(raw_data)
        
        # ==================== SAFETY CHECKS ====================
        
        st.markdown("---")
        st.header("🔍 Data Quality & Safety Checks")
        
        # 1. Data quality validation
        try:
            enhanced_data_validation(data, ticker)
            st.success("✅ Data quality validation passed")
        except ValueError as e:
            st.error(f"❌ Data quality issue: {e}")
            return

        # 2. Black swan detection
        anomaly_check = detect_anomalies_and_black_swans(data)
        if anomaly_check['alerts']:
            for alert in anomaly_check['alerts']:
                st.warning(alert)
        else:
            st.success("✅ No extreme market conditions detected")
        
        st.success(f"✅ Successfully loaded {len(data)} days of data")
        
        currency = get_currency_symbol(ticker)
        
        # ==================== MARKET REGIME DETECTION ====================
        
        if enable_regime_detection:
            st.markdown("---")
            st.header("🌐 Market Regime Analysis")
            
            regime, regime_conf, recommendation, metrics = MarketRegimeDetector.detect_regime(data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                regime_color = {
                    "BULL_STABLE": "🟢",
                    "BULL_VOLATILE": "🟡",
                    "BEAR": "🔴",
                    "SIDEWAYS": "🟡",
                    "CHOPPY": "🔴"
                }
                st.metric("Current Regime", f"{regime_color.get(regime, '⚪')} {regime}")
            
            with col2:
                st.metric("Regime Confidence", f"{regime_conf*100:.0f}%")
            
            with col3:
                st.metric("Short-Term Trend", f"{metrics['short_trend']:.2f}%")
            
            with col4:
                st.metric("Volatility", f"{metrics['volatility']:.2f}%")
            
            if regime in ["BULL_STABLE"]:
                st.markdown(f'<div class="success-box">✅ {recommendation}</div>', 
                        unsafe_allow_html=True)
            elif regime in ["BEAR", "CHOPPY"]:
                st.markdown(f'<div class="warning-box">⚠️ {recommendation}</div>', 
                        unsafe_allow_html=True)
            else:
                st.info(recommendation)
            
            strategy_params = MarketRegimeDetector.get_regime_appropriate_strategy(regime)
            
            with st.expander("📋 Regime-Based Strategy Parameters"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Position Size Adjustment", 
                        f"{strategy_params['position_size_multiplier']*100:.0f}%")
                col2.metric("Min Confidence Required", 
                        f"{strategy_params['confidence_threshold']*100:.0f}%")
                col3.metric("Preferred Direction", strategy_params['preferred_direction'])
        
        # ==================== MODEL TRAINING ====================
        
        st.markdown("---")
        st.header("🤖 Model Training & Evaluation")
        
        with st.spinner("Training models..."):
            train_data, test_data = proper_time_series_split(data, test_size)
            
            result = train_all_models(train_data, test_data)
            
            if result[0] is None:
                st.error("Model training failed. Please check your data.")
                return
            
            best_model, rmse_scores, best_model_name, all_models, best_model_type = result
        
        st.success(f"🏆 Best Model: **{best_model_name}** (RMSE: {rmse_scores[best_model_name]:.2f})")
        
        # 3. Overfitting detection
        if len(rmse_scores) >= 2:
            train_scores = [v for k, v in rmse_scores.items() if 'train' in k.lower() or k == best_model_name]
            test_scores = [v for k, v in rmse_scores.items() if 'test' in k.lower() or k != best_model_name]
            
            if train_scores and test_scores:
                train_rmse = min(train_scores)
                test_rmse = min(test_scores)
                
                overfitting_status, overfitting_msg = OverfittingDetector.check_overfitting(
                    train_rmse, test_rmse
                )
                
                if "HIGH" in overfitting_status:
                    st.error(f"🚨 {overfitting_msg} - DO NOT TRADE")
                elif "MEDIUM" in overfitting_status:
                    st.warning(f"⚠️ {overfitting_msg} - Trade with caution")
                else:
                    st.success(f"✅ {overfitting_msg}")
        
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
        
        # ==================== WALK-FORWARD VALIDATION ====================
        
        if enable_walk_forward:
            st.markdown("---")
            st.header("🔄 Walk-Forward Validation")
            
            st.info("""
            **What is Walk-Forward Validation?**
            
            Instead of training once and testing once, this method:
            1. Trains on historical data (e.g., 180 days)
            2. Tests on next period (e.g., 7 days)
            3. Moves forward and repeats
            
            This simulates real trading where you only have past data, giving honest performance metrics.
            """)
            
            with st.spinner("Running walk-forward validation... This may take a minute..."):
                backtest_results = walk_forward_backtest(
                    data=data,
                    model_type=best_model_type,
                    window=min(180, int(len(data) * 0.6)),
                    step=7,
                    initial_capital=initial_capital if use_risk_mgmt else 100000
                )
            
            performance = backtest_results['performance']
            
            st.subheader("💰 Backtest Performance")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric(
                "Total Return",
                f"{performance['total_return_pct']:.2f}%",
                delta=f"₹{performance['total_pnl']:.2f}"
            )
            
            col2.metric(
                "Win Rate",
                f"{performance['win_rate']:.1f}%",
                delta="Good" if performance['win_rate'] > 55 else "Poor"
            )
            
            winning_trades = performance.get('winning_trades', 0)
            losing_trades = performance.get('losing_trades', 0)

            col3.metric(
                "Total Trades",
                f"{performance['total_trades']}",
                delta=f"{winning_trades}W / {losing_trades}L"
            )
            
            # FIXED: Proper profit factor display
            profit_factor_display = display_profit_factor(performance['profit_factor'])
            profit_factor_delta = "Perfect" if performance['profit_factor'] == float('inf') else ("Good" if performance['profit_factor'] > 1.5 else "Poor")
            
            col4.metric(
                "Profit Factor",
                profit_factor_display,
                delta=profit_factor_delta
            )
            
            col5.metric(
                "Avg Profit/Loss",
                f"₹{performance['avg_profit']:.0f}",
                delta=f"-₹{performance['avg_loss']:.0f}"
            )
            
            st.markdown("---")
            st.subheader("📊 Model Readiness Assessment")
            
            # FIXED: Proper profit factor check
            profit_factor_check = (performance['profit_factor'] == float('inf') or performance['profit_factor'] > 1.5)
            
            checks = {
                "Win Rate > 55%": performance['win_rate'] > 55,
                "Profit Factor > 1.5": profit_factor_check,
                "Total Return > 10%": performance['total_return_pct'] > 10,
                "At Least 10 Trades": performance['total_trades'] >= 10
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                for check, passed in checks.items():
                    if passed:
                        st.markdown(f"✅ **{check}**")
                    else:
                        st.markdown(f"❌ **{check}**")
            
            with col2:
                readiness_score = (passed_checks / total_checks) * 100
                st.metric("Readiness Score", f"{readiness_score:.0f}%")
                
                if readiness_score >= 75:
                    st.success("✅ Model Ready")
                elif readiness_score >= 50:
                    st.warning("⚠️ Use Caution")
                else:
                    st.error("❌ Not Ready")
            
            if performance['win_rate'] < 55:
                st.error("""
                ⚠️ **WARNING: Low Win Rate**
                
                A win rate below 55% means the model is not consistently accurate.
                - Do NOT use for real trading yet
                - Try different parameters
                - Collect more data
                - Consider market regime
                """)
            
            if not profit_factor_check:
                st.warning("""
                ⚠️ **WARNING: Low Profit Factor**
                
                Profit factor < 1.5 means wins aren't big enough compared to losses.
                - Adjust take profit targets
                - Tighten stop losses
                - Trade only high-confidence signals
                """)
            
            if len(backtest_results['results']['portfolio_values']) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=backtest_results['results']['dates'],
                    y=backtest_results['results']['portfolio_values'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00ff41', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(
                    y=initial_capital if use_risk_mgmt else 100000,
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Starting Capital"
                )
                
                fig.update_layout(
                    title="Portfolio Value Over Time (Walk-Forward Test)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (₹)",
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show trade details
            if backtest_results['results']['trades']:
                with st.expander("📜 Trade History"):
                    trades_df = pd.DataFrame(backtest_results['results']['trades'])
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
        
        # ==================== FORECASTING ====================
        
        st.markdown("---")
        st.header("🔮 Price Forecast")
        
        with st.spinner("Generating forecasts..."):
            model_mapping = {
                'rmse_linear': 'linear',
                'rmse_Exp': 'exp',
                'rmse_Quad': 'quad',
                'rmse_add_sea': 'add_sea',
                'rmse_Mult_sea': 'mul_sea',
                'rmse_add_sea_quad': 'add_sea_quad',
                'rmse_Mult_add_sea': 'mul_add_sea'
            }
            
            production_model_type = model_mapping[best_model_name]
            
            if production_model_type == 'linear':
                production_model = smf.ols("Price ~ t", data=data).fit()
            elif production_model_type == 'exp':
                production_model = smf.ols("log_Price ~ t", data=data).fit()
            elif production_model_type == 'quad':
                production_model = smf.ols("Price ~ t + t_square", data=data).fit()
            elif production_model_type == 'add_sea':
                production_model = smf.ols('Price ~ Sin + Cos', data=data).fit()
            elif production_model_type == 'mul_sea':
                production_model = smf.ols('log_Price ~ Sin + Cos', data=data).fit()
            elif production_model_type == 'add_sea_quad':
                production_model = smf.ols('Price ~ t + t_square + Sin + Cos', data=data).fit()
            elif production_model_type == 'mul_add_sea':
                production_model = smf.ols('log_Price ~ t + Sin + Cos', data=data).fit()
            else:
                production_model = best_model
            
            production_residuals = calculate_residuals(data, production_model, production_model_type)
            
            if production_residuals is None:
                st.error("Error calculating residuals")
                return
            
            future_df = create_future_dataframe(data, forecast_days)
            
            prophet_forecast = None
            prophet_model = None
            full_prophet_forecast = None
            
            if "AutoReg" in forecast_methods:
                future_df, ar_model = forecast_with_autoreg(
                    production_model, production_model_type, data, data, 
                    future_df, production_residuals
                )
            
            if "ARIMA" in forecast_methods:
                future_df, arima_model = forecast_with_arima(
                    production_model, production_model_type, data, data, 
                    future_df, production_residuals, arima_order
                )
            
            if "SARIMAX" in forecast_methods:
                future_df, sarimax_model = forecast_with_sarimax(
                    production_model, production_model_type, data, data, 
                    future_df, production_residuals, arima_order
                )
            
            if "Prophet" in forecast_methods and PROPHET_AVAILABLE:
                prophet_forecast, prophet_model, full_prophet_forecast = forecast_with_prophet(
                    data, forecast_days
                )
                if prophet_forecast is not None:
                    future_df = future_df.merge(prophet_forecast, on='Date', how='left')
        
        st.plotly_chart(
            plot_forecast_comparison(data, future_df, ticker, prophet_forecast),
            use_container_width=True
        )
        
        if "Prophet" in forecast_methods and prophet_model is not None:
            with st.expander("📊 Prophet Model Components (Trend & Seasonality)", expanded=False):
                components_fig = plot_prophet_components(prophet_model, full_prophet_forecast)
                if components_fig:
                    st.plotly_chart(components_fig, use_container_width=True)
        
        # ==================== TRADING SIGNAL GENERATION ====================
        
        st.markdown("---")
        st.header("🎯 Trading Signal Analysis")
        
        predictions_dict = {}
        for method in forecast_methods:
            col_name = f"{method}_Price"
            if col_name in future_df.columns and not future_df[col_name].isna().all():
                predictions_dict[method] = future_df[col_name].iloc[-1]
        
        if predictions_dict:
            current_price = data['Price'].iloc[-1]
            
            regime_info = (regime, regime_conf, recommendation, metrics) if enable_regime_detection else ("UNKNOWN", 0.5, "No regime detection", {})
            
            signal_info = generate_trading_signal(
                predictions_dict,
                current_price,
                data,
                regime_info
            )
            
            can_trade = False
            reason = "No risk assessment performed"
            risk_score = 0.0
            shares = 0
            position_value = 0
            position_pct = 0
            stop_loss = 0
            take_profit = 0
            tp_pct = 0
            risk_reward = 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_color = {
                    "STRONG BUY": "🟢",
                    "WEAK BUY": "🟡",
                    "HOLD": "⚪",
                    "WEAK SELL": "🟠",
                    "STRONG SELL": "🔴",
                    "NO SIGNAL": "⚫"
                }
                st.metric(
                    "Signal",
                    f"{signal_color.get(signal_info['signal'], '⚪')} {signal_info['signal']}"
                )
            
            with col2:
                st.metric(
                    "Signal Strength",
                    f"{signal_info['signal_strength']*100:.0f}%"
                )
            
            with col3:
                st.metric(
                    "Model Agreement",
                    f"{signal_info['model_agreement']:.0f}%"
                )
            
            with col4:
                st.metric(
                    "Expected Return",
                    f"{signal_info['expected_return']:.2f}%"
                )
            
            # ==================== TRADING SAFETY CHECKS ====================
            
            if use_risk_mgmt and signal_info['signal'] in ["STRONG BUY", "WEAK BUY"]:
                st.markdown("---")
                st.subheader("💼 Position Sizing & Safety Checks")
                
                can_trade, reason, risk_score = risk_mgr.can_trade(
                    signal_info['confidence'],
                    current_price,
                    signal_info['avg_prediction']
                )
                
                if can_trade:
                    shares, position_value, position_pct = risk_mgr.calculate_position_size(
                        signal_info['confidence'],
                        current_price,
                        risk_score
                    )
                    
                    # 4. Liquidity check
                    if shares > 0:
                        liquidity_analyzer = LiquidityAnalyzer()
                        liquidity_status = liquidity_analyzer.check_liquidity(data, shares)
                        
                        if not liquidity_status['position_size_reasonable']:
                            st.error(f"❌ Position too large: {liquidity_status['position_volume_ratio']:.2%} of daily volume")
                            can_trade = False
                            reason = "Position size exceeds liquidity constraints"
                        elif not liquidity_status['sufficient_volume']:
                            st.warning(f"⚠️ Low liquidity stock: avg volume {liquidity_status['avg_daily_volume']:,.0f} shares")
                        else:
                            st.success(f"✅ Position size OK: {liquidity_status['position_volume_ratio']:.2%} of daily volume")
                    
                    # 5. Circuit breaker
                    if can_trade:
                        circuit_breaker = CircuitBreaker()
                        should_halt, halt_reason = circuit_breaker.should_halt_trading(
                            performance if 'performance' in locals() else {},
                            anomaly_check
                        )
                        if should_halt:
                            st.error(f"🚨 TRADING HALTED: {halt_reason}")
                            signal_info['signal'] = "NO SIGNAL - SAFETY HALT"
                            can_trade = False
                            reason = halt_reason
                
                # Display trade recommendation
                if can_trade:
                    stop_loss = risk_mgr.calculate_stop_loss(current_price)
                    take_profit, tp_pct = risk_mgr.calculate_take_profit(
                        current_price,
                        signal_info['avg_prediction']
                    )
                    
                    risk_reward = risk_mgr.calculate_risk_reward_ratio(
                        current_price,
                        stop_loss,
                        take_profit
                    )
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    col1.metric("Buy Shares", f"{shares}")
                    col2.metric("Position Value", f"₹{position_value:,.0f}")
                    col3.metric("Stop Loss", f"₹{stop_loss:.2f}")
                    col4.metric("Take Profit", f"₹{take_profit:.2f}")
                    col5.metric("Risk:Reward", f"1:{risk_reward:.2f}")
                    
                    st.info(f"""
                    **📋 Trade Summary:**
                    - **Entry Price:** ₹{current_price:.2f}
                    - **Target Price:** ₹{take_profit:.2f} (+{tp_pct:.2f}%)
                    - **Stop Loss:** ₹{stop_loss:.2f} (-{stop_loss_pct*100:.2f}%)
                    - **Position Size:** {position_pct:.1f}% of capital
                    - **Max Loss:** ₹{(current_price - stop_loss) * shares:.2f}
                    - **Expected Profit:** ₹{(take_profit - current_price) * shares:.2f}
                    """)
                    
                    if risk_reward >= 2.0:
                        st.success("✅ Good Risk-Reward Ratio (>2:1)")
                    elif risk_reward >= 1.5:
                        st.warning("⚠️ Acceptable Risk-Reward (1.5:1 to 2:1)")
                    else:
                        st.error("❌ Poor Risk-Reward (<1.5:1) - Consider avoiding")
                    
                else:
                    st.error(f"❌ Trade Not Approved: {reason}")
            
            with st.expander("📊 Individual Model Predictions"):
                pred_df = pd.DataFrame({
                    'Model': list(signal_info['all_predictions'].keys()),
                    'Prediction': [f"₹{v:.2f}" for v in signal_info['all_predictions'].values()],
                    'Change': [f"{((v/current_price)-1)*100:+.2f}%" 
                            for v in signal_info['all_predictions'].values()]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # ==================== DIAGNOSTICS ====================
        
        if show_diagnostics:
            st.markdown("---")
            st.header("🔍 Model Diagnostics")
            
            residuals = calculate_residuals(train_data, best_model, best_model_type)
            
            if residuals is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Residual", f"{residuals.mean():.4f}")
                col2.metric("Std Dev", f"{residuals.std():.4f}")
                col3.metric("Max Abs Error", f"{abs(residuals).max():.2f}")
                
                st.plotly_chart(plot_residuals_analysis(residuals, train_data), 
                            use_container_width=True)
                
                st.subheader("Autocorrelation Analysis")
                acf_pacf_fig = create_acf_pacf_plots(residuals)
                st.plotly_chart(acf_pacf_fig, use_container_width=True)
        
        # ==================== HISTORICAL DATA TAB ====================
        
        st.markdown("---")
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Latest Price", f"{currency}{data['Price'].iloc[-1]:.2f}")
        col2.metric("Highest", f"{currency}{data['High'].max():.2f}")
        col3.metric("Lowest", f"{currency}{data['Low'].min():.2f}")
        col4.metric("Avg Volume", f"{data['Volume'].mean()/1e6:.2f}M")
        col5.metric("Data Points", f"{len(data)}")
        
        st.plotly_chart(plot_price_trends(data, ticker), use_container_width=True)
        
        with st.expander("🕯️ Candlestick Chart"):
            st.plotly_chart(plot_candlestick(data, ticker), use_container_width=True)
        
        with st.expander("📋 Recent Data"):
            st.dataframe(
                data[['Date', 'Open', 'High', 'Low', 'Price', 'Volume']].tail(20),
                use_container_width=True
            )
        
        # ==================== EXPORT ====================
        
        st.markdown("---")
        st.header("💾 Export Data & Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Historical Data")
            historical_csv = convert_to_csv(data)
            st.download_button(
                label="📥 Download CSV",
                data=historical_csv,
                file_name=f"{ticker}_historical.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🔮 Forecast Data")
            forecast_csv = convert_to_csv(future_df)
            st.download_button(
                label="📥 Download CSV",
                data=forecast_csv,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.markdown("#### 📈 Complete Report")
            excel_data = create_excel_download(data, future_df, rmse_scores)
            if excel_data:
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{ticker}_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        st.markdown("---")
        st.subheader("📄 Analysis Report")
        
        # FIXED: Proper profit factor display in report
        profit_factor_report = display_profit_factor(performance['profit_factor']) if 'performance' in locals() else "N/A"
        
        report = f"""
Stock Analysis Report - {ticker}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Market Conditions
=================
Current Regime: {regime if enable_regime_detection else 'N/A'}
Regime Confidence: {regime_conf*100:.0f}%
Recommendation: {recommendation if enable_regime_detection else 'N/A'}

Data Summary
============
Analysis Period: {start_date} to {end_date}
Total Days: {len(data)}
Current Price: ₹{data['Price'].iloc[-1]:.2f}
52-Week High: ₹{data['High'].max():.2f}
52-Week Low: ₹{data['Low'].min():.2f}

Model Performance
=================
Best Model: {best_model_name}
RMSE: {rmse_scores[best_model_name]:.2f}
"""
        
        if enable_walk_forward:
            report += f"""
Backtesting Results (Walk-Forward)
===================================
Total Return: {performance['total_return_pct']:.2f}%
Win Rate: {performance['win_rate']:.1f}%
Profit Factor: {profit_factor_report}
Total Trades: {performance['total_trades']}
Readiness Score: {readiness_score:.0f}%
"""
        
        if predictions_dict:
            report += f"""
Trading Signal
==============
Signal: {signal_info['signal']}
Signal Strength: {signal_info['signal_strength']*100:.0f}%
Model Agreement: {signal_info['model_agreement']:.0f}%
Expected Return: {signal_info['expected_return']:.2f}%

Predictions ({forecast_days} days ahead)
"""
            for model, pred in predictions_dict.items():
                change = ((pred / current_price) - 1) * 100
                report += f"- {model}: ₹{pred:.2f} ({change:+.2f}%)\n"
        
        if use_risk_mgmt and can_trade:
            report += f"""
Risk Management Recommendation
===============================
Position Size: {shares} shares (₹{position_value:,.0f})
Entry Price: ₹{current_price:.2f}
Stop Loss: ₹{stop_loss:.2f} (-{stop_loss_pct*100:.1f}%)
Take Profit: ₹{take_profit:.2f} (+{tp_pct:.1f}%)
Risk:Reward Ratio: 1:{risk_reward:.2f}
Maximum Loss: ₹{(current_price - stop_loss) * shares:,.2f}
Expected Profit: ₹{(take_profit - current_price) * shares:,.2f}
"""
        
        report += """
Disclaimer
==========
⚠️ IMPORTANT WARNINGS:
This analysis is for EDUCATIONAL PURPOSES ONLY and is NOT financial advice.

• Stock trading involves substantial risk of loss
• Past performance does NOT guarantee future results
• Machine learning models CAN and WILL be wrong
• Black swan events are unpredictable
• You can lose 100% of your invested capital

Before Trading Real Money:
• Paper trade for minimum 3-6 months
• Start with money you can afford to lose
• Never use borrowed money or emergency funds
• Consult a licensed financial advisor
• Do your own research (DYOR)
• Understand the risks completely

Risk Management is MANDATORY:
• Always use stop losses
• Never risk more than 2% per trade
• Diversify your portfolio
• Keep detailed trading journal
• Review and learn from losses

The creators and distributors of this tool accept NO LIABILITY for any
financial losses incurred through the use of this software.
USE AT YOUR OWN RISK.
"""
        
        st.markdown(report)
        
        st.download_button(
            label="📥 Download Full Report (TXT)",
            data=report,
            file_name=f"{ticker}_full_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # ==================== EDUCATIONAL SECTION ====================
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Learn More")

    with st.sidebar.expander("⚠️ Risk Warnings"):
        st.markdown("""
        **CRITICAL WARNINGS:**
        
        ❌ This is NOT financial advice
        ❌ You CAN lose all your money
        ❌ Models CAN be completely wrong
        ❌ Past results ≠ Future results
        
        ✅ **Before Real Trading:**
        - Paper trade 3-6 months
        - Start with ₹5,000-10,000 max
        - ALWAYS use stop losses
        - Never risk >2% per trade
        - Consult financial advisor
        """)

    with st.sidebar.expander("📖 Key Terms"):
        st.markdown("""
        **RMSE:** Lower = Better accuracy
        
        **Win Rate:** >55% is good
        
        **Profit Factor:** >1.5 needed (∞ = Perfect)
        
        **Confidence:** >70% to trade
        
        **Risk:Reward:** >1.5:1 minimum
        
        **Stop Loss:** Auto-exit on loss
        
        **Take Profit:** Auto-exit on gain
        """)

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **🎯 Features:**

    ✅ Multiple ML models
    ✅ Market regime detection
    ✅ Walk-forward validation
    ✅ Risk management system
    ✅ Position sizing calculator
    ✅ Confidence scoring
    ✅ Stop loss / Take profit

    **⚠️ Remember:**
    - Paper trade FIRST
    - Start SMALL
    - Use STOP LOSSES
    - This is NOT advice!
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with ❤️ using Streamlit | Data: Yahoo Finance")
    st.sidebar.caption("⚠️ For Educational Purposes Only")
    st.sidebar.caption("Version 2.2 - Fixed Profit Factor Calculation")

if __name__ == "__main__":
    main()
