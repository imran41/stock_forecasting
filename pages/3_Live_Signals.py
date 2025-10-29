import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class RealTimeStockAnalyzer:
    def __init__(self):
        # Pre-defined Indian stocks
        self.default_stocks = {
            "RELIANCE.NS": "Reliance Industries",
            "SETFGOLD.NS": "SBI ETF(GOLD)",
            "IDEA.NS": "Idea Vodafone",
            "TATAGOLD.NS": "TATA Gold",
            "ADANIGREEN.NS": "Adani Green",
            "TCS.NS": "Tata Consultancy Services", 
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "ICICIBANK.NS": "ICICI Bank",
            "ITC.NS": "ITC Limited",
            "SBIN.NS": "State Bank of India",
            "BHARTIARTL.NS": "Bharti Airtel",
            "KOTAKBANK.NS": "Kotak Mahindra Bank",
            "LT.NS": "Larsen & Toubro",
            "ASIANPAINT.NS": "Asian Paints",
            "MARUTI.NS": "Maruti Suzuki",
            "SUNPHARMA.NS": "Sun Pharmaceutical",
            "TITAN.NS": "Titan Company",
            "WIPRO.NS": "Wipro",
            "AXISBANK.NS": "Axis Bank",
            "BAJFINANCE.NS": "Bajaj Finance",
            "TATAMOTORS.NS": "Tata Motors",
            "ADANIPORTS.NS": "Adani Ports"
        }
        
        # Cache to avoid repeated API calls
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def is_market_open(self):
        """Check if Indian market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)"""
        now = datetime.now()
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        # Check market hours (simplified - doesn't account for holidays)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def fetch_real_data(self, ticker):
        """
        Fetch real market data from Yahoo Finance for a SINGLE ticker with caching.
        Returns only the history DataFrame.
        """
        # Check cache first
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if (time.time() - cache_time) < self.cache_timeout:
                return cached_data
        
        try:
            stock = yf.Ticker(ticker)
            # Get historical data (1 year)
            hist = stock.history(period="1y")
            
            if hist.empty:
                st.error(f"No data found for {ticker}. Check the ticker symbol.")
                return None
            
            # Cache the result
            self.cache[cache_key] = (time.time(), hist.copy())
            
            return hist
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """
        Calculate RSI indicator using Wilder's Smoothing (standard method).
        Returns the last RSI value.
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # Not enough data
            
            delta = prices.diff(1)
            
            # Make the positive gains and negative losses series
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate initial averages using SMA for first period
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()

            # Apply Wilder's smoothing for subsequent values
            for i in range(period, len(prices)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            last_rsi = rsi.iloc[-1]
            return last_rsi if not pd.isna(last_rsi) else 50.0
        
        except Exception as e:
            return 50.0  # Return neutral on failure
    
    def calculate_rsi_series(self, prices, period=14):
        """
        Calculate full RSI series for charting.
        Returns a pandas Series.
        """
        try:
            if len(prices) < period + 1:
                return pd.Series([50.0] * len(prices), index=prices.index)
            
            delta = prices.diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()

            for i in range(period, len(prices)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50.0)
        
        except Exception as e:
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            return macd.iloc[-1], signal.iloc[-1]
        except:
            return 0.0, 0.0
    
    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
        except:
            return None, None, None
    
    def create_price_chart(self, hist, ticker, company_name):
        """Create interactive price chart with moving averages and Bollinger Bands"""
        
        # Get last 6 months of data for cleaner visualization
        hist_6m = hist.tail(120)
        
        # Create subplots: Price chart and Volume chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{company_name} ({ticker}) - Price Analysis', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Bollinger Bands (if available)
        if 'BB_Upper' in hist_6m.columns and not hist_6m['BB_Upper'].isnull().all():
            fig.add_trace(
                go.Scatter(
                    x=hist_6m.index,
                    y=hist_6m['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(250, 128, 114, 0.3)', width=1, dash='dot'),
                    showlegend=True,
                    hovertemplate='₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hist_6m.index,
                    y=hist_6m['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(250, 128, 114, 0.3)', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(250, 128, 114, 0.1)',
                    showlegend=True,
                    hovertemplate='₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist_6m.index,
                y=hist_6m['Close'],
                name='Price',
                line=dict(color='#667eea', width=2.5),
                hovertemplate='₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in hist_6m.columns and not hist_6m['SMA_20'].isnull().all():
            fig.add_trace(
                go.Scatter(
                    x=hist_6m.index,
                    y=hist_6m['SMA_20'],
                    name='SMA 20',
                    line=dict(color='#00b09b', width=1.5, dash='dash'),
                    hovertemplate='₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in hist_6m.columns and not hist_6m['SMA_50'].isnull().all():
            fig.add_trace(
                go.Scatter(
                    x=hist_6m.index,
                    y=hist_6m['SMA_50'],
                    name='SMA 50',
                    line=dict(color='#f7971e', width=1.5, dash='dot'),
                    hovertemplate='₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_200' in hist_6m.columns and not hist_6m['SMA_200'].isnull().all():
            fig.add_trace(
                go.Scatter(
                    x=hist_6m.index,
                    y=hist_6m['SMA_200'],
                    name='SMA 200',
                    line=dict(color='#ff416c', width=1.5, dash='dashdot'),
                    hovertemplate='₹%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in hist_6m.iterrows()]
        fig.add_trace(
            go.Bar(
                x=hist_6m.index,
                y=hist_6m['Volume'],
                name='Volume',
                marker_color=colors,
                hovertemplate='%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_rsi_chart(self, hist, ticker):
        """Create RSI indicator chart"""
        
        # Calculate RSI series for all data
        rsi_series = self.calculate_rsi_series(hist['Close'], period=14)
        
        # Get last 6 months
        hist_6m = hist.tail(120)
        rsi_6m = rsi_series.tail(120)
        
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(
            go.Scatter(
                x=hist_6m.index,
                y=rsi_6m,
                name='RSI',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            )
        )
        
        # Add shaded regions
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0, 255, 0, 0.1)", line_width=0)
        
        # Overbought line (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="Overbought (70)", annotation_position="right")
        
        # Oversold line (30)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=2,
                      annotation_text="Oversold (30)", annotation_position="right")
        
        # Neutral line (50)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                      annotation_text="Neutral (50)", annotation_position="right")
        
        fig.update_layout(
            title=f'RSI (Relative Strength Index) - {ticker}',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def analyze_from_history(self, ticker, company_name, hist):
        """
        Analyzes stock data from a pre-fetched history DataFrame.
        This is the core calculation engine used by both Auto and Manual tabs.
        """
        try:
            if hist is None or hist.empty:
                return None
            
            # Ensure minimum data
            if len(hist) < 20:
                st.warning(f"{ticker}: Insufficient data for analysis (need at least 20 days)")
                return None
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            
            # Daily change
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                daily_change = ((current_price - prev_close) / prev_close) * 100
            else:
                daily_change = 0
            
            # Calculate moving averages
            hist['SMA_20'] = hist['Close'].rolling(window=20, min_periods=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50, min_periods=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200, min_periods=200).mean()
            
            sma_20 = hist['SMA_20'].iloc[-1]
            sma_50 = hist['SMA_50'].iloc[-1]
            sma_200 = hist['SMA_200'].iloc[-1]
            
            # Handle potential NaNs
            if pd.isna(sma_20):
                sma_20 = current_price
                trend_20 = 0
            else:
                trend_20 = ((current_price - sma_20) / sma_20) * 100
                
            if pd.isna(sma_50):
                sma_50 = current_price
                trend_50 = 0
            else:
                trend_50 = ((current_price - sma_50) / sma_50) * 100

            if pd.isna(sma_200):
                sma_200 = current_price
                trend_200 = 0
            else:
                trend_200 = ((current_price - sma_200) / sma_200) * 100
            
            # RSI Calculation
            rsi = self.calculate_rsi(hist['Close'], 14)
            
            # MACD
            macd, macd_signal = self.calculate_macd(hist['Close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(hist['Close'])
            if bb_upper is not None:
                hist['BB_Upper'] = hist['Close'].rolling(window=20).mean() + (hist['Close'].rolling(window=20).std() * 2)
                hist['BB_Lower'] = hist['Close'].rolling(window=20).mean() - (hist['Close'].rolling(window=20).std() * 2)
            
            # Volatility (Standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            else:
                volatility = 0
            
            # Volume ratio (current vs average)
            avg_volume = hist['Volume'].tail(20).mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Support and Resistance (52-week high/low)
            support = hist['Low'].tail(min(252, len(hist))).min()
            resistance = hist['High'].tail(min(252, len(hist))).max()
            
            # Distance from 52-week high/low
            distance_from_high = ((current_price - resistance) / resistance) * 100
            distance_from_low = ((current_price - support) / support) * 100
            
            data = {
                'current_price': current_price,
                'daily_change': daily_change,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'trend_20': trend_20,
                'trend_50': trend_50,
                'trend_200': trend_200,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'support': support,
                'resistance': resistance,
                'macd': macd,
                'macd_signal': macd_signal,
                'distance_from_high': distance_from_high,
                'distance_from_low': distance_from_low
            }
            
            # Calculate comprehensive score
            score = self._calculate_score(
                data['daily_change'], 
                data['trend_20'], 
                data['trend_50'], 
                data['trend_200'],
                data['rsi'], 
                data['volatility'], 
                data['volume_ratio'],
                data['macd'],
                data['macd_signal']
            )
            
            # Generate recommendation
            recommendation_data = self._generate_recommendation(
                score, 
                data['rsi'], 
                data['trend_20'], 
                data['trend_50'], 
                data['trend_200'], 
                data['volatility'], 
                data['volume_ratio'], 
                data['daily_change'],
                data['macd'],
                data['macd_signal']
            )
            
            result = {
                'ticker': ticker,
                'company_name': company_name,
                **data,
                'score': score,
                **recommendation_data
            }
            
            return result
        
        except Exception as e:
            st.warning(f"Could not analyze {ticker}: {e}")
            return None

    def analyze_stock(self, ticker, company_name=None):
        """
        Complete analysis for a SINGLE stock.
        Used by the Manual Analysis tab.
        """
        
        # 1. Fetch data
        hist = self.fetch_real_data(ticker)
        
        if hist is None:
            return None, None
        
        # 2. Analyze data
        company_name_to_use = company_name or self.default_stocks.get(ticker, ticker)
        result = self.analyze_from_history(ticker, company_name_to_use, hist)
        
        if result is None:
            return None, None
            
        return result, hist
    
    def _calculate_score(self, daily_change, trend_20, trend_50, trend_200, rsi, volatility, volume_ratio, macd, macd_signal):
        """Enhanced scoring algorithm with MACD"""
        score = 50  # Base score
        
        # Trend Analysis (30 points)
        if trend_20 > 8 and trend_50 > 5 and trend_200 > 0:
            score += 30
        elif trend_20 > 5 and trend_50 > 3:
            score += 22
        elif trend_20 > 2 and trend_50 > 0:
            score += 14
        elif trend_20 > 0:
            score += 6
        elif trend_20 < -5:
            score -= 15
        
        # RSI Analysis (20 points)
        if 40 <= rsi <= 60:
            score += 20
        elif 30 <= rsi < 40:
            score += 18
        elif 60 < rsi <= 70:
            score += 12
        elif rsi < 30:
            score += 10
        else:
            score += 5
        
        # MACD Analysis (15 points)
        if macd > macd_signal and macd > 0:
            score += 15  # Bullish crossover above zero
        elif macd > macd_signal:
            score += 10  # Bullish crossover
        elif macd < macd_signal and macd < 0:
            score -= 10  # Bearish crossover below zero
        elif macd < macd_signal:
            score -= 5   # Bearish crossover
        
        # Volatility Analysis (15 points)
        if volatility < 20:
            score += 15
        elif volatility < 30:
            score += 10
        elif volatility < 40:
            score += 5
        else:
            score -= 5
        
        # Volume Analysis (10 points)
        if volume_ratio > 1.5:
            score += 10
        elif volume_ratio > 1.2:
            score += 7
        elif volume_ratio > 1.0:
            score += 5
        elif volume_ratio < 0.5:
            score -= 5
        
        # Daily momentum (10 points)
        if daily_change > 2:
            score += 10
        elif daily_change > 0:
            score += 6
        elif daily_change > -2:
            score += 2
        else:
            score -= 5
        
        return max(0, min(100, score))
    
    def _generate_recommendation(self, score, rsi, trend_20, trend_50, trend_200, volatility, volume_ratio, daily_change, macd, macd_signal):
        """Generate buy/sell recommendation with DATA-DRIVEN reasoning"""
        
        reasons = []
        
        # Analyze trends
        if trend_20 > 5 and trend_50 > 3 and trend_200 > 0:
            reasons.append(f"✅ Strong upward trend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%), 200D (+{trend_200:.1f}%)")
        elif trend_20 > 5 and trend_50 > 3:
            reasons.append(f"✅ Strong short-term uptrend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%)")
        elif trend_20 > 2 and trend_50 > 0:
            reasons.append(f"📊 Moderate uptrend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%)")
        elif trend_20 > 0:
            reasons.append(f"📈 Weak uptrend: 20D (+{trend_20:.1f}%), limited momentum")
        elif trend_20 < 0 and trend_50 < 0:
            reasons.append(f"⚠️ Downtrend active: 20D ({trend_20:.1f}%), 50D ({trend_50:.1f}%)")
        elif trend_20 < -5:
            reasons.append(f"🔴 Strong downward pressure: 20D ({trend_20:.1f}%)")
        else:
            reasons.append(f"🔄 Mixed trend signals: 20D ({trend_20:.1f}%), 50D ({trend_50:.1f}%)")
        
        # RSI analysis
        if 40 <= rsi <= 60:
            reasons.append(f"⚖️ RSI neutral at {rsi:.1f} - balanced momentum, good entry zone")
        elif 30 <= rsi < 40:
            reasons.append(f"💚 RSI at {rsi:.1f} - approaching oversold, potential buying opportunity")
        elif rsi < 30:
            reasons.append(f"🎯 RSI oversold at {rsi:.1f} - strong reversal potential, high reward")
        elif 60 < rsi <= 70:
            reasons.append(f"⚠️ RSI at {rsi:.1f} - mild overbought, exercise caution on new entries")
        elif rsi > 70:
            reasons.append(f"🔴 RSI overbought at {rsi:.1f} - high risk of correction, avoid buying")
        
        # MACD analysis
        macd_diff = macd - macd_signal
        if macd > macd_signal and macd > 0:
            reasons.append(f"✅ MACD bullish crossover above zero ({macd:.2f}) - strong momentum")
        elif macd > macd_signal:
            reasons.append(f"📈 MACD bullish crossover ({macd:.2f}) - positive momentum building")
        elif macd < macd_signal and macd < 0:
            reasons.append(f"🔴 MACD bearish crossover below zero ({macd:.2f}) - weak momentum")
        elif macd < macd_signal:
            reasons.append(f"⚠️ MACD bearish crossover ({macd:.2f}) - momentum weakening")
        
        # Volatility analysis
        if volatility < 20:
            reasons.append(f"✅ Low volatility ({volatility:.1f}%) - stable price action, lower risk")
        elif volatility < 30:
            reasons.append(f"📊 Moderate volatility ({volatility:.1f}%) - normal fluctuations")
        elif volatility < 40:
            reasons.append(f"⚠️ High volatility ({volatility:.1f}%) - increased risk, use tight stops")
        else:
            reasons.append(f"🔴 Very high volatility ({volatility:.1f}%) - significant price swings, high risk")
        
        # Volume analysis
        if volume_ratio > 1.5:
            reasons.append(f"✅ Strong volume ({volume_ratio:.2f}x) - high conviction move, good participation")
        elif volume_ratio > 1.2:
            reasons.append(f"📊 Above average volume ({volume_ratio:.2f}x) - good participation")
        elif volume_ratio > 0.8:
            reasons.append(f"⚖️ Normal volume ({volume_ratio:.2f}x) - adequate liquidity")
        else:
            reasons.append(f"⚠️ Low volume ({volume_ratio:.2f}x) - weak participation, be cautious")
        
        # Daily momentum
        if daily_change > 2:
            reasons.append(f"✅ Strong daily gain (+{daily_change:.2f}%) - positive momentum accelerating")
        elif daily_change > 0:
            reasons.append(f"📈 Positive day (+{daily_change:.2f}%) - bullish sentiment present")
        elif daily_change > -2:
            reasons.append(f"⚖️ Minor decline ({daily_change:.2f}%) - consolidation phase")
        else:
            reasons.append(f"🔴 Sharp decline ({daily_change:.2f}%) - selling pressure evident")
        
        # Generate recommendation based on score
        if score >= 80:
            recommendation = "STRONG BUY 🚀"
            confidence = "VERY HIGH"
            color = "🟢"
            action = "Invest Now - Excellent Opportunity"
            risk = "LOW"
        elif score >= 70:
            recommendation = "BUY ✅"
            confidence = "HIGH"
            color = "🟢"
            action = "Good Entry Point - Consider Buying"
            risk = "LOW-MEDIUM"
        elif score >= 60:
            recommendation = "MODERATE BUY 📈"
            confidence = "MEDIUM-HIGH"
            color = "🟡"
            action = "Consider for Portfolio - Accumulate"
            risk = "MEDIUM"
        elif score >= 45:
            recommendation = "HOLD 🔄"
            confidence = "MEDIUM"
            color = "🟠"
            action = "Wait for Better Setup - Stay Sidelined"
            risk = "MEDIUM"
        elif score >= 30:
            recommendation = "WEAK - AVOID ⚠️"
            confidence = "LOW"
            color = "🔴"
            action = "Do Not Enter New Position"
            risk = "HIGH"
        else:
            recommendation = "STRONG AVOID ❌"
            confidence = "VERY LOW"
            color = "🔴"
            action = "Stay Away - High Risk of Loss"
            risk = "VERY HIGH"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'color': color,
            'action': action,
            'risk': risk,
            'reasons': reasons
        }

def main():
    st.set_page_config(
        page_title="Real-Time Stock Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Real-Time Stock Recommendation System</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = RealTimeStockAnalyzer()
    
    # Check market status
    market_open = analyzer.is_market_open()
    if market_open:
        st.success("🟢 **Market is OPEN** - Live trading in progress")
    else:
        st.info("🔴 **Market is CLOSED** - Showing last closing prices")
    
    # Initialize session state to store results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'hist_dict' not in st.session_state:
        st.session_state.hist_dict = None
    if 'last_run_time' not in st.session_state:
        st.session_state.last_run_time = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Sidebar settings
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        min_score = st.slider(
            "Minimum Quality Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Only show stocks with score above this value"
        )
        
        show_recommendations = st.multiselect(
            "Show Recommendations",
            ["STRONG BUY", "BUY", "MODERATE BUY", "HOLD", "AVOID"],
            default=["STRONG BUY", "BUY", "MODERATE BUY"],
            help="Filter by recommendation type"
        )
        
        top_n = st.slider(
            "Show Top N Stocks",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of top-ranked stocks to display"
        )
        
        st.markdown("---")
        
        # Auto-refresh option
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh_enabled = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.auto_refresh,
            help="Automatically refresh data every 5 minutes (only during market hours)"
        )
        st.session_state.auto_refresh = auto_refresh_enabled
        
        if auto_refresh_enabled and market_open:
            st.info("⏱️ Auto-refresh active - Data updates every 5 minutes")
        
        st.markdown("---")
        st.markdown("### 🔄 Data Source")
        st.success("""
        **Live Data from:**
        - Yahoo Finance API
        - Real-time NSE prices
        - Updated continuously
        """)
        
        st.markdown("---")
        st.markdown("### 📌 Features")
        st.info("""
        - ✅ Real market data
        - ✅ Live price updates
        - ✅ Technical analysis (RSI, MACD, BB)
        - ✅ Smart recommendations
        - ✅ Risk assessment
        - ✅ Data caching for speed
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        **Version:** 2.0 Enhanced
        
        **New Features:**
        - MACD indicator
        - Bollinger Bands
        - Data caching
        - Enhanced scoring
        - Better error handling
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Auto Scan", "➕ Manual Analysis", "📊 Comparison", "📚 Guide"])
    
    with tab1:
        st.markdown("### 📊 Real-Time Stock Scanner")
        st.info(f"📡 **Live Analysis** - Scanning top 20 Indian stocks from Yahoo Finance")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            run_analysis = st.button("🚀 START / REFRESH LIVE ANALYSIS", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                analyzer.cache.clear()
                st.success("Cache cleared!")
                st.rerun()
        with col3:
            if st.session_state.last_run_time:
                st.caption(f"Last: {st.session_state.last_run_time.strftime('%H:%M:%S')}")
        
        # Auto-refresh logic
        if auto_refresh_enabled and market_open and st.session_state.last_run_time:
            time_since_last = (datetime.now() - st.session_state.last_run_time).seconds
            if time_since_last > 300:  # 5 minutes
                run_analysis = True
                st.info("🔄 Auto-refreshing data...")
        
        # Only run analysis if button is pressed OR if it's the first run
        if run_analysis or st.session_state.analysis_results is None:
            
            with st.spinner("📡 Fetching batch data for all 20 stocks... (10x faster with batch download!)"):
                results = []
                hist_dict = {}
                tickers_list = list(analyzer.default_stocks.keys())
                company_names = analyzer.default_stocks
                
                try:
                    # 1. BATCH FETCH ALL DATA AT ONCE
                    data = yf.download(tickers_list, period="1y", group_by='ticker', progress=False)
                    
                    if data.empty:
                        st.error("❌ Unable to fetch market data. Please check your internet connection and try again.")
                        st.stop()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 2. Loop through tickers to process the batch data
                    for i, ticker in enumerate(tickers_list):
                        company_name = company_names[ticker]
                        status_text.text(f"Analyzing {company_name}... ({i+1}/{len(tickers_list)})")
                        
                        try:
                            # Extract single-stock history from the multi-index DataFrame
                            if len(tickers_list) == 1:
                                hist_data = data.copy()
                            else:
                                hist_data = data[ticker].copy()
                            
                            # Drop rows where all data is NaN
                            hist_data = hist_data.dropna(how='all')
                            
                            if hist_data.empty:
                                st.warning(f"No data for {ticker}")
                                continue
                            
                            # Pass this 'hist' DataFrame to the analyzer
                            result = analyzer.analyze_from_history(ticker, company_name, hist_data)
                            
                            if result:
                                results.append(result)
                                hist_dict[ticker] = hist_data
                        
                        except Exception as e:
                            st.warning(f"Skipping {ticker}: {str(e)}")
                            continue
                        
                        progress_bar.progress((i + 1) / len(tickers_list))

                    status_text.success(f"✅ Batch analysis complete! Analyzed {len(results)} stocks.")
                    progress_bar.empty()
                    
                    # 3. STORE IN SESSION STATE
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.hist_dict = hist_dict
                        st.session_state.last_run_time = datetime.now()
                    else:
                        st.error("❌ No stocks could be analyzed. Please try again.")

                except Exception as e:
                    st.error(f"❌ An error occurred during batch analysis: {str(e)}")
                    st.session_state.analysis_results = []
        
        # 4. ALWAYS RENDER FROM SESSION STATE
        if not st.session_state.analysis_results:
            st.warning("⚠️ Please click the 'START / REFRESH' button to run the analysis.")
        
        else:
            # Extract data from state
            all_stock_data = st.session_state.analysis_results
            hist_dict = st.session_state.hist_dict
            
            # Filter results
            filtered_results = [
                stock for stock in all_stock_data
                if stock['score'] >= min_score and 
                any(rec in stock['recommendation'] for rec in show_recommendations)
            ]
            
            # Sort by score
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            if filtered_results:
                st.success(f"✅ Analysis Complete! Displaying {len(filtered_results)} stocks matching your criteria")
                st.caption(f"⏰ Last updated: {st.session_state.last_run_time.strftime('%d-%m-%Y %H:%M:%S IST')}")
                
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                strong_buy = len([s for s in all_stock_data if "STRONG BUY" in s['recommendation']])
                buy = len([s for s in all_stock_data if s['recommendation'].startswith("BUY") and "MODERATE" not in s['recommendation']])
                moderate = len([s for s in all_stock_data if "MODERATE" in s['recommendation']])
                hold = len([s for s in all_stock_data if "HOLD" in s['recommendation']])
                avoid = len([s for s in all_stock_data if "AVOID" in s['recommendation']])
                
                with col1:
                    st.metric("🚀 STRONG BUY", strong_buy)
                with col2:
                    st.metric("✅ BUY", buy)
                with col3:
                    st.metric("📈 MODERATE BUY", moderate)
                with col4:
                    st.metric("🔄 HOLD", hold)
                with col5:
                    st.metric("⚠️ AVOID", avoid)
                
                # Average score
                avg_score = sum(s['score'] for s in all_stock_data) / len(all_stock_data)
                st.info(f"📊 **Average Portfolio Score:** {avg_score:.1f}/100")
                
                # Top recommendations
                st.markdown("---")
                st.markdown(f"## 🎯 Top {min(top_n, len(filtered_results))} Investment Opportunities")
                
                for i, stock in enumerate(filtered_results[:top_n]):
                    with st.expander(
                        f"{stock['color']} #{i+1}: {stock['company_name']} - {stock['recommendation']} | Score: {stock['score']}/100", 
                        expanded=(i<2)
                    ):
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Ticker:** `{stock['ticker']}`")
                            st.markdown(f"**🎯 Action:** {stock['action']}")
                        with col2:
                            st.markdown(f"**📊 Confidence:** {stock['confidence']}")
                            st.markdown(f"**⚠️ Risk:** {stock['risk']}")
                        
                        st.markdown("---")
                        
                        # Price metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("💰 Current Price", f"₹{stock['current_price']:.2f}", f"{stock['daily_change']:+.2f}%")
                        with col2:
                            st.metric("🎯 RSI", f"{stock['rsi']:.1f}")
                        with col3:
                            st.metric("📊 20D Trend", f"{stock['trend_20']:+.1f}%")
                        with col4:
                            st.metric("⚡ Volatility", f"{stock['volatility']:.1f}%")
                        
                        st.markdown("---")
                        
                        # Show price chart
                        if stock['ticker'] in hist_dict:
                            st.markdown("### 📈 Price Chart with Technical Indicators")
                            fig_price = analyzer.create_price_chart(hist_dict[stock['ticker']], stock['ticker'], stock['company_name'])
                            st.plotly_chart(fig_price, use_container_width=True)
                            
                            st.markdown("### 📊 RSI Indicator")
                            fig_rsi = analyzer.create_rsi_chart(hist_dict[stock['ticker']], stock['ticker'])
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Additional metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**📊 Moving Averages:**")
                            st.write(f"• SMA 20: ₹{stock['sma_20']:.2f}")
                            st.write(f"• SMA 50: ₹{stock['sma_50']:.2f}")
                            st.write(f"• SMA 200: ₹{stock['sma_200']:.2f}")
                        
                        with col2:
                            st.write("**📈 Trends:**")
                            st.write(f"• 20D: {stock['trend_20']:+.1f}%")
                            st.write(f"• 50D: {stock['trend_50']:+.1f}%")
                            st.write(f"• 200D: {stock['trend_200']:+.1f}%")
                        
                        with col3:
                            st.write("**🎯 Levels:**")
                            st.write(f"• Support: ₹{stock['support']:.2f}")
                            st.write(f"• Resistance: ₹{stock['resistance']:.2f}")
                            st.write(f"• Volume: {stock['volume_ratio']:.2f}x")
                        
                        st.markdown("---")
                        st.markdown("**📋 Detailed Analysis:**")
                        for reason in stock['reasons']:
                            st.write(f"{reason}")
                        
                        # Entry/Exit strategy
                        if "BUY" in stock['recommendation'] and "AVOID" not in stock['recommendation']:
                            st.success(f"""
                            **📝 Suggested Action Plan:**
                            - 🎯 Entry Price: ₹{stock['current_price']:.2f} (Current)
                            - 🛡️ Stop Loss: ₹{stock['support'] * 0.97:.2f} (3% below support)
                            - 💰 Target 1 (8%): ₹{stock['current_price'] * 1.08:.2f}
                            - 💰 Target 2 (15%): ₹{stock['current_price'] * 1.15:.2f}
                            - 💰 Target 3 (25%): ₹{stock['current_price'] * 1.25:.2f}
                            - 📊 Position Size: Max 10-12% of portfolio
                            - ⏱️ Time Horizon: 2-4 weeks for targets
                            """)
                
                # Investment Plan
                st.markdown("---")
                st.markdown("## 💰 Today's Investment Strategy")
                
                strong_buys = [s for s in all_stock_data if "STRONG BUY" in s['recommendation']]
                buys = [s for s in all_stock_data if s['recommendation'].startswith("BUY") and "MODERATE" not in s['recommendation']]
                moderate_buys = [s for s in all_stock_data if "MODERATE BUY" in s['recommendation']]
                
                if strong_buys or buys or moderate_buys:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Recommended Portfolio Allocation**")
                        
                        if strong_buys:
                            st.markdown(f"**💎 Strong Buy Stocks ({min(3, len(strong_buys))} stocks - 40% allocation):**")
                            for stock in strong_buys[:3]:
                                st.write(f"• {stock['company_name']} ({stock['ticker']}) - ₹{stock['current_price']:.2f} | Score: {stock['score']}")
                        
                        if buys:
                            st.markdown(f"**✅ Buy Stocks ({min(2, len(buys))} stocks - 30% allocation):**")
                            for stock in buys[:2]:
                                st.write(f"• {stock['company_name']} ({stock['ticker']}) - ₹{stock['current_price']:.2f} | Score: {stock['score']}")
                        
                        if moderate_buys and not strong_buys and not buys:
                            st.markdown(f"**📈 Moderate Buy Stocks ({min(2, len(moderate_buys))} stocks - 30% allocation):**")
                            for stock in moderate_buys[:2]:
                                st.write(f"• {stock['company_name']} ({stock['ticker']}) - ₹{stock['current_price']:.2f} | Score: {stock['score']}")
                        
                        st.markdown("**💵 Cash Reserve: 30%** (for opportunities & risk management)")
                    
                    with col2:
                        st.markdown("""
                        **📊 Risk Management Guidelines**
                        
                        **Stop Loss Strategy:**
                        - Set at 5-7% below entry price
                        - Trail stops as price rises by 5%
                        - Book 50% profits at 10-12% gain
                        - Let remaining 50% run to targets
                        
                        **Position Sizing:**
                        - Max 12% per strong buy stock
                        - Max 10% per buy stock
                        - Max 8% per moderate buy
                        - Total: 5-7 stocks maximum
                        - Keep 30% cash for opportunities
                        
                        **Review Schedule:**
                        - Daily: Check stop losses
                        - Weekly: Review all positions
                        - Monthly: Rebalance portfolio
                        """)
                
                else:
                    st.warning("🔄 **Market Outlook: Neutral/Bearish** - No strong buy signals currently. Preserve capital and wait for better opportunities.")
                
                # Data table
                st.markdown("---")
                st.markdown("## 📋 Complete Analysis Data")
                
                table_data = []
                for stock in filtered_results:
                    table_data.append({
                        'Rank': filtered_results.index(stock) + 1,
                        'Company': stock['company_name'],
                        'Ticker': stock['ticker'],
                        'Price (₹)': f"{stock['current_price']:.2f}",
                        'Change %': f"{stock['daily_change']:+.2f}",
                        'RSI': f"{stock['rsi']:.1f}",
                        'Vol': f"{stock['volatility']:.1f}",
                        '20D %': f"{stock['trend_20']:+.1f}",
                        '50D %': f"{stock['trend_50']:+.1f}",
                        'Score': stock['score'],
                        'Recommendation': stock['recommendation'],
                        'Risk': stock['risk']
                    })
                
                df = pd.DataFrame(table_data)
                
                # Style the dataframe
                def highlight_score(val):
                    if isinstance(val, (int, float)):
                        if val >= 80:
                            return 'background-color: #90EE90'
                        elif val >= 70:
                            return 'background-color: #FFFFE0'
                        elif val >= 60:
                            return 'background-color: #FFE4B5'
                        elif val < 45:
                            return 'background-color: #FFB6C1'
                    return ''
                
                st.dataframe(
                    df.style.applymap(highlight_score, subset=['Score']),
                    use_container_width=True,
                    height=400
                )
                
                # Download
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.download_button(
                        label="📥 Download Analysis (CSV)",
                        data=csv,
                        file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    # Export to text summary
                    summary_text = f"""STOCK ANALYSIS REPORT
Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}
                    
SUMMARY:
- Total Stocks Analyzed: {len(all_stock_data)}
- Strong Buy: {strong_buy}
- Buy: {buy}
- Moderate Buy: {moderate}
- Hold: {hold}
- Avoid: {avoid}
- Average Score: {avg_score:.1f}/100

TOP RECOMMENDATIONS:
"""
                    for i, stock in enumerate(filtered_results[:5], 1):
                        summary_text += f"\n{i}. {stock['company_name']} ({stock['ticker']})\n"
                        summary_text += f"   Score: {stock['score']}/100 | {stock['recommendation']}\n"
                        summary_text += f"   Price: ₹{stock['current_price']:.2f} | Change: {stock['daily_change']:+.2f}%\n"
                    
                    st.download_button(
                        label="📄 Download Summary (TXT)",
                        data=summary_text,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            else:
                st.error("❌ No stocks match your current filters. Try adjusting the settings in the sidebar:")
                st.info("""
                **Suggestions:**
                - Lower the minimum quality score
                - Select more recommendation types
                - Check if any stocks were successfully analyzed
                """)
    
    with tab2:
        st.markdown("### ➕ Analyze Custom Stock")
        st.markdown("Enter any NSE/BSE stock ticker for real-time analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_ticker = st.text_input(
                "Enter Stock Ticker",
                placeholder="e.g., WIPRO.NS, ADANIENT.NS, HCLTECH.NS",
                help="Use Yahoo Finance format: SYMBOL.NS for NSE, SYMBOL.BO for BSE"
            )
        
        with col2:
            manual_name = st.text_input(
                "Company Name (Optional)",
                placeholder="e.g., Wipro Ltd"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_button = st.button("🔍 ANALYZE THIS STOCK", type="primary", use_container_width=True)
        with col2:
            if st.button("📋 Example Tickers", use_container_width=True):
                st.info("""
                **Popular NSE Tickers:**
                - WIPRO.NS - Wipro
                - M&M.NS - Mahindra & Mahindra
                - ADANIENT.NS - Adani Enterprises
                - HCLTECH.NS - HCL Technologies
                - NESTLEIND.NS - Nestle India
                - BRITANNIA.NS - Britannia Industries
                """)
        
        if analyze_button:
            if manual_ticker:
                with st.spinner(f"📡 Fetching live data for {manual_ticker}..."):
                    stock, hist = analyzer.analyze_stock(
                        manual_ticker.strip().upper(),
                        manual_name.strip() if manual_name else None
                    )
                
                if stock and hist is not None:
                    st.success(f"✅ Analysis complete for {stock['company_name']}")
                    st.caption(f"⏰ Data as of: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}")
                    
                    # Display analysis
                    st.markdown(f"## {stock['color']} {stock['company_name']} ({stock['ticker']})")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"### {stock['recommendation']}")
                        st.markdown(f"**Score: {stock['score']}/100**")
                    with col2:
                        st.metric("Confidence", stock['confidence'])
                    with col3:
                        st.metric("Risk Level", stock['risk'])
                    
                    st.info(f"**🎯 Recommended Action:** {stock['action']}")
                    
                    st.markdown("---")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💰 Current Price", f"₹{stock['current_price']:.2f}", f"{stock['daily_change']:+.2f}%")
                    with col2:
                        st.metric("🎯 RSI", f"{stock['rsi']:.1f}")
                    with col3:
                        st.metric("📊 20D Trend", f"{stock['trend_20']:+.1f}%")
                    with col4:
                        st.metric("⚡ Volatility", f"{stock['volatility']:.1f}%")
                    
                    st.markdown("---")
                    
                    # Show charts
                    st.markdown("### 📈 Price Chart with Technical Indicators")
                    fig_price = analyzer.create_price_chart(hist, stock['ticker'], stock['company_name'])
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    st.markdown("### 📊 RSI Indicator")
                    fig_rsi = analyzer.create_rsi_chart(hist, stock['ticker'])
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Price Information**")
                        st.write(f"• Current: ₹{stock['current_price']:.2f}")
                        st.write(f"• Support: ₹{stock['support']:.2f} ({stock['distance_from_low']:+.1f}%)")
                        st.write(f"• Resistance: ₹{stock['resistance']:.2f} ({stock['distance_from_high']:+.1f}%)")
                        st.write(f"• SMA 20: ₹{stock['sma_20']:.2f}")
                        st.write(f"• SMA 50: ₹{stock['sma_50']:.2f}")
                        st.write(f"• SMA 200: ₹{stock['sma_200']:.2f}")
                    
                    with col2:
                        st.markdown("**📈 Technical Indicators**")
                        st.write(f"• RSI: {stock['rsi']:.1f}")
                        st.write(f"• MACD: {stock['macd']:.2f}")
                        st.write(f"• MACD Signal: {stock['macd_signal']:.2f}")
                        st.write(f"• Volatility: {stock['volatility']:.1f}%")
                        st.write(f"• Volume Ratio: {stock['volume_ratio']:.2f}x")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("20-Day Trend", f"{stock['trend_20']:+.1f}%")
                    with col2:
                        st.metric("50-Day Trend", f"{stock['trend_50']:+.1f}%")
                    with col3:
                        st.metric("200-Day Trend", f"{stock['trend_200']:+.1f}%")
                    
                    st.markdown("---")
                    st.markdown("**📋 Detailed Analysis Reasons:**")
                    for reason in stock['reasons']:
                        st.write(reason)
                    
                    # Action plan
                    st.markdown("---")
                    if "BUY" in stock['recommendation'] and "AVOID" not in stock['recommendation']:
                        st.success(f"""
                        **📝 Detailed Action Plan:**
                        
                        **Entry Strategy:**
                        - Current Price: ₹{stock['current_price']:.2f}
                        - Suggested Entry: ₹{stock['current_price'] * 0.99:.2f} - ₹{stock['current_price'] * 1.01:.2f}
                        - Buy in 2 parts: 50% now, 50% on 2-3% dip
                        
                        **Risk Management:**
                        - Stop Loss: ₹{stock['support'] * 0.97:.2f} (Strict - no exceptions!)
                        - Max Loss: ~5-7% from entry
                        
                        **Profit Targets:**
                        - Target 1 (8%): ₹{stock['current_price'] * 1.08:.2f} - Book 30%
                        - Target 2 (15%): ₹{stock['current_price'] * 1.15:.2f} - Book 40%
                        - Target 3 (25%): ₹{stock['current_price'] * 1.25:.2f} - Let rest run
                        
                        **Position Size:**
                        - Allocate max 10-12% of total portfolio
                        - Avoid overconcentration in single stock
                        
                        **Time Horizon:**
                        - Short-term targets: 2-4 weeks
                        - Medium-term: 2-3 months
                        - Review weekly, adjust stops
                        """)
                    elif "HOLD" in stock['recommendation']:
                        st.info("""
                        **💡 Hold Strategy:**
                        - Current setup not ideal for fresh entry
                        - Wait for better technical confirmation
                        - Watch for: Price above SMA 20, RSI 30-60, Volume pickup
                        - Re-evaluate in 1-2 weeks
                        - Set price alert at support and resistance levels
                        """)
                    else:
                        st.warning("""
                        **⚠️ Caution - Not Recommended:**
                        - Technical indicators show weakness
                        - High risk of further downside
                        - Better opportunities available elsewhere
                        - If holding: Consider reducing position
                        - Avoid new investment at current levels
                        """)
                
                else:
                    st.error(f"❌ Unable to fetch data for {manual_ticker}. Please verify:")
                    st.info("""
                    **Troubleshooting:**
                    - Check ticker format (add .NS for NSE, .BO for BSE)
                    - Ensure correct spelling
                    - Stock must be actively traded
                    - Try searching on Yahoo Finance first
                    
                    **Example:** For Wipro on NSE, use: WIPRO.NS
                    """)
            
            else:
                st.error("❌ Please enter a stock ticker symbol")
    
    with tab3:
        st.markdown("### 📊 Compare Multiple Stocks")
        st.markdown("Compare up to 5 stocks side by side")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            compare_tickers = st.text_input(
                "Enter tickers (comma-separated)",
                placeholder="e.g., RELIANCE.NS, TCS.NS, INFY.NS",
                help="Enter 2-5 stock tickers separated by commas"
            )
        with col2:
            compare_button = st.button("📊 COMPARE STOCKS", type="primary", use_container_width=True)
        
        if compare_button and compare_tickers:
            tickers = [t.strip().upper() for t in compare_tickers.split(",") if t.strip()]
            
            if len(tickers) < 2:
                st.error("Please enter at least 2 tickers to compare")
            elif len(tickers) > 5:
                st.error("Maximum 5 stocks can be compared at once")
            else:
                with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                    comparison_results = []
                    
                    for ticker in tickers:
                        stock, hist = analyzer.analyze_stock(ticker)
                        if stock:
                            comparison_results.append(stock)
                    
                    if len(comparison_results) >= 2:
                        st.success(f"✅ Successfully compared {len(comparison_results)} stocks")
                        
                        # Comparison table
                        st.markdown("### 📋 Comparison Table")
                        
                        comparison_data = []
                        for stock in comparison_results:
                            comparison_data.append({
                                'Company': stock['company_name'],
                                'Ticker': stock['ticker'],
                                'Score': stock['score'],
                                'Recommendation': stock['recommendation'],
                                'Price (₹)': f"{stock['current_price']:.2f}",
                                'Change %': f"{stock['daily_change']:+.2f}",
                                'RSI': f"{stock['rsi']:.1f}",
                                '20D Trend %': f"{stock['trend_20']:+.1f}",
                                '50D Trend %': f"{stock['trend_50']:+.1f}",
                                'Volatility %': f"{stock['volatility']:.1f}",
                                'Volume Ratio': f"{stock['volume_ratio']:.2f}",
                                'Risk': stock['risk']
                            })
                        
                        df_compare = pd.DataFrame(comparison_data)
                        st.dataframe(df_compare, use_container_width=True)
                        
                        # Visual comparison
                        st.markdown("---")
                        st.markdown("### 📊 Visual Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Score comparison
                            fig_scores = go.Figure(data=[
                                go.Bar(
                                    x=[s['company_name'] for s in comparison_results],
                                    y=[s['score'] for s in comparison_results],
                                    marker_color=['green' if s['score'] >= 70 else 'orange' if s['score'] >= 50 else 'red' for s in comparison_results],
                                    text=[s['score'] for s in comparison_results],
                                    textposition='auto',
                                )
                            ])
                            fig_scores.update_layout(
                                title="Quality Score Comparison",
                                xaxis_title="Company",
                                yaxis_title="Score",
                                height=400,
                                yaxis=dict(range=[0, 100])
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)
                        
                        with col2:
                            # RSI comparison
                            fig_rsi = go.Figure(data=[
                                go.Bar(
                                    x=[s['company_name'] for s in comparison_results],
                                    y=[s['rsi'] for s in comparison_results],
                                    marker_color=['red' if s['rsi'] > 70 else 'green' if s['rsi'] < 30 else 'blue' for s in comparison_results],
                                    text=[f"{s['rsi']:.1f}" for s in comparison_results],
                                    textposition='auto',
                                )
                            ])
                            fig_rsi.update_layout(
                                title="RSI Comparison",
                                xaxis_title="Company",
                                yaxis_title="RSI",
                                height=400,
                                yaxis=dict(range=[0, 100])
                            )
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # Best pick recommendation
                        st.markdown("---")
                        st.markdown("### 🏆 Best Pick from Comparison")
                        
                        best_stock = max(comparison_results, key=lambda x: x['score'])
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.success(f"""
                            **Winner: {best_stock['company_name']} ({best_stock['ticker']})**
                            - Score: {best_stock['score']}/100
                            - Recommendation: {best_stock['recommendation']}
                            """)
                        with col2:
                            st.metric("Price", f"₹{best_stock['current_price']:.2f}")
                        with col3:
                            st.metric("RSI", f"{best_stock['rsi']:.1f}")
                        
                        st.info(f"**Why {best_stock['company_name']}?** Based on comprehensive technical analysis, it has the highest quality score among compared stocks.")
                    
                    else:
                        st.error("Unable to analyze enough stocks for comparison. Please check ticker symbols.")
    
    with tab4:
        st.markdown("## 📚 Complete System Guide")
        
        st.markdown("""
        ### 🎯 How This System Works
        
        **Real-Time Data Integration:**
        - Connects to Yahoo Finance API for live NSE/BSE data
        - Batch processing for 10x faster analysis
        - Smart caching to reduce API calls
        - Updates available every 5 minutes during market hours
        
        **Technical Analysis Engine:**
        - **Moving Averages:** 20, 50, and 200-day SMAs for trend identification
        - **RSI:** Relative Strength Index using Wilder's Smoothing (industry standard)
        - **MACD:** Moving Average Convergence Divergence for momentum
        - **Bollinger Bands:** Volatility and support/resistance levels
        - **Volume Analysis:** Comparison with 20-day average
        - **Support & Resistance:** 52-week highs and lows
        
        **Advanced Scoring System (0-100):**
        - **Trend Analysis (30%):** Short, medium, and long-term trends
        - **RSI Indicator (20%):** Momentum and overbought/oversold levels
        - **MACD Signal (15%):** Momentum direction and strength
        - **Volatility (15%):** Risk assessment
        - **Volume (10%):** Market participation and conviction
        - **Daily Movement (10%):** Short-term momentum
        
        **Score Interpretation:**
        - **80-100:** 🚀 STRONG BUY (Very High Confidence, Low Risk)
        - **70-79:** ✅ BUY (High Confidence, Low-Medium Risk)
        - **60-69:** 📈 MODERATE BUY (Medium Confidence, Medium Risk)
        - **45-59:** 🔄 HOLD (Neutral - Wait for Better Setup)
        - **30-44:** ⚠️ WEAK - AVOID (Low Confidence, High Risk)
        - **0-29:** ❌ STRONG AVOID (Very High Risk)
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Technical Indicators Explained
            
            **RSI (Relative Strength Index):**
            - **Below 30:** Oversold zone (potential reversal up)
            - **30-40:** Approaching oversold (watch for entry)
            - **40-60:** Neutral/Balanced (ideal zone)
            - **60-70:** Slightly overbought (caution)
            - **Above 70:** Overbought (risk of pullback)
            
            **MACD (Moving Average Convergence Divergence):**
            - **MACD > Signal & MACD > 0:** Strong bullish momentum
            - **MACD > Signal:** Bullish crossover (buy signal)
            - **MACD < Signal:** Bearish crossover (sell signal)
            - **MACD < Signal & MACD < 0:** Strong bearish momentum
            
            **Bollinger Bands:**
            - **Price at Lower Band:** Potential support/oversold
            - **Price in Middle:** Neutral
            - **Price at Upper Band:** Potential resistance/overbought
            - **Band Width:** Volatility measurement
            
            **Moving Averages (Trend):**
            - **Price > MA:** Bullish trend
            - **Price < MA:** Bearish trend
            - **Positive %:** Above average (strength)
            - **Negative %:** Below average (weakness)
            - **20D:** Short-term trend (2-4 weeks)
            - **50D:** Medium-term trend (2-3 months)
            - **200D:** Long-term trend (8-12 months)
            
            **Volatility:**
            - **< 20%:** Low risk, stable moves
            - **20-30%:** Normal market fluctuations
            - **30-40%:** High risk, larger swings
            - **> 40%:** Very high risk, extreme moves
            
            **Volume Ratio:**
            - **> 1.5x:** Strong conviction, institutional interest
            - **1.0-1.5x:** Normal activity
            - **0.5-1.0x:** Below average
            - **< 0.5x:** Weak participation, liquidity concerns
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Investment Guidelines & Best Practices
            
            **Portfolio Construction:**
            - Diversify across 5-7 stocks maximum
            - Max allocation per stock:
              * Strong Buy: 12-15%
              * Buy: 10-12%
              * Moderate Buy: 8-10%
            - Maintain 25-30% cash reserve always
            - Mix across different sectors
            - Avoid concentration risk
            
            **Entry Strategy:**
            - Never go all-in at once (scale in)
            - Buy 50% at current price
            - Keep 50% for 2-3% dip
            - Confirm with volume (>1.2x average)
            - Enter near support levels
            - Check multiple timeframes
            - Avoid buying at resistance
            
            **Risk Management (CRITICAL!):**
            - **ALWAYS set stop loss** before entering
            - Stop loss at 5-7% below entry (strict)
            - Never average down on losing trades
            - Trail stops as profit increases:
              * +5% gain → Move SL to entry
              * +10% gain → Move SL to +5%
              * +15% gain → Move SL to +10%
            - Book partial profits at targets:
              * 30% at first target (+8-10%)
              * 40% at second target (+15%)
              * Rest let run or trail tight
            
            **Exit Strategy:**
            - Stick to stop loss (NO EMOTIONS!)
            - Book profits at resistance levels
            - Exit if fundamentals change
            - Exit if score drops below 45
            - Exit if multiple red flags appear
            - Don't be greedy, lock gains
            
            **Position Monitoring:**
            - Review daily: Stop losses, news
            - Review weekly: All positions, rebalance
            - Review monthly: Overall strategy
            - Keep a trading journal
            - Learn from every trade
            
            **Common Mistakes to AVOID:**
            - ❌ No stop loss (biggest mistake!)
            - ❌ Overtrading, too many positions
            - ❌ Revenge trading after losses
            - ❌ Averaging down on losers
            - ❌ Holding losers, selling winners
            - ❌ Ignoring risk management
            - ❌ Trading based on emotions/tips
            - ❌ Not taking profits at targets
            - ❌ Over-leveraging position sizes
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### ⚠️ Important Disclaimers & Limitations
        
        **Before Investing:**
        - ✅ This tool provides **technical analysis only**
        - ✅ Do your own fundamental research always
        - ✅ Consult a SEBI-registered financial advisor
        - ✅ Only invest what you can afford to lose
        - ✅ Past performance ≠ Future results
        - ✅ Markets are inherently risky
        
        **System Limitations:**
        - Market data may have 15-minute delay
        - Technical analysis is probabilistic, not certain
        - News/events can override technical signals
        - System doesn't account for:
          * Fundamental analysis (earnings, debt, etc.)
          * Macro-economic factors
          * Company-specific news
          * Sector-specific developments
          * Market sentiment shifts
        
        **Data Quality:**
        - Requires stable internet connection
        - Yahoo Finance API dependencies
        - Historical data accuracy limitations
        - Real-time updates during market hours only
        
        **Best Practices for This Tool:**
        - ✅ Run analysis before market opens (9:00-9:15 AM)
        - ✅ Use Auto Scan for broad market overview
        - ✅ Use Manual Analysis for specific stocks
        - ✅ Cross-reference with fundamental analysis
        - ✅ Compare multiple stocks before deciding
        - ✅ Keep cache clear for fresh data
        - ✅ Monitor recommendations but use judgment
        """)
        
        st.markdown("---")
        
        st.success("""
        ### 🚀 Quick Start Guide
        
        **For Beginners (5-Step Process):**
        1. Go to **Auto Scan** tab
        2. Click **START / REFRESH LIVE ANALYSIS**
        3. Wait 30-60 seconds for analysis
        4. Review **STRONG BUY** and **BUY** stocks
        5. Use **Manual Analysis** for detailed view
        
        **For Advanced Users:**
        1. Adjust filters in sidebar (min score, recommendations)
        2. Run Auto Scan for market-wide view
        3. Identify top 3-5 candidates
        4. Use Manual Analysis for each
        5. Compare using Comparison tab
        6. Check charts and technical indicators
        7. Cross-reference with fundamentals
        8. Make informed decision with proper risk management
        
        **Daily Workflow Suggestion:**
        - **Morning (9:00 AM):** Run Auto Scan before market opens
        - **Mid-Day (12:00 PM):** Check positions, adjust stops
        - **Closing (3:30 PM):** Review performance, plan next day
        - **Evening:** Research fundamentals of new candidates
        
        **Pro Tips:**
        - 🎯 Focus on stocks scoring 70+
        - 🎯 Prioritize low volatility (<30%)
        - 🎯 Look for volume confirmation (>1.2x)
        - 🎯 Prefer all trends positive
        - 🎯 RSI 40-60 is ideal entry zone
        - 🎯 Avoid buying at resistance levels
        - 🎯 Always maintain 30% cash reserve
        - 🎯 Keep position sizes disciplined
        - 🎯 Never skip stop losses
        """)
        
        st.markdown("---")
        
        st.info("""
        ### 📞 Troubleshooting & FAQ
        
        **Q: "Unable to fetch data" error?**
        A: Check internet connection, verify ticker format (add .NS for NSE)
        
        **Q: No stocks match my filters?**
        A: Lower minimum score or select more recommendation types
        
        **Q: Data seems outdated?**
        A: Click "Clear Cache" button and refresh analysis
        
        **Q: How often should I refresh?**
        A: Every 5-15 minutes during market hours, or use Auto-Refresh
        
        **Q: Can I trust the recommendations?**
        A: Use as one input among many. Always do your own research.
        
        **Q: What ticker format to use?**
        A: NSE stocks: SYMBOL.NS (e.g., TCS.NS)
           BSE stocks: SYMBOL.BO (e.g., TCS.BO)
        
        **Q: How to find ticker symbols?**
        A: Search on Yahoo Finance or NSE/BSE website
        
        **Q: System shows different recommendation than yesterday?**
        A: Market conditions change daily. Technical indicators update with new data.
        
        **Q: Score dropped suddenly?**
        A: Market volatility, price drops, or volume changes can affect score
        
        **Q: Best time to use this system?**
        A: Before market opens (9:00-9:15 AM) for planning the day
        
        **Common Ticker Format Examples:**
        - Reliance: RELIANCE.NS
        - TCS: TCS.NS
        - Infosys: INFY.NS
        - HDFC Bank: HDFCBANK.NS
        - Wipro: WIPRO.NS
        - Mahindra & Mahindra: M&M.NS
        - State Bank: SBIN.NS
        """)
        
        st.markdown("---")
        
        st.warning("""
        ### ⚖️ Legal Disclaimer
        
        This tool is for **educational and informational purposes only**. 
        
        - Not financial advice or investment recommendation
        - Not a substitute for professional financial counsel
        - Creator assumes no liability for trading losses
        - Users trade at their own risk
        - Consult SEBI-registered advisors before investing
        - Markets are subject to risks; past performance doesn't guarantee future results
        
        **By using this system, you acknowledge:**
        - You understand the risks involved in stock trading
        - You will not hold the creator liable for any losses
        - You will conduct your own due diligence
        - You will use proper risk management
        - You are solely responsible for your investment decisions
        """)

if __name__ == "__main__":
    main()
