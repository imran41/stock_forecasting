import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RealTimeStockAnalyzer:
    def __init__(self):
        # Pre-defined Indian stocks
        self.default_stocks = {
            "RELIANCE.NS": "Reliance Industries",
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
    
    def fetch_real_data(self, ticker):
        """Fetch real market data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data (1 year)
            hist = stock.history(period="1y")
            
            if hist.empty:
                return None, None
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            
            # Daily change
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                daily_change = ((current_price - prev_close) / prev_close) * 100
            else:
                daily_change = 0
            
            # Calculate moving averages and trends
            if len(hist) >= 200:
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                
                sma_20 = hist['SMA_20'].iloc[-1]
                sma_50 = hist['SMA_50'].iloc[-1]
                sma_200 = hist['SMA_200'].iloc[-1]
                
                trend_20 = ((current_price - sma_20) / sma_20) * 100
                trend_50 = ((current_price - sma_50) / sma_50) * 100
                trend_200 = ((current_price - sma_200) / sma_200) * 100
            elif len(hist) >= 50:
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = current_price
                
                sma_20 = hist['SMA_20'].iloc[-1]
                sma_50 = hist['SMA_50'].iloc[-1]
                sma_200 = current_price
                
                trend_20 = ((current_price - sma_20) / sma_20) * 100
                trend_50 = ((current_price - sma_50) / sma_50) * 100
                trend_200 = 0
            else:
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = current_price
                hist['SMA_200'] = current_price
                
                sma_20 = hist['SMA_20'].iloc[-1] if not pd.isna(hist['SMA_20'].iloc[-1]) else current_price
                sma_50 = sma_200 = current_price
                trend_20 = trend_50 = trend_200 = 0
            
            # RSI Calculation
            rsi = self.calculate_rsi(hist['Close'], 14)
            
            # Volatility (Standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Volume ratio (current vs average)
            avg_volume = hist['Volume'].tail(20).mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Support and Resistance (52-week high/low)
            support = hist['Low'].tail(252).min()
            resistance = hist['High'].tail(252).max()
            
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
                'resistance': resistance
            }
            
            return data, hist
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None, None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def create_price_chart(self, hist, ticker, company_name):
        """Create interactive price chart with moving averages"""
        
        # Get last 6 months of data for cleaner visualization
        hist_6m = hist.tail(120)
        
        # Create subplots: Price chart and Volume chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{company_name} ({ticker}) - Price & Moving Averages', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist_6m.index,
                y=hist_6m['Close'],
                name='Price',
                line=dict(color='#667eea', width=2),
                hovertemplate='₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in hist_6m.columns:
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
        
        if 'SMA_50' in hist_6m.columns:
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
        
        if 'SMA_200' in hist_6m.columns:
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
        
        # Calculate RSI for all historical data
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        
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
        
        # Overbought line (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="Overbought (70)", annotation_position="right")
        
        # Oversold line (30)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
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
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def analyze_stock(self, ticker, company_name=None):
        """Complete analysis with real data"""
        
        # Fetch real market data
        data, hist = self.fetch_real_data(ticker)
        
        if data is None:
            return None, None
        
        # Calculate comprehensive score
        score = self._calculate_score(
            data['daily_change'], 
            data['trend_20'], 
            data['trend_50'], 
            data['trend_200'],
            data['rsi'], 
            data['volatility'], 
            data['volume_ratio']
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
            data['daily_change']
        )
        
        result = {
            'ticker': ticker,
            'company_name': company_name or self.default_stocks.get(ticker, ticker),
            **data,
            'score': score,
            **recommendation_data
        }
        
        return result, hist
    
    def _calculate_score(self, daily_change, trend_20, trend_50, trend_200, rsi, volatility, volume_ratio):
        """Advanced scoring algorithm"""
        score = 50  # Base score
        
        # Trend Analysis (35 points)
        if trend_20 > 8 and trend_50 > 5 and trend_200 > 0:
            score += 35
        elif trend_20 > 5 and trend_50 > 3:
            score += 25
        elif trend_20 > 2 and trend_50 > 0:
            score += 15
        elif trend_20 > 0:
            score += 5
        elif trend_20 < -5:
            score -= 15
        
        # RSI Analysis (25 points)
        if 40 <= rsi <= 60:
            score += 25
        elif 30 <= rsi < 40:
            score += 20
        elif 60 < rsi <= 70:
            score += 15
        elif rsi < 30:
            score += 10
        else:
            score += 5
        
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
        elif volume_ratio > 1.0:
            score += 5
        
        # Daily momentum (15 points)
        if daily_change > 2:
            score += 15
        elif daily_change > 0:
            score += 8
        elif daily_change > -2:
            score += 3
        else:
            score -= 5
        
        return max(0, min(100, score))
    
    def _generate_recommendation(self, score, rsi, trend_20, trend_50, trend_200, volatility, volume_ratio, daily_change):
        """Generate buy/sell recommendation with DATA-DRIVEN reasoning"""
        
        reasons = []
        
        # Analyze trends
        if trend_20 > 5 and trend_50 > 3 and trend_200 > 0:
            reasons.append(f"Strong upward trend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%), 200D (+{trend_200:.1f}%)")
        elif trend_20 > 5 and trend_50 > 3:
            reasons.append(f"Strong short-term uptrend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%)")
        elif trend_20 > 2 and trend_50 > 0:
            reasons.append(f"Moderate uptrend: 20D (+{trend_20:.1f}%), 50D (+{trend_50:.1f}%)")
        elif trend_20 > 0:
            reasons.append(f"Weak uptrend: 20D (+{trend_20:.1f}%), momentum limited")
        elif trend_20 < 0 and trend_50 < 0:
            reasons.append(f"Downtrend active: 20D ({trend_20:.1f}%), 50D ({trend_50:.1f}%)")
        elif trend_20 < -5:
            reasons.append(f"Strong downward pressure: 20D ({trend_20:.1f}%)")
        else:
            reasons.append(f"Mixed trend signals: 20D ({trend_20:.1f}%), 50D ({trend_50:.1f}%)")
        
        # RSI analysis
        if 40 <= rsi <= 60:
            reasons.append(f"RSI neutral at {rsi:.1f} - balanced momentum")
        elif 30 <= rsi < 40:
            reasons.append(f"RSI at {rsi:.1f} - approaching oversold, potential buying opportunity")
        elif rsi < 30:
            reasons.append(f"RSI oversold at {rsi:.1f} - strong reversal potential")
        elif 60 < rsi <= 70:
            reasons.append(f"RSI at {rsi:.1f} - mild overbought, caution on entry")
        elif rsi > 70:
            reasons.append(f"RSI overbought at {rsi:.1f} - high risk of correction")
        
        # Volatility analysis
        if volatility < 20:
            reasons.append(f"Low volatility ({volatility:.1f}%) - stable price action")
        elif volatility < 30:
            reasons.append(f"Moderate volatility ({volatility:.1f}%) - normal fluctuations")
        elif volatility < 40:
            reasons.append(f"High volatility ({volatility:.1f}%) - increased risk")
        else:
            reasons.append(f"Very high volatility ({volatility:.1f}%) - significant price swings")
        
        # Volume analysis
        if volume_ratio > 1.5:
            reasons.append(f"Strong volume ({volume_ratio:.2f}x) - high conviction move")
        elif volume_ratio > 1.2:
            reasons.append(f"Above average volume ({volume_ratio:.2f}x) - good participation")
        elif volume_ratio > 0.8:
            reasons.append(f"Normal volume ({volume_ratio:.2f}x) - adequate liquidity")
        else:
            reasons.append(f"Low volume ({volume_ratio:.2f}x) - weak participation, be cautious")
        
        # Daily momentum
        if daily_change > 2:
            reasons.append(f"Strong daily gain (+{daily_change:.2f}%) - positive momentum")
        elif daily_change > 0:
            reasons.append(f"Positive day (+{daily_change:.2f}%) - bullish sentiment")
        elif daily_change > -2:
            reasons.append(f"Minor decline ({daily_change:.2f}%) - consolidation phase")
        else:
            reasons.append(f"Sharp decline ({daily_change:.2f}%) - selling pressure")
        
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
            action = "Good Entry Point"
            risk = "LOW-MEDIUM"
        elif score >= 60:
            recommendation = "MODERATE BUY 📈"
            confidence = "MEDIUM-HIGH"
            color = "🟡"
            action = "Consider for Portfolio"
            risk = "MEDIUM"
        elif score >= 45:
            recommendation = "HOLD 🔄"
            confidence = "MEDIUM"
            color = "🟠"
            action = "Wait for Better Setup"
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
            action = "Stay Away - High Risk"
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Real-Time Stock Recommendation System</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = RealTimeStockAnalyzer()
    
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
        - ✅ Technical analysis
        - ✅ Smart recommendations
        - ✅ Risk assessment
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Auto Scan", "➕ Manual Analysis", "📚 Guide"])
    
    with tab1:
        st.markdown("### 📊 Real-Time Stock Scanner")
        st.info(f"📡 **Live Analysis** - Fetching real-time data from Yahoo Finance for top 20 Indian stocks")
        
        if st.button("🚀 START LIVE ANALYSIS", type="primary", use_container_width=True):
            
            with st.spinner("📡 Fetching real-time market data..."):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_stocks = len(analyzer.default_stocks)
                
                for i, (ticker, name) in enumerate(analyzer.default_stocks.items()):
                    status_text.text(f"Analyzing {name}... ({i+1}/{total_stocks})")
                    result, hist = analyzer.analyze_stock(ticker)
                    
                    if result:
                        results.append({'data': result, 'hist': hist})
                    
                    progress_bar.progress((i + 1) / total_stocks)
                
                status_text.empty()
                progress_bar.empty()
            
            if not results:
                st.error("❌ Unable to fetch market data. Please check your internet connection and try again.")
                return
            
            # Extract just the data for filtering
            all_stock_data = [item['data'] for item in results]
            
            # Filter results
            filtered_results = [
                stock for stock in all_stock_data
                if stock['score'] >= min_score and 
                any(rec in stock['recommendation'] for rec in show_recommendations)
            ]
            
            # Create a dict for easy hist lookup
            hist_dict = {item['data']['ticker']: item['hist'] for item in results}
            
            # Sort by score
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            if filtered_results:
                st.success(f"✅ Analysis Complete! Found {len(filtered_results)} stocks matching your criteria")
                st.caption(f"⏰ Last updated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                strong_buy = len([s for s in all_stock_data if "STRONG BUY" in s['recommendation']])
                buy = len([s for s in all_stock_data if s['recommendation'].startswith("BUY")])
                moderate = len([s for s in all_stock_data if "MODERATE" in s['recommendation']])
                hold = len([s for s in all_stock_data if "HOLD" in s['recommendation']])
                
                with col1:
                    st.metric("🚀 STRONG BUY", strong_buy)
                with col2:
                    st.metric("✅ BUY", buy)
                with col3:
                    st.metric("📈 MODERATE BUY", moderate)
                with col4:
                    st.metric("🔄 HOLD/AVOID", hold)
                
                # Top recommendations
                st.markdown("---")
                st.markdown(f"## 🎯 Top {min(top_n, len(filtered_results))} Investment Opportunities")
                
                for i, stock in enumerate(filtered_results[:top_n]):
                    with st.expander(f"{stock['color']} #{i+1}: {stock['company_name']} - {stock['recommendation']} | Score: {stock['score']}/100", expanded=(i<3)):
                        
                        st.markdown(f"**Ticker:** {stock['ticker']}")
                        st.markdown(f"**🎯 Action:** {stock['action']}")
                        st.markdown(f"**📊 Confidence:** {stock['confidence']} | **⚠️ Risk:** {stock['risk']}")
                        
                        st.markdown("---")
                        
                        # Show price chart
                        if stock['ticker'] in hist_dict:
                            st.markdown("### 📈 Price Chart with Moving Averages")
                            fig_price = analyzer.create_price_chart(hist_dict[stock['ticker']], stock['ticker'], stock['company_name'])
                            st.plotly_chart(fig_price, use_container_width=True)
                            
                            st.markdown("### 📊 RSI Indicator")
                            fig_rsi = analyzer.create_rsi_chart(hist_dict[stock['ticker']], stock['ticker'])
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("💰 Current Price", f"₹{stock['current_price']:.2f}", f"{stock['daily_change']:+.2f}%")
                            st.metric("🎯 RSI", f"{stock['rsi']:.1f}")
                        
                        with col2:
                            st.metric("📊 Trend (20D)", f"{stock['trend_20']:+.1f}%")
                            st.metric("📊 Trend (50D)", f"{stock['trend_50']:+.1f}%")
                        
                        with col3:
                            st.metric("📊 Trend (200D)", f"{stock['trend_200']:+.1f}%")
                            st.metric("⚡ Volatility", f"{stock['volatility']:.1f}%")
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**📦 Volume Ratio:** {stock['volume_ratio']:.2f}x")
                            st.write(f"**🛡️ Support:** ₹{stock['support']:.2f}")
                            st.write(f"**🎯 Resistance:** ₹{stock['resistance']:.2f}")
                        
                        with col2:
                            st.write(f"**SMA 20:** ₹{stock['sma_20']:.2f}")
                            st.write(f"**SMA 50:** ₹{stock['sma_50']:.2f}")
                            st.write(f"**SMA 200:** ₹{stock['sma_200']:.2f}")
                        
                        st.markdown("---")
                        st.markdown("**📋 Analysis Reasons:**")
                        for reason in stock['reasons']:
                            st.write(f"• {reason}")
                
                # Investment Plan
                st.markdown("---")
                st.markdown("## 💰 Today's Investment Strategy")
                
                strong_buys = [s for s in all_stock_data if "STRONG BUY" in s['recommendation']]
                buys = [s for s in all_stock_data if s['recommendation'].startswith("BUY") and "MODERATE" not in s['recommendation']]
                
                if strong_buys or buys:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Recommended Portfolio**")
                        
                        if strong_buys:
                            st.markdown("**💎 Strong Buy Stocks (40% allocation):**")
                            for stock in strong_buys[:3]:
                                st.write(f"• {stock['company_name']} - ₹{stock['current_price']:.2f} (Score: {stock['score']})")
                        
                        if buys:
                            st.markdown("**✅ Buy Stocks (30% allocation):**")
                            for stock in buys[:2]:
                                st.write(f"• {stock['company_name']} - ₹{stock['current_price']:.2f} (Score: {stock['score']})")
                        
                        st.markdown("**💵 Cash Reserve: 30%**")
                    
                    with col2:
                        st.markdown("""
                        **📊 Risk Management**
                        
                        **Stop Loss:**
                        - Set at 5-7% below entry
                        - Trail as price rises
                        - Book profits at 10-15%
                        
                        **Position Size:**
                        - Max 15% per stock
                        - 5-7 stocks total
                        - 30% cash reserve
                        """)
                
                else:
                    st.warning("🔄 **Market Outlook: Neutral** - No strong buy signals currently. Wait for better opportunities.")
                
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
                        'Change': f"{stock['daily_change']:+.2f}%",
                        'RSI': f"{stock['rsi']:.1f}",
                        'Volatility': f"{stock['volatility']:.1f}%",
                        'Trend 20D': f"{stock['trend_20']:+.1f}%",
                        'Score': stock['score'],
                        'Recommendation': f"{stock['color']} {stock['recommendation']}",
                        'Risk': stock['risk']
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, height=400)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis (CSV)",
                    data=csv,
                    file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            else:
                st.error("❌ No stocks match your filters. Try adjusting the settings in the sidebar.")
    
    with tab2:
        st.markdown("### ➕ Analyze Custom Stock")
        st.markdown("Enter any NSE stock ticker for real-time analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_ticker = st.text_input(
                "Enter Stock Ticker",
                placeholder="e.g., WIPRO.NS, ADANIENT.NS, M&M.NS",
                help="Use Yahoo Finance format: SYMBOL.NS for NSE stocks"
            )
        
        with col2:
            manual_name = st.text_input(
                "Company Name (Optional)",
                placeholder="e.g., Wipro Ltd"
            )
        
        if st.button("🔍 ANALYZE THIS STOCK", type="primary", use_container_width=True):
            if manual_ticker:
                with st.spinner(f"📡 Fetching live data for {manual_ticker}..."):
                    stock, hist = analyzer.analyze_stock(
                        manual_ticker.upper(),
                        manual_name if manual_name else None
                    )
                
                if stock:
                    st.success(f"✅ Analysis complete for {stock['company_name']}")
                    st.caption(f"⏰ Data as of: {datetime.now().strftime('%d-%m-%Y %H:%M:%S IST')}")
                    
                    # Display analysis
                    st.markdown(f"## {stock['color']} {stock['company_name']} ({stock['ticker']})")
                    st.markdown(f"### {stock['recommendation']} | Score: {stock['score']}/100")
                    st.markdown(f"**🎯 Action:** {stock['action']}")
                    st.markdown(f"**Confidence:** {stock['confidence']} | **Risk:** {stock['risk']}")
                    
                    st.markdown("---")
                    
                    # Show charts
                    if hist is not None:
                        st.markdown("### 📈 Price Chart with Moving Averages")
                        fig_price = analyzer.create_price_chart(hist, stock['ticker'], stock['company_name'])
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                        st.markdown("### 📊 RSI Indicator")
                        fig_rsi = analyzer.create_rsi_chart(hist, stock['ticker'])
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Price Information**")
                        st.metric("Current Price", f"₹{stock['current_price']:.2f}", f"{stock['daily_change']:+.2f}%")
                        st.write(f"**Support:** ₹{stock['support']:.2f}")
                        st.write(f"**Resistance:** ₹{stock['resistance']:.2f}")
                    
                    with col2:
                        st.markdown("**📈 Technical Indicators**")
                        st.write(f"**RSI:** {stock['rsi']:.1f}")
                        st.write(f"**Volatility:** {stock['volatility']:.1f}%")
                        st.write(f"**Volume Ratio:** {stock['volume_ratio']:.2f}x")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("20-Day Trend", f"{stock['trend_20']:+.1f}%")
                    with col2:
                        st.metric("50-Day Trend", f"{stock['trend_50']:+.1f}%")
                    with col3:
                        st.metric("200-Day Trend", f"{stock['trend_200']:+.1f}%")
                    
                    st.markdown("---")
                    st.markdown("**📋 Analysis Reasons:**")
                    for reason in stock['reasons']:
                        st.write(f"• {reason}")
                    
                    # Action plan
                    if "BUY" in stock['recommendation']:
                        st.success(f"""
                        **📝 Action Plan:**
                        - Entry: Around ₹{stock['current_price']:.2f}
                        - Stop Loss: ₹{stock['support'] * 0.97:.2f}
                        - Target 1 (10%): ₹{stock['current_price'] * 1.10:.2f}
                        - Target 2 (15%): ₹{stock['current_price'] * 1.15:.2f}
                        - Max allocation: 10-15% of portfolio
                        """)
                    elif "HOLD" in stock['recommendation']:
                        st.info("Strategy: Wait for better technical setup. Re-evaluate in 1-2 weeks.")
                    else:
                        st.warning("Caution: Not recommended for new investment based on current analysis.")
                
                else:
                    st.error(f"❌ Unable to fetch data for {manual_ticker}. Please check the ticker symbol and try again.")
            
            else:
                st.error("❌ Please enter a stock ticker symbol")
    
    with tab3:
        st.markdown("## 📚 System Guide")
        
        st.markdown("""
        ### 🎯 How This Works
        
        **Data Source:**
        - Real-time data from Yahoo Finance API
        - Live NSE stock prices
        - Historical data for technical analysis
        
        **Technical Analysis:**
        - Moving Averages (20, 50, 200 day)
        - RSI (Relative Strength Index)
        - Volatility measurement
        - Volume analysis
        - Support & Resistance levels
        
        **Scoring System (0-100):**
        - 80-100: STRONG BUY 🚀 (Very High Confidence, Low Risk)
        - 70-79: BUY ✅ (High Confidence, Low-Medium Risk)
        - 60-69: MODERATE BUY 📈 (Medium Confidence, Medium Risk)
        - 45-59: HOLD 🔄 (Wait for better setup)
        - 0-44: AVOID ❌ (High Risk)
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Key Indicators Explained
            
            **RSI (Relative Strength Index):**
            - Below 30: Oversold (potential buy)
            - 30-40: Approaching oversold
            - 40-60: Neutral/Balanced
            - 60-70: Slightly overbought
            - Above 70: Overbought (caution)
            
            **Trend Analysis:**
            - Positive % = Price above moving average (bullish)
            - Negative % = Price below moving average (bearish)
            - 20D: Short-term trend
            - 50D: Medium-term trend
            - 200D: Long-term trend
            
            **Volatility:**
            - Below 20%: Stable, low risk
            - 20-30%: Normal fluctuations
            - 30-40%: High risk
            - Above 40%: Very high risk
            
            **Volume Ratio:**
            - Above 1.5x: High conviction
            - 1.0-1.5x: Normal
            - Below 1.0x: Low participation
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Investment Guidelines
            
            **Portfolio Construction:**
            - Diversify across 5-7 stocks
            - Max 15% per stock
            - Keep 20-30% cash reserve
            - Mix of sectors
            
            **Risk Management:**
            - Always set stop loss (5-7% below entry)
            - Trail stop loss as price rises
            - Book partial profits at targets
            - Review positions weekly
            
            **Entry Strategy:**
            - Buy near support levels
            - Enter gradually (50% first, add later)
            - Confirm with volume
            - Check multiple timeframes
            
            **Exit Strategy:**
            - Stick to stop loss (no emotions!)
            - Take profits at resistance
            - Exit if fundamentals change
            - Don't average down losses
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### ⚠️ Important Notes
        
        **Before Investing:**
        - This tool provides technical analysis only
        - Do your own fundamental research
        - Consult a financial advisor
        - Only invest what you can afford to lose
        - Past performance ≠ future results
        
        **Best Practices:**
        - ✅ Use stop losses always
        - ✅ Diversify your portfolio
        - ✅ Follow systematic approach
        - ✅ Keep emotions out
        - ✅ Review and learn
        - ❌ Don't chase losses
        - ❌ Don't overtrade
        - ❌ Don't ignore warnings
        
        **Data Limitations:**
        - Market data may have 15-min delay
        - Technical analysis is not 100% accurate
        - News/events can override technicals
        - Always verify ticker symbols
        """)
        
        st.markdown("---")
        
        st.success("""
        ### 🚀 Getting Started
        
        1. **Auto Scan Tab:** Click "START LIVE ANALYSIS" to scan 20 top stocks
        2. **Review Results:** Check recommendations and scores
        3. **Read Analysis:** Understand the reasons for each recommendation
        4. **Make Decision:** Use the data to make informed investment choices
        5. **Manual Analysis:** Analyze any specific stock by ticker
        6. **Download Data:** Export analysis for your records
        
        **Pro Tips:**
        - Run analysis daily before market opens
        - Focus on STRONG BUY and BUY recommendations
        - Cross-check with fundamental analysis
        - Start with small positions
        - Keep a trading journal
        """)
        
        st.markdown("---")
        
        st.info("""
        ### 📞 Need Help?
        
        **Common Issues:**
        - **"Unable to fetch data"**: Check internet connection, verify ticker format
        - **"No stocks match filters"**: Lower minimum score or change recommendation filters
        - **Wrong ticker**: Use Yahoo Finance format (SYMBOL.NS for NSE)
        
        **Ticker Format Examples:**
        - Reliance: RELIANCE.NS
        - TCS: TCS.NS
        - Infosys: INFY.NS
        - HDFC Bank: HDFCBANK.NS
        
        For any stock, add ".NS" suffix for NSE or ".BO" for BSE.
        """)

if __name__ == "__main__":
    main()
