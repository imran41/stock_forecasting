import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# ==================== UTILITY FUNCTIONS ====================

def format_currency(value):
    """Format value as Indian currency"""
    if value is None:
        return "N/A"
    return f"₹{value:.2f}"

def format_percentage(value, is_already_percent=False):
    """Format value as percentage"""
    if value is None:
        return "N/A"
    if is_already_percent:
        return f"{value:.2f}%"
    return f"{value * 100:.2f}%"

def format_ratio(value):
    """Format value as ratio"""
    if value is None:
        return "N/A"
    return f"{value:.2f}"

def format_market_cap(value):
    """Format market cap in crores"""
    if value is None:
        return "N/A"
    return f"₹{value / 1e7:.2f} Cr"

def calculate_distance_from_52w(current_price, week_52_low, week_52_high):
    """Calculate percentage distance from 52-week low and high"""
    pct_from_low = None
    pct_from_high = None
    
    if current_price and week_52_low:
        pct_from_low = ((current_price - week_52_low) / week_52_low * 100)
    
    if current_price and week_52_high:
        pct_from_high = ((current_price - week_52_high) / week_52_high * 100)
    
    return pct_from_low, pct_from_high

def fetch_stock_data(ticker):
    """Fetch stock data from yfinance"""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    return stock, info, hist

def extract_stock_fundamentals(info):
    """Extract fundamental data from stock info"""
    div_yield = info.get("dividendYield")
    debt_to_equity = info.get("debtToEquity")
    
    fundamentals = {
        'current_price': info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"),
        'week_52_low': info.get("fiftyTwoWeekLow"),
        'week_52_high': info.get("fiftyTwoWeekHigh"),
        'PE': info.get("trailingPE"),
        'PB': info.get("priceToBook"),
        'div_yield': div_yield/100 if div_yield is not None else None,
        'ROE': info.get("returnOnEquity"),
        'DE_ratio': debt_to_equity/100 if debt_to_equity is not None else None,
        'market_cap': info.get("marketCap"),
        'EPS': info.get("trailingEps"),
        'company_name': info.get("longName") or info.get("shortName") or "N/A",
        'sector': info.get("sector", "N/A"),
        'industry': info.get("industry", "N/A"),
        'industry_pe': info.get("industryPE")  # Try to fetch industry PE
    }
    return fundamentals

def calculate_moving_average(hist, window=50):
    """Calculate moving average"""
    if len(hist) >= window:
        return hist['Close'].rolling(window=window).mean().iloc[-1]
    return None

def evaluate_criterion_near_52w_low(current_price, week_52_low):
    """Evaluate if stock is near 52-week low"""
    if current_price and week_52_low and current_price <= week_52_low * 1.10:
        return 1.0, "✅ Trading within 10% of 52-week low (+1.0)"
    return 0.0, "❌ Not near 52-week low (0)"

def evaluate_criterion_pe_ratio(PE, industry_pe):
    """Evaluate P/E ratio criterion"""
    if PE and PE < industry_pe:
        return 1.0, f"✅ P/E ({PE:.2f}) below industry P/E ({industry_pe:.2f}) (+1.0)"
    elif PE:
        return 0.0, f"❌ P/E ({PE:.2f}) above industry average (0)"
    else:
        return 0.0, "⚠️ P/E data not available (0)"

def evaluate_criterion_pb_ratio(PB):
    """Evaluate P/B ratio criterion"""
    if PB and PB <= 3:
        return 1.0, f"✅ P/B ratio ({PB:.2f}) ≤ 3 (+1.0)"
    elif PB:
        return 0.0, f"❌ P/B ratio ({PB:.2f}) > 3 (0)"
    else:
        return 0.0, "⚠️ P/B data not available (0)"

def evaluate_criterion_dividend_yield(div_yield):
    """Evaluate dividend yield criterion"""
    if div_yield:
        # Handle both formats: decimal (0.41) or percentage (41.0)
        if div_yield > 1:  # It's already a percentage
            normalized_yield = div_yield / 100
            display_yield = div_yield
        else:  # It's a decimal
            normalized_yield = div_yield
            display_yield = div_yield * 100
        
        if normalized_yield >= 0.02:  # 2% threshold
            return 0.5, f"✅ Dividend yield ({display_yield:.2f}%) ≥ 2% (+0.5)"
        else:
            return 0.0, f"❌ Dividend yield ({display_yield:.2f}%) < 2% (0)"
    else:
        return 0.0, "⚠️ Dividend data not available (0)"

def evaluate_criterion_roe(ROE):
    """Evaluate ROE criterion"""
    if ROE and ROE >= 0.10:
        return 1.0, f"✅ ROE ({ROE * 100:.2f}%) ≥ 10% (+1.0)"
    elif ROE:
        return 0.0, f"❌ ROE ({ROE * 100:.2f}%) < 10% (0)"
    else:
        return 0.0, "⚠️ ROE data not available (0)"

def evaluate_criterion_debt_equity(DE_ratio):
    """Evaluate Debt/Equity criterion"""
    if DE_ratio is not None:
        # yfinance returns D/E as percentage (244 for 2.44), need to divide by 100
        normalized_de = DE_ratio / 100 if DE_ratio > 10 else DE_ratio
        
        if normalized_de <= 0.5:
            return 1.0, f"✅ Debt/Equity ({normalized_de:.2f}) ≤ 0.5 (+1.0)"
        else:
            return 0.0, f"❌ Debt/Equity ({normalized_de:.2f}) > 0.5 (0)"
    else:
        return 0.0, "⚠️ Debt/Equity data not available (0)"

def evaluate_criterion_market_cap(market_cap):
    """Evaluate market cap criterion"""
    if market_cap and market_cap >= 20_000 * 1e7:
        return 1.0, f"✅ Large-cap stock (Market Cap ≥ ₹20,000 Cr) (+1.0)"
    elif market_cap:
        return 0.0, f"❌ Not a large-cap stock (0)"
    else:
        return 0.0, "⚠️ Market cap data not available (0)"

def evaluate_criterion_momentum(current_price, ma_50):
    """Evaluate price momentum criterion"""
    if ma_50 and current_price and current_price > ma_50:
        return 0.5, f"✅ Price above 50-day MA (₹{ma_50:.2f}) (+0.5)"
    elif ma_50:
        return 0.0, f"❌ Price below 50-day MA (₹{ma_50:.2f}) (0)"
    else:
        return 0.0, "⚠️ Not enough data for 50-day MA (0)"

def calculate_investment_score(fundamentals, industry_pe, hist):
    """Calculate investment score based on all criteria"""
    score = 0
    criteria_results = []
    
    # Criterion 1: Near 52-week low
    points, message = evaluate_criterion_near_52w_low(fundamentals['current_price'], fundamentals['week_52_low'])
    score += points
    criteria_results.append(message)
    
    # Criterion 2: P/E ratio
    points, message = evaluate_criterion_pe_ratio(fundamentals['PE'], industry_pe)
    score += points
    criteria_results.append(message)
    
    # Criterion 3: P/B ratio
    points, message = evaluate_criterion_pb_ratio(fundamentals['PB'])
    score += points
    criteria_results.append(message)
    
    # Criterion 4: Dividend yield
    points, message = evaluate_criterion_dividend_yield(fundamentals['div_yield'])
    score += points
    criteria_results.append(message)
    
    # Criterion 5: ROE
    points, message = evaluate_criterion_roe(fundamentals['ROE'])
    score += points
    criteria_results.append(message)
    
    # Criterion 6: Debt/Equity
    points, message = evaluate_criterion_debt_equity(fundamentals['DE_ratio'])
    score += points
    criteria_results.append(message)
    
    # Criterion 7: Market cap
    points, message = evaluate_criterion_market_cap(fundamentals['market_cap'])
    score += points
    criteria_results.append(message)
    
    # Criterion 8: Price momentum
    ma_50 = calculate_moving_average(hist, 50)
    points, message = evaluate_criterion_momentum(fundamentals['current_price'], ma_50)
    score += points
    criteria_results.append(message)
    
    return score, criteria_results

def get_recommendation(score, max_score):
    """Get investment recommendation based on score"""
    score_percentage = (score / max_score) * 100
    
    if score >= 6:
        return "BUY / ACCUMULATE", "Strong fundamentals with favorable entry point near 52-week low.", "success"
    elif score >= 4:
        return "WAIT / PARTIAL BUY", "Moderate fundamentals. Consider dollar-cost averaging or wait for better entry.", "warning"
    else:
        return "AVOID / SKIP", "Weak fundamentals or unfavorable risk-reward. Look for better opportunities.", "error"

def display_header():
    """Display application header"""
    st.markdown("# 📈 NSE Stock Analyzer")
    st.markdown("**Analyze NSE stocks trading near 52-week lows with fundamental insights**")
    st.markdown("---")

def display_company_info(ticker, fundamentals):
    """Display company information"""
    st.subheader(f"🏢 {fundamentals['company_name']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sector", fundamentals['sector'])
    with col2:
        st.metric("Industry", fundamentals['industry'])
    with col3:
        st.metric("Ticker", ticker)

def display_price_analysis(fundamentals, pct_from_low, pct_from_high):
    """Display price analysis section"""
    st.markdown("---")
    st.subheader("💰 Price Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", format_currency(fundamentals['current_price']))
    
    with col2:
        delta_str = f"{pct_from_low:.1f}%" if pct_from_low is not None else None
        st.metric("52W Low", format_currency(fundamentals['week_52_low']), delta=delta_str)
    
    with col3:
        delta_str = f"{pct_from_high:.1f}%" if pct_from_high is not None else None
        st.metric("52W High", format_currency(fundamentals['week_52_high']), delta=delta_str)
    
    with col4:
        st.metric("Market Cap", format_market_cap(fundamentals['market_cap']))

def display_fundamentals(fundamentals, industry_pe):
    """Display fundamental metrics"""
    st.markdown("---")
    st.subheader("📊 Fundamental Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("P/E Ratio", format_ratio(fundamentals['PE']))
    with col2:
        st.metric("P/B Ratio", format_ratio(fundamentals['PB']))
    with col3:
        st.metric("EPS", format_currency(fundamentals['EPS']))
    with col4:
        st.metric("ROE", format_percentage(fundamentals['ROE']))
    with col5:
        # Handle dividend yield - check if already percentage
        div_val = fundamentals['div_yield']
        if div_val is not None and div_val > 1:
            st.metric("Div Yield", f"{div_val:.2f}%")
        else:
            st.metric("Div Yield", format_percentage(div_val))
    
    col1, col2 = st.columns(2)
    with col1:
        # Normalize D/E ratio for display
        de_display = fundamentals['DE_ratio']
        if de_display is not None and de_display > 10:
            de_display = de_display / 100
        st.metric("Debt/Equity", format_ratio(de_display))
    with col2:
        # Show if industry PE was auto-fetched or manual
        industry_pe_label = "Industry P/E"
        if fundamentals.get('industry_pe'):
            industry_pe_label = "Industry P/E (Auto)"
        else:
            industry_pe_label = "Industry P/E (Manual)"
        st.metric(industry_pe_label, format_ratio(industry_pe))

def display_scoring_analysis(criteria_results):
    """Display scoring criteria analysis"""
    st.markdown("---")
    st.subheader("🎯 Investment Score Analysis")
    
    for criterion in criteria_results:
        st.markdown(f"**{criterion}**")

def display_recommendation_section(score, max_score):
    """Display final recommendation"""
    st.markdown("---")
    
    recommendation, description, msg_type = get_recommendation(score, max_score)
    score_percentage = (score / max_score) * 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Investment Score", f"{score:.1f} / {max_score}", delta=f"{score_percentage:.0f}%")
    
    with col2:
        if msg_type == "success":
            st.success(f"🟢 **Recommendation: {recommendation}**")
        elif msg_type == "warning":
            st.warning(f"🟡 **Recommendation: {recommendation}**")
        else:
            st.error(f"🔴 **Recommendation: {recommendation}**")
        
        st.write(description)

def display_charts(hist, ticker):
    """Display price charts"""
    st.markdown("---")
    st.subheader("📈 Price Charts")
    
    # Create tabs for different chart types
    tab1, tab2 = st.tabs(["📊 Line Chart", "🕯️ Candlestick Chart"])
    
    with tab1:
        line_chart = create_price_line_chart(hist, ticker)
        if line_chart:
            st.plotly_chart(line_chart, use_container_width=True)
        else:
            st.warning("No historical data available for line chart")
    
    with tab2:
        candle_chart = create_candlestick_chart(hist, ticker)
        if candle_chart:
            st.plotly_chart(candle_chart, use_container_width=True)
        else:
            st.warning("No historical data available for candlestick chart")

def display_disclaimer():
    """Display disclaimer"""
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** This analysis is for educational purposes only. Always conduct thorough research and consult with a financial advisor before making investment decisions.")

def create_price_line_chart(hist, ticker):
    """Create interactive line chart for price history"""
    if hist.empty:
        return None
    
    fig = go.Figure()
    
    # Add closing price line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86DE', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
    ))
    
    # Add 50-day moving average
    if len(hist) >= 50:
        ma_50 = hist['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma_50,
            mode='lines',
            name='50-Day MA',
            line=dict(color='#FFA502', width=1.5, dash='dash'),
            hovertemplate='<b>50-Day MA</b>: ₹%{y:.2f}<extra></extra>'
        ))
    
    # Add 200-day moving average
    if len(hist) >= 200:
        ma_200 = hist['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma_200,
            mode='lines',
            name='200-Day MA',
            line=dict(color='#FF6348', width=1.5, dash='dot'),
            hovertemplate='<b>200-Day MA</b>: ₹%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{ticker} - Price History (1 Year)',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

def create_candlestick_chart(hist, ticker):
    """Create interactive candlestick chart with volume"""
    if hist.empty:
        return None
    
    # Create subplots: candlestick on top, volume on bottom
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} - Candlestick Chart', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC',
            increasing_line_color='#26DE81',
            decreasing_line_color='#FC5C65',
            hovertext=[f'Open: ₹{o:.2f}<br>High: ₹{h:.2f}<br>Low: ₹{l:.2f}<br>Close: ₹{c:.2f}' 
                      for o, h, l, c in zip(hist['Open'], hist['High'], hist['Low'], hist['Close'])]
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26DE81' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] 
              else '#FC5C65' for i in range(len(hist))]
    
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False,
            hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig
    """Display disclaimer"""
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** This analysis is for educational purposes only. Always conduct thorough research and consult with a financial advisor before making investment decisions.")

def display_sidebar():
    """Display sidebar with instructions"""
    with st.sidebar:
        st.header("📖 How to Use")
        st.markdown("""
        1. Enter NSE ticker (e.g., **RELIANCE.NS**)
        2. Adjust industry P/E if needed
        3. Click **Analyze Stock**
        4. Review the investment score
        
        **Popular NSE Tickers:**
        - RELIANCE.NS
        - TCS.NS
        - INFY.NS
        - HDFCBANK.NS
        - WIPRO.NS
        - TATAMOTORS.NS
        - ITC.NS
        - SBIN.NS
        """)
        
        st.markdown("---")
        st.markdown("**Scoring Criteria:**")
        st.markdown("""
        - Near 52W low: 1.0
        - P/E < Industry: 1.0
        - P/B ≤ 3: 1.0
        - Div Yield ≥ 2%: 0.5
        - ROE ≥ 10%: 1.0
        - D/E ≤ 0.5: 1.0
        - Large cap: 1.0
        - Above 50-MA: 0.5
        
        **Max Score: 7.5**
        """)

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""
    display_header()
    display_sidebar()
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Enter NSE Ticker Symbol",
            placeholder="e.g., WIPRO.NS, TCS.NS, RELIANCE.NS",
            help="Add .NS suffix for NSE stocks"
        ).upper()
    
    with col2:
        industry_pe_input = st.number_input(
            "Industry P/E", 
            value=25.0, 
            min_value=0.0, 
            step=0.5,
            help="Will auto-update if available from yfinance"
        )
    
    with col3:
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    # Process analysis
    if ticker_input and analyze_btn:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            try:
                # Fetch data
                stock, info, hist = fetch_stock_data(ticker_input)
                
                # Validate data
                if not info or info.get('regularMarketPrice') is None:
                    st.error(f"❌ Could not fetch data for {ticker_input}. Please verify the ticker symbol.")
                    return
                
                # Extract fundamentals
                fundamentals = extract_stock_fundamentals(info)
                
                # Update industry PE if available from yfinance
                if fundamentals.get('industry_pe'):
                    industry_pe_input = float(fundamentals['industry_pe'])
                    st.info(f"ℹ️ Industry P/E auto-updated to {industry_pe_input:.2f} from yfinance data")
                
                # Calculate distances from 52-week high/low
                pct_from_low, pct_from_high = calculate_distance_from_52w(
                    fundamentals['current_price'],
                    fundamentals['week_52_low'],
                    fundamentals['week_52_high']
                )
                
                # Display results
                display_company_info(ticker_input, fundamentals)
                display_price_analysis(fundamentals, pct_from_low, pct_from_high)
                
                # Display charts
                display_charts(hist, ticker_input)
                
                display_fundamentals(fundamentals, industry_pe_input)
                
                # Calculate score
                max_score = 7.5
                score, criteria_results = calculate_investment_score(fundamentals, industry_pe_input, hist)
                
                # Display analysis
                display_scoring_analysis(criteria_results)
                display_recommendation_section(score, max_score)
                display_disclaimer()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.write("Please verify the ticker symbol and try again.")

# Run the application
if __name__ == "__main__":
    main()
