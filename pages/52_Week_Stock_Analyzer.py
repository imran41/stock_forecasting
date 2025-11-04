import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== PAGE CONFIGURATION ====================

def configure_page():
    """Configure Streamlit page settings with professional styling."""
    st.set_page_config(
        page_title="52-Week Stock Analyzer | NSE Market Insights",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Professional 52-Week High/Low Stock Analyzer for NSE stocks with AI-powered insights."
        }
    )
    
    st.markdown("""
        <style>
        :root {
            --primary-color: #1f77b4;
            --success-color: #26DE81;
            --danger-color: #EE5A6F;
            --warning-color: #FFA502;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            color: rgba(255,255,255,0.9);
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }
        
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: #1f1f1f !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #1f1f1f !important;
        }
        
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #1f1f1f !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #1f1f1f !important;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'analysis_results': None,
        'analysis_stats': None,
        'analysis_time': None,
        'selected_stock_list': 'NIFTY 100',
        'analysis_complete': False,
        'last_update': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data(ttl=3600)
def load_stock_symbols(option: str) -> List[str]:
    """Load stock symbols based on selected option with error handling."""
    url_map = {
        "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "NIFTY 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "All NSE Stocks": "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    }
    
    try:
        df = pd.read_csv(url_map[option])
        col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
        symbols = df[col].dropna().unique().tolist()
        symbols = [s.strip() for s in symbols if s and isinstance(s, str) and s.strip()]
        return symbols
    except Exception as e:
        st.error(f"❌ Error loading stock list: {str(e)}")
        return []

def fetch_stock_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Fetch both historical data and info in a single optimized call."""
    try:
        symbol = str(symbol).strip().upper()
        if not symbol:
            return None, {}
        
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1y", auto_adjust=True, actions=False)
        
        if hist is None or hist.empty or len(hist) < 30:
            return None, {}
        
        try:
            info = ticker.info
            if not info or not isinstance(info, dict):
                info = {}
        except:
            info = {}
        
        stock_info = {
            "pe": info.get("trailingPE", info.get("forwardPE")),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", info.get("industry", "Unknown")),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "52w_change": info.get("52WeekChange")
        }
        
        return hist, stock_info
    except Exception:
        return None, {}

def calculate_52_week_metrics(hist: pd.DataFrame) -> Optional[Tuple[float, float, float, float, float]]:
    """Calculate comprehensive 52-week metrics."""
    try:
        if len(hist) < 30:
            return None
            
        high_52w = hist["Close"].max()
        low_52w = hist["Close"].min()
        current = hist["Close"].iloc[-1]
        first_price = hist["Close"].iloc[0]
        avg_volume = hist["Volume"].mean()
        
        if pd.isna(high_52w) or pd.isna(low_52w) or pd.isna(current) or pd.isna(first_price):
            return None
        
        if first_price == 0:
            return None
            
        ret_1y = ((current - first_price) / first_price) * 100
        
        return high_52w, low_52w, current, ret_1y, avg_volume
    except Exception:
        return None

def determine_stock_status(current: float, high_52w: float, low_52w: float) -> Optional[Tuple[str, str, float]]:
    """Determine if stock is near 52W high or low with smart thresholds."""
    if high_52w <= 0 or low_52w <= 0 or current <= 0:
        return None
    
    distance_to_low = ((current - low_52w) / low_52w) * 100
    distance_to_high = ((high_52w - current) / high_52w) * 100
    
    if distance_to_low <= 5:
        return ("52W LOW", "Near support zone - potential value opportunity", low_52w)
    elif distance_to_high <= 5:
        return ("52W HIGH", "Near resistance zone - consider profit booking", high_52w)
    
    return None

def calculate_strength(pe: Optional[float], market_cap: Optional[float], beta: Optional[float]) -> str:
    """Enhanced strength calculation with beta consideration."""
    score = 0
    
    if pe and 0 < pe < 20:
        score += 3
    elif pe and 20 <= pe < 30:
        score += 2
    elif pe and 30 <= pe < 50:
        score += 1
    
    if market_cap:
        if market_cap > 5e11:
            score += 3
        elif market_cap > 1e11:
            score += 2
        else:
            score += 1
    
    if beta:
        if 0.8 <= beta <= 1.2:
            score += 1
    
    if score >= 6:
        return "Strong"
    elif score >= 3:
        return "Moderate"
    else:
        return "Weak"

def generate_ai_summary(stock_type: str, strength: str, return_1y: float, pe: Optional[float], 
                        beta: Optional[float], dividend_yield: Optional[float]) -> str:
    """Generate comprehensive AI investment summary."""
    summary_parts = []
    
    if stock_type == "52W LOW":
        if strength == "Strong" and return_1y < -10:
            summary_parts.append("🎯 **Strong Buy Signal** - Quality stock at discount")
        elif strength == "Strong":
            summary_parts.append("✅ **Accumulation Zone** - Good entry point for long-term")
        elif strength == "Moderate" and return_1y < -20:
            summary_parts.append("⏳ **Wait & Watch** - Needs trend reversal confirmation")
        else:
            summary_parts.append("⚠️ **Caution** - Analyze fundamentals deeply before entry")
        
        if dividend_yield and dividend_yield > 0.02:
            summary_parts.append(f"💰 Dividend yield: {dividend_yield*100:.2f}%")
            
    elif stock_type == "52W HIGH":
        if return_1y > 100:
            summary_parts.append("🔴 **Highly Overextended** - Very high profit booking risk")
        elif return_1y > 50:
            summary_parts.append("⚠️ **Overbought Territory** - Use strict stop loss")
        elif strength == "Strong":
            summary_parts.append("✅ **Strong Momentum** - Can hold with trailing SL")
        else:
            summary_parts.append("⏳ **Momentum Play** - Short-term only, be cautious")
    
    if beta and beta > 1.5:
        summary_parts.append("⚡ High volatility stock")
    elif beta and beta < 0.5:
        summary_parts.append("🛡️ Low volatility stock")
    
    return " | ".join(summary_parts) if summary_parts else "⚠️ Neutral - Monitor closely"

def analyze_single_stock(symbol: str) -> Optional[Dict]:
    """Comprehensive single stock analysis with enhanced metrics."""
    hist, info = fetch_stock_data(symbol)
    if hist is None:
        return None
    
    metrics = calculate_52_week_metrics(hist)
    if metrics is None:
        return None
    
    high_52w, low_52w, current, ret_1y, avg_volume = metrics
    
    status_result = determine_stock_status(current, high_52w, low_52w)
    if status_result is None:
        return None
    
    status, remark, ref_value = status_result
    strength = calculate_strength(info["pe"], info["market_cap"], info["beta"])
    ai_summary = generate_ai_summary(status, strength, ret_1y, info["pe"], 
                                     info["beta"], info["dividend_yield"])
    
    market_cap_display = "N/A"
    if info["market_cap"]:
        if info["market_cap"] >= 1e12:
            market_cap_display = f"₹{info['market_cap']/1e12:.2f}T"
        elif info["market_cap"] >= 1e9:
            market_cap_display = f"₹{info['market_cap']/1e9:.2f}B"
        else:
            market_cap_display = f"₹{info['market_cap']/1e6:.2f}M"
    
    return {
        "Symbol": symbol,
        "Type": status,
        "Current_Price": round(current, 2),
        "52W_Value": round(ref_value, 2),
        "Distance_%": round(((current - ref_value) / ref_value) * 100, 2),
        "P/E": round(info["pe"], 2) if info["pe"] and info["pe"] > 0 else "N/A",
        "Market_Cap": market_cap_display,
        "Sector": info["sector"],
        "1Y_Return_%": round(ret_1y, 2),
        "Beta": round(info["beta"], 2) if info["beta"] else "N/A",
        "Div_Yield_%": round(info["dividend_yield"]*100, 2) if info["dividend_yield"] else "N/A",
        "Avg_Volume": f"{avg_volume/1e6:.2f}M",
        "Strength": strength,
        "Remarks": remark,
        "AI_Summary": ai_summary,
        "History": hist
    }

def analyze_stocks_parallel(symbols: List[str], max_workers: int = 10) -> Tuple[List[Dict], Dict]:
    """Parallel stock analysis with detailed progress tracking."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    original_count = len(symbols)
    if len(symbols) > 500:
        st.warning(f"⚠️ Analyzing first 500 stocks out of {len(symbols)} for optimal performance.")
        symbols = symbols[:500]
    
    stats = {
        "total": len(symbols),
        "completed": 0,
        "found": 0,
        "failed": 0,
        "not_extreme": 0
    }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(analyze_single_stock, sym): sym 
            for sym in symbols
        }
        
        for future in as_completed(future_to_symbol):
            stats["completed"] += 1
            
            try:
                result = future.result(timeout=15)
                if result:
                    results.append(result)
                    stats["found"] += 1
                else:
                    stats["not_extreme"] += 1
            except:
                stats["failed"] += 1
            
            if stats["completed"] % 5 == 0 or stats["completed"] == stats["total"]:
                progress = stats["completed"] / stats["total"]
                progress_bar.progress(progress)
                status_text.markdown(f"""
                    📊 **Progress**: {stats['completed']}/{stats['total']} 
                    | ✅ **Found**: {stats['found']} 
                    | ⏭️ **Not Extreme**: {stats['not_extreme']} 
                    | ❌ **Failed**: {stats['failed']}
                """)
    
    progress_bar.empty()
    status_text.empty()
    return results, stats

def create_interactive_chart(symbol: str, hist: pd.DataFrame, ref_value: float, 
                            stock_type: str, current_price: float):
    """Create professional interactive chart using Plotly with 3-year data."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist_3y = ticker.history(period="3y", auto_adjust=True, actions=False)
        
        if hist_3y is None or hist_3y.empty or len(hist_3y) < 30:
            hist_3y = hist
    except:
        hist_3y = hist
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - Price Trend (3 Years)', 'Candlestick Chart', 'Volume'),
        row_heights=[0.4, 0.4, 0.2]
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist_3y.index,
            y=hist_3y['Close'],
            name='Close Price',
            line=dict(color='#2E86DE', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    color = '#EE5A6F' if stock_type == "52W HIGH" else '#26DE81'
    fig.add_trace(
        go.Scatter(
            x=[hist_3y.index[0], hist_3y.index[-1]],
            y=[ref_value, ref_value],
            name=f'{stock_type}: ₹{ref_value:.2f}',
            line=dict(color=color, width=2, dash='dash'),
            hovertemplate=f'<b>{stock_type}</b>: ₹{ref_value:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[hist_3y.index[0], hist_3y.index[-1]],
            y=[current_price, current_price],
            name=f'Current: ₹{current_price:.2f}',
            line=dict(color='orange', width=1, dash='dot'),
            hovertemplate=f'<b>Current</b>: ₹{current_price:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Candlestick(
            x=hist_3y.index,
            open=hist_3y['Open'],
            high=hist_3y['High'],
            low=hist_3y['Low'],
            close=hist_3y['Close'],
            name='Candlestick',
            increasing_line_color='#26DE81',
            decreasing_line_color='#EE5A6F'
        ),
        row=2, col=1
    )
    
    colors = ['#EE5A6F' if hist_3y['Close'].iloc[i] < hist_3y['Close'].iloc[i-1] else '#26DE81' 
              for i in range(1, len(hist_3y))]
    colors.insert(0, 'gray')
    
    fig.add_trace(
        go.Bar(
            x=hist_3y.index,
            y=hist_3y['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6,
            hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#1f1f1f', size=12)
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f1f1f', size=12)
    )
    
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200,200,200,0.3)',
        title_font=dict(color='#1f1f1f'),
        tickfont=dict(color='#1f1f1f')
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200,200,200,0.3)',
        title_font=dict(color='#1f1f1f'),
        tickfont=dict(color='#1f1f1f')
    )
    
    fig.update_annotations(font=dict(color='#1f1f1f', size=14))
    
    st.plotly_chart(fig, use_container_width=True)

def render_header():
    """Render professional application header."""
    st.markdown("""
        <div class="main-header">
            <h1>📊 52-Week Stock Analyzer</h1>
            <p>Professional NSE Market Analysis Tool with AI-Powered Insights</p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render comprehensive sidebar with settings and info."""
    with st.sidebar:
        st.image("istockphoto-1153657433-612x612.jpg", 
                 use_container_width=True)
        
        st.markdown("### 📋 Analysis Settings")
        
        option = st.radio(
            "Select Stock Universe:",
            ["NIFTY 50", "NIFTY 100", "NIFTY 200", "All NSE Stocks"],
            index=1,
            help="Choose the set of stocks to analyze"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Advanced Settings")
        
        max_workers = st.slider(
            "Parallel Threads:",
            min_value=5,
            max_value=25,
            value=15,
            help="Higher values = faster analysis"
        )
        
        threshold = st.slider(
            "Distance Threshold (%):",
            min_value=2,
            max_value=10,
            value=5,
            help="Max distance from 52W high/low"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.analysis_results:
            df = pd.DataFrame(st.session_state.analysis_results)
            st.metric("Total Found", len(df))
            st.metric("52W Highs", len(df[df["Type"] == "52W HIGH"]))
            st.metric("52W Lows", len(df[df["Type"] == "52W LOW"]))
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This professional tool analyzes NSE stocks to identify those "
            "near their 52-week highs or lows, providing AI-powered investment insights."
        )
        
        st.markdown("### 🔒 Disclaimer")
        st.caption(
            "For educational purposes only. Not financial advice. "
            "Always consult a certified financial advisor."
        )
        
        return option, max_workers, threshold

def render_summary_dashboard(df: pd.DataFrame):
    """Render comprehensive summary dashboard."""
    st.markdown("### 📊 Analysis Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Stocks",
            len(df),
            delta=None,
            help="Total stocks found near 52W extremes"
        )
    
    with col2:
        lows = len(df[df["Type"] == "52W LOW"])
        st.metric(
            "52W Lows",
            lows,
            delta=f"{(lows/len(df)*100):.1f}%",
            help="Stocks near 52-week low"
        )
    
    with col3:
        highs = len(df[df["Type"] == "52W HIGH"])
        st.metric(
            "52W Highs",
            highs,
            delta=f"{(highs/len(df)*100):.1f}%",
            help="Stocks near 52-week high"
        )
    
    with col4:
        strong = len(df[df["Strength"] == "Strong"])
        st.metric(
            "Strong Stocks",
            strong,
            delta=f"{(strong/len(df)*100):.1f}%",
            help="Fundamentally strong stocks"
        )
    
    with col5:
        avg_return = df["1Y_Return_%"].mean()
        st.metric(
            "Avg 1Y Return",
            f"{avg_return:.1f}%",
            delta=None,
            help="Average 1-year return"
        )

def render_sector_analysis(df: pd.DataFrame):
    """Render sector-wise analysis."""
    st.markdown("### 🏭 Sector Analysis")
    
    sector_stats = df.groupby('Sector').agg({
        'Symbol': 'count',
        '1Y_Return_%': 'mean'
    }).round(2)
    sector_stats.columns = ['Stock Count', 'Avg Return %']
    sector_stats = sector_stats.sort_values('Stock Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            sector_stats.head(10),
            use_container_width=True,
            height=300
        )
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sector_stats.head(10)['Stock Count'].plot(
            kind='barh',
            ax=ax,
            color='#667eea'
        )
        ax.set_xlabel('Number of Stocks')
        ax.set_title('Top 10 Sectors by Stock Count')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def render_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Render interactive results table with advanced filters."""
    st.markdown("### 📈 Stock Analysis Results")
    
    with st.expander("🔍 Advanced Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            type_filter = st.multiselect(
                "Type:",
                options=["52W LOW", "52W HIGH"],
                default=["52W LOW", "52W HIGH"],
                key="type_filter"
            )
        
        with col2:
            strength_filter = st.multiselect(
                "Strength:",
                options=["Strong", "Moderate", "Weak"],
                default=["Strong", "Moderate", "Weak"],
                key="strength_filter"
            )
        
        with col3:
            sectors = sorted(df["Sector"].unique().tolist())
            sector_filter = st.multiselect(
                "Sector:",
                options=sectors,
                default=sectors,
                key="sector_filter"
            )
        
        with col4:
            return_range = st.slider(
                "1Y Return % Range:",
                min_value=float(df["1Y_Return_%"].min()),
                max_value=float(df["1Y_Return_%"].max()),
                value=(float(df["1Y_Return_%"].min()), float(df["1Y_Return_%"].max())),
                key="return_filter"
            )
    
    filtered_df = df.copy()
    if type_filter:
        filtered_df = filtered_df[filtered_df["Type"].isin(type_filter)]
    if strength_filter:
        filtered_df = filtered_df[filtered_df["Strength"].isin(strength_filter)]
    if sector_filter:
        filtered_df = filtered_df[filtered_df["Sector"].isin(sector_filter)]
    filtered_df = filtered_df[
        (filtered_df["1Y_Return_%"] >= return_range[0]) & 
        (filtered_df["1Y_Return_%"] <= return_range[1])
    ]
    
    if len(filtered_df) < len(df):
        st.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** stocks after filtering")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["1Y_Return_%", "Current_Price", "Distance_%", "Market_Cap"],
            key="sort_filter"
        )
    with col2:
        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True, key="sort_order")
    
    filtered_df = filtered_df.sort_values(
        by=sort_by,
        ascending=(sort_order == "Ascending")
    )
    
    display_df = filtered_df.drop(columns=["History"]).copy()
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Current_Price": st.column_config.NumberColumn("Current (₹)", format="₹%.2f"),
            "52W_Value": st.column_config.NumberColumn("52W Value (₹)", format="₹%.2f"),
            "Distance_%": st.column_config.NumberColumn("Distance %", format="%.2f%%"),
            "1Y_Return_%": st.column_config.NumberColumn("1Y Return", format="%.2f%%"),
            "P/E": st.column_config.TextColumn("P/E", width="small"),
            "Beta": st.column_config.TextColumn("Beta", width="small"),
            "AI_Summary": st.column_config.TextColumn("AI Insights", width="large"),
        }
    )
    
    return filtered_df

def render_stock_detail(filtered_df: pd.DataFrame):
    """Render detailed stock analysis with interactive chart."""
    st.markdown("### 🔍 Detailed Stock Analysis")
    
    if filtered_df.empty:
        st.info("ℹ️ No stocks match the selected filters.")
        return
    
    selected = st.selectbox(
        "Search and select a stock:",
        options=filtered_df["Symbol"].tolist(),
        key="stock_selector",
        help="Type to search or select from dropdown"
    )
    
    if selected:
        row = filtered_df[filtered_df["Symbol"] == selected].iloc[0]
        
        st.markdown(f"#### {selected} - {row['Sector']}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{row['Current_Price']}")
        with col2:
            st.metric("52W Value", f"₹{row['52W_Value']}")
        with col3:
            return_value = row['1Y_Return_%']
            if return_value >= 0:
                st.metric("1Y Return", f"{return_value}%", 
                         delta=f"{return_value:.2f}%", 
                         delta_color="normal")
            else:
                st.metric("1Y Return", f"{return_value}%", 
                         delta=f"{return_value:.2f}%", 
                         delta_color="normal")
        with col4:
            st.metric("P/E Ratio", row['P/E'])
        with col5:
            st.metric("Market Cap", row['Market_Cap'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Beta", row['Beta'])
        with col2:
            st.metric("Div Yield", f"{row['Div_Yield_%']}%")
        with col3:
            st.metric("Avg Volume", row['Avg_Volume'])
        with col4:
            st.metric("Strength", row['Strength'])
        
        st.info(f"**🤖 AI Analysis:** {row['AI_Summary']}")
        st.warning(f"**📝 Remarks:** {row['Remarks']}")
        
        create_interactive_chart(
            selected,
            row["History"],
            row["52W_Value"],
            row["Type"],
            row["Current_Price"]
        )
        
        with st.expander("📊 Technical Summary", expanded=False):
            hist = row["History"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price Statistics:**")
                st.write(f"- **Highest**: ₹{hist['Close'].max():.2f}")
                st.write(f"- **Lowest**: ₹{hist['Close'].min():.2f}")
                st.write(f"- **Average**: ₹{hist['Close'].mean():.2f}")
                st.write(f"- **Volatility (Std)**: ₹{hist['Close'].std():.2f}")
            
            with col2:
                st.markdown("**Volume Statistics:**")
                st.write(f"- **Avg Volume**: {hist['Volume'].mean()/1e6:.2f}M")
                st.write(f"- **Max Volume**: {hist['Volume'].max()/1e6:.2f}M")
                st.write(f"- **Min Volume**: {hist['Volume'].min()/1e6:.2f}M")
                
                recent_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / 
                               hist['Close'].iloc[-30] * 100)
                st.write(f"- **30D Return**: {recent_return:.2f}%")

def render_export_options(filtered_df: pd.DataFrame):
    """Render professional export options."""
    st.markdown("### 📥 Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_df.drop(columns=["History"]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download filtered results as CSV"
        )
    
    with col2:
        excel_data = filtered_df.drop(columns=["History"]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True,
            help="Download filtered results as Excel"
        )
    
    with col3:
        report = generate_text_report(filtered_df)
        st.download_button(
            label="📋 Download Report",
            data=report,
            file_name=f"stock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download summary report as text"
        )

def generate_text_report(df: pd.DataFrame) -> str:
    """Generate a comprehensive text report."""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          52-WEEK STOCK ANALYSIS REPORT                       ║
║          Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                     ║
╚══════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════
Total Stocks Analyzed: {len(df)}
52-Week Lows: {len(df[df['Type'] == '52W LOW'])}
52-Week Highs: {len(df[df['Type'] == '52W HIGH'])}
Strong Stocks: {len(df[df['Strength'] == 'Strong'])}
Average 1Y Return: {df['1Y_Return_%'].mean():.2f}%

TOP 10 PERFORMERS (by 1Y Return)
═══════════════════════════════════════════════════════════════
"""
    
    top_performers = df.nlargest(10, '1Y_Return_%')[['Symbol', 'Type', '1Y_Return_%', 'Strength']]
    for idx, row in top_performers.iterrows():
        report += f"{row['Symbol']:15} | {row['Type']:10} | {row['1Y_Return_%']:8.2f}% | {row['Strength']}\n"
    
    report += f"""
SECTOR DISTRIBUTION
═══════════════════════════════════════════════════════════════
"""
    sector_counts = df['Sector'].value_counts().head(10)
    for sector, count in sector_counts.items():
        report += f"{sector:30} : {count:3} stocks\n"
    
    report += f"""
INVESTMENT OPPORTUNITIES (Strong + 52W Low)
═══════════════════════════════════════════════════════════════
"""
    opportunities = df[(df['Type'] == '52W LOW') & (df['Strength'] == 'Strong')]
    if len(opportunities) > 0:
        for idx, row in opportunities.iterrows():
            report += f"""
Symbol: {row['Symbol']}
Sector: {row['Sector']}
Current Price: ₹{row['Current_Price']}
1Y Return: {row['1Y_Return_%']}%
AI Summary: {row['AI_Summary']}
───────────────────────────────────────────────────────────────
"""
    else:
        report += "No strong opportunities found at 52W low.\n"
    
    report += f"""
DISCLAIMER
═══════════════════════════════════════════════════════════════
This report is for educational and informational purposes only.
It does not constitute financial advice. Always consult with a
certified financial advisor before making investment decisions.
Past performance does not guarantee future results.

═══════════════════════════════════════════════════════════════
                    End of Report
═══════════════════════════════════════════════════════════════
"""
    
    return report

def render_watchlist_feature(filtered_df: pd.DataFrame):
    """Render watchlist management feature."""
    st.markdown("### ⭐ Watchlist Manager")
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stocks = st.multiselect(
            "Add stocks to watchlist:",
            options=filtered_df["Symbol"].tolist(),
            default=st.session_state.watchlist,
            key="watchlist_selector"
        )
    
    with col2:
        if st.button("💾 Save Watchlist", use_container_width=True):
            st.session_state.watchlist = selected_stocks
            st.success("✅ Watchlist saved!")
    
    if selected_stocks:
        watchlist_df = filtered_df[filtered_df["Symbol"].isin(selected_stocks)]
        st.dataframe(
            watchlist_df.drop(columns=["History"])[
                ["Symbol", "Type", "Current_Price", "1Y_Return_%", "Strength", "AI_Summary"]
            ],
            use_container_width=True,
            height=300
        )
        
        watchlist_csv = watchlist_df.drop(columns=["History"]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Watchlist",
            data=watchlist_csv,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def render_comparison_tool(filtered_df: pd.DataFrame):
    """Render stock comparison tool."""
    st.markdown("### 🔄 Compare Stocks")
    
    if len(filtered_df) < 2:
        st.info("Need at least 2 stocks to compare.")
        return
    
    compare_stocks = st.multiselect(
        "Select stocks to compare (max 4):",
        options=filtered_df["Symbol"].tolist(),
        max_selections=4,
        key="compare_selector"
    )
    
    if len(compare_stocks) >= 2:
        compare_df = filtered_df[filtered_df["Symbol"].isin(compare_stocks)]
        
        comparison_cols = ["Symbol", "Type", "Current_Price", "52W_Value", "1Y_Return_%", 
                          "P/E", "Market_Cap", "Beta", "Strength"]
        st.dataframe(
            compare_df[comparison_cols].set_index("Symbol"),
            use_container_width=True
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        compare_df.plot(x="Symbol", y="1Y_Return_%", kind="bar", ax=axes[0], 
                       color='#667eea', legend=False)
        axes[0].set_title("1-Year Return Comparison")
        axes[0].set_ylabel("Return %")
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        compare_df.plot(x="Symbol", y="Distance_%", kind="bar", ax=axes[1], 
                       color='#764ba2', legend=False)
        axes[1].set_title("Distance from 52W Value")
        axes[1].set_ylabel("Distance %")
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def render_help_section():
    """Render comprehensive help and documentation."""
    st.markdown("### 📚 User Guide")
    
    st.markdown("""
    #### 🎯 How to Use This Tool
    
    1. **Select Stock Universe**: Choose from NIFTY 50, 100, 200, or All NSE stocks in the sidebar
    2. **Adjust Settings**: Configure parallel threads and distance threshold
    3. **Start Analysis**: Click the 'Start Analysis' button and wait for results
    4. **Filter Results**: Use advanced filters to narrow down stocks
    5. **Analyze Stocks**: Click on any stock to view detailed charts and insights
    6. **Export Data**: Download results as CSV, Excel, or text report
    
    #### 📊 Understanding the Metrics
    
    - **52W LOW**: Stock is trading within 5% of its 52-week low
    - **52W HIGH**: Stock is trading within 5% of its 52-week high
    - **Strength**: Based on P/E ratio, market cap, and beta
    - **1Y Return**: Percentage return over the past year
    - **Distance %**: How far the stock is from the 52W reference value
    - **Beta**: Measure of stock volatility (>1 = more volatile)
    - **P/E Ratio**: Price-to-Earnings ratio
    
    #### 🤖 AI Summary Interpretation
    
    - **🎯 Strong Buy Signal**: High-quality stock at attractive valuation
    - **✅ Accumulation Zone**: Good for long-term investors
    - **⏳ Wait & Watch**: Needs confirmation before entry
    - **⚠️ Caution**: Risky, analyze fundamentals deeply
    - **🔴 Highly Overextended**: High profit-booking risk
    
    #### ⚙️ Advanced Features
    
    - **Multi-filter System**: Combine type, strength, sector, and return filters
    - **Stock Comparison**: Compare up to 4 stocks side-by-side
    - **Watchlist Manager**: Save and track your favorite stocks
    - **Interactive Charts**: 3-year price trend, candlestick, and volume charts
    - **Export Options**: Multiple format support for further analysis
    
    #### ⚠️ Important Notes
    
    - Analysis uses real-time data from Yahoo Finance
    - Data may have slight delays or inaccuracies
    - Past performance does not guarantee future results
    - Always conduct your own research before investing
    - Consult a certified financial advisor for personalized advice
    
    #### 🔒 Data Privacy
    
    - No user data is collected or stored
    - All analysis happens locally in your browser session
    - Data is cleared when you close the browser
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **For Value Investors:**
        - Focus on 52W LOW + Strong stocks
        - Look for low P/E ratios (<15)
        - Check dividend yield
        - Analyze sector trends
        """)
    
    with col2:
        st.info("""
        **For Momentum Traders:**
        - Focus on 52W HIGH + Strong stocks
        - Look for high 1Y returns
        - Check beta for volatility
        - Use strict stop-loss
        """)

def main():
    """Main application entry point with professional structure."""
    configure_page()
    initialize_session_state()
    render_header()
    
    option, max_workers, threshold = render_sidebar()
    
    symbols = load_stock_symbols(option)
    
    if not symbols:
        st.error("❌ Failed to load stock list. Please refresh and try again.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analysis", 
        "📊 Dashboard", 
        "⭐ Watchlist", 
        "📚 Help"
    ])
    
    with tab1:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.info(f"📈 Ready to analyze **{len(symbols)}** stocks from {option}")
        
        with col2:
            if st.session_state.analysis_results is not None:
                if st.button("🗑️ Clear", use_container_width=True, help="Clear current results"):
                    st.session_state.analysis_results = None
                    st.session_state.analysis_stats = None
                    st.session_state.analysis_time = None
                    st.session_state.analysis_complete = False
                    st.rerun()
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True, 
                    disabled=st.session_state.analysis_complete):
            start_time = time.time()
            
            with st.spinner("🔄 Analyzing stocks... Please wait..."):
                results, stats = analyze_stocks_parallel(symbols, max_workers=max_workers)
            
            elapsed_time = time.time() - start_time
            
            st.session_state.analysis_results = results
            st.session_state.analysis_stats = stats
            st.session_state.analysis_time = elapsed_time
            st.session_state.analysis_complete = True
            st.session_state.last_update = datetime.now()
            
            st.rerun()
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            stats = st.session_state.analysis_stats
            elapsed_time = st.session_state.analysis_time
            
            st.success(f"✅ Analysis completed in {elapsed_time:.1f} seconds!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyzed", stats['total'])
            with col2:
                st.metric("Found", stats['found'], delta=f"{stats['found']/stats['total']*100:.1f}%")
            with col3:
                st.metric("Not Extreme", stats['not_extreme'])
            with col4:
                st.metric("Failed", stats['failed'])
            
            if not results:
                st.warning(
                    "⚠️ **No stocks found near 52-week extremes.**\n\n"
                    "Try adjusting the threshold in sidebar or selecting a different stock list."
                )
            else:
                df = pd.DataFrame(results)
                
                st.markdown("---")
                
                filtered_df = render_results_table(df)
                
                st.markdown("---")
                
                render_stock_detail(filtered_df)
                
                st.markdown("---")
                
                render_comparison_tool(filtered_df)
                
                st.markdown("---")
                
                render_export_options(filtered_df)
    
    with tab2:
        if st.session_state.analysis_results is not None and len(st.session_state.analysis_results) > 0:
            df = pd.DataFrame(st.session_state.analysis_results)
            
            render_summary_dashboard(df)
            
            st.markdown("---")
            
            render_sector_analysis(df)
            
            st.markdown("---")
            
            st.markdown("### 💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Best Value Picks")
                value_picks = df[
                    (df['Type'] == '52W LOW') & 
                    (df['Strength'] == 'Strong')
                ].nsmallest(5, '1Y_Return_%')
                
                if len(value_picks) > 0:
                    st.dataframe(
                        value_picks[['Symbol', 'Sector', '1Y_Return_%', 'P/E']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No strong value picks found.")
            
            with col2:
                st.markdown("#### 🚀 Momentum Leaders")
                momentum = df[
                    (df['Type'] == '52W HIGH') & 
                    (df['Strength'] == 'Strong')
                ].nlargest(5, '1Y_Return_%')
                
                if len(momentum) > 0:
                    st.dataframe(
                        momentum[['Symbol', 'Sector', '1Y_Return_%', 'P/E']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No strong momentum stocks found.")
        else:
            st.info("👈 Run an analysis first to view the dashboard.")
    
    with tab3:
        if st.session_state.analysis_results is not None and len(st.session_state.analysis_results) > 0:
            df = pd.DataFrame(st.session_state.analysis_results)
            render_watchlist_feature(df)
        else:
            st.info("👈 Run an analysis first to create a watchlist.")
    
    with tab4:
        render_help_section()
    
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'Not yet run'}<br>"
            "Made with ❤️ using Streamlit | © 2024"
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
