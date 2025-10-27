import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 100px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>📈 Stock Forecasting Dashboard</h1>
        <p>Select a version to access the forecasting application</p>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
### Welcome to Stock Forecasting Platform
Choose from three versions of our stock forecasting application. Each version offers unique features 
and improvements to help you analyze and predict stock market trends.
""")

st.markdown("---")

# Create three columns for the buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Advanced Stock Forecasting")
    st.markdown("Advanced Stock Forecasting Platform")
    if st.button("🚀 Launch V1", key="v1", help="Open Version 1"):
        st.markdown("""
            <meta http-equiv="refresh" content="0;url=https://stock-forecasting-imran.streamlit.app/">
        """, unsafe_allow_html=True)
        st.info("Opening Advanced Stock Forecasting Platform...")
        st.link_button("Click here if not redirected", "https://stock-forecasting-imran.streamlit.app/")

with col2:
    st.markdown("### 📊 Professional Forecasting")
    st.markdown("Professional Stock Forecasting Platform")
    if st.button("🚀 Launch V2", key="v2", help="Open Version 2"):
        st.markdown("""
            <meta http-equiv="refresh" content="0;url=https://stock-forecasting-imran-v2.streamlit.app/">
        """, unsafe_allow_html=True)
        st.info("Opening Professional Stock Forecasting Platform...")
        st.link_button("Click here if not redirected", "https://stock-forecasting-imran-v2.streamlit.app/")

with col3:
    st.markdown("### 📊 Real-Time Recommendations")
    st.markdown("Real-Time Stock Recommendation System")
    if st.button("🚀 Launch V3", key="v3", help="Open Version 3"):
        st.markdown("""
            <meta http-equiv="refresh" content="0;url=https://stock-forecasting-imran-v3.streamlit.app/">
        """, unsafe_allow_html=True)
        st.info("Opening Real-Time Stock Recommendation System...")
        st.link_button("Click here if not redirected", "https://stock-forecasting-imran-v3.streamlit.app/")

st.markdown("---")

# Features section
st.markdown("### 🎯 Key Features")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    #### 📈 Accurate Predictions
    - Machine learning models
    - Historical data analysis
    - Real-time updates
    """)

with col_b:
    st.markdown("""
    #### 📊 Interactive Charts
    - Customizable visualizations
    - Multiple timeframes
    - Technical indicators
    """)

with col_c:
    st.markdown("""
    #### 💡 User-Friendly
    - Intuitive interface
    - Easy navigation
    - Responsive design
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Made with ❤️ by Imran | © 2024 Stock Forecasting Platform</p>
    </div>
""", unsafe_allow_html=True)
