import streamlit as st

# Page setup
st.set_page_config(page_title="Stock Forecast Hub", page_icon="📈", layout="centered")

# Title and description
st.title("📈 Stock Forecast Hub")
st.markdown("Your one-stop platform for AI-driven market insights and trading tools.")

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

# Create three columns for feature sections
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🤖 AI-Powered Forecasts")
    st.markdown("Get intelligent predictions and trend analysis for your favorite stocks.")
    if st.button("Open Advanced Analytics", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Advanced_Analytics.py")

with col2:
    st.markdown("### 📊 Pro Trading Dashboard")
    st.markdown("Access in-depth technical indicators, tools, and portfolio insights.")
    if st.button("Open Professional Tools", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Professional_Tools.py")

with col3:
    st.markdown("### ⚡ Live Market Signals")
    st.markdown("Stay ahead with real-time buy/sell alerts and market momentum updates.")
    if st.button("Open Live Signals", use_container_width=True, type="primary"):
        st.switch_page("pages/3_Live_Signals.py")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("*Choose any section above to begin your stock forecasting journey!*")
