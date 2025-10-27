import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")

# Title and description
st.title("🏠 Welcome to the Application Hub")
st.markdown("Select an application to get started:")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Create three columns for buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Advanced Stock Forecasting")
    st.markdown("Advanced AI-powered stock prediction and analysis")
    if st.button("Advanced Analytics", use_container_width=True, type="primary"):
        st.switch_page("pages/app.py")

with col2:
    st.markdown("### 💼 Professional Stock Forecasting")
    st.markdown("Professional-grade forecasting tools and insights")
    if st.button("Professional Tools", use_container_width=True, type="primary"):
        st.switch_page("pages/trail.py")

with col3:
    st.markdown("### 📈 Real-Time Recommendations")
    st.markdown("Live stock recommendations and market signals")
    if st.button("Live Signals", use_container_width=True, type="primary"):
        st.switch_page("pages/Real-Time-stock.py")

# Add footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("*Select any application above to begin*")
