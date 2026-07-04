import streamlit as st
import pandas as pd

st.set_page_config(page_title="Overview", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

st.title("🏠 A/B Testing Experiment Overview")
st.markdown("""
Welcome to the A/B Testing Analysis Dashboard. This application evaluates the performance 
of our new landing page (Treatment) against the existing design (Control).
""")

st.header("High-Level KPIs")

# Calculate metrics
total_users = len(df)
total_conversions = df['converted'].sum()
overall_cr = df['converted'].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Users", value=f"{total_users:,}")
    
with col2:
    st.metric(label="Total Conversions", value=f"{total_conversions:,}")
    
with col3:
    st.metric(label="Overall Conversion Rate", value=f"{overall_cr:.2%}")

st.divider()

st.header("Group Breakdown")
group_summary = df.groupby('group').agg(
    Users=('user_id', 'count'),
    Conversions=('converted', 'sum'),
    Conversion_Rate=('converted', 'mean')
).reset_index()

group_summary['Conversion_Rate'] = group_summary['Conversion_Rate'].apply(lambda x: f"{x:.2%}")

st.dataframe(group_summary, use_container_width=True)

st.info("👈 Navigate through the sidebar to view deep-dive analytics, statistical test results, and final business recommendations.")