import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Experiment Analysis", page_icon="🧪", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

st.title("🧪 Experiment Analysis")
st.markdown("Detailed breakdown of how the Treatment group performed against the Control group.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversion Rate Comparison")
    fig_cr, ax_cr = plt.subplots(figsize=(6, 4))
    sns.barplot(x='group', y='converted', data=df, errorbar=None, ax=ax_cr, palette="muted")
    ax_cr.set_ylabel("Conversion Rate")
    ax_cr.set_xlabel("Group")
    
    # Annotate bars
    for p in ax_cr.patches:
        ax_cr.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    st.pyplot(fig_cr)

with col2:
    st.subheader("Revenue Comparison")
    if 'purchase_amount' in df.columns:
        fig_rev, ax_rev = plt.subplots(figsize=(6, 4))
        sns.barplot(x='group', y='purchase_amount', data=df, errorbar=None, ax=ax_rev, palette="muted")
        ax_rev.set_ylabel("Average Purchase Amount ($)")
        ax_rev.set_xlabel("Group")
        st.pyplot(fig_rev)
    else:
        st.warning("Revenue data not available in this dataset.")

st.divider()

st.subheader("User Engagement (Session Duration)")
if 'session_duration' in df.columns:
    fig_box, ax_box = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='session_duration', y='group', data=df, ax=ax_box, palette="pastel")
    ax_box.set_xlabel("Session Duration (Minutes)")
    ax_box.set_ylabel("Group")
    st.pyplot(fig_box)
else:
    st.warning("Session duration data not available in this dataset.")