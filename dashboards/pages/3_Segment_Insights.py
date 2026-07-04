import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Segment Insights", page_icon="👥", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

st.title("👥 Segment Insights")
st.markdown("Investigate how the new landing page performed across different user demographics to ensure consistent performance and detect potential Simpson's Paradox.")

segment_option = st.selectbox(
    "Select a dimension to analyze:",
    options=['device_type', 'gender', 'location']
)

if segment_option in df.columns:
    st.subheader(f"Conversion Rate by {segment_option.replace('_', ' ').title()}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=segment_option, y='converted', hue='group', data=df, errorbar=None, ax=ax, palette="muted")
    ax.set_ylabel("Conversion Rate")
    ax.set_xlabel(segment_option.replace('_', ' ').title())
    st.pyplot(fig)
    
    st.subheader("Segment Data Table")
    segment_table = df.groupby([segment_option, 'group'])['converted'].mean().unstack()
    segment_table['Absolute Difference'] = segment_table['treatment'] - segment_table['control']
    segment_table['Relative Lift'] = segment_table['Absolute Difference'] / segment_table['control']
    
    # Format table for display
    display_table = segment_table.copy()
    for col in display_table.columns:
        display_table[col] = display_table[col].apply(lambda x: f"{x:.2%}")
        
    st.dataframe(display_table, use_container_width=True)

else:
    st.error(f"The column '{segment_option}' was not found in the dataset.")