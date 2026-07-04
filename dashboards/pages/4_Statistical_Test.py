import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

st.set_page_config(page_title="Statistical Test", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

st.title("📊 Statistical Significance Test")
st.markdown("Using a Two-Proportion Z-Test to determine if the observed lift in conversion rate is statistically significant.")

control_conv = df[df['group'] == 'control']['converted']
treatment_conv = df[df['group'] == 'treatment']['converted']

n_control = control_conv.count()
n_treatment = treatment_conv.count()
successes = np.array([treatment_conv.sum(), control_conv.sum()])
nobs = np.array([n_treatment, n_control])

# Perform Z-Test
z_stat, p_value = proportions_ztest(count=successes, nobs=nobs, alternative='larger')

alpha = 0.05
is_significant = p_value < alpha

st.subheader("Hypotheses")
st.markdown("""
* **Null Hypothesis ($H_0$):** $p_{treatment} - p_{control} \le 0$ (The new page performs the same or worse)
* **Alternative Hypothesis ($H_1$):** $p_{treatment} - p_{control} > 0$ (The new page strictly performs better)
""")

st.subheader("Test Results")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Z-Score", value=f"{z_stat:.4f}")
with col2:
    st.metric(label="P-Value", value=f"{p_value:.4f}")
with col3:
    st.metric(label="Alpha Level", value=f"{alpha}")

st.divider()

if is_significant:
    st.success(f"✅ **Statistically Significant!** The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis.")
else:
    st.error(f"❌ **Not Statistically Significant.** The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis.")