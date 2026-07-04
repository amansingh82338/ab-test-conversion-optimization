import streamlit as st

st.set_page_config(page_title="Business Recommendations", page_icon="💼", layout="wide")

st.title("💼 Business Recommendations")
st.markdown("Based on the data analysis and statistical testing, here is the executive summary and recommended next steps.")

st.header("1. Final Verdict: **Launch** 🚀")
st.markdown("""
The Treatment variant (new landing page) proved to be highly successful. It generated a statistically significant lift in overall conversion rates without negatively impacting secondary metrics like session duration or average order value.
""")

st.header("2. Strategic Rollout Plan")
st.markdown("""
* **Phase 1 (Immediate):** Increase traffic allocation to the new landing page from 50% to 80%.
* **Phase 2 (7 Days Later):** Pending no technical regressions or server load issues, scale to 100% of all user traffic.
* **Deprecation:** Sunset the codebase for the legacy landing page to reduce technical debt.
""")

st.header("3. Risk Mitigation & Monitoring")
st.markdown("""
* **Novelty Effect:** The lift might be partially driven by returning users experiencing something "new." We must monitor the conversion rate over the next 30 days to establish the true new baseline.
* **Mobile Performance:** While overall metrics were positive, segment analysis showed slower growth on Mobile devices compared to Desktop. 
""")

st.header("4. Next Recommended Experiments")
st.info("""
1. **Mobile Optimization:** Run an A/B test specifically targeting the mobile UI to close the performance gap.
2. **Call to Action (CTA) Tuning:** Test different copy text on the primary checkout button of the new winning page.
""")