"""
A/B Testing Analysis Dashboard - Main Entry Point
File: dashboards/streamlit_app.py

This script serves as the main entry point for the Streamlit multi-page application.
It configures the global page settings, injects custom CSS for styling, and provides
a welcoming landing page that directs users to the detailed analytical pages in the sidebar.

To run the application, execute the following command from the root directory:
$ streamlit run dashboards/streamlit_app.py
"""

import streamlit as st
import os

def load_css(file_path: str) -> None:
    """
    Reads a CSS file and injects it into the Streamlit application.
    
    Args:
        file_path (str): The relative path to the CSS file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Warning: Custom CSS file not found at '{file_path}'. Proceeding with default styles.")

def main():
    # 1. Configure the main page settings
    # This must be the first Streamlit command executed
    st.set_page_config(
        page_title="A/B Testing Analysis Dashboard",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Load custom styling
    # Adjust the path assuming the app is run from the project root
    css_path = os.path.join("dashboards", "assets", "style.css")
    load_css(css_path)

    # 3. Main Landing Page Content
    st.title("🧪 A/B Testing Analysis & Executive Dashboard")
    
    st.markdown("""
    ### Welcome to the A/B Testing Results Application
    
    This interactive dashboard presents the end-to-end findings of our recent landing page A/B test. 
    It is designed to give stakeholders, product managers, and marketing teams transparent, 
    data-driven insights into how the **Treatment** (new landing page) performed against the 
    **Control** (existing landing page).
    
    ---
    
    ### 📂 How to Navigate this Dashboard
    
    Please use the **sidebar on the left** to explore the different phases of the analysis:
    
    *   **1. Overview:** High-level key performance indicators (KPIs) and traffic breakdown.
    *   **2. Experiment Analysis:** Direct comparisons of Conversion Rates, Revenue, and Engagement metrics.
    *   **3. Segment Insights:** Deep dives into demographic data to see *who* responded best to the changes and check for Simpson's Paradox.
    *   **4. Statistical Test:** The mathematical proof (Two-Proportion Z-Test) ensuring our results are not just random chance.
    *   **5. Business Recommendations:** The final executive verdict, estimated impact, and strategic next steps for rollout.
    
    ---
    """)
    
    st.info("👈 **Please select '1_Overview' from the sidebar to begin reviewing the data.**")

if __name__ == "__main__":
    main()