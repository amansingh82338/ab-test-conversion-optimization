# A/B Testing Analysis: Landing Page Conversion Optimization

## Overview

This project analyzes the results of an A/B test designed to evaluate whether a redesigned landing page (the **Treatment** experience) improves user conversion relative to the existing landing page (the **Control** experience). The project follows a complete data analytics workflow — from raw data cleaning through statistical hypothesis testing, customer segmentation, and executive-level business recommendations — and includes an interactive Streamlit dashboard for exploring the results.

## Business Problem

Conversion rate is one of the clearest signals of how effectively a digital product turns visitors into engaged or paying customers. The business suspected that friction in the existing landing page design was suppressing conversion, revenue, and overall user engagement. To test this, a randomly assigned subset of users (Treatment) were shown a redesigned landing page, while the remaining users (Control) continued to see the original page. This project analyzes the resulting experiment data to determine whether the redesign should be rolled out at scale.

## Objectives

- Determine whether the new landing page produces a statistically significant improvement in conversion rate.
- Quantify the size and reliability of the observed effect using formal statistical tests and confidence intervals.
- Assess the impact of the new page on secondary metrics, including purchase amount and session duration.
- Identify customer segments (device, gender, age, location) that respond most and least favorably to the redesign.
- Translate statistical findings into clear, actionable business recommendations.

## Dataset

The dataset contains user-level records from the experiment, including:

- **group** — experiment assignment (`control` or `treatment`)
- **converted** — binary conversion flag
- **purchase_amount** — value of purchase made, if any
- **session_duration** — time spent in the session
- **device_type, gender, age / age_group, location, traffic_source** — demographic and behavioral attributes

*(Exact row counts and summary statistics are available in `data/processed/cleaned_data.csv` and `data/processed/experiment_summary.csv`.)*

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Jupyter Notebook
- Streamlit

## Project Workflow

1. **Data Cleaning** — handling missing values, removing duplicates, correcting data types, and preparing derived features.
2. **Exploratory Data Analysis** — examining conversion, revenue, and user behavior patterns before formal testing.
3. **Statistical Testing** — hypothesis testing using a Two-Proportion Z-Test, Chi-Square Test, confidence intervals, and effect size (Cohen's h).
4. **Customer Segmentation** — breaking down performance by device, gender, age, and location.
5. **Business Recommendations** — translating findings into an executive-ready report.
6. **Dashboard Development** — building an interactive Streamlit application for ongoing exploration of results.

## Key Analyses

- **Conversion Rate Analysis** — comparing Control vs. Treatment conversion rates at the aggregate level.
- **Revenue Analysis** — comparing average purchase amount between groups.
- **User Behaviour** — comparing session duration and engagement patterns across groups.
- **Statistical Tests** — Two-Proportion Z-Test, Chi-Square Test of Independence, confidence interval estimation, and effect size calculation.
- **Customer Segmentation** — conversion and revenue performance broken down by device type, gender, age group, and location, with Control vs. Treatment comparisons within each segment.

## Key Findings

- Overall conversion rate: **[Insert Overall Conversion Rate]**
- Control group conversion rate: **[Insert Control Conversion Rate]**
- Treatment group conversion rate: **[Insert Treatment Conversion Rate]**
- Observed conversion lift: **[Insert Conversion Lift]%**
- Statistical significance (p-value): **[Insert P-Value]**
- 95% confidence interval for the difference in conversion rates: **[Insert CI Range]**
- Effect size (Cohen's h): **[Insert Effect Size Value and Interpretation]**
- Best performing segment: **[Insert Segment]**
- Weakest performing segment: **[Insert Segment]**

## Business Recommendations

- Roll out the new landing page in a phased manner, prioritizing the segments that responded most positively.
- Monitor core KPIs (conversion rate, purchase amount, session duration) for at least two weeks post-launch to rule out a novelty effect.
- Run follow-up experiments on specific page elements (e.g., call-to-action design) and consider personalized variants by traffic source or device.
- Focus marketing spend on the highest-lift segments while investigating friction points in underperforming segments.

## Dashboard Features

The accompanying Streamlit dashboard (`dashboards/streamlit_app.py`) provides interactive, multi-page access to the analysis:

- **Overview** — high-level KPIs and experiment summary.
- **Experiment Analysis** — detailed conversion, revenue, and engagement comparisons between groups.
- **Segment Insights** — interactive breakdowns by device, gender, age, and location.
- **Statistical Test** — interactive hypothesis testing results, confidence intervals, and effect size.
- **Business Recommendations** — summarized, stakeholder-ready recommendations and next steps.

## Project Structure

```
ab_testing_analysis/
│
├── data/
│   ├── raw/
│   │   └── ab_testing.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       ├── experiment_summary.csv
│       └── segment_summary.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_statistical_testing.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_business_recommendations.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── statistics.py
│   ├── visualization.py
│   └── utils.py
│
├── dashboards/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Experiment_Analysis.py
│   │   ├── 3_Segment_Insights.py
│   │   ├── 4_Statistical_Test.py
│   │   └── 5_Business_Recommendations.py
│   │
│   └── assets/
│       └── style.css
│
├── reports/
│   ├── figures/
│   ├── ab_testing_report.pdf
│   └── project_summary.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Hypothesis Testing
- A/B Testing
- Statistical Analysis
- Customer Segmentation
- Business Analytics
- Streamlit
- Git
- GitHub

## Future Improvements

- Incorporate additional behavioral and demographic features for richer segmentation.
- Extend post-launch monitoring to better separate genuine effects from novelty effects.
- Explore multi-armed bandit or sequential testing approaches for faster future experiments.
- Add automated data refresh and alerting to the Streamlit dashboard for ongoing monitoring.

## Conclusion

This project demonstrates a complete, end-to-end A/B testing workflow — from raw data to statistically grounded, business-ready recommendations. By combining rigorous hypothesis testing with granular segmentation analysis, the project provides a template for data-driven decision-making that balances statistical confidence with practical business impact.
