import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
try:
    df_cleaned = pd.read_csv('cleaned_aapl_forecast_data.csv')
    # Convert date columns back to datetime if necessary for plotting or filtering
    date_cols = ['ACTDATS', 'ANNDATS', 'REVDATS', 'FPEDATS', 'ANNDATS_ACT']
    for col in date_cols:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col])

except FileNotFoundError:
    st.error("Error: cleaned_aapl_forecast_data.csv not found. Please ensure the data cleaning step was run.")
    st.stop()

# Define the Streamlit app
st.title("AAPL Analyst Forecast Analysis")

st.header("Introduction")
st.write("""
This application analyzes analyst forecast behavior and accuracy for Apple (AAPL) stock using the IBES dataset.
The goal is to understand the relationship between forecasted and actual earnings per share (EPS),
identify patterns in forecast errors, and evaluate the accuracy of different forecasting agents (Estimators/Analysts).
""")

st.header("Data Exploration")
st.write("Explore the distribution of forecasted and actual EPS, and the relationship between them.")

# Add a filter for FPI
fpi_options = df_cleaned['FPI'].unique().tolist()
selected_fpi = st.selectbox("Select Forecast Period (FPI):", fpi_options)

# Filter data based on selected FPI
df_filtered = df_cleaned[df_cleaned['FPI'] == selected_fpi]


# Univariate Analysis
st.subheader(f"Univariate Analysis: Distribution of EPS Values for FPI = {selected_fpi}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_filtered['VALUE'], kde=True)
plt.title('Distribution of Forecasted EPS (VALUE)')
plt.xlabel('Forecasted EPS')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df_filtered['ACTUAL'], kde=True)
plt.title('Distribution of Actual EPS (ACTUAL)')
plt.xlabel('Actual EPS')
plt.ylabel('Frequency')

st.pyplot(plt)
plt.close() # Close the figure to prevent it from displaying twice

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df_filtered['VALUE'])
plt.title('Box Plot of Forecasted EPS (VALUE)')
plt.xlabel('Forecasted EPS')

plt.subplot(1, 2, 2)
sns.boxplot(x=df_filtered['ACTUAL'])
plt.title('Box Plot of Actual EPS (ACTUAL)')
plt.xlabel('Actual EPS')

st.pyplot(plt)
plt.close() # Close the figure to prevent it from displaying twice


# Bivariate Analysis
st.subheader(f"Bivariate Analysis: Forecasted vs Actual EPS for FPI = {selected_fpi}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='VALUE', y='ACTUAL', data=df_filtered)
plt.title('Scatter Plot of Forecasted EPS (VALUE) vs Actual EPS (ACTUAL)')
plt.xlabel('Forecasted EPS (VALUE)')
plt.ylabel('Actual EPS (ACTUAL)')
st.pyplot(plt)
plt.close() # Close the figure to prevent it from displaying twice


# Correlation Heatmap (Excluding identifier columns)
st.subheader(f"Multivariate Analysis: Correlation Heatmap for FPI = {selected_fpi}")
exclude_cols = ['ESTIMATOR', 'ANALYS', 'FPI'] # Keep FPI in exclude_cols as it's categorical
numeric_cols = [
    col for col in df_filtered.select_dtypes(include=['float64', 'int64']).columns
    if col not in exclude_cols
]
plt.figure(figsize=(10, 8))
sns.heatmap(df_filtered[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Variables')
st.pyplot(plt)
plt.close() # Close the figure to prevent it from displaying twice


st.header("Insights")
st.write("""
Based on the analysis:
- Both forecasted ('VALUE') and actual ('ACTUAL') EPS distributions are skewed with a concentration at lower values and a tail towards higher values.
- Box plots reveal the presence of significant outliers in both forecasted and actual EPS, indicating instances of extreme earnings or forecasts.
- The scatter plot shows a positive relationship between forecasted and actual EPS, but with considerable spread, suggesting that forecasts are not always perfectly accurate.
- The correlation heatmap shows the relationships between numerical variables, highlighting the positive correlation between 'VALUE' and 'ACTUAL'.
""")

st.header("Recommendations")
st.write("""
- **Focus on Agents with Lower MAE:** Agents with lower Mean Absolute Error (MAE) are more accurate. Identifying these agents and potentially understanding their forecasting methodologies could be beneficial.
- **Investigate Outliers:** Further analysis of instances with large forecast errors (outliers) could reveal specific events or circumstances that led to significant discrepancies between forecasts and actuals.
- **Analyze by Forecast Period (FPI):** Explore if forecast accuracy varies significantly based on the forecast period (FPI). Longer-term forecasts might inherently have higher errors.
""")