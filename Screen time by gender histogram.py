import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')
# Merge datasets on 'ID'
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Descriptive Analysis 1: Average screen time per day by gender
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['avg_screen_time'] = df[screen_time_cols].mean(axis=1)

# Group by gender and calculate mean screen time
avg_screen_time_by_gender = df.groupby('gender')['avg_screen_time'].mean()
print("Average Screen Time by Gender:\n", avg_screen_time_by_gender)


# Visualizations
# Histogram of screen time by gender
sns.histplot(data=df, x='avg_screen_time', hue='gender', element="step", stat="density", common_norm=False)
plt.title('Distribution of Average Screen Time by Gender')
plt.xlabel('Average Screen Time (hours)')
plt.ylabel('Density')
plt.show()