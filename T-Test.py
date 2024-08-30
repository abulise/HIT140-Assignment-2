import pandas as pd
from scipy import stats

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Calculate average screen time
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['avg_screen_time'] = df[screen_time_cols].mean(axis=1)

# Calculate average well-being score
wellbeing_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                  'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df['avg_wellbeing_score'] = df[wellbeing_cols].mean(axis=1)

# Split data into high and low screen time groups
high_screen_time = df[df['avg_screen_time'] > df['avg_screen_time'].median()]['avg_wellbeing_score']
low_screen_time = df[df['avg_screen_time'] <= df['avg_screen_time'].median()]['avg_wellbeing_score']

# Perform t-test
t_stat, p_value = stats.ttest_ind(high_screen_time, low_screen_time)

print(f"T-test Results: t-statistic = {t_stat}, p-value = {p_value}")