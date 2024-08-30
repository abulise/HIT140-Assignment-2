import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Calculate average screen time
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['avg_screen_time'] = df[screen_time_cols].mean(axis=1)

# Calculate average wellbeing score
wellbeing_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                  'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df['avg_wellbeing_score'] = df[wellbeing_cols].mean(axis=1)

# Create screen time bins
screen_time_bins = pd.qcut(df['avg_screen_time'], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"])

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_screen_time', y='avg_wellbeing_score', hue=screen_time_bins)
plt.title('Screen Time vs Well-Being Score')
plt.xlabel('Average Screen Time (hours)')
plt.ylabel('Average Well-Being Score')
plt.legend(title='Screen Time Category')
plt.tight_layout()
plt.show()