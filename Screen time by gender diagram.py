import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Calculate average screen time
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['avg_screen_time'] = df[screen_time_cols].mean(axis=1)

# Group by gender and calculate mean screen time
avg_screen_time_by_gender = df.groupby('gender')['avg_screen_time'].mean()

# Create a bar plot
avg_screen_time_by_gender.plot(kind='bar')
plt.title('Average Screen Time by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Screen Time (hours)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print the numerical results
print("Average Screen Time by Gender:\n", avg_screen_time_by_gender)