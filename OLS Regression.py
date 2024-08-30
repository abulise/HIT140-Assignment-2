import pandas as pd
import statsmodels.api as sm

# Load the datasets (assuming you've already done this)
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

# Prepare data for regression
X = df['avg_screen_time']
y = df['avg_wellbeing_score']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())