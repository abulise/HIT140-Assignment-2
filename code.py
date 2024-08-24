import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and merge datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')

# Calculate total screen time and average well-being score
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
wellbeing_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thkclr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

df['total_screen_time'] = df[screen_time_cols].sum(axis=1)
df['avg_wellbeing'] = df[wellbeing_cols].mean(axis=1)

# Descriptive Analysis 1: Screen Time Patterns
def analyze_screen_time():
    plt.figure(figsize=(12,6))
    sns.histplot(df['total_screen_time'], kde=True)
    plt.title('Distribution of Total Screen Time')
    plt.xlabel('Total Screen Time (hours per week)')
    plt.savefig('screen_time_distribution.png')
    plt.close()

    device_types = ['Computer', 'Games', 'Smartphone', 'TV']
    weekday_cols = ['C_wk', 'G_wk', 'S_wk', 'T_wk']
    weekend_cols = ['C_we', 'G_we', 'S_we', 'T_we']

    weekday_means = df[weekday_cols].mean()
    weekend_means = df[weekend_cols].mean()

    plt.figure(figsize=(12,6))
    x = np.arange(len(device_types))
    width = 0.35
    plt.bar(x - width/2, weekday_means, width, label='Weekday')
    plt.bar(x + width/2, weekend_means, width, label='Weekend')
    plt.xlabel('Device Type')
    plt.ylabel('Average Hours per Day')
    plt.title('Average Screen Time by Device Type and Day Type')
    plt.xticks(x, device_types)
    plt.legend()
    plt.savefig('screen_time_by_device.png')
    plt.close()

# Descriptive Analysis 2: Well-being Score Analysis
def analyze_wellbeing():
    plt.figure(figsize=(12,6))
    sns.histplot(df['avg_wellbeing'], kde=True)
    plt.title('Distribution of Average Well-being Scores')
    plt.xlabel('Average Well-being Score')
    plt.savefig('wellbeing_distribution.png')
    plt.close()

    wellbeing_means = df[wellbeing_cols].mean().sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    wellbeing_means.plot(kind='bar')
    plt.title('Average Scores of Individual Well-being Indicators')
    plt.xlabel('Well-being Indicator')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('wellbeing_indicators.png')
    plt.close()

# Inferential Analysis 1: Impact of Demographics on Screen Time
def analyze_demographics():
    # T-test for gender differences in screen time
    male_screen_time = df[df['gender'] == 1]['total_screen_time']
    female_screen_time = df[df['gender'] == 0]['total_screen_time']
    t_stat, p_value = stats.ttest_ind(male_screen_time, female_screen_time)
    print(f"T-test for gender differences in screen time: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # ANOVA for demographic groups
    demographic_factors = ['gender', 'minority', 'deprived']
    for factor in demographic_factors:
        groups = [group for _, group in df.groupby(factor)['total_screen_time']]
        f_value, p_value = stats.f_oneway(*groups)
        print(f"ANOVA for {factor}: F-value = {f_value:.4f}, p-value = {p_value:.4f}")

# Inferential Analysis 2: Relationship between Screen Time and Well-being
def analyze_screen_time_wellbeing():
    correlation = df['total_screen_time'].corr(df['avg_wellbeing'])
    print(f"Correlation between total screen time and average well-being: {correlation:.4f}")

    plt.figure(figsize=(10,6))
    plt.scatter(df['total_screen_time'], df['avg_wellbeing'], alpha=0.5)
    plt.xlabel('Total Screen Time (hours per week)')
    plt.ylabel('Average Well-being Score')
    plt.title('Screen Time vs Well-being')
    plt.savefig('screen_time_vs_wellbeing.png')
    plt.close()

    # Multiple Linear Regression
    X = df[['total_screen_time', 'gender', 'minority', 'deprived']]
    y = df['avg_wellbeing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Multiple Linear Regression Results:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R-squared: {model.score(X_test, y_test):.4f}")

# Run all analyses
analyze_screen_time()
analyze_wellbeing()
analyze_demographics()
analyze_screen_time_wellbeing()