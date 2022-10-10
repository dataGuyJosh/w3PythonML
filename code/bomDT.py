import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# read CSV into dataframe
df = pd.read_csv("~/w3PythonML/data/melbourneDailyRainfall.csv")

# handle null values & map categorical to numeric
quality_map = {'Y': 1, 'N': 0}
df['Quality'] = df['Quality'].fillna('N').map(quality_map)
df['Rainfall (mm)'] = df['Rainfall (mm)'].fillna(0.0)

print(df)

features = ['Year', 'Month', 'Day', 'Quality']
x = df[features]
y = df['Rainfall (mm)']

dt = DecisionTreeRegressor()
dt = dt.fit(x, y)

# tree.plot_tree(dt, feature_names=features)
# plt.show()
print('Rainfall (mm): ', dt.predict([[2022, 10, 12, 1]]))
