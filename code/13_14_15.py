import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# read CSV into dataframe
df = pd.read_csv("~/w3PythonML/data/13_comedy.csv")

# map categorical to numeric data ('Go' indicates whether the individual attended this event)
nat_map = {'UK': 0, 'USA': 1, 'N': 2}
go_map = {'YES': 1, 'NO': 0}
df['Nationality'] = df['Nationality'].map(nat_map)
df['Go'] = df['Go'].map(go_map)

print(df)

features = ['Age', 'Experience', 'Rank', 'Nationality']
x = df[features]
y = df['Go']

dt = DecisionTreeClassifier()
dt = dt.fit(x, y)

tree.plot_tree(dt, feature_names=features)
# Would this person watch a rank 7, 40 year old American with 10 years of experience?
# (0 = no, 1 = yes)
print('40 year old, 10 years experience, rank 7, American: ', dt.predict([[40, 10, 7, 1]]),
      '\nAs above but rank 6: ', dt.predict([[40, 10, 6, 1]]))
plt.show()
