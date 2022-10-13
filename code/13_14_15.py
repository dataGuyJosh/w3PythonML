import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Part 13
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
# plt.show()


# Part 14
# Generate some values for random/actual i.e. as though a model were being tested
actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Sensitivity_recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F1_score = metrics.f1_score(actual, predicted)

# metrics
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})

cm = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[False, True])

cm_display.plot()
plt.show()