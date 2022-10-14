import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering

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
# using x.values to placate python warning message
dt = dt.fit(x.values, y)

tree.plot_tree(dt, feature_names=features)
# Would this person watch a rank 7, 40 year old American with 10 years of experience?
# (0 = no, 1 = yes)
print('40 year old, 10 years experience, rank 7, American: ', dt.predict([[40, 10, 7, 1]]),
      '\nAs above but rank 6: ', dt.predict([[40, 10, 6, 1]]))
# plt.show()
plt.close()

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
print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall":
      Sensitivity_recall, "Specificity": Specificity, "F1_score": F1_score})

cm = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[False, True])

cm_display.plot()
# plt.show()
plt.close()


# Part 15
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

# Dendrogram
# Compute the linkage between points, here we use simple euclidean distance 
# and Ward's linkage, which seeks to minimize the variance between clusters.
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()

# X-Y scatter plot
# Initialize AgglomerativeClustering class with 2 clusters,
# using euclidean distance & Ward linkage
hierarchical_cluster = AgglomerativeClustering(
    n_clusters=2, affinity='euclidean', linkage='ward')
# Call fit_predict to compute clusters using defined parameters above,
# this generates a list of labels i.e. which cluster each point belongs to.
labels = hierarchical_cluster.fit_predict(data)
# Plot values using labels to colour-code points
# based on the cluster they're assigned to.
plt.scatter(x, y, c=labels)
plt.show()