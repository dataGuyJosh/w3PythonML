import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree


# setup mock data (note that we use only two variables but this method works with more)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# setup data points
data = list(zip(x, y))

inertias = []
indexes = range(1, len(y))

'''
In order to find the best value for K,
we need to run K-means across our data for a range of possible values.
We only have 10 data points, so the maximum number of clusters is 10.
So for each value K in range(1,11),
we train a K-means model and plot the intertia at that number of clusters.
'''
for i in indexes:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# plt.plot(indexes, inertias, marker='o')
# plt.title('Inertia vs Clusters (Elbow Method)')
# plt.xlabel('Clusters')
# plt.ylabel('Inertia')
# plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
# plt.scatter(x, y, c=kmeans.labels_)
# plt.show()


# Part 20
# load dataset into X/y parameters
data = datasets.load_wine(as_frame=True)
X = data.data
y = data.target

# split data into training/test subsets
# shuffles data by default, setting random_state makes the results reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22)
# instantiate base classifier and fit to training data
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)

# predict y values (type of wine) based on test x values (features)
y_pred = dtree.predict(X_test)

print('Train data accuracy:', accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)),
      '\nTest data accuracy:', accuracy_score(y_true=y_test, y_pred=y_pred))

'''
The base classifier performs reasonably well on the dataset,
achieving 82% accuracy on the test dataset with the current parameters.
(Different results may occur if you do not have the random_state parameter set)

Now that we have a baseline accuracy for the test dataset,
we can see how the Bagging Classifier out performs a single Decision Tree Classifier.
'''

estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
models = []
scores = []

for n_estimators in estimator_range:
    # create bagging classifier
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    # fit model
    clf.fit(X_train, y_train)
    # append model and score to respective lits
    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

# # generate plot of scores vs n_estimators
# plt.figure(figsize=(9, 6))
# plt.plot(estimator_range, scores)
# # adjust labels and font
# plt.xlabel('n_estimators', fontsize=18)
# plt.ylabel('score', fontsize=18)
# plt.tick_params(labelsize=16)
# # visualize plot
# plt.show()
'''
By iterating through different values for the number of estimators,
we can see an increase in model performance from 82.2% to 95.5%.
After 14 estimators accuracy begins to drop, this drop can be mitigated using cross validation.
This shows a 13.3% increase in accuracy over the base model!
'''

# Out-of-bag validation using 12 estimators
# (heighest observed accuracy in previous tests on wine dataset)
oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)

oob_model.fit(X_train, y_train)

print(oob_model.oob_score_)

# plt.figure(figsize=(10, 10))
# # plots the first (0th) tree used to vote on the final prediction
# plot_tree(clf.estimators_[0], feature_names=X.columns)
# plt.show()

# Part 21
X, y = datasets.load_iris(return_X_y=True)

# K-Fold
'''
As we will be trying to classify different species of iris flowers,
we will need to import a classifier model,
for this exercise we will be using a DecisionTreeClassifier.
We will also need to import CV modules from sklearn.
'''
clf = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=k_folds)

print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))

# Stratified K-Fold
sk_folds = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=sk_folds)
print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))

# Leave-One-Out
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)
print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))

# Leave-P-Out
lpo = LeavePOut(p=2)
scores = cross_val_score(clf, X, y, cv=lpo)
print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      # Unique Pairs (Combinations) of 2 observations
      # C(n,r)=n!/(r!(n−r)!)
      # 150!÷(2!(150−2)!) = 11175
      '\nNumber of scores used in Average:', len(scores))

# Shuffle Split
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores = cross_val_score(clf, X, y, cv=ss)

print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))