https://www.w3schools.com/python/python_ml_getting_started.asp

# Part 1: Getting Started
In Machine Learning it is common to work with very large data sets. In this tutorial we will try to make it as easy as possible to understand the different concepts of machine learning, and we will work with small easy-to-understand data sets.

Data Types
- Numerical
  - Discrete: integers e.g. number of cars on a given street --> 25
  - Continuous: integers + complex numbers e.g. length of a string --> 25.009852mm
- Categorical
  - Nominal: no intrinsic ordering e.g. apple, orange, tomato
  - Ordinal: clear ordering e.g. small, medium, large


# Part 2: Mean, Median & Mode
Mean: "average" value
Mediam: middle value
Mode: most frequent value


# Part 3: Standard Deviation
The standard deviation describes how "spread out" values are; a lower standard deviation implies that values are closer to the mean than a higher standard deviation.

For example, a list of numbers [86,87,88,86,87,85,86] has a standard deviation of 0.9. This means most values are within 0.9 units of the mean (86.4). Whereas in a list of numbers [32,111,138,28,59,77,97] the standard deviation is 37.85 with a mean of 77.4.

Variance: another measure of spread, the square root of the variance is the standard deviation. To calculate variance:
- take a list: [1, 2, 3]
- find the mean: 2
- for each values, find the differene from the mean: [1, 0, 1]
- square each difference: [1, 0, 1]
- find the mean, this is the variance: 0.667


# Part 4: Percentile
Percentiles are used in statistics and are a number that describes the value that a given percent of the values are lower than. For example, for the first 10 fibonacci numbers [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], the 75th percentile is 11.75 meaning 75% of the first 10 fibonacci numbers are less than or equal to 11.75.


# Part 5: Data Distribution
## How do you generate large data sets?
It can be difficult to gather real world data, at least at an early stage of a project. To create big data sets for testing, we use the Python module NumPy, which comes with a number of methods to create random data sets, of any size.

## Histogram vs Bar Graph
A bar graph represents categorical data i.e. where each bar represents a category. A histogram represents continuous data where each bar aggregates a range of values.

# Part 6 - Normal Data Distribution
A normal or Gaussian distribution defines a set of data who's values are groups around a central point. Using numpy we can specify a normal distribution by the number of values, mean and standard deviation.

For example, the following python code produces a set of 100,000 values where the mean is 5 and the standard deviation is 1 meaning values should be concentrated around 5 and rarely outside the range 4 to 6.
```python
import numpy as np
x = np.random.normal(5.0, 1.0, 100000)
```

# Part 7: Scatter Plot
You can draw scatter plots with Matplotlib using two arrays (x, y).


# Part 8: Linear Regression
- Regression: a measure of the relation between the mean value of one variable (e.g. output) and corresponding values of other variables (e.g. time and cost)
- Linear Regression: estimates a "straight-line" relationship between variables (y = m * x + c)

Given a linear function derived from real data, we can make predictions about future data. For example, given x/y data as follows:

x = [1, 2, 3]
y = [2, 3, 4]

The linear function would be y = 1 * x + 1, we can therefore calculate y at x = 4:
y = 1 * 4 + 1 = 5

Linear regression is not always the best fit, this will result in a value of r close to 0 (rather than -1/1).


# Part 9: Polynomial Regression
Non-linear data may fit a polynomial regression better. We often square the value of r (giving the r^2 value).

y = a_0 + a_1 * x + a_2 * x ^ 2 + ... + a_n * x ^ n
Where n is considered the "degree of polynomial".


# Part 10: Multiple Regression
Multiple regression is similar to linear regression, however it uses multiple independent variables instead of one. For example, with linear regression we can predict the CO2 emission of a car based on the size of the engine, but with multiple regression we can throw in more variables, like the weight of the car, to make the prediction more accurate.


# Part 11: Scale
It is sometimes important to adjust disparate variables to a similar scale for comparison. This can be acheived using the z-score, the formula for which is
z = (x - u) / s
where z = the new value
      x = original value
      u = mean
      s = standard deviation


# Part 12: Train/Test
Machine learning models are created to predict the outcome of certain events e.g. in previous sections we predicted vehicle CO2 emissions based on weight & engine size. One way of testing model "accuracy" is to train on a subset of data then test on the other subset (train/test datasets).


# Part 13: Decision Tree
A decision tree is a flow chart which can "help decision making processes based on previous experience". 
Decision trees generally only work on numerical data, as such we often "map" categorical data to numerical data e.g. `{'UK': 0, 'USA': 1, 'N': 2}`.

Target Column: variable being predicted
Feature Columns: variable(s) which (should) influence the target variable


# Part 14: Confusion Matrix
What is a confusion matrix?
- table used in classification problems to assess where errors were made in a model
- rows represent actual values while columns represent predictions
- can be made using predictions from a logistic regression

A confusion matrix may have 4 quadrants:
- True Negative (Top-Left Quadrant)
- False Positive (Top-Right Quadrant)
- False Negative (Bottom-Left Quadrant)
- True Positive (Bottom-Right Quadrant)

|       |   |           |       |
|-------|---|-----------|-------|
| True  | F | 8         | 83    |
| Label | T | 93        | 816   |
|       |   | F         | T     |
|       |   | Predicted | Label |

The prefix (True/False) indicates whether the prediction was accurate, the suffix (Positive/Negative) indicates the prediction itself.

A confusion matrix can be used to quantify the quality of a model using metrics such as:
- accuracy
  - how often is the model correct?
  - (TP + TN) / Total Predictions
- precision
  - of the positive predictions, how many were correct?
  - TP / (TP + FP)
- recall (sensitivity)
  - of all positive cases, what percentage were predicted positive?
  - TP / (TP + FN)
- specificity
  - of all negative cases, what pecentage were predicted negative?
  - TN / (TN + FP)
- F-score
  - harmonic mean of precision & recall
  - considers both FP & FN, good for "imbalancd" datasets
  - 2 * ((P * R) / (P + R))
  - note that because the formula uses precision and recall, it does not take false negatives into account


# Part 15: Hierarchical Clustering
What is hierarchical clustering?
- an unsupervised (no training or target variable) learning method for clustering (categorizing) data points
- generates "clusters" by measuring data dissimilarities
- helps vizualize and interpret relationships between individual datapoints

How does clustering work?
Agglomerative Clustering: a bottom-up approach with the following steps
- treat each data point as its own cluster
- join clusters together based on distance (creating larger clusters)
- continue this process until all points are part of the same cluster

Hierarchical clustering has several methods for calculating distance & linkage, this tutorial uses 
- euclidean distance
- ward linkage method --> tries to minimize variance between clusters

Dendrogram: a diagram which shows hierarchical relationships between objects, used to work out object allocation in clusters (https://www.displayr.com/what-is-dendrogram/)


# Part 16: Logistic Regression
Logistic regression aims to solve classification problems by predicting categorical outcomes, unlike linear regression which predicts continuous outcomes. Cases with two outcomes are considered binomial, while cases with more outcomes are multinomial (e.g. different species of iris flower).

In logistict regression, the coefficient is the expected change in "log-odds" of having the outcome per unit change in X. Note that we can exponentiate log-odds to find odds (odds = e ^ log_odds).

The coefficient and intercept values can be used to find the probability of an outcome: probability = odds / (1 + odds)


# Part 17: Grid Search
The majority of machine learning models contain parameters that can be adjusted to vary how the model learns. For example, the logistic regression model, from sklearn, has a parameter C that controls regularization, which affects the "complexity" of the model.

Higher values of C tell the model, the training data resembles real world information, place a greater weight on the training data. While lower values of C do the opposite.

How do we pick the best value for C? The best value is dependent on the data used to train the model.

One method is to try out different values and then pick the value that gives the best score. This technique is known as a grid search. If we had to select the values for two or more parameters, we would evaluate all combinations of the sets of values thus forming a grid of values.

Note on Best Practices:

We scored our logistic regression model by using the same data that was used to train it. If the model corresponds too closely to that data, it may not be great at predicting unseen data. This statistical error is known as over fitting.

To avoid being misled by the scores on the training data, we can put aside a portion of our data and use it specifically for the purpose of testing the model. Refer to the lecture on train/test splitting to avoid being misled and overfitting.


# Part 18: Preprocessing Categorical Data
Data represented by strings cause difficulties in training models which only accept numeric data. Instead of ignoring such (categorical) data, excluding information from the model, it can be transformed.

One Hot Encoding: A linear relationship cannot be determined between categorical & numeric variables. One solution is to create a column representing each group in the category. For each column, the values 1 & 0 will represent the inclusion & exlcusion of the group respectively (one hot encoding).

Use label encoding when
- number of categories is large (one hot encoding can lead to high memory consumption)
- order matters e.g. satisfaction rating

Use one hot encoding when
- order does not matter for a feature e.g. car models
- feature has few categories


# Part 19: K-means
K-means is an unsupervised learning method for clustering data points. The algorithm iteratively divides data points into K clusters by minimizing the variance in each cluster.

First, each data point is randomly assigned to one of the K clusters. Then, we compute the centroid (functionally the center) of each cluster, and reassign each data point to the cluster with the closest centroid. We repeat this process until the cluster assignments for each data point are no longer changing.

K-means clustering requires us to select K, the number of clusters we want to group the data into. The elbow method lets us graph the inertia (a distance-based metric) and visualize the point at which it starts decreasing linearly. This point is referred to as the "eblow" and is a good estimate for the best value for K based on our data.


# Part 20: Bootstrap Aggregation (Bagging)
Methods such as Decision Trees can be prone to overfitting on training data, leading to poor performance on new data. Bootstrap Aggregation (bagging) is an ensembling method which attempts to resolve overfitting for classification/regression problems.

Bagging aims to improve accuracy and performance of machine learning algorithms. This is done by taking random subsets of a dataset, with replacement, fitting either a classifier (for classification) or regressor (for regression) to each subset.

The predictions for each subset are aggregated through majority vote or averaging for classification and regression respectively, increasing prediction accuracy.

Another form of evaluation
As bootstrapping chooses random subsets of observations to create classifiers, there are observations that are left out in the selection process. These "out-of-bag" observations can then be used to evaluate the model, similarly to that of a test set. Keep in mind, that out-of-bag estimation can overestimate error in binary classification problems and should only be used as a compliment to other metrics.

Since the samples used in OOB and the test set are different, and the dataset is relatively small, there is a difference in the accuracy. It is rare that they would be exactly the same, OOB should be used as a quick means for estimating error, but is not the only evaluation metric.

Generating Decision Trees from a Bagging Classifier
As shown previously, it's possible to graph a generated decision tree classifier. It's also possible to see individual trees in an aggregated classifier. This can help users gain understanding on how the bagging model arrives at its predictions.

Note that this is only really functional on smaller datasets, where trees are relatively shallow and narrow making it easy to visualize.


# Part 21: Cross Validation
Hyperparameter tuning can lead to better model performance on test sets, however optimizing parameters to test sets can lead to information leakage. This causes the model to perform worse on unseen data, cross validation is used to correct this issue.

## K-Fold
Training data is split into k subsets, models are trained on k-1 subsets then validated against the remaining subset.

## Stratified K-Fold
In cases where classes are imbalanced we need a way to account for the imbalance in both the train and validation sets. To do so we can stratify the target classes, meaning that both sets will have an equal proportion of all classes.

## Leave-One-Out (LOO)

Use 1 observation to validate and n-1 observations to train, this technique is exhaustive.

## Leave-P-Out (LPO)
Use P observations to validate and n-p observations to train, this technique is exhaustive. As we are comparing unique combinations (not to be confused with permutations), this process grows quickly with dataset size.

For example, unique combinations of 2 observations in a dataset of 150 observations:
- C(n,r)=n!/(r!(n−r)!)
- 150!÷(2!(150−2)!) = 11175

## Shuffle Split
Unlike KFold, ShuffleSplit leaves out a percentage of the data, not to be used in the train or validation sets. To do so we must decide what the train and test sizes are, as well as the number of splits.

## When should each technique be used?
One of the most important factors to consider when choosing a cross validation technique is dataset size. The following list is arranged from most to least intensive, if memory usage/processing time is a concern, consider using techniques towards the bottom. Otherwise the higher techniques will usually offer better results.
- Leave-P-Out
- Leave-One-Out
- K-Fold/Stratified K-Fold
- Shuffle Split

These are just a few of the CV methods that can be applied to models. There are many more cross validation classes, with most models having their own class.


# Part 22: AUC - ROC Curve
In classification, there are many different evaluation metrics. The most popular is accuracy, which measures how often the model is correct. This is a great metric because it is easy to understand and getting the most correct guesses is often desired. There are some cases where you might consider using another evaluation metric.

Another common metric is AUC, area under the receiver operating characteristic (ROC) curve. The Reciever operating characteristic curve plots the true positive (TP) rate versus the false positive (FP) rate at different classification thresholds. The thresholds are different probability cutoffs that separate the two classes in binary classification. It uses probability to tell us how well a model separates the classes.

An AUC score of around .5 would mean that the model is unable to make a distinction between the two classes and the curve would look like a line with a slope of 1. An AUC score closer to 1 means that the model has the ability to separate the two classes and the curve would come closer to the top left corner of the graph.

Because AUC is a metric that utilizes probabilities of the class predictions, we can be more confident in a model that has a higher AUC score than one with a lower score even if they have similar accuracies.

To reiterate, models with higher AUC scores are more likely produce higher accuracy on future data.