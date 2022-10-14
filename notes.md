https://www.w3schools.com/python/python_ml_getting_started.asp

# Part 1 - Getting Started
In Machine Learning it is common to work with very large data sets. In this tutorial we will try to make it as easy as possible to understand the different concepts of machine learning, and we will work with small easy-to-understand data sets.

Data Types
- Numerical
  - Discrete: integers e.g. number of cars on a given street --> 25
  - Continuous: integers + complex numbers e.g. length of a string --> 25.009852mm
- Categorical
  - Nominal: no intrinsic ordering e.g. apple, orange, tomato
  - Ordinal: clear ordering e.g. small, medium, large


# Part 2 - Mean, Median & Mode
Mean: "average" value
Mediam: middle value
Mode: most frequent value


# Part 3 - Standard Deviation
The standard deviation describes how "spread out" values are; a lower standard deviation implies that values are closer to the mean than a higher standard deviation.

For example, a list of numbers [86,87,88,86,87,85,86] has a standard deviation of 0.9. This means most values are within 0.9 units of the mean (86.4). Whereas in a list of numbers [32,111,138,28,59,77,97] the standard deviation is 37.85 with a mean of 77.4.

Variance: another measure of spread, the square root of the variance is the standard deviation. To calculate variance:
- take a list: [1, 2, 3]
- find the mean: 2
- for each values, find the differene from the mean: [1, 0, 1]
- square each difference: [1, 0, 1]
- find the mean, this is the variance: 0.667


# Part 4 - Percentile
Percentiles are used in statistics and are a number that describes the value that a given percent of the values are lower than. For example, for the first 10 fibonacci numbers [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], the 75th percentile is 11.75 meaning 75% of the first 10 fibonacci numbers are less than or equal to 11.75.


# Part 5 - Data Distribution
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

# Part 7 - Scatter Plot
You can draw scatter plots with Matplotlib using two arrays (x, y).


# Part 8 - Linear Regression
- Regression: a measure of the relation between the mean value of one variable (e.g. output) and corresponding values of other variables (e.g. time and cost)
- Linear Regression: estimates a "straight-line" relationship between variables (y = m * x + c)

Given a linear function derived from real data, we can make predictions about future data. For example, given x/y data as follows:

x = [1, 2, 3]
y = [2, 3, 4]

The linear function would be y = 1 * x + 1, we can therefore calculate y at x = 4:
y = 1 * 4 + 1 = 5

Linear regression is not always the best fit, this will result in a value of r close to 0 (rather than -1/1).


# Part 9 - Polynomial Regression
Non-linear data may fit a polynomial regression better. We often square the value of r (giving the r^2 value).

y = a_0 + a_1 * x + a_2 * x ^ 2 + ... + a_n * x ^ n
Where n is considered the "degree of polynomial".


# Part 10 - Multiple Regression
Multiple regression is similar to linear regression, however it uses multiple independent variables instead of one. For example, with linear regression we can predict the CO2 emission of a car based on the size of the engine, but with multiple regression we can throw in more variables, like the weight of the car, to make the prediction more accurate.


# Part 11 - Scale
It is sometimes important to adjust disparate variables to a similar scale for comparison. This can be acheived using the z-score, the formula for which is
z = (x - u) / s
where z = the new value
      x = original value
      u = mean
      s = standard deviation


# Part 12 - Train/Test
Machine learning models are created to predict the outcome of certain events e.g. in previous sections we predicted vehicle CO2 emissions based on weight & engine size. One way of testing model "accuracy" is to train on a subset of data then test on the other subset (train/test datasets).


# Part 13 - Decision Tree
A decision tree is a flow chart which can "help decision making processes based on previous experience". 
Decision trees generally only work on numerical data, as such we often "map" categorical data to numerical data e.g. `{'UK': 0, 'USA': 1, 'N': 2}`.

Target Column: variable being predicted
Feature Columns: variable(s) which (should) influence the target variable


# Part 14 - Confusion Matrix
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


# Part 15 - Hierarchical Clustering
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
