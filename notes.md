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