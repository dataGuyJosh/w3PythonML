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