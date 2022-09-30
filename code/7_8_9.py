import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# Part 7
# these parameters will produce a scatter plot centered around (5, 10)
x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)


# Part 8
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# define parameters for linear model using linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)

# define function y = m * x + c
# where     m = gradient, c = y-intercept


def f(x):
    return slope * x + intercept


# Run each value of the x array through the function,
# this will result in a new array with new values for the y-axis.
model = list(map(f, x))

plt.scatter(x, y)
plt.plot(x, model)
# plt.show()

print('Function: ', 'y = {} * x + {}'.format(round(slope, 3), round(intercept, 3)),
      '\nCoefficient of correlation (r): ', r,
      '\nPredicted y at x = 10: ', f(10))


# Part 9
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# define a polynomial model based on the x/y data (3 represents "degree of fitting")
model = np.poly1d(np.polyfit(x, y, 3))

# define the limits of a line between 1 & 22 with 100 samples
line = np.linspace(1, 22, 100)

plt.scatter(x, y)
# draw the polynomial with limits defined in 'line'
plt.plot(line, model(line))
print('Function:\n', model,
      '\nR^2: ', r2_score(y, model(x)))

plt.show()
