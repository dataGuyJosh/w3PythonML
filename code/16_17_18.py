import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression

# Part 16
# represents size of tumor (cm) i.e. independent variable
# reshaped into a column for LogisticRegression() function to work
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92,
             4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)

# represents whether or not the tumor is cancerous (0 = no, 1 = yes) i.e. dependent variable
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

log_reg = linear_model.LogisticRegression()
log_reg.fit(X, y)

# predict whether a 3.46mm tumor is cancerous
prediction = log_reg.predict(np.array([3.46]).reshape(-1, 1))

# determine odds i.e. if x increases by 1, y increases by a factor of "odds"
log_odds = log_reg.coef_
odds = np.exp(log_odds)

# coefficient & intercept values can be used to find the probability of each tumor being cancerous


def logit2prob(log_reg, x):
    # find log-odds for each observation
    log_odds = log_reg.coef_ * x + log_reg.intercept_
    # exponentiate to odds
    odds = np.exp(log_odds)
    # convert to probability
    probability = odds / (1 + odds)
    return (probability)


print("Is a 3.46mm tumor likely to be cancerous?", prediction,
      "\n1mm in growth increases the liklihood of a tumor being cancerous by a factor of", odds,
      "\nProbability each tumor is cancerous:\n", logit2prob(log_reg, X)
      )

# Results Explained
# 3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.
# 2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.
# 2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.


# Part 17
iris = datasets.load_iris()
# define independent variables (X) and dependant variable (y)
X = iris['data']
y = iris['target']
# set model's max_iter to a "higher" values to ensure it finds results
# the default value for C in a logistic regression model is 1
# max_iter determines the number of unique C values to try
log_reg = LogisticRegression(max_iter=10000)

# Fit model according to given training data
log_reg.fit(X, y)

print('Mean Accuracy: ', log_reg.score(X, y))

'''
Improving accuracy using grid search (instead of random selection)
- same steps as before, however we set a range of values for C
- knowing which values to set for the searched parameters requires domain knowledge and practice
'''
# Default value for C is 1, therefore we set a range surrounding it
C_range = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# Create empty list to store scores
scores = []
# loop over C range updating parameters each time
for C_value in C_range:
    log_reg.set_params(C=C_value)
    log_reg.fit(X, y)
    scores.append(log_reg.score(X, y))

# Notice how lower C values performed worse than the base value (1).
# However increasing it helped to increase accuracy!
print(scores)


# Part 18
cars = pd.read_csv("~/w3PythonML/data/11_cars1.csv")
# one hot encode the car manufacturer such that there is a column per manufacturer
# (indicating whether a given car is/isn't of that type)
ohe_cars = pd.get_dummies(cars[['Car']])

# select independent variables (X) adding one hot encoded variables columnwise (axis=1)
X = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
y = cars['CO2']

# Fit data using linear regression
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X.values, y)

# Predict CO2 emissions based on car weight, volume & manufacturer
# Predict the CO2 emission of a 2300kg, 1300cm3 Volvo:
print(lin_reg.predict(
    [[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]))


colours = pd.DataFrame({'colour': ['red', 'green', 'blue']})
# to save memory, one hot encoded data can use one less column than the number of categories
# i.e. if a row holds 0 for each column, it must be the other category
ohe_colours = pd.get_dummies(colours, drop_first=True)
ohe_colours['colour'] = colours['colour']
print(ohe_colours)