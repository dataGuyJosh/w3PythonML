import numpy as np
from sklearn import linear_model

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