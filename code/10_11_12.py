import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Part 10
# load a CSV as a python dataframe
df = pd.read_csv("~/w3PythonML/data/11_cars1.csv")

# It is common to name the list of independent values with a upper case X,
# and the list of dependent values with a lower case y.

# independent variables
# taking "values" here to circumvent a warning relating to using a dataframe with a list
# (not actually necessary)
X = df[['Weight', 'Volume']].values
# dependent variable
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)
# predict CO^2 emission of a car where
print(
    'Coefficients (Weight vs CO2, Volume vs CO2): ', regr.coef_,
    # weight = 2300kg & volume = 1300cm^3
    '\nLight Car: ', regr.predict([[2300, 1300]]),
    # weight = 3300kg & volume = 1300cm^3
    '\nHeavy Car: ', regr.predict([[3300, 1300]])
)
'''
The result array represents the coefficient values of weight and volume.

Weight: 0.00755095
Volume: 0.00780526

These values tell us that if the weight increase by 1kg,
the CO2 emission increases by 0.00755095g.

If the engine size (Volume) increases by 1 cm3,
the CO2 emission increases by 0.00780526g.

We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, 
will release approximately 115 grams of CO2 for every kilometer it drives.

Which shows that the coefficient of 0.00755095 is correct:
lightCarCO2 + (weightDifference * coefficient) = heavyCarCO2
107.2087328 + (1000 * 0.00755095) = 114.75968
'''


# Part 11
scale = StandardScaler()
# important to note that this dataset
# has the volume converted to litres for the purposes of this example
df = pd.read_csv("~/w3PythonML/data/11_cars2.csv")
X = df[['Weight', 'Volume']].values
y = df['CO2']
scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
# scale 1.3 liters and 2300kg to a similar magnitude
scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regr.predict([scaled[0]])
print('Light Car (using scaled values): ', predictedCO2)


# Part 12
np.random.seed(2)

# create a random normal dataset
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

# split into test/train data sets
# train set: dataset ending at index 80
train_x = x[:80]
train_y = y[:80]
# test set: dataset starting at index 80
test_x = x[80:]
test_y = y[80:]


# generate polynomial function using training data
model = np.poly1d(np.polyfit(train_x, train_y, 4))

# calculate R^2 values for training data i.e. how well the model estimates the training data
r2_train = r2_score(train_y, model(train_x))
r2_test = r2_score(test_y, model(test_x))
print('R^2 Train: ', r2_train,
      '\nR^2 Test: ', r2_test,
      '\ny @ x = 5: ', model(5))


fit_line = np.linspace(0, 6, 100)
# plt.scatter(test_x, test_y)
# plot training data
plt.scatter(train_x, train_y)
# plot fit line where
# x = list of 100 floast from 0 to 6
# y = model(x)
plt.plot(fit_line, model(fit_line))

plt.show()
