import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# create an imbalanced dataset
# where class 0 observations make up 5% of all observations
# while class 1 observations make up 95%
n = 10000
ratio = 0.95
n_0 = int((1 - ratio) * n)
n_1 = int(ratio * n)
y = np.array([0] * n_0 + [1] * n_1)

# below are the probabilities obtained from a hypothetical model
# which always predicts the majority class,
# i.e. the probability of predicting class 1 is 100%

y_proba = np.array([1]*n)
y_pred = y_proba > .5

print('accuracy score:', {accuracy_score(y, y_pred)})
cf_mat = confusion_matrix(y, y_pred)
print('Confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

'''
Although we obtain a very high accuracy, the model provided no information about the data so it's not useful.
We accurately predict class 1 100% of the time while accurately predicting class 0 0% of the time.

To reiterate, we have simulated a model which predicted that 100% of the data is of class 1.
This means we get an accuracy of 95%, as 5% are actually class 0.

At the expense of accuracy, it might be better to have a model that can somewhat separate the two classes.
'''

# below are the probabilities obtained from a hypothetical model that doesn't always predict the mode
y_proba_2 = np.array(
    np.random.uniform(0, .7, n_0).tolist() +
    np.random.uniform(.3, 1, n_1).tolist()
)
y_pred_2 = y_proba_2 > .5

print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print('Confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

'''
For the second set of predictions, we do not have as high of an accuracy score as the first
but the accuracy for each class is more balanced.
Using accuracy as an evaluation metric we would rate the first model higher than the second
even though it doesn't tell us anything about the data.

In cases like this, using another evaluation metric like AUC would be preferred.
'''


def plot_roc_curve(true_y, y_prob):
    # plot the roc curve based of the probabilities
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(y, y_proba)
print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}')
plot_roc_curve(y, y_proba_2)
print(f'model 2 AUC score: {roc_auc_score(y, y_proba_2)}')
plt.show()

'''
In the data below, we have two sets of probabilites from hypothetical models.
The first has probabilities that are not as "confident" when predicting the two classes
(the probabilities are close to .5).

The second has probabilities that are more "confident" when predicting the two classes
(the probabilities are close to the extremes of 0 or 1).
'''

y = np.array([0] * n + [1] * n)
#
y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.6, 1, n//2).tolist()
)

# print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}')
# print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}')

# print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
# print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')

print('model 1 accuracy:', accuracy_score(y, y_prob_1 > .5),
      '\nmodel 2 accuracy:', accuracy_score(y, y_prob_2 > .5),
      '\nmodel 1 AUC:', roc_auc_score(y, y_prob_1),
      '\nmodel 2 AUC:', roc_auc_score(y, y_prob_2))

plot_roc_curve(y, y_prob_1)
fpr, tpr, thresholds = roc_curve(y, y_prob_2)
plt.plot(fpr, tpr)
plt.show()

'''
Even though the accuracies for the two models are similar,
the model with the higher AUC score will be more reliable
because it takes into account the predicted probability.
It is more likely to give you higher accuracy when predicting future data.
'''