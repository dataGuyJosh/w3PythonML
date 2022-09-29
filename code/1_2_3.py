import numpy as np
from scipy import stats
# Part 1
# Part 2
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

print(np.mean(speed),
      np.median(speed),
      stats.mode(speed, keepdims=True))

# Part 3
list1 = [86, 87, 88, 86, 87, 85, 86]
list2 = [32, 111, 138, 28, 59, 77, 97]

print('List 1 STD: ', np.std(list1),
      '\nList 1 Var: ', np.var(list1),
      '\nList 2 STD: ', np.std(list2),
      '\nList 2 Var: ', np.var(list2))
