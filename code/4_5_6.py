import numpy as np
import matplotlib.pyplot as plt
# Part 4
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39,
        80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]
fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(np.percentile(ages, 75),  # 75% of people in this list are 43 or younger
      np.percentile(fibonacci, 75))  # 75% of the first 10 fibonacci numbers are smaller than 11.75

# Part 5
# create an array containing 250 random floats between 0 and 5
x = np.random.uniform(0.0, 5.0, 250)

print(x)

# generate a histogram with 5 categories
# each bar represents values in ranges
# 0-1, 1-2, 2-3, 3-4, 4-5
plt.hist(x, 5)
plt.show()

# Part 6
x = np.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()