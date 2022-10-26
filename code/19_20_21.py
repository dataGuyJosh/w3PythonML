import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# setup mock data (note that we use only two variables but this method works with more)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# setup data points
data = list(zip(x, y))

inertias = []
indexes = range(1, len(y))

'''
In order to find the best value for K,
we need to run K-means across our data for a range of possible values.
We only have 10 data points, so the maximum number of clusters is 10.
So for each value K in range(1,11),
we train a K-means model and plot the intertia at that number of clusters.
'''
for i in indexes:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(indexes, inertias, marker='o')
plt.title('Inertia vs Clusters (Elbow Method)')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()
