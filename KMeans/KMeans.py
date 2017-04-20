import numpy as np
import random

def createClusteredData(N,k):
    random.seed(5)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 100000.0), np.random.normal(ageCentroid, 2.0)])
    X = np.array(X)
    return X

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

data = createClusteredData(100,5)
model = KMeans(n_clusters=5)
model = model.fit(scale(data))

print model.labels_

plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float))
plt.show()