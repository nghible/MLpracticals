# k-means-starter.py
# parsons/28-feb-2017
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import random
import pandas as pd

#%%

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()

#%% k-Means manually, 3 clusters

# Initial random clustres center:

u1 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])
u2 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])
u3 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])

u = np.array([u1, u2, u3])

#%% Expectation-Maximization Algorithm 

# Step 1. Expectation: find the assignment of data points to clusters

# 1. Calculate distance between the data point to three cluster centers.
# 2. Assign the data point to cluster that has the shortest distance.
# 3. Repeat for each points.

# To assign data points to each cluster center we need to implement
# 1-of-K encoding scheme

r = [] # This vector hold information of which clusters each data point belong to.

for data_point in X:
    d = []
    for mean in u:
        d.append(np.linalg.norm(data_point - mean))
    r.append(np.argmin(d))

r_1ofK = pd.get_dummies(r) # Convert the vector into 1-of-K coding scheme

#Be careful that this matrix is a pandas dataframe.

#%%

# Step 2. Maximization: find the new cluster centers.

for i in range(len(u)):
    u[i] = np.matmul(r_1ofK[i],X) / r_1ofK[i].sum() 

# The new means are found by averaging the values of data points belong to that 
# specific cluster over the number of data point belong to that cluster.

#%% Putting the 2 EM steps together.

def MyKMeans (n_iterations):
    
    u1 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])
    u2 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])
    u3 = np.array([random.uniform(x0_min, x0_max), random.uniform(x1_min, x1_max)])
    u = np.array([u1, u2, u3])
    
    for iteration in range(n_iterations):
        r = [] # This vector hold information of which clusters each data point belong to.
        for data_point in X:
            d = []
            for mean in u:
                d.append(np.linalg.norm(data_point - mean))
            r.append(np.argmin(d))
                
        r_1ofK = pd.get_dummies(r)
    
        for i in range(len(u)):
            u[i] = np.matmul(r_1ofK[i] , X) / r_1ofK[i].sum()
    return(np.array(r))

#%% Run the algorithm

my_clusterings = MyKMeans(50)

#%% Rand-Score metric to compare

from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y, my_clusterings)

# Rand-Score is 1 when two vectors are identical.
# We got 0.797

#%% Let's compare our algorithm with sklearn's K-Means

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X)

#%% Rand-Score again for sklearn

adjusted_rand_score(y, kmeans.labels_)

# Exactly the same performance to our algorithm.

#%% Plot everything

plt.subplot( 1, 3, 1 )

# Plot the original data 

plt.scatter(X[:, 0], X[:, 1], c= y.astype(np.float))

# Label axes

plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

# Plot my clustering results

plt.subplot( 1, 3, 2)

plt.scatter(X[:, 0], X[:, 1], c= my_clusterings.astype(np.float))

plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

# Plot sklearn results

plt.subplot( 1, 3, 3)

plt.scatter(X[:, 0], X[:, 1], c= kmeans.labels_.astype(np.float))

plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()

# Similar performances between our K-Means and Sk-learn.


