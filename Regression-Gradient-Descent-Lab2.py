#%% Linear Regression Manually

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#%% First, generate the data and then split the data.

X, y = make_regression(n_samples=100, n_features=1, noise = 5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%%

# Say we are trying to manually fit a simple y = w0 + w1*x

# Steps:
# 1. Initialize random weights.
# 2. Use X_train sample to get p for that sample, using the random weights.
# 3. Find the error between p and y_train, in this case we use mean squared
# error.
# 4. Use stochastic gradient descent to update the weights
# 5. Put the new weights to find the new error of the new points.

#%% Mean squared error derivatives:

# The mean squared error is L = sum((y_train - p)^2), 
# with the sum over the samples.

# The derivative of L w.r.t to w0 is: y_train - p
# The derivative of L w.r.t to w1 is: (y_train - p)*X_train

# So for stochastic gradient descent, we do this sequentially by each sample
# picked randomly without replacement. The key is randomness.

#%% Stochastic gradient descent to train the model:

w0, w1 = np.random.uniform( -0.2, 0.2, 2) #Step 1
w_iterations = pd.DataFrame() #To keep track of w0 and w1 over iterations
error_iterations = pd.DataFrame()
alpha = 0.1 #Random pick

for iteration in range(1): #Only need to go through the dataset once to converge.
    random_index = np.random.choice(range(len(X_train)), len(X_train), replace = False) #
    for i in random_index:
        error = y_train[i] - (w0 + w1 * X_train[i]) #Step 2, 3 and 5
        error_iterations = error_iterations.append(pd.Series(error ** 2), ignore_index = True)
        w0 = w0 + alpha * error #Step 4
        w1 = w1 + alpha * error * X_train[i] #Step 4
        w_iterations = pd.concat([w_iterations,pd.DataFrame([w0, w1])], axis = 1)

#%%     
        
y_hat = w0 + w1 * X_test
y_hat = y_hat.reshape(y_test.shape) #Reshape to normalize data.

#%%

#Error plot over iterations
        
plt.subplot(3, 1, 1)
plt.plot(error_iterations) 

#Regression plots of y hat and y test

plt.subplot(3, 1, 2)
plt.scatter(X_test, y_test, color="red")
plt.scatter(X_test, y_hat, color="blue")

#Convergence graph of w0 and w1

plt.subplot(3,1,3)
plt.plot(w_iterations.loc[0,:].reset_index(drop = True))
plt.plot(w_iterations.loc[1,:].reset_index(drop = True))

plt.show()

#%%

#Calculate mean squared error:

print("Mean squared error: %.2f"
      % np.mean((y_hat - y_test) ** 2))

#Mean square error of sickit-learn regression:

print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))

# The results are close to each other.

#%% We extend the model to multivariate case: 
# y = w0 + w1 * x1 + w2 * x2 + w3 * x3

#Data generation and split

X, y = make_regression(n_samples = 100, n_features = 3, noise = 3)
w0, w1, w2, w3 = np.random.uniform( -0.2, 0.2, 4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%%

w_iterations = pd.DataFrame() #To keep track of w0 and w1 over iterations
error_iterations = pd.DataFrame()
alpha = 0.1 #Random pick

for iteration in range(5): #Only need to go through the dataset once to converge.
    random_index = np.random.choice(range(len(X_train)), len(X_train), replace = False) #
    for i in random_index:
        error = y_train[i] - (w0 + w1 * X_train[i][0] + w2 * X_train[i][1] + w3 * X_train[i][2]) #Step 2, 3 and 5
        error_iterations = error_iterations.append(pd.Series(error ** 2), ignore_index = True)
        w0 = w0 + alpha * error #Step 4
        w1 = w1 + alpha * error * X_train[i][0] #Step 4
        w2 = w2 + alpha * error * X_train[i][1]
        w3 = w3 + alpha * error * X_train[i][2]
        w_iterations = pd.concat([w_iterations,pd.DataFrame([w0, w1, w2, w3])], axis = 1)


#%%
        
y_hat = w0 + w1 * X_test[:,0] + w2 * X_test[:,1] + w3 * X_test[:,2]
y_hat = y_hat.reshape(y_test.shape)

#%%

# Error plot over iterations:
        
plt.subplot(2, 1, 1)
plt.plot(error_iterations) 

# Weights convergence graphs:

plt.subplot(2, 1, 2)
plt.plot(w_iterations.loc[0,:].reset_index(drop = True))
plt.plot(w_iterations.loc[1,:].reset_index(drop = True))
plt.plot(w_iterations.loc[2,:].reset_index(drop = True))
plt.plot(w_iterations.loc[3,:].reset_index(drop = True))
plt.show()

#%%

print("Mean squared error: %.2f"
      % np.mean((y_hat - y_test) ** 2))


