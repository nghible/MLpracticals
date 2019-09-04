#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:02:03 2019

@author: K1898955
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#%%
bc = load_breast_cancer()

#%%
num_features = 6
n_classes = 2
plot_colors = "bry"

#%%

plt.figure()
for i in range(0, num_features):
    for j in range(i+1,num_features):
        # classify using two corresponding features
        pair = [i, j]
        X = bc.data[:, pair]
        y = bc.target
        # plot the (learned) decision boundaries
        plt.subplot( num_features, num_features, j*num_features+i+1 )
        x_min, x_max = X[:, 0].min()- 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        plt.xlabel( bc.feature_names[pair[0]], fontsize=8 )
        plt.ylabel( bc.feature_names[pair[1]], fontsize=8 )
        plt.axis( "tight" )
        # plot the training points
        for ii, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == ii)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=bc.target_names[ii],cmap=plt.cm.Paired)
        plt.axis("tight")
plt.show()

#%%

bc.feature_names[3] #mean_area
bc.feature_names[1] #mean_texture

#%%

X = bc.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = bc.target

#%%

x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 100, X[:, 1].max() + 100
xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))
# Mesh grid explanation:
# https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy


#%%

Z = bc_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#%%

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired) 

plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))

# Label axes
plt.xlabel( bc.feature_names[1], fontsize=10 )
plt.ylabel( bc.feature_names[3], fontsize=10 )
plt.axis('tight')
plt.show()







#%% Cross-validation

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.1, random_state=0) #random_state is the seed

#This splits the data set accoring to 90% training points and 10% test points

#%% Build 10 trees and take the average

score = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)
    bc_tree = DecisionTreeClassifier().fit( X_train,  y_train )
    score.append(bc_tree.score(X_test, y_test))
    
#%% Mean of 10 trees scores
    
np.array(score).mean()

#%% n-fold cross validation

from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
import random

#%% 5-fold

cv_scores = cross_val_score(bc_tree, X, y, cv = 5)
print(cv_scores.mean())
print(cv_scores.std())

#%% 10-fold

cv10_scores = cross_val_score(bc_tree, X, y, cv = 10)
print(cv10_scores.mean())
print(cv10_scores.std())

#0.85 for 10-fold cv, and 0.87 for 10 trees






#%% Conversion to Data Frame and rename columns

X = pd.DataFrame(X)
y = pd.DataFrame(y)
X.columns = ['mean texture', 'mean area']
y.columns = ['target']
Xy = pd.concat([X,y], axis = 1)

#%% Preprocessing

# Shuffle the data frame
shuffled_Xy = Xy.sample(frac = 1) #Frac is fraction, 1 means returning the whole df.


# Assigning index to groups
groups = np.arange(len(shuffled_Xy)) // round(0.1 * len(shuffled_Xy))


# Divide the dataframe in to almost even group according to the index created
grouped_Xy = shuffled_Xy.groupby(groups)

#%% Code for 10-fold validation manually

scores_kfold = []

for i in range(0,len(grouped_Xy.groups)):
    test_X = grouped_Xy.get_group(i)[['mean texture','mean area']]
    test_y = grouped_Xy.get_group(i)['target']
    train_X = shuffled_Xy.drop(test_X.index)[['mean texture','mean area']]
    train_y = shuffled_Xy.drop(test_y.index)['target']
    bc_tree = DecisionTreeClassifier().fit(train_X, train_y)
    scores_kfold.append(bc_tree.score(test_X, test_y))
    
#%%
np.array(scores_kfold).mean() #0.85 so yeah.








#%% Resplit data

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.1, random_state=0)

#%% Train yr_tree

yr_tree = DecisionTreeClassifier().fit(X_train, y_train)

#%%

predicted_y = pd.DataFrame(yr_tree.predict(X_test))
predicted_y.columns = ['target']
y_test = y_test.reset_index(drop = True)

#Drop is to drop the old index that was kept in the dataframe.


#%% Creating the confusion matrix

def Performance(test, predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(predicted)):
        if ((test.loc[i] == predicted.loc[i]) & (predicted.loc[i] == True)).bool():
            TP += 1
        if ((test.loc[i] == predicted.loc[i]) & (predicted.loc[i] == False)).bool():
            TN += 1
        if ((test.loc[i] != predicted.loc[i]) & (predicted.loc[i] == True)).bool():
            FP += 1
        if ((test.loc[i] != predicted.loc[i]) & (predicted.loc[i] == False)).bool():
            FN += 1
            
    return(TP,FP,FN,TN)
            
#We have to use dataframe.bool() to return boolean value contained
#within the dataframe.
#%%

confusion_matrix = Performance(y_test,predicted_y)
confusion_matrix = np.matrix(confusion_matrix).reshape(2,2)
confusion_matrix

#%% How many 'yes' prediction is correct.

precision = float(confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[1,0])
precision

#Remember the float !

#%% How many of the actual 'yes' examples are predicted correctly.

recall = float(confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[0,1])
recall

#%% F1 score

F = 2* ((precision * recall) / (precision + recall))
F

#%% Do all the scoring for the 10 trees in cross validation

scores_kfold = []
precision_vector = []
recall_vector = []

for i in range(0,len(grouped_Xy.groups)):
    test_X = grouped_Xy.get_group(i)[['mean texture','mean area']]
    test_y = grouped_Xy.get_group(i)['target']
    test_y = test_y.reset_index(drop = True)
    train_X = shuffled_Xy.drop(test_X.index)[['mean texture','mean area']]
    train_y = shuffled_Xy.drop(test_y.index)['target']
    bc_tree = DecisionTreeClassifier().fit(train_X, train_y)
    predicted_y = pd.DataFrame(bc_tree.predict(test_X))
    confusion_matrix = Performance(test_y,predicted_y)
    confusion_matrix = np.matrix(confusion_matrix).reshape(2,2)
    precision = float(confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[1,0])
    recall = float(confusion_matrix[0,0]) / (confusion_matrix[0,0] + confusion_matrix[0,1])
    precision_vector.append(precision)
    recall_vector.append(recall)

#%%
np.array(precision_vector).mean()
#%%
np.array(recall_vector).mean()
#%%
np.array(precision_vector).std()
#%%
np.array(recall_vector).std()






