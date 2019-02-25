#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

#%%

with open('data_banknote_authentication.txt', 'r') as filedata:
    bank_data = pd.read_table(filedata, delimiter = ',', 
                              dtype = float, header = None)

#%%
    
with open('winequality-white.csv', 'r') as filedata:
    wine_data = pd.read_csv(filedata, dtype = float) 

#%% 
    
bank_data, drop_data = train_test_split(bank_data, test_size = 0.8)
bank_train = bank_data.iloc[:,0:2]

#%%

clf = svm.SVC(kernel = 'linear')

#%%

clf.fit()
