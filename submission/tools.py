import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import cross_val_score, train_test_split

def load_data(file_name):
    # Load training data
    data = pd.read_csv(file_name)
    # Replace spaces with underscores in column names
    data.columns = [c.replace(' ', '_') for c in data.columns]
    
    return data



def check_values(series, cat=True) :
    # Check number of null values and 
    # Unique values for categorical data
    print("Dtype : ", series.dtype)
    print("N_null = ", series.isnull().sum())   
    if (cat):
        print("Unique : ", series.unique())
        print('mode :', stats.mode(series).mode[0])

    else:
        print('mean :', np.mean(series))
        print('median :', np.median(series))
        print('mode :', stats.mode(series).mode[0])
        
        series.hist(bins=20)
   
    pass


def corr_map(data, figsize=(15,10), mask_val = None):
    
    corr = data.corr()    
    mask = np.zeros_like(corr)
    
    mask[np.triu_indices_from(corr)] = True
    
    if (mask_val != None):
        mask[abs(corr) < mask_val] = True
    
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2g')
    pass


def corr_bar(data, figsize=(15,5)):
    corr = data.corr()
    
    f, ax = plt.subplots(figsize=figsize)
    corr.plot(x=corr.columns, y='SalePrice', kind='bar', ax=ax)

    pass
    

