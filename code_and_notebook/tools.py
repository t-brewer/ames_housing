'''
Collection of functions to help visualize data and stuff.
'''

import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import cross_val_score, train_test_split

def load_csv(file_name):
    # Small function to load .csv files as dataframes and replace spaces in column names with
    # Unders
    data = pd.read_csv(file_name)
    data.columns = [c.replace(' ', '_') for c in data.columns]
 
    return data


def check_values(series, cat=True, bins=30) :
    # Function for a quick glance at value distributions
    # Needs to be a pandas series
    
    # Get data type and number of null values
    print("Dtype : ", series.dtype)
    print("N_null = ", series.isnull().sum())
    
    # Conditional display
    if (cat):
        # Categorical Feature
        # Print different categories, and the most common one.
        print("Unique : ", series.unique())
        print('mode :', stats.mode(series).mode[0])
        
        # If categorical data, make barplot with counts
        # for each category
        series.value_counts().plot(kind='bar')

    else:
        # Numerical data
        # Get some stats
        print('mean :', np.mean(series))
        print('median :', np.median(series))
        print('mode :', stats.mode(series).mode[0])
        
        # Checkout the distribution
        ax = series.hist(bins=bins)
        ax.set_xlabel(series.name)
        ax.set_ylabel('Count')
   
    pass


def corr_map(data, figsize=(15,10), mask_val = None):
    # Make correlation heat map from pandas dataframe
    # pass a value to mask_val to only see correlations
    # above a certain value (the absolute value of the correlation)
    corr = data.corr()    
    
    # Make mask to only show bottom triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(corr)] = True
    
    # Add to the mask (if mask_val passed)
    if (mask_val != None):
        mask[abs(corr) < mask_val] = True
    
    # Make heatmap
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2g')
    pass


def corr_bar(data, column, figsize=(15,5)):
    # Make barplot of correlations with a single feature
    # Pass DataFrame and column name
    corr = data.corr()    
    f, ax = plt.subplots(figsize=figsize)
    corr.plot(x=corr.columns, y=column, kind='bar', ax=ax)

    pass


def correlated_columns(data, column, corr_val=0):
    # Make list of columns that only has a correlation to
    # to a given column greater than corr_val (absolute value really)
    corr = data.corr()
    keep = [c for c in corr.columns
            if abs(corr[column][c]) >= corr_val]
    
    return keep
    

def get_imputers(data, strategy_dict) :         
    # Get a dictionary of imputers for different features of a 
    # dataframe.  This is to help using different strategies on 
    # different columns.
    # Pass the DataFrame, and a strategy_dict as such:
        # strategy_dict = {column_name : imputing_strategy}
    # Returns a dictionary as such:
        # {column_name : Imputer object}
    
    imputer_dict = {}
    # Loop through strategy_dict and fit an imputer with the data        
    for k,s in strategy_dict.items():
        try :
            imputer = Imputer(strategy=s)
            imputer.fit(data[k].values.reshape(-1,1))
            imputer_dict.update({k : imputer})
        except:
            continue
    
    return imputer_dict