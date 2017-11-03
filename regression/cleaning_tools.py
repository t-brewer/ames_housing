#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import cross_val_score, train_test_split



def check_values(data, column, cat=True) :
    # Check number of null values and 
    # Unique values for categorical data
    print("Dtype : ", data[column].dtype)
    print("N_null = ", data[column].isnull().sum())   
    if (cat):
        print("Unique : ", data[column].unique())
        print('mode :', stats.mode(data[column]).mode[0])

    else:
        print('mean :', np.mean(data[column]))
        print('median :', np.median(data[column]))
        print('mode :', stats.mode(data[column]).mode[0])
        
        data[column].hist(bins=20)

def load_data():
    # Load training data
    data = pd.read_csv('../data/train.csv')
    # Replace spaces with underscores in column names
    data.columns = [c.replace(' ', '_') for c in data.columns]
    return data


def dummy_drop(data, column, prefix=None, drop_first=False):
    # Create dummy columns for dataframe, and drop original oone
    
    if (prefix == None):
        prefix = column
    
    dummies = pd.get_dummies(data[column],
                             prefix=prefix,
                             drop_first=drop_first)
    
    data = pd.concat([data, dummies], axis=1)
    
    
    data.drop(column, axis=1, inplace=True)
    return data              

def fill_na(data, column, strategy='mean', verbose=False, inplace=True):
    # Fill Missing Values in a column with a set strategy :
        # mean, median, or mode
    
    if (inplace == False):
        data = data.copy()
    
    values = data[column].copy().dropna().values
    
    if (strategy == 'mean'):
        filler = np.mean(values)
        
    elif (strategy == 'median'):
        filler = np.median(values)
    
    elif(strategy == 'mode'):
        filler = stats.mode(values).mode[0]
    
    else:
        print('Not a valid strategy.  Use : "mean", "median", or "mode".')
        return data
    
    if (verbose == True):
        print('column : {}, strategy : {}, value : {}'.format(column, strategy, filler))
    
    return data[column].fillna(filler, inplace=True)


def map_to_number(data, column, keys, start=1, inplace=True):

    # Make the dictionary
    d = {}
    for i,k in enumerate(keys):
        d.update({k : start + i})
        
    data['dummy'] = data[column].map(d)
    data.drop(column, axis=1, inplace=inplace)
    data[column] = data['dummy']
    data.drop('dummy', axis=1, inplace=inplace)
    
    return data

# Clean data
def make_clean() :
    
    # Load data
    data = load_data()
    
    # Map to numbers :
    map_to_number(data, 'Heating_QC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
    map_to_number(data, 'Kitchen_Qual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
    map_to_number(data, 'Central_Air', ['N', 'Y'], start=0)
    
    
    # Fireplace Quality 
    data['Fireplace_Qu'].fillna('0', inplace=True)
    map_to_number(data, 'Fireplace_Qu',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)

    data['Garage_Finish'].fillna('0', inplace=True)
    map_to_number(data, 'Garage_Finish',
                  ['0', 'Unf', 'RFn', 'Fin'], start=0)
    
    data['Garage_Qual'].fillna('0', inplace=True)
    map_to_number(data, 'Garage_Qual',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)
    
    data['Garage_Cond'].fillna('0', inplace=True)
    map_to_number(data, 'Garage_Cond',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)
    
    data['Pool_QC'].fillna('0', inplace=True)
    map_to_number(data, 'Pool_QC', ['0', 'Fa', 'TA', 'Gd'], start=0)
    
    # Fill null values 
    fill_na(data, 'Lot_Frontage', strategy='mean')
    fill_na(data, 'Mas_Vnr_Type', strategy='mode')
    fill_na(data, 'Mas_Vnr_Area', strategy='median')
    fill_na(data, 'BsmtFin_SF_1', strategy='median')
    fill_na(data, 'BsmtFin_SF_1', strategy='mode')
    fill_na(data, 'Total_Bsmt_SF', strategy='mean')
    data['Bsmt_Full_Bath'].fillna(0, inplace=True)
    data['Bsmt_Half_Bath'].fillna(0, inplace=True)
    
    
    # Categorical columns that I don't want to drop_first.
    cat_data1 = ['MS_SubClass', 'MS_Zoning', 'Alley', 'Lot_Shape',
                 'Land_Contour', 'Lot_Config', 'Land_Slope',
                'Neighborhood', 'Condition_1', 'Condition_2',
                'Bldg_Type', 'House_Style', 'Roof_Style',
                'Roof_Matl', 'Exterior_1st', 'Exterior_2nd',
                 'Mas_Vnr_Type', 'Exter_Qual', 'Exter_Cond', 
                'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 
                'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2',
                'Heating', 'Electrical', 'Functional', 'Paved_Drive',
                 'Garage_Type', 'Fence', 'Sale_Type']
    
    #            'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Total_Bsmaaaat ]

    
    # Categorical data where I do want to drop_first
    cat_data2 = ['Street', 'Utilities']
    
    # Replace categorical columns with dummy columns
    for c in cat_data1 :
        data = dummy_drop(data, c)

    for c in cat_data2 :
        data = dummy_drop(data, c, drop_first=True)

    # Other :
    data = dummy_drop(data, 'Misc_Feature', prefix='m')
    
    # Completly Drop some columns :
    data.drop('Garage_Cars', axis=1, inplace=True)

     
    return data
