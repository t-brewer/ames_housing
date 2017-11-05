#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import cross_val_score, train_test_split

import tools

num_cols = ['Lot_Frontage', 'Lot_Area', 'BsmtFin_SF_1',
           'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF',
           '1st_Flr_SF', '2nd_Flr_SF', 'Low_Qual_Fin_SF',
           'Gr_Liv_Area', 'Garage_Area', 'Wood_Deck_SF', 
           'Open_Porch_SF', 'Enclosed_Porch', '3Ssn_Porch',
           'Screen_Porch', 'Pool_Area', 'Misc_Val']



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
        

# Clean data
def make_dummies(data):
# Function to Split categorical data into dummy variables

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

    # Columns where I want to keep all columns : N_dummies = N_categories.
    cat_data1 = ['MS_SubClass', 'MS_Zoning', 'Alley', 'Lot_Shape',
                 'Land_Contour', 'Lot_Config', 'Land_Slope',
                'Neighborhood', 'Condition_1', 'Condition_2',
                'Bldg_Type', 'House_Style', 'Roof_Style',
                'Roof_Matl', 'Exterior_1st', 'Exterior_2nd',
                 'Mas_Vnr_Type', 'Sale_Condition',
                'Foundation', 
                'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2',
                'Heating', 'Electrical', 'Functional', 'Paved_Drive',
                 'Garage_Type', 'Fence', 'Sale_Type']

    
    # Categorical data where I want to drop a column:
    # N_dummies = N_categories - 1
    cat_data2 = ['Street', 'Utilities']
    
    # Replace categorical columns with dummy columns
    for c in cat_data1 :
        data = dummy_drop(data, c)

    for c in cat_data2 :
        data = dummy_drop(data, c, drop_first=True)

    # Other : wanted to change the prefix, so I didn't include in the 
    # the list.
    data = dummy_drop(data, 'Misc_Feature', prefix='m')
    return data

def map_to_number(data):

    def mapper(data, column, keys, start=1, inplace=True):

        # Make the dictionary
        d = {}
        for i,k in enumerate(keys):
            d.update({k : start + i})

        data['dummy'] = data[column].map(d)
        data.drop(column, axis=1, inplace=inplace)
        data[column] = data['dummy']
        data.drop('dummy', axis=1, inplace=inplace)

        return data   

    map1 = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    map2 = [np.nan, 'Po', 'Fa', 'TA', 'Gd', 'Ex']
   
    # Map to numbers :
    mapper(data, 'Heating_QC', map1)
    mapper(data, 'Kitchen_Qual', map1)
    mapper(data, 'Central_Air', ['N', 'Y'], start=0)
    mapper(data, 'Exter_Qual', map1)
    mapper(data, 'Exter_Cond', map1)
    mapper(data, 'Bsmt_Qual', map2, start=0)
    mapper(data, 'Bsmt_Cond', map2, start=0)

    
    #Exter_Qual', 'Exter_Cond
    

    # Fireplace Quality 
    data['Fireplace_Qu'].fillna('0', inplace=True)
    mapper(data, 'Fireplace_Qu',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)

    data['Garage_Finish'].fillna('0', inplace=True)
    mapper(data, 'Garage_Finish',
                  ['0', 'Unf', 'RFn', 'Fin'], start=0)
    
    data['Garage_Qual'].fillna('0', inplace=True)
    mapper(data, 'Garage_Qual',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)
    
    data['Garage_Cond'].fillna('0', inplace=True)
    mapper(data, 'Garage_Cond',
                  ['0', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], start=0)
    
    data['Pool_QC'].fillna('0', inplace=True)
    mapper(data, 'Pool_QC', ['0', 'Fa', 'TA', 'Gd'], start=0)
    
   
    return data


    
    
def get_imputers(data) :
         
    # Dictionary of columns to impute and their strategy :
    strategy_dict = {'Lot_Frontage' : 'mean',
                     'Mas_Vnr_Area' : 'median',
                     'BsmtFin_SF_1' : 'median',
                     'BsmtFin_SF_2' : 'most_frequent',
                     'Total_Bsmt_SF' : 'mean',
                     'Bsmt_Unf_SF' : 'median',
                     'Garage_Area' : 'mean'
                    }
    
    imputer_dict = {}
    # Loop through strategy_dict and fit an imputer with the data        
    for k,s in strategy_dict.items():
        imputer = Imputer(strategy=s)
        imputer.fit(data[k].values.reshape(-1,1))
        imputer_dict.update({k : imputer})
    
    return imputer_dict
        

def impute_columns(data, imputer_dict):
    
    for k, v in imputer_dict.items():
        data[k] = v.transform(data[k].values.reshape(-1,1))
        
    pass

def get_scalers(data):
# Same idea as for imputers
    scaler_dict = {}
    for c in num_cols:
        scaler = StandardScaler()
        try:
            scaler.fit(data[c].values.reshape(-1,1))
            scaler_dict.update({c : scaler})
        except:
            print(c)
        
    return scaler_dict

def scale_columns(data, scaler_dict):
    for k, v in scaler_dict.items():
        data[k] = v.transform(data[k].values.reshape(-1,1))
    pass

def corr_columns(data, corr_val=0):
    # Make list of columns that only has a correlation to
    # to sale price greater than corr_val (absolute value really)
    corr = data.corr()
    keep = [c for c in corr.columns
            if abs(corr['SalePrice'][c]) >= corr_Val]
    
    return keep
        
def make_clean(corr_val=0.4):
    
    # load data
    data = tools.load_data()
                
        
    # Get Dummy columns for categorical columns
    data = make_dummies(data)

    # Change Some categorical columns to a number scale
    data = map_to_number(data)

    # Fill NaN values in these columns with zeros (no need
    # for imputing since NaN represents that there is none).
    data['Bsmt_Full_Bath'].fillna(0, inplace=True)
    data['Bsmt_Half_Bath'].fillna(0, inplace=True)

    # Get imputers 
    imputers = get_imputers(data)
    impute_columns(data, imputers)

    # Standard Scaling
    scalers = get_scalers(data)
    scale_columns(data, scalers)

    # Scale SalePrice (divide by 100,000)
    # data['SalePrice'] = data['SalePrice'].apply(lambda x: x/100000)
    
    # Delete Columns that have less than a corr_val correlation to SalePrice
    # keep = 
    # corr = data.corr()
    # del_cols = [c for c in corr.columns
    #            if (abs(corr['SalePrice'][c]) < corr_val)]
    
    for c in del_cols:
        data.drop(c, inplace=True, axis=1)

    return data


def clean_data(data, imputers, scalers):
    
    # Drop Columns I decided not to keep
    data.drop(['Garage_Yr_Blt', 'Garage_Cars'], axis=1, inplace=True)    
        
    # Get Dummy columns for categorical columns
    data = make_dummies(data)

    # Change Some categorical columns to a number scale
    data = map_to_number(data)

    # Fill NaN values in these columns with zeros (no need
    # for imputing since NaN represents that there is none).
    data['Bsmt_Full_Bath'].fillna(0, inplace=True)
    data['Bsmt_Half_Bath'].fillna(0, inplace=True)

    # Impute and scale
    impute_columns(data, imputers)
    scale_columns(data, scalers)
    
    # Delete Columns that have less than a corr_val correlation to SalePrice
    corr = data.corr()
    del_cols = [c for c in corr.columns
                if (abs(corr['SalePrice'][c]) < corr_val)]
    
    for c in del_cols:
        data.drop(c, inplace=True, axis=1)

    return data


def basic_clean(data):
    # Cleaning procedures where it doesn't matter if the 
    # data is split before or not.

    # Scale SalePrice (divide by 100,000)
    data['SalePrice'] = data['SalePrice'].apply(lambda x: x/100000)
    
    # Drop columns I don't want to keep
    data.drop(['Garage_Yr_Blt', 'Garage_Cars'], axis=1, inplace=True)
    
    # Get Dummy columns for categorical columns
    data = make_dummies(data)    

    # Change Some categorical columns to a number scale
    data = map_to_number(data)

    # Fill NaN values in these columns with zeros (no need
    # split NaN represents that there is none).
    data['Bsmt_Full_Bath'].fillna(0, inplace=True)
    data['Bsmt_Half_Bath'].fillna(0, inplace=True)
       
    return data
    
    
    
    
    