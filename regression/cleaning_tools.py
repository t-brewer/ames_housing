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
           'Screen_Porch', 'Pool_Area', 'Misc_Val', 'SalePrice']



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
def make_dummies(data, train=True):
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
                 'Mas_Vnr_Type',
                'Foundation', 
                'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2',
                'Heating', 'Electrical', 'Functional', 'Paved_Drive',
                 'Garage_Type', 'Fence', 'Sale_Type']

    if (train):
        cat_data1.append('Sale_Condition')
    
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
    mapper(data, 'Exter_Qual', map1)
    mapper(data, 'Exter_Cond', map1)    
    
    mapper(data, 'Bsmt_Qual', map2, start=0)
    mapper(data, 'Bsmt_Cond', map2, start=0)
    mapper(data, 'Fireplace_Qu', map2, start=0)
    mapper(data, 'Garage_Qual', map2, start=0)
    mapper(data, 'Garage_Cond', map2, start=0)    
    mapper(data, 'Pool_QC', map2, start=0)    
    
    mapper(data, 'Central_Air', ['N', 'Y'], start=0)
    mapper(data, 'Garage_Finish', [np.nan, 'Unf', 'RFn', 'Fin'], start=0)
    
       
    
    return data


    
    
def get_imputers(data) :
         
    # Dictionary of columns to impute and their strategy :
    strategy_dict = {'Lot_Frontage' : 'mean',
                     'Mas_Vnr_Area' : 'median',
                     'BsmtFin_SF_1' : 'median',
                     'Bsmt_Unf_SF' : 'median',
                     'Total_Bsmt_SF' : 'mean',
                     'Bsmt_Unf_SF' : 'median',
                     'Garage_Area' : 'mean'
                    }
    
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
        

def impute_columns(data, imputer_dict):
    
    for k, v in imputer_dict.items():
        data[k] = v.transform(data[k].values.reshape(-1,1))
        
    pass

def get_scalers(data):
# Same idea as for imputers
    dont_scale = ['Id', 'SalePrice']
    scaler_dict = {}
    for c in num_cols:
        
        if c in dont_scale:
            continue
            
        scaler = StandardScaler()
        try:
            scaler.fit(data[c].values.reshape(-1,1))
            scaler_dict.update({c : scaler})
        except:
            continue
        
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
            if abs(corr['SalePrice'][c]) >= corr_val]
    
    return keep
       

def basic_clean(data, train=True, final=False):
    # Cleaning procedures where it doesn't matter if the 
    # data is split before or not.
    
    # Columns to drop:
    cols2drop = ['Garage_Yr_Blt', 'Garage_Cars', 'PID']
    if(final == False):
        cols2drop.append('Id')
        cols2drop.append('Sale_Condition')
        
    data.drop(cols2drop, axis=1, inplace=True)

    # Get Dummy columns for categorical columns
    # data = make_dummies(data, train=train)    

    # Change Some categorical columns to a number scale
    data = map_to_number(data)

    # Fill NaN values in these columns with zeros (no need
    # split NaN represents that there is none).
    data['Bsmt_Full_Bath'].fillna(0, inplace=True)
    data['Bsmt_Half_Bath'].fillna(0, inplace=True)
       
    return data
    
def train_clean(data):
    # Get imputers for Null values (also fits them)
    # Assign to a dictionary : column_name : Imputer(strategy)
    imputers = get_imputers(data)
    impute_columns(data, imputers)
    
    # 
    scalers = get_scalers(data)
    scale_columns(data, scalers)
    
    # Return clean data, imputers, and scalers
    return [data, imputers, scalers]
    

def reshape_data(data):
    data.rename(columns={'Street':'paved', 'Lot_Shape': 'reg_lot',
                         'Condition_2' : 'posAN',
                         'Roof_Matl':'roof_wd',
                         'Exter_Cond':'ExtCond_Fair',
                         'Foundation':'foundation_Pconc',
                         'Bsmt_Cond' : 'Bsmt_fair',
                         'Bsmt_Exposure' : 'Bsmt_expGd',
                         'BsmtFin_Type_1': 'Bsmt_GLQ',
                         'Electrical' : 'Electric_SBrkr',
                         '2nd_Flr_SF' : '2nd_Flr_bg',
                         'Full_Bath' : 'full_bath_2plus',
                         'Functional': 'Func_Fair',
                         'Garage_Type' : 'Garage_AB',
                         'MS_Zoning' : 'residential',
                         'Mas_Vnr_Type' : 'masVnr_solid'
                                          
                        },
                inplace=True)
    
    # Binarize
    data['residential'] = data['residential'].apply(lambda x : 1 if x in ['RH', 'RL', 'RM', 'FV'] else 0)
    data['paved'] = data['paved'].apply(lambda x: 1 if x == "Pave" else 0)
    data['reg_lot'] = data['reg_lot'].apply(lambda x: 1 if x == 'Reg' else 0) 
    data['posAN'] = data['posAN'].apply(lambda x: 1 if (x=='PosA' or x=='PosN') else 0)
    data['bldg_Fam_TwnhsE'] = data['Bldg_Type'].apply(lambda x: 1 if (x=='1Fam' or x=='TwnhsE') else 0)
    
    data['roof_wd'] = data['roof_wd'].apply(lambda x: 1 if (x=='WdShngl') else 0)
    
    data['ExtCond_Fair'] = data['ExtCond_Fair'].apply(lambda x: 1 if (x >= 3) else 0)
    
    data['foundation_Pconc'] = data['foundation_Pconc'].apply(lambda x: 1 if (x == 'PConc') else 0)
    
    data['Bsmt_fair'] = data['Bsmt_fair'].apply(lambda x: 1 if (x >= 3) else 0)
    
    data['Bsmt_expGd'] = data['Bsmt_expGd'].apply(lambda x: 1 if (x == 'Gd') else 0)

    data['Bsmt_GLQ'] = data['Bsmt_GLQ'].apply(lambda x: 1 if (x == 'GLQ') else 0)
    
    data['Electric_SBrkr'] = data['Electric_SBrkr'].apply(lambda x: 1 if x=='SBrkr' else 0)
    
    data['2nd_Flr_bg'] = data['2nd_Flr_bg'].apply(lambda x: 1 if x > 1000 else 0)

    data['full_bath_2plus'] = data['full_bath_2plus'].apply(lambda x: 1 if x >= 2 else 0)
    
    data['Func_Fair'] = data['Func_Fair'].apply(lambda x: 0 if x in ['Sev', 'Sal', 'Maj2'] else 1)
    
    data['Garage_AB'] = data['Garage_AB'].apply(lambda x : 1 if x in ['BuiltIn', 'Attchd'] else 0)

    data['Paved_Drive'] = data['Paved_Drive'].apply(lambda x : 1 if x == 'Y' else 0)
    
    data['has_deck'] = data['Wood_Deck_SF'].apply(lambda x: 1 if x > 0 else 0)
    data['masVnr_solid'] = data['masVnr_solid'].apply(lambda x: 1 if x in ['BrkFace', 'Stone'] else 0)
  


    # Porch Stuff
    data['porch_area']  = data['Open_Porch_SF'] + data['Enclosed_Porch'] + data['3Ssn_Porch'] + data['Screen_Porch']

    data['has_porch'] = data['porch_area'].apply(lambda x: 1 if x > 0 else 0)
    

    # Alley Stuff
    data['Alley'].fillna('None', inplace=True)
    
    
    
    # Delete Columns :
    
    # I've turned them into something else :
    data.drop('Open_Porch_SF', axis=1, inplace=True)    
    data.drop('Enclosed_Porch', axis=1, inplace=True) 
    data.drop('Screen_Porch', axis=1, inplace=True)
    data.drop('porch_area', axis=1, inplace=True)

    # reason: only two rows don't have all
    data.drop('Utilities', axis=1, inplace=True)
    
    # Reason : Not enough chnage between different types
    data.drop('Lot_Config', axis=1, inplace=True)
    
    # Doesn't seem to have a correlation with saleprice
    data.drop('BsmtFin_SF_2', axis=1, inplace=True)
    data.drop('Low_Qual_Fin_SF', axis=1, inplace=True)
    data.drop('3Ssn_Porch', axis=1, inplace=True)
    data.drop('Misc_Feature', axis=1, inplace=True)
    data.drop('Misc_Val', axis=1, inplace=True)
    data.drop('Mo_Sold', axis=1, inplace=True)
    data.drop('Yr_Sold', axis=1, inplace=True)
    data.drop('Land_Slope', axis=1, inplace=True)
    data.drop('Roof_Style', axis=1, inplace=True)
    data.drop('BsmtFin_Type_2', axis=1, inplace=True)
    data.drop('Heating', axis=1, inplace=True)
    
    # Reason : Not enough examples with feature
    data.drop('Pool_Area', axis=1, inplace=True)
    data.drop('Pool_QC', axis=1, inplace=True)
    data.drop('Fence', axis=1, inplace=True)

    # Reason : Cant figure out how to deal with it
    data.drop('Alley', axis=1, inplace=True)
    data.drop('Land_Contour', axis=1, inplace=True)
    data.drop('Neighborhood', axis=1, inplace=True)
    data.drop('Condition_1', axis=1, inplace=True)
    data.drop('Bldg_Type', axis=1, inplace=True)
    data.drop('House_Style', axis=1, inplace=True)
    data.drop('Exterior_1st', axis=1, inplace=True)
    data.drop('Exterior_2nd', axis=1, inplace=True)
    data.drop('Sale_Type', axis=1, inplace=True)
    
    return data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    