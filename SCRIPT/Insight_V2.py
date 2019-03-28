# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:16:26 2019

@author: kennedy
"""

import pandas as pd
import numpy as np
import os
from os.path import dirname, join
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
#stop runtime error
np.seterr(divide='ignore', invalid='ignore')
rcParams['figure.figsize'] = 20, 25
__FILE__ = 'D:\\FREELANCER\\DATAMINING AND INSIGHTHOUSE PRICES\\' #e.g D:\\Gulsha_Salmut\\ ensure it ends with \\
DATASET_ = join(dirname(__FILE__), 'DATASET')

hosue_df = pd.read_csv(os.path.join(DATASET_, 'Villas For Sale.csv'))
#drop na values
hosue_df.dropna(inplace = True)
if 'Create Time' in hosue_df.columns:
  hosue_df.rename(columns = {'Create Time': 'date'}, inplace = True)
  hosue_df.date = pd.to_datetime(hosue_df.date, unit = 's')
  hosue_df.sort_values(by = 'date', inplace = True)
  hosue_df.set_index(['date'], inplace = True)

print('See data descroiption: {}'.format(hosue_df.describe()))
print('Skew of data: {}'.format(hosue_df.skew()))
print('Kurt of data: {}'.format(hosue_df.kurt()))
hosue_df.hist()


#%%

def standardize_houseprize(df, standardize = None, 
                           logg = None, normalize = None):
  df = df.copy(deep = True)
  #drop all objects
  #and leaving all float64 and int64 datatypes
  for ii in df.columns:
    if df[ii].dtype == object:
      df = df.drop(ii, axis = 1)
  
  '''
  #standardize values
        x - mean of x
  z = --------------------
          sd of x
          
  #log values
  
  z = log(x)
  
  #normalize values
  
          x - min(x)
  z = --------------------
          max(x) - min(x)
  '''
  
  #standard deviation
  def stdev(df):
    return np.std(df, axis = 0)
  #mean deviation
  def mean_dev(df):
    return df - np.mean(df, axis = 0)
  #log of data
  def logg_dat(df):
    return np.log(df)
  
  #standardized values for columns
  if standardize:
    for ii, ij in enumerate(df.columns):
      print(ii, ij)
      df['{}'.format(ij)] = mean_dev(df.loc[:, '{}'.format(ij)])/stdev(df.loc[:, '{}'.format(ij)])
  elif logg:
    df = logg_dat(df)
    df = df.replace([np.inf, -np.inf, np.nan], 0)
  elif normalize:
    for ii, ij in enumerate(df.columns):
      df['{}'.format(ij)] = (df.loc[:, '{}'.format(ij)] - min(df.loc[:, '{}'.format(ij)]))/\
      (max(df.loc[:, '{}'.format(ij)]) - min(df.loc[:, '{}'.format(ij)]))
  else:
    pass
    
  return df

df = standardize_houseprize(hosue_df)
df_standard = standardize_houseprize(hosue_df, standardize = True)
log_data = standardize_houseprize(hosue_df, logg=True)
df_normal = standardize_houseprize(hosue_df, normalize = True)


#%%


log_data.describe()
#plot log_price
#after_outl = log_data[(log_data.price < 20.0) & (log_data.price > 2.5)]
plt.scatter(np.arange(log_data.shape[0]), log_data.Price, s = .5)
plt.title('Plot of count against price on a log scale')
#plt.axhline(y = 20, linewidth=1, color='r')
#plt.axhline(y = 2.5, linewidth=1, color='r')
#plt.axhline(y = 12.159753818376581, linewidth=1, color='r')

#%% plots and correlation

df_standard.hist()
log_data.hist()
df_normal.hist()


#%% 
sns.violinplot(df_standard)
sns.pairplot(df_standard)
sns.pairplot(log_data)
sns.pairplot(df_normal)
sns.heatmap(df_standard)
sns.heatmap(log_data)
sns.heatmap(df_normal)
sns.heatmap(hosue_df.corr(), annot=True);plt.show()
sns.heatmap(df_standard.corr(), annot=True);plt.show()
sns.heatmap(log_data.corr(), annot=True);plt.show()
sns.heatmap(df_normal.corr(), annot=True);plt.show()
#%% DEALING WITH OUTLIERS

def remove_outliers(df, standardize = None, remove_objects = True,
                    logg = None, normalize = None, 
                    lower_quartile = None, upper_quartile = None, multiplier = None):
  
  #drop all objects
  #and leaving all float64 and int64 datatypes
  if remove_objects:
    for ii in df.columns:
      if df[ii].dtype == object:
        df = df.drop(ii, axis = 1)
  else:
    df = df
    df = pd.get_dummies(df, dtype = float)
    
    
    
  df = df.copy(deep = True)
  quart_1 = df.quantile(lower_quartile)
  quart_2 = df.quantile(upper_quartile)
  diff_quart = abs(quart_1 - quart_2)
  df = df[~((df < (quart_1 - 1.5 * diff_quart)) | (df > (quart_2 + 1.5 * diff_quart))).any(axis=1)]
  '''
  #standardize values
        x - mean of x
  z = --------------------
          sd of x
          
  #log values
  
  z = log(x)
  
  #normalize values
  
          x - min(x)
  z = --------------------
          max(x) - min(x)
  '''
  #standard deviation
  def stdev(df):
    return np.std(df, axis = 0)
  #mean deviation
  def mean_dev(df):
    return df - np.mean(df, axis = 0)
  #log of data
  def logg_dat(df):
    return np.log(df)
  
  #standardized values for columns
  if standardize:
    for ii, ij in enumerate(df.columns):
      print(ii, ij)
      df['{}'.format(ij)] = mean_dev(df.loc[:, '{}'.format(ij)])/stdev(df.loc[:, '{}'.format(ij)])
      df = df.replace([np.inf, -np.inf, np.nan], 0)
  elif logg:
    df = logg_dat(df)
    df = df.replace([np.inf, -np.inf, np.nan], 0)
  elif normalize:
    for ii, ij in enumerate(df.columns):
      df['{}'.format(ij)] = (df.loc[:, '{}'.format(ij)] - min(df.loc[:, '{}'.format(ij)]))/\
      (max(df.loc[:, '{}'.format(ij)]) - min(df.loc[:, '{}'.format(ij)]))
      df = df.replace([np.inf, -np.inf, np.nan], 0)
  else:
    pass
    
  return df

lower_quart = .25
upper_quart = .75
multiplier = 1.5
df_no_out = remove_outliers(hosue_df, remove_objects = True, lower_quartile = lower_quart, upper_quartile = upper_quart, multiplier = multiplier)
df_standard_no_out = remove_outliers(hosue_df, remove_objects = True, standardize = True, lower_quartile = lower_quart, upper_quartile = upper_quart, multiplier = multiplier)
log_data_no_out = remove_outliers(hosue_df, remove_objects = True, logg=True, lower_quartile = lower_quart, upper_quartile = upper_quart, multiplier = multiplier)
df_normal_no_out = remove_outliers(hosue_df, remove_objects = True, normalize = True, lower_quartile = lower_quart, upper_quartile = upper_quart, multiplier = multiplier)

#%%
from scipy import stats
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

#plot feature importance
def plot_features(model):
  figsize = [20, 16]
  fig, ax = plt.subplots(1, 1, figsize = figsize)
  return plot_importance(model)


def train_test(df, split = None, test_siz = None):
  if not split:
    return df.Price.values, df.drop(['Price'], axis = 1)
  else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(['Price'], axis = 1), df.price.values, test_size = test_siz)
#    X_train, X_test = standize_it(X_train), standize_it(X_test)
    return X_train, X_test, Y_train, Y_test

df_y, df_X = train_test(df_standard_no_out)


def Grid_Search_CV_RFR(X_train, y_train):
  #model
  model = XGBRegressor()
  #parameters
  param_grid = { 
          "n_estimators" : [10,20,30, 50],
          'max_depth': [4, 5, 6],
          'min_child_weight': [11],
          }

  grid = GridSearchCV(model, param_grid,
                      cv=StratifiedKFold(df_y, n_folds=10, shuffle=True),
                      n_jobs=-1)

  grid.fit(df_X, df_y)
  return grid.best_estimator_, grid.best_score_ , grid.best_params_

estimator, score_, params_ = Grid_Search_CV_RFR(df_X, df_y)

#plot importance
plot_features(estimator)

#%% MODELING 
#Modeling
X_train, X_test, Y_train, Y_test = train_test(df_standard, split = True, test_siz = 0.3)

def predic_prices(X_train, X_test, Y_train, Y_test):
  model = XGBRegressor()
  #parameters
  param_grid = { 
          "n_estimators" : [10,20,30, 50],
          'max_depth': [4, 5, 6],
          'min_child_weight': [11],
          }

  grid = GridSearchCV(model, param_grid,
                      cv=StratifiedKFold(Y_train, n_folds=10, shuffle=True),
                      n_jobs=-1)

  grid.fit(X_train, Y_train)
  prediction = grid.predict(X_test)
  return prediction

predicted = predic_prices(X_train, X_test, Y_train, Y_test)