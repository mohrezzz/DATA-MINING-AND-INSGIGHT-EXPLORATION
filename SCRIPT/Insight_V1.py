# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:57:52 2019

@author: kennedy
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


path = 'D:\\FREELANCER\\DATAMINING AND INSIGHTHOUSE PRICES'
os.chdir(path)
hosue_df = pd.read_csv(os.path.join('DATASET', 'Al-Muzahmiyya.csv'))
hosue_df['last_updated'] = pd.to_datetime(hosue_df.last_updated)
hosue_df.set_index('last_updated', inplace = True)
hosue_df.sort_values(by = 'last_updated', inplace = True)
#sort the data
print('See data descroiption: {}'.format(hosue_df.describe()))
print('Skew of data: {}'.format(hosue_df.skew()))
print('Kurt of data: {}'.format(hosue_df.kurt()))
hosue_df.hist()


#standardize numeric dataset
def standardize_houseprize(df, standardize = None, 
                           logg = None, normalize = None):
  df = df.copy(deep = True)
  #drop all objects
  #and leaving all float64 and int64 datatypes
  for ii in hosue_df.columns:
    if hosue_df[ii].dtype == object:
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

df = standardize_houseprize(hosue_df.iloc[:, 1:])
df_standard = standardize_houseprize(hosue_df.iloc[:, 1:], standardize = True)
log_data = standardize_houseprize(hosue_df.iloc[:, 1:], logg=True)
df_normal = standardize_houseprize(hosue_df.iloc[:, 1:], normalize = True)


#%% Dealing withh outliers
log_data.describe()
#plot log_price
after_outl = log_data[(log_data.price < 20.0) & (log_data.price > 2.5)]
plt.scatter(np.arange(after_outl.shape[0]), after_outl.price, s = .5)
plt.axhline(y = 20, linewidth=1, color='r')
plt.axhline(y = 2.5, linewidth=1, color='r')
#%% plots and correlation

df_standard.hist()
log_data.hist()
df_normal.hist()
df.hist()
sns.violinplot(df_standard)
sns.pairplot(df_standard)
sns.pairplot(log_data)
sns.pairplot(df_normal)
sns.heatmap(df_standard)
sns.heatmap(log_data)
sns.heatmap(df_normal)
sns.heatmap(df_standard.corr(), annot=True);plt.show()
sns.heatmap(log_data.corr(), annot=True);plt.show()
sns.heatmap(df_normal.corr(), annot=True);plt.show()
#box plot
log_data.plot(kind='box')
df.groupby('price').mean().plot()
plt.title('Plot of features against price')
#data exploration
color = ['red', 'green', 'brown', 'black', 'blue', 'indigo']
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex= True)
ax1.scatter(log_data.index, log_data.price.values, s = .5, color = color[1], label='Price')
ax1.legend()
ax2.scatter(log_data.index, log_data.meter_price.values, s = .5, color = color[2], label='meter_price')
ax2.legend()
ax3.scatter(log_data.index, log_data.area.values, s = .5, color = color[3], label='area')
ax3.legend()
ax4.scatter(log_data.index, log_data.wc.values, s = .5, color = color[4], label='wc')
ax4.legend()
ax5.scatter(log_data.index, log_data.street_width.values, s = .5, color = color[5], label='street_width')
ax5.legend()

#regression line
sns.lmplot('area', 'price', log_data)

#%%
#create syntetic variables
def moving_av(df, n):
  '''
  :params
    :df: feature, can be price, area or any numerical value 
    :n: period we want to check price
  '''
  return pd.DataFrame({'MA_'+str(n): df.rolling(n).mean()})

def expmoving_av(df, n):
  '''
  :params
    :df: feature, can be price, area or any numerical value 
    :n: period we want to check price
  '''
  return pd.DataFrame({'MA_'+str(n): df.ewm(n).mean()})


ma = moving_av(df.price, 2)
ema = expmoving_av(df.price, 2)

ma_log = moving_av(df.price, 2)

sns.lmplot('area', 'price', df)
plt.scatter(df.iloc[:, [3]], df.iloc[:, [0]], s = .5)


#%% Feature engineering/ selection

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
  
 












#%% Analysis