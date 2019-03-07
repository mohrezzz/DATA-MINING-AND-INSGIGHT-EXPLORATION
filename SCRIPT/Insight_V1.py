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
from pylab import rcParams
#stop runtime error
np.seterr(divide='ignore', invalid='ignore')
rcParams['figure.figsize'] = 20, 25

path = 'D:\\FREELANCER\\DATAMINING AND INSIGHTHOUSE PRICES'
os.chdir(path)
hosue_df = pd.read_csv(os.path.join('DATASET', 'Al-Muzahmiyya.csv'))
hosue_df['last_updated'] = pd.to_datetime(hosue_df.last_updated)
hosue_df = hosue_df.iloc[:, 1:]
hosue_df = hosue_df.drop(['created_at', 'address'], axis = 1)
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

#%% Dealing withh outliers
log_data.describe()
#plot log_price
after_outl = log_data[(log_data.price < 20.0) & (log_data.price > 2.5)]
plt.scatter(np.arange(after_outl.shape[0]), after_outl.price, s = .5)
plt.title('Plot of count against price on a log scale')
plt.axhline(y = 20, linewidth=1, color='r')
plt.axhline(y = 2.5, linewidth=1, color='r')
plt.axhline(y = 12.159753818376581, linewidth=1, color='r')
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
sns.heatmap(df_standard.corr(), annot=True);plt.show()
sns.heatmap(log_data.corr(), annot=True);plt.show()
sns.heatmap(df_normal.corr(), annot=True);plt.show()

#%%
#box plot
log_data.plot(kind='box')
plt.title('log data contaning outliers')
log_data.groupby('price').mean().plot()
plt.title('line chart of features against price')
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
sns.lmplot('street_width', 'price', log_data)

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

sns.lmplot('area', 'price', log_data)
plt.scatter(df.iloc[:, [3]], df.iloc[:, [0]], s = .5)


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
    dum = pd.get_dummies(df, dtype = float)
    
    
    
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
#df_dum_dum = remove_outliers(hosue_df, remove_objects = False, standardize = True, lower_quartile = lower_quart, upper_quartile = upper_quart, multiplier = multiplier)

plt.scatter(np.arange(df_no_out.shape[0]), df_no_out.price, s = 1.5)
sns.lmplot('area', 'price', df_no_out)

#%% plots with and without outliers
#plot log_price
rcParams['figure.figsize'] = 20, 14
plt.scatter(np.arange(log_data_no_out.shape[0]), log_data_no_out.price, s = 2.5)
plt.title('Plot of count against price on a log scale without outliers')
plt.axhline(y = 20, linewidth=1, color='r')
plt.axhline(y = 2.5, linewidth=1, color='r')
plt.axhline(y = 12.159753818376581, linewidth=1, color='r')

##plot log_price  using price range
plt.scatter(np.arange(after_outl.shape[0]), after_outl.price, s = 2.5)
plt.title('Plot of count against price on a log scale with outliers')
plt.axhline(y = 20, linewidth=1, color='r')
plt.axhline(y = 2.5, linewidth=1, color='r')
plt.axhline(y = 12.159753818376581, linewidth=1, color='r')

#%% plot price with all other numeric features

def plotit(df):
  if 'price' in df.columns:
    for ii in df.columns[1:]:
      plt.scatter(df['price'], df[ii], cmap='Sequential', s = 2.5)
      plt.legend()
      plt.ylabel('Price')
      plt.xlabel('features')
      plt.title('Price against other numerical variables')
plotit(log_data_no_out)

#%% Feature engineering/ selection
from scipy import stats
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#plot feature importance
def plot_features():
  fig, ax = plt.subplots(1, 1, figsize = figsize)
  return plot_importance()


def categorical_handler(df, standardize = None, remove_objects = True,
                    logg = None, normalize = None, 
                    lower_quartile = None, upper_quartile = None, multiplier = None):
  df_dum = hosue_df.copy(deep = True)
  df_num = hosue_df.copy(deep = True)
  #seperate numerical variables
  for ii in df_num.columns:
    if df_num[ii].dtypes == object:
      df_num = df_num.drop(ii, axis = 1)
  #seperate categories
  for ii in df_dum.columns:
    if df_dum[ii].dtypes != object:
      df_dum = df_dum.drop(ii, axis = 1)
  
  
  
  quart_1 = df_num.quantile(lower_quart)
  quart_2 = df_num.quantile(upper_quart)
  diff_quart = abs(quart_1 - quart_2)
  df_num = df_num[~((df_num < (quart_1 - 1.5 * diff_quart)) | (df_num > (quart_2 + 1.5 * diff_quart))).any(axis=1)]
  
  df_dum = pd.get_dummies(df_dum, dtype = float)
  #create additional time features
  df_num['date'] = df_num.index.date
  df_num['time'] = df_num.index.time
  df_num['day'] = df_num['date'].map(str) + df_num['time'].map(str)
  df_dum['date'] = df_dum.index.date
  df_dum['time'] = df_dum.index.time
  df_dum['day'] = df_dum['date'].map(str) + df_num['time'].map(str)
  #concat
  df = pd.concat([df_num, df_dum], axis = 1)
  df_new = pd.merge(df_num, df_dum, left_on = df_num.index, right_on = df_dum.index, how = "right")
  df_new = df_num.merge(df_dum, how = 'right')
  
  print(len([x for x in df_dum.index]))
  print(len([x for x in df_num.index]))
  print(len([x for x in df_dum.index if x in df_num.index]))
  df_new = df_dum[[x for x in df_dum.index if x in df_num.index]]
  for ij in df_dum.index:
    print(len(ij))
    for ii in df_num.index:
      print(ii)
  
  for ii in df_dum.index:
    for ij in df_num.index:
      if ii == ij:
        pass
 
  #join both dataframes on index
  concatt_frame = pd.merge(df_num, df_dum, indicator=True)
  concatt_frame = pd.concat([df_num, df_dum], axis = 1, join_axes = [df_num.index])
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

#get the standard catefgorical variable
df_cat_stan = categorical_handler(hosue_df, standardize = True, lower_quartile = lower_quart, \
                             upper_quartile = upper_quart, multiplier = multiplier)
#standardize df_dummy





#%% Analysis