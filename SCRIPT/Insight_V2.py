# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:16:26 2019

@author: kennedy
"""

import pandas as pd
import numpy as np
from os.path import dirname, join
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
#stop runtime error
np.seterr(divide='ignore', invalid='ignore')
rcParams['figure.figsize'] = 20, 25
__FILE__ = '/home/kenneth/Documents/GIT_PROJECTS/DATA-MINING-AND-INSGIGHT-EXPLORATION/' 
DATASET_ = join(dirname(__FILE__), 'DATASET')

hosue_df = pd.read_csv(join(DATASET_, 'Villas For Sale.csv'))
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
plt.scatter(np.arange(df_standard.shape[0]), df_standard.Price, s = .5)
#plt.axhline(y = .08e13, linewidth=1, color='r')
plt.title('Plot of count against price on a standardized scale')
#plt.axhline(y = 20, linewidth=1, color='r')
#plt.axhline(y = 2.5, linewidth=1, color='r')
#plt.axhline(y = 12.159753818376581, linewidth=1, color='r')
#%%

log_data.plot(kind='box')
#%% plots and correlation
hosue_df.hist()
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

#%%
x = 'District_id'
sns.countplot(x = x, data = hosue_df)
plt.title('Count of %s'%x)

x = 'District_id'
y = 'Living Rooms'
sns.countplot(x = x, data = hosue_df, hue = y)
plt.title('%s count vs %s'%(x, y))

#%%

df_standard.plot(kind='box')
plt.title('log data contaning outliers')

#data exploration
color = ['red', 'green', 'brown', 'black', 'blue', 'indigo']


#%%
def moving_av(df, n):
  '''
  :params
    :df: feature, can be price, area or any numerical value 
    :n: period we want to check price
  '''
  return pd.DataFrame({str(n)+'_day_average': df.rolling(n).mean()})

def expmoving_av(df, n):
  '''
  :params
    :df: feature, can be price, area or any numerical value 
    :n: period we want to check price
  '''
  return pd.DataFrame({'MA_'+str(n): df.ewm(n).mean()})


ma = moving_av(hosue_df.Price, 6)
ema = expmoving_av(hosue_df.Price, 6)
ma_plot = pd.concat([ma, hosue_df.Price], axis = 1)
ma_plot.plot()
#----------------
ma_log = moving_av(log_data.price, 6)
ma_log_plot = pd.concat([ma_log, log_data.price], axis = 1)
ma_log_plot.plot()

def plot_ma(df, n):
  ma_perd = moving_av(df.Price, n)
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
  ax1.plot(df.index, df.Price, lw = .5, color = color[1], label = 'Price')
  ax1.legend()
  ax2.plot(df.index, ma_perd, lw = .5, color = color[2], label = str(n)+'day_MA')
  ax2.legend()
  plt.title(str(n)+' day_Moving Average')
  
def plot_ma_all(df, n):
  ma = []
  for ii in n:
    ma.append(moving_av(df.Price, ii))
  fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex= True)
  ax1.plot(df.index, df.Price, lw = .5, color = color[1], label = 'Price')
  ax1.legend()
  ax1.set_title('Price')
  ax2.plot(df.index, ma[0], lw = .5, color = color[2], label = str(30)+'day_MA')
  ax2.legend()
  ax2.set_title(str(30)+' day_Moving Average')
  ax3.plot(df.index, ma[1], lw = .5, color = color[2], label = str(60)+'day_MA')
  ax3.legend()
  ax3.set_title(str(60)+' day_Moving Average')
  ax4.plot(df.index, ma[2], lw = .5, color = color[2], label = str(120)+'day_MA')
  ax4.legend()
  ax4.set_title(str(120)+' day_Moving Average')
  ax5.plot(df.index, ma[3], lw = .5, color = color[2], label = str(240)+'day_MA')
  ax5.legend()
  ax5.set_title(str(240)+' day_Moving Average')
  ax6.plot(df.index, ma[4], lw = .5, color = color[2], label = str(360)+'day_MA')
  ax6.legend()
  ax6.set_title(str(365)+' day_Moving Average')
  ax7.plot(df.index, ma[5], lw = .5, color = color[2], label = str(730)+'day_MA')
  ax7.legend()
  ax7.set_title(str(730)+' day_Moving Average')
  
  
plot_ma(log_data, 730)
plot_ma_all(log_data, [30, 60, 120, 240, 365, 730])
#------------------------------
ema_log = expmoving_av(log_data.price, 2)

sns.lmplot('area', 'price', log_data)
plt.scatter(df.iloc[:, [3]], df.iloc[:, [0]], s = .5)

#%% MACD

def ema(df, n):
  '''
  :params
    :df: feature, can be price, area or any numerical value 
    :n: duration we want to check price
  '''
  return df.ewm(n).mean()
def MACD(price, n_fast, n_slow, signal):
    '''
    :Arguments:
      :n_fast: <integer> representing fast exponential
              moving average
              
      :n_slow: <integer> representing slow exponential
              moving average
              
      :signal: Signal line
      
    :Return:
      MACD: fast, slow and signal.
    '''
    
    n_fast = n_fast
    n_slow = n_slow
    signal = signal
    #defin MACD
    macd = ema(price, n_fast) - ema(price, n_slow)
    #MACD signal
    macd_signal = ema(macd, signal)
    #MACD histo
    macd_histo_ = macd - macd_signal
    return pd.DataFrame({'MACD': macd, 'MACD_HIST': macd_histo_,
                         'MACD_SIGNAL': macd_signal})

FAST = 365
SLOW = 765
SIGNAL = 9
macd = MACD(log_data.Price, FAST, SLOW, SIGNAL)
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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

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
                      cv=StratifiedKFold(df_y, n_splits=10, shuffle=True),
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