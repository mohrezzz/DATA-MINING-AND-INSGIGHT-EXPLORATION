#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:40:10 2019

@author: kenneth
"""

import numpy as np
import pandas as pd

class Regression(object):
    def __init__(self):
        return
    
    def fit_predict(self, X, Y):
        self.X = X
        self.Y = Y
        #--either beta syntax below would work
        #beta = np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.Y))
        beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)
        Y_hat = self.X.dot(beta)
        return Y_hat
    
    #-Mean Square Error
    def RMSE(self, yh, y):
        return np.sqrt(np.square(yh - y).mean())
    #-Mean Square Error
    def MSE(self, yh, y):
        return np.square(yh - y).mean()
    #-Mean Absolute Error
    def MAE(self, yh, y):
        return np.abs(yh - y).mean()
    #-R-squared Error
    def R_squared(self, yh, y):
        #-- R_square = 1 - (SS[reg]/SS[total])
        # 1 - (y-yh).dot(y-yh)/(y - y.mean()).dot(y - y.mean()) OR
        return (1 -(np.sum(np.square(y - yh))/np.sum(np.square(y - y.mean()))))
        
    def summary(self, X, y, y_hat):
        #y_hat = self.fit_predict(self.X, self.Y)
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('RMSE: %s'%(self.RMSE(y_hat,  Y)))
        print('*'*40)
        print('MSE: %s'%(self.MSE(y_hat,  Y)))
        print('*'*40)
        print('MAE: %s'%(self.MAE(y_hat,  Y)))
        print('*'*40)
        print('R_squared = %s'%(self.R_squared(y_hat,  Y)))
        print('*'*40)
    
    def plot(self, X, Y, y_hat):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(X.shape[0]), Y)
        plt.plot(np.arange(X.shape[0]), y_hat)
        plt.legend(loc = 2)
        plt.title('True vlaue vs Predicted value')
        plt.xlabel('Data point')
        plt.ylabel('True vlaue vs Predicted value')
        
class GradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def GD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        cost_rec = np.zeros(iterations)
        beta_rec = np.zeros((iterations, X.shape[1]))
        if early_stopping:
            for ii in range(iterations):
                #compute gradient
                beta = beta - (1/len(Y)) *(alpha) * (np.dot(X.T, (np.dot(X,beta) - Y)))
                beta_rec[ii, :] = beta.T
                cost_rec[ii] = self.cost(X, Y, beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))
                #--compare last and previous value. stop if they are the same
                if not cost_rec[ii] == cost_rec[ii -1]:
                    continue
                else:
                    break
            y_hat = X.dot(beta)
            return beta, cost_rec[:ii], beta_rec, y_hat, ii
        else:
            for ii in range(iterations):
                #compute gradient
                beta = beta - (1/len(Y)) *(alpha) * (np.dot(X.T, (np.dot(X,beta) - Y)))
                beta_rec[ii, :] = beta.T
                cost_rec[ii] = self.cost(X, Y, beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))
            print('*'*40)
            y_hat = X.dot(beta)
            return beta, cost_rec, beta_rec, y_hat, ii
        
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
class StochasticGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def StochGD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        cost_rec = np.zeros(iterations)
        len_y = len(Y)
        if early_stopping:
            for ii in range(iterations):
                #compute gradient
                cost_val = []
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    beta = beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp,beta) - Y_samp)))
                    cost_val.append(self.cost(X_samp, Y_samp, beta))
                    if cost_val[ij] == cost_val[ij -1]:
                        break
                    else:
                        continue
                cost_rec[ii] = np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))    
                #--compare last and previous value. stop if they are the same
                if not cost_rec[ii] == cost_rec[ii -1]:
                    continue
                else:
                    break
            print('*'*40)
            y_hat = X.dot(beta)
            return beta, cost_rec[:ii], y_hat, ii
        else:
            for ii in range(iterations):
                #compute gradient
                cost_val = 0.0
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    beta = beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp,beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, beta)
                cost_rec[ii] = cost_val
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))    
            print('*'*40)
            y_hat = X.dot(beta)
            return beta, cost_rec, y_hat, ii
        
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
class MinibatchGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def minbatchGD(self, X, Y, beta, alpha, iterations, batch_size = None, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        cost_rec = np.zeros(iterations)
        len_y = len(Y)
        number_batches = int(len_y/batch_size)
        if early_stopping:
            for ii in range(iterations):
                cost_val = 0
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    beta = beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp,beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, beta)
#                    if cost_val[ij] == cost_val[ij -1]:
#                        break
#                    else:
#                        continue
                cost_rec[ii] = cost_val #np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))    
                #--compare last and previous value. stop if they are the same
                if not cost_rec[ii] == cost_rec[ii -1]:
                    continue
                else:
                    break
            print('*'*40)
            y_hat = X.dot(beta)
            return beta, cost_rec[:ii], y_hat, ii
        else:
            for ii in range(iterations):
                cost_val = 0
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    beta = beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp,beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, beta)
                cost_rec[ii] = cost_val 
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, cost_rec[ii]))    
            print('*'*40)
            y_hat = X.dot(beta)
            return beta, cost_rec, y_hat, ii
        
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
#%%

from collections import Counter
def extractFeatures(model, k_features, fscore = None):
    features = Counter(model.get_booster().get_score())
    features = features.most_common(k_features)
    features = [x[0] for x in features if x[1] >= fscore]
    return features

features = extractFeatures(estimator, 20, 1)
features = ['Area m2',
             'District_id',
             'Street Width',
             'Driver Room',
             'Extra Unit',
             'Apartments',
             'Bed Rooms',
             'WC',
             'With Stairs',
             'Living Rooms',
             'Servant Room']

import statsmodels.api as sm
X = df_standard_no_out[features]
X = np.c_[np.ones((X.shape[0], 1)), X]    
Y = df_standard_no_out[['Price']].values

#--Multivariant regression
lm = Regression()
yhat = lm.fit_predict(X, Y)
lm.summary(X, Y, yhat)
gd.plot(X[:200], Y[:200], y_hat[:200])

#--Gradient descent
iterations = 1000
gd = GradientDescent()
beta,cost_rec,theta_rec, yhat, stopping = gd.GD(X, Y, beta = np.zeros(X.shape[1]).reshape(-1, 1), alpha = 0.1, iterations = iterations, early_stopping=True)
gd.summary(X, Y, yhat)
gd.plot_cost(cost_rec, stopping)

#--stochastic gradient descent
stgrad = StochasticGradientDescent()
beta,cost_rec, yhat, stopping = stgrad.StochGD(X, Y, beta = np.zeros(X.shape[1]).reshape(-1, 1), alpha = 0.8, iterations = iterations, early_stopping=True)
stgrad.summary(X, Y, yhat)
stgrad.plot_cost(cost_rec, stopping)

#--stochastic gradient descent
minibatch = MinibatchGradientDescent()
beta,cost_rec, yhat, stopping = minibatch.minbatchGD(X, Y, beta = np.zeros(X.shape[1]).reshape(-1, 1), alpha = 0.01, iterations = iterations, batch_size = 20, early_stopping=True)
minibatch.summary(X, Y, yhat)
minibatch.plot_cost(cost_rec, stopping)

#%% ADDISTIONAL FEATURES FOR REGRESSION






















