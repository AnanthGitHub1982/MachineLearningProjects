# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:55:00 2020

@author: ananth.vaidyanathan

Course: CS 767 - Machine Learning

Boston University MET - Applied Data Analytics
"""

#pip install --upgrade tensorflow
#pip install numpy scipy
#pip install scikit-learn
#pip install pillow
#pip install h5py
#pip install keras
#pip install --ignore-installed --upgrade tensorflow-gpu
#pip install sklearn-genetic

from numba import jit, cuda 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras import layers
import random as python_random

import os 
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
    
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def reset_seeds():
   np.random.seed(123) 
   python_random.seed(123)
   tf.random.set_seed(1234)
   
try:
    # Open the dataset from the CSV input file from the <Current Working Directory>\datasets
    here =os.path.dirname(os.path.abspath("__file__"))
    input_dir = os.path.abspath (here) 
    input_file = os.path.join(input_dir, 'kc_house_data.csv')
   
    # Load from the Comma Separated Input file
    housing_data = pd.read_csv(input_file)

    #Exploratory data analysis

    # Check for null values
    housing_data.isnull().sum()
    housing_data.info()

    # Distribution of numerical features across samples
    housing_data.describe().transpose()
    
    housing_data['date'] = pd.to_datetime(housing_data['date'])
    housing_data['month'] = housing_data['date'].apply(lambda date:date.month)
    housing_data['year'] = housing_data['date'].apply(lambda date:date.year)
    housing_data = housing_data.drop('date',axis=1)

    # House price distribution- outliers?
    
    fig = plt.figure(figsize=(10,7))
    fig.add_subplot(2,1,1)
    sns.distplot(housing_data['price'])
    fig.add_subplot(2,1,2)
    sns.boxplot(housing_data['price'])
    plt.tight_layout()

    # Trend based on Year and months
    
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    sns.boxplot(x='year',y='price',data=housing_data, ax=axes[0])
    sns.boxplot(x='month',y='price',data=housing_data, ax=axes[1])
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Year', ylabel='Price', title='Price by Year')
    axes[1].set(xlabel='Month', ylabel='Price', title='Price by Month')
    
    f, axe = plt.subplots(1, 1,figsize=(15,5))
    housing_data.groupby('month').mean()['price'].plot()
    sns.despine(left=True, bottom=True)
    axe.set(xlabel='Month', ylabel='Price', title='Price Trends')
    
    fig = plt.figure(figsize=(16,5))
    fig.add_subplot(1,2,1)
    housing_data.groupby('month').mean()['price'].plot()

    # Correlation between parameters
    
    sns.set(style="whitegrid", font_scale=1)
    
    plt.figure(figsize=(13,13))
    plt.title('Correlation Matrix',fontsize=25)
    sns.heatmap(housing_data.corr(),linewidths=0.25,vmax=0.7,linecolor='w',
                annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})

    price_corr = housing_data.corr()['price'].sort_values(ascending=False)
    print(price_corr)

    # Impact of month and year on price. Mean price is same and hence might remove it.

    f, axes = plt.subplots(1, 2,figsize=(15,5))
    sns.boxplot(x='year',y='price',data=housing_data, ax=axes[0])
    sns.boxplot(x='month',y='price',data=housing_data, ax=axes[1])
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Year', ylabel='Price', title='Price by Year Box Plot')
    axes[1].set(xlabel='Month', ylabel='Price', title='Price by Month Box Plot')
    
    # Drop some features - ID and Zip code
   
    housing_data = housing_data.drop('id',axis=1)
    housing_data = housing_data.drop('zipcode',axis=1)

    # Splitting Train and Test     
    X = housing_data.drop('price',axis =1).values
    Y = housing_data['price'].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=101)

    
    # Scale the input data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)   
    

    # Neural networks training model:
    reset_seeds() # Set the random seed to get consistent results between runs
    model = Sequential()
    model.add(Dense(19,activation='relu'))
    model.add(Dense(14,activation='relu'))
    model.add(Dense(14,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    
    a = datetime.datetime.now()
    
    model.fit(x=X_train,y=Y_train,
              validation_data=(X_val,Y_val),
              batch_size=128,epochs=500)
    model.summary()

    b = datetime.datetime.now()

    print("Time for training =", b-a)
    
    loss_df = pd.DataFrame(model.history.history)
    loss_df.plot(figsize=(12,8))

    plt.show()
    
    Y_pred = model.predict(X_test)


    # visualizing residuals
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    
    # Our model predictions
    plt.scatter(Y_test,Y_pred)
    
    # Perfect predictions
    plt.plot(Y_test,Y_test,'r')
    
    errors = Y_test - Y_pred
    sns.distplot(errors, ax=axes[0])
    
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
    axes[1].set(xlabel='Target Y', ylabel='Actual Y [Predicted]', title='Predicted vs Target output')     
    
    # Evaluate the performance of the algorithm (MAE - MSE - RMSE - R2)
    r2 = r2_score(Y_test, Y_pred)
    adj_r2 = 1 - (1-r2)*(len(housing_data) - 1) / (len(housing_data) - (housing_data.shape[1] - 1) - 1)

    print('MAE:', round(metrics.mean_absolute_error(Y_test, Y_pred),2))  
    print('MSE:', round(metrics.mean_squared_error(Y_test, Y_pred),2))  
    print('RMSE:', round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2))
    print('R2 Score [%]:', round(r2*100,2))    
    print('Adjusted R2 Score [%]:', round(adj_r2*100,2))    
    
    # Multiple Liner Regression model
    
    regressor = LinearRegression()  
    regressor.fit(X_train, Y_train)
    
    #evaluate the model (intercept and slope)
    print("Intercept:", regressor.intercept_)
    print("Coefficients", regressor.coef_)
    #predicting the test set result
    Y_pred = regressor.predict(X_test)
    
    
    # visualizing residuals
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    
    # Our model predictions
    plt.scatter(Y_test,Y_pred)
    
    # Perfect predictions
    plt.plot(Y_test,Y_test,'r')
    
    errors = Y_test - Y_pred
    sns.distplot(errors, ax=axes[0])
    
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
    axes[1].set(xlabel='Target Y', ylabel='Actual Y [Predicted]', title='Predicted vs Target output')   
    
    # Evaluate the performance of the algorithm (MAE - MSE - RMSE - R2)
    r2 = r2_score(Y_test, Y_pred)
    adj_r2 = 1 - (1-r2)*(len(housing_data) - 1) / (len(housing_data) - (housing_data.shape[1] - 1) - 1)

    print('MAE:', round(metrics.mean_absolute_error(Y_test, Y_pred),2))  
    print('MSE:', round(metrics.mean_squared_error(Y_test, Y_pred),2))  
    print('RMSE:', round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2))
    print('R2 Score [%]:', round(r2*100,2))    
    print('Adjusted R2 Score [%]:', round(adj_r2*100,2))    

    
    # Feature selection
    housingdata_temp = housing_data.drop('price',axis =1)
    
    # 1) Find all features with more than 90% variance in values:
    
    threshold = 0.90
    vt = VarianceThreshold().fit(X_train)
    feat_var_threshold = housingdata_temp.columns[vt.variances_ > threshold * (1-threshold)]
    print("Features with > 90% Variance:", feat_var_threshold[0:20])
 
    # 2) Select K best features
    X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train, Y_train)
    feature_scoring = pd.DataFrame({
            'feature': housingdata_temp.columns,
            'score': X_scored.scores_
        })
    
    feat_scored_10= feature_scoring.sort_values('score', ascending=False).head(10)['feature'].values
    feat_scored_10    
    
    X_train_fs, X_test_fs, fs = select_features(X_train, Y_train, X_test)
    for i in range(len(fs.scores_)):
    	print('Feature %d [%s]: %f' % (i+1, housingdata_temp.columns[i], fs.scores_[i]))
        
        
    features = list(housingdata_temp.columns) 
    scores = fs.scores_ 
       
    fig = plt.figure(figsize = (10, 5)) 
      
    # creating the bar plot 
    plt.bar(features, scores, color ='maroon',  
            width = 0.4) 
      
    plt.xlabel("Features") 
    plt.ylabel("Scores") 
    plt.title("Scores for different features") 
    plt.xticks(rotation=90)
    plt.show() 
    
    
    # Drop the features with low k values and > 90% Variance - 'long', 'month', 'year'
    features_to_drop = {'long', 'month', 'year'}
    housing_data_wo_features = housing_data.copy(True)
    housing_data_wo_features = housing_data_wo_features.drop(features_to_drop,axis=1)        
    
    X = housing_data_wo_features.drop('price',axis =1).values
    Y = housing_data_wo_features['price'].values
    
    # Splitting Train and Test 
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=101)

    scaler = MinMaxScaler()
    
    # Scale the input data
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)   
    
    reset_seeds() # Set the random seed to get consistent results between runs
    
    # Neural networks:
    model = Sequential()
    model.add(Dense(12,activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    
    a = datetime.datetime.now()
    

    model.fit(x=X_train,y=Y_train,
              validation_data=(X_val,Y_val),
              batch_size=128,epochs=500)
    model.summary()

    b = datetime.datetime.now()

    print("Time for training =", b-a)
    
    loss_df = pd.DataFrame(model.history.history)
    loss_df.plot(figsize=(12,8))

    plt.show()
    
    Y_pred = model.predict(X_test)


    # visualizing residuals
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    
    # Our model predictions
    plt.scatter(Y_test,Y_pred)
    
    # Perfect predictions
    plt.plot(Y_test,Y_test,'r')
    
    errors = Y_test - Y_pred
    sns.distplot(errors, ax=axes[0])
    
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
    axes[1].set(xlabel='Target Y', ylabel='Actual Y [Predicted]', title='Predicted vs Target output')     
    
    # Evaluate the performance of the algorithm (MAE - MSE - RMSE - R2)
    r2 = r2_score(Y_test, Y_pred)
    adj_r2 = 1 - (1-r2)*(len(housing_data) - 1) / (len(housing_data) - (housing_data.shape[1] - 1) - 1)

    print('MAE:', round(metrics.mean_absolute_error(Y_test, Y_pred),2))  
    print('MSE:', round(metrics.mean_squared_error(Y_test, Y_pred),2))  
    print('RMSE:', round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2))
    print('R2 Score [%]:', round(r2*100,2))    
    print('Adjusted R2 Score [%]:', round(adj_r2*100,2))   
    

    # Multiple Liner Regression model

    regressor = LinearRegression()  
    regressor.fit(X_train, Y_train)
    
    #evaluate the model (intercept and slope)
    print("Intercept:", regressor.intercept_)
    print("Coefficients", regressor.coef_)
    #predicting the test set result
    Y_pred = regressor.predict(X_test)
    
    
    # visualizing residuals
    f, axes = plt.subplots(1, 2,figsize=(15,5))
    
    # Our model predictions
    plt.scatter(Y_test,Y_pred)
    
    # Perfect predictions
    plt.plot(Y_test,Y_test,'r')
    
    errors = Y_test - Y_pred
    sns.distplot(errors, ax=axes[0])
    
    sns.despine(left=True, bottom=True)
    axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
    axes[1].set(xlabel='Target Y', ylabel='Actual Y [Predicted]', title='Predicted vs Target output')   
    
    # evaluate the performance of the algorithm (MAE - MSE - RMSE - R2)
    r2 = r2_score(Y_test, Y_pred)
    adj_r2 = 1 - (1-r2)*(len(housing_data) - 1) / (len(housing_data) - (housing_data.shape[1] - 1) - 1)

    print('MAE:', round(metrics.mean_absolute_error(Y_test, Y_pred),2))  
    print('MSE:', round(metrics.mean_squared_error(Y_test, Y_pred),2))  
    print('RMSE:', round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2))
    print('R2 Score [%]:', round(r2*100,2))    
    print('Adjusted R2 Score [%]:', round(adj_r2*100,2))     
    
    """
    # GA based feature selection -- under development and commented out for now.
    
    a = datetime.datetime.now()    
    print("Start Time for feature selection =", a)
    
    estimator = MLPRegressor(hidden_layer_sizes=(14,14),
                             activation='relu',
                             solver='adam',
                             alpha=0.0001,
                             batch_size='auto',
                             learning_rate='constant',
                             learning_rate_init=0.001,
                             power_t=0.5,
                             max_iter=1000,
                             shuffle=True,
                             random_state=1,
                             tol=0.0001,
                             verbose=False,
                             warm_start=False,
                             momentum=0.9,
                             nesterovs_momentum=True,
                             early_stopping=False,
                             validation_fraction=0.1,
                             beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-08)

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="r2",
                                  max_features=5,
                                  n_population=19,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=20,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X_train, Y_train)
    
    
    print(selector.support_)

    b = datetime.datetime.now()
    print("End Time for feature selection =", b)
    
    print("Time for feature selection =", b-a)    
    
    """

# =============================================================================
except Exception as e:
    print(e)
     

