#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:36:31 2017

@author: d29905p
"""

import pandas as pd
import os
from  sklearn import linear_model
from sklearn.metrics import confusion_matrix
try: 
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data')))
    
except NameError:
    
    os.chdir("C:/Users/d29905P/Dropbox/m106_backup/m106proj/")
    
def create_feature_categories():
    pass
    #CARRIER FEATURES
    #DB1B FEATURES
    #COOPERATIVE FEATURES
    #COMPETITIVE FEATURES
    #OWN FEATURES
    #OTHER FEATURES
    #SEGMENT FEATURES
    #CARRIER SEGMENT FEATURES
    #TEMPORAL FEATURES
    #CATEGORICAL FEATURES
    #NODE MULTIPLEX FEATURES
    #NODE CARRRIER FEATURES
    #LOAD FEATURES
def data_load():
    session="ope35_m
    years = range(2007,2017)   
    file_name = ("../processed_data/regression_file_%s" % session)+ "_%s.csv"
    
    features = ['DAILY_FREQ','COMPETING_FREQ','SEG_PAX','GC_DISTANCE','SHARED_CITY_MARKET_ID_SEGS','ENPLANEMENTS_LARGER','ENPLANEMENTS_SMALLER','EDGE_COUNT','NODE_COUNT','IS_LOWCOST','MAIN_LINE','JACCARD_COEF','DEG_CENT_SMALLER','DEG_CENT_LARGER','SEG_PORTS_OCCUPIED']
    structural = ['YEAR','MONTH','QUARTER','UNIQUE_CARRIER','A1','A2','SEG_PRESENCE']
    full_df = []
    for year in years:
        full_df.append(pd.read_csv(file_name % year,usecols=structural+features))
    full_df = pd.concat(full_df).sort_values(['YEAR','MONTH'],ascending=True)
    
    return full_df

def lagged_vars(full_df,features):
    forecast_timestep = 'MONTH'
    #dictionary of variables and lists of offsets to create
    lagged_features = {'SEG_PAX':[1,2],'DAILY_FREQ':[1,2],'COMPETING_FREQ':[1,2] }
    training_target_offset = 1
    training_target = 'SEG_PRESENCE'
    #create predication (for training) variable
    lag_gb = full_df.groupby(['UNIQUE_CARRIER','A1','A2'])
    
    full_df[training_target+'_FORECAST'] = lag_gb[training_target].shift(-training_target_offset)
    #remove final date data (with no forecast values..)
    full_df = full_df.dropna(subset=[training_target+'_FORECAST'])
    ###full_df = pd.merge(full_df,shift_target.reset_index().drop('level_3',1),how='right',on=['UNIQUE_CARRIER','A1','A2'])
    for key, vals in lagged_features.items():
        for val in vals:
            lag_gb = full_df.groupby(['UNIQUE_CARRIER','A1','A2'])
            full_df[key+'_LAG'+ str(val)] = lag_gb[key].shift(val)       
            full_df = full_df.dropna(subset=[key+'_LAG'+ str(val)])
    return full_df   

def diagnostic_regression(full_df,training_target,features):  
   
    X = full_df.loc[:,features].values
    Y = full_df.loc[:,[training_target+'_FORECAST']].values.flatten()
    #NORMALIZE DATA!!
  
    logreg = linear_model.LogisticRegression(class_weight ='balanced')
    #fit model with data
    logreg.fit(X,Y)
    Y_pred_train = logreg.predict(X)
    #make a coef printer
    return logreg, confusion_matrix(Y,Y_pred_train)

def create_entries():
  
    pass

def create_exits():
  
    pass

    
def exploratory():
    pass