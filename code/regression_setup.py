#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:36:31 2017

@author: d29905p

COMMENT THESE FUNCTIONS MORE EXTENSIVELY 
"""

import pandas as pd
import os
from  sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np
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
    # DEMOGRAPHIC FEATURES
    ####### dONT FOREGET THOSE NEW FEATURES 
    #LOAD FEATURES
def data_load():
    session="ope35_m"
    years = range(2007,2017)   
    file_name = ("../processed_data/regression_files/regression_file_%s" % session)+ "_%s.csv"
    
    features = ['DAILY_FREQ','COMPETING_FREQ','SEG_PAX','GC_DISTANCE','SHARED_CITY_MARKET_ID_SEGS','ENPLANEMENTS_LARGER','ENPLANEMENTS_SMALLER','EDGE_COUNT','NODE_COUNT','IS_LOWCOST','MAIN_LINE','JACCARD_COEF','DEG_CENT_SMALLER','DEG_CENT_LARGER','SEG_PORTS_OCCUPIED']
    structural = ['YEAR','MONTH','QUARTER','UNIQUE_CARRIER','A1','A2','SEG_PRESENCE']
 
    fullfeat = ['DEP_PASSENGERS_AIRPORT_LARGER_ACR',
                'ENPLANEMENTS_LARGER_ACR',
                'DEP_CAPACITY_AIRPORT_LARGER_ACR',
                'DEP_PASSENGERS_AIRPORT_SMALLER_ACR',
                'ENPLANEMENTS_SMALLER_ACR',
                'DEP_CAPACITY_AIRPORT_SMALLER_ACR',
                'AVG_ENPLANEMENTS_ACR',
                'DIFF_ENPLANEMENTS_ACR',
                'RATIO_ENPLANEMENTS_ACR',
                'SEG_PORTS_OCCUPIED',
                'DEG_CENT_SMALLER',
                'DEG_CENT_LARGER',
                'DEPARTURES_SCHEDULED',
                'DEPARTURES_PERFORMED',
                'SEATS',
                'FLIGHT_TIME_CARRIER_WAVG',
                'SEATS_PER_FLIGHT_CARRIER_WAVG',
                'PASSENGERS',
                'DAILY_FREQ',
                'SEG_PAX_SHARE',
                'SEG_FREQ_SHARE',
                'SEG_CAPACITY_SHARE',
                'JACCARD_COEF',
                'SEG_PRESENCE',
                'SEG_PLAYERS',
                'SEG_PAX',
                'SEG_FREQ',
                'SEG_SEATS',
                'SEG_LOAD_FACTOR',
                'SEG_LOWCOST',
                'SEG_MAIN_LINE',
                'SEG_HHI',
                'IS_LOWCOST',
                'MAIN_LINE',
                'TRANSITIVITY',
                'EDGE_COUNT',
                'NODE_COUNT',
                'EDGE_NODE_RATIO',
                'CARRIER_FLIGHTS',
                'CARRIER_PAX',
                'CARRIER_SEATS',
                'CARRIER_SEATS_FLIGHT_RATIO',
                'CARRIER_WAVG_PORT_SIZE',
                'ALLIED_FREQ',
                'ALLIED_SEATS',
                'ALLIED_PAX',
                'ALLIED_PLAYERS',
                'MERGING_FREQ',
                'MERGING_SEATS',
                'MERGING_PLAYERS',
                'MERGING_PAX',
                'MERGING_SEGMENT',
                'OTHER_SEATS',
                'OTHER_FREQ',
                'OTHER_PAX',
                'OTHER_PLAYERS',
                'COMPETING_SEATS',
                'COMPETING_FREQ',
                'COMPETING_PLAYERS',
                'COMPETING_PAX',
                'OTHER_LOAD_FACTOR',
                'COMPETING_LOAD_FACTOR',
                'MERGING_LOAD_FACTOR',
                'ALLIED_LOAD_FACTOR',
                'OTHER_SEATS_PER_FLIGHT_AVG',
                'WAVG_MKT_DISTANCE',
                'PASSENGERS_CX',
                'COUNT_CX',
                'SEG_PAX_CXN_RATIO',
                'AVG_MKT_FARE',
                'AVG_MKT_FARE_0S',
                'AVG_MKT_FARE_1S',
                'PASSENGERS_CX_CR',
                'COUNT_CX_CR',
                'SEG_PAX_CXN_RATIO_CR',
                'CR_AVG_MKT_FARE',
                'CXN_RATIO_LARGER',
                'CXN_RATIO_SMALLER',
                'MAX_CXN_LARGER',
                'MAX_CXN_SMALLER',
                'GC_DISTANCE',
                'DEP_PASSENGERS_AIRPORT_LARGER',
                'ENPLANEMENTS_LARGER',
                'DEP_CAPACITY_AIRPORT_LARGER',
                'DEP_PASSENGERS_AIRPORT_SMALLER',
                'ENPLANEMENTS_SMALLER',
                'DEP_CAPACITY_AIRPORT_SMALLER',
                'AVG_ENPLANEMENTS_AR',
                'RATIO_ENPLANEMENTS_AR',
                'DIFF_ENPLANEMENTS_AR',
                'SHARED_CITY_MARKET_ID_SEGS']
    features = fullfeat
    full_df = []
    for year in years:
        full_df.append(pd.read_csv(file_name % year,usecols=structural+features))
    full_df = pd.concat(full_df).sort_values(['YEAR','MONTH'],ascending=True)
    return full_df

def lagged_vars(full_df,features):
    forecast_timestep = 'MONTH' ## IMPLLEMENT QUARTER AS WELL
    #dictionary of variables and lists of offsets to create
    lagged_features = {'SEG_PAX':[1,2],'DAILY_FREQ':[1,2],'COMPETING_FREQ':[1,2] }
    training_target_offset = 3 # one quarter ahead
    training_target = 'SEG_PRESENCE'
    #create predication (for training) variable
    lag_gb = full_df.groupby(['UNIQUE_CARRIER','A1','A2'])
    
    full_df[training_target+'_FORECAST'] = lag_gb[training_target].shift(-training_target_offset)
    #remove final date data (with no forecast values..)
    full_df = full_df.dropna(subset=[training_target+'_FORECAST'])
    ###full_df = pd.merge(full_df,shift_target.reset_index().drop('level_3',1),how='right',on=['UNIQUE_CARRIER','A1','A2'])
    lagged_var_list = []
    for key, vals in lagged_features.items():
        for val in vals:
            lag_gb = full_df.groupby(['UNIQUE_CARRIER','A1','A2'])
            full_df[key+'_LAG'+ str(val)] = lag_gb[key].shift(val)       
            full_df = full_df.dropna(subset=[key+'_LAG'+ str(val)])
            lagged_var_list.append(key+'_LAG'+ str(val))
    return full_df, lagged_var_list   

def diagnostic_regression(full_df,training_target,lagged_var_list,features):  
   
    X = full_df.loc[:,features +lagged_var_list].values
    Y = full_df.loc[:,[training_target+'_FORECAST']].values.flatten()
    #NORMALIZE DATA!!
  
    logreg = linear_model.LogisticRegression(class_weight ='balanced')
    #fit model with data
    logreg.fit(X,Y)
    Y_pred_train = logreg.predict(X)
    #make a coef printer
    return logreg, confusion_matrix(Y,Y_pred_train)

#create entries, then create entr window (i.e. entry in next six months)
def create_entries(df,pre_window=12,post_window=3,entry_col='ENTRY5_3',presence_col='SEG_PRESENCE_FORECAST'):
    #testing dataframe
    '''
    test_series = [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0]
    df = pd.DataFrame({presence_col: test_series +  test_series,'UNIQUE_CARRIER': [1]*len(test_series) + [2]*len(test_series), 'A1':[1]*(len(test_series)*2),'A2':[2]*(len(test_series)*2)})
    '''
    #moving avg calc
    df['mavg']=df.groupby(['UNIQUE_CARRIER','A1','A2'])[presence_col].transform(pd.rolling_mean,window = pre_window)
    df['entry_possible']=df.groupby(['UNIQUE_CARRIER','A1','A2'])['mavg'].shift(1)
    df['entry_pre_check'] = np.where( (df['entry_possible']==0) & (df['mavg']>0),1,0)
    df['presence_reverse'] = 1- df[presence_col]
    df['mavg_reverse']=df.groupby(['UNIQUE_CARRIER','A1','A2'])['presence_reverse'].transform(pd.rolling_mean,window = post_window)
    df['mavg_reverse_shift'] =  df.groupby(['UNIQUE_CARRIER','A1','A2'])['mavg_reverse'].shift(-(post_window-1))
    df[entry_col] = np.where((df['mavg_reverse_shift']==0) & (df['entry_pre_check']==1),1,0)
    #remove undetectable time steps
    df = df.dropna(subset=['entry_possible','mavg_reverse_shift'])
    del df['entry_possible'], df['presence_reverse'], df['mavg_reverse'], df['mavg_reverse_shift'], df['mavg'], df['entry_pre_check']
    return df


# presumably, want entry, plus segment presence after entry unless leave? combine these metrics
# remove frequency from presence predictors for the purpose of predicting entry?
# need a date indexer to plot entries, entries by year to decide evaluation properties, also want to enumerate stats about entry: frequency of entry, change of this frequency if any, changes in competitor frequencies, how long they stay in, have they every been in this segment before, 
# how many competitors on this segment/market, DB1B for market entry (including complex market entry, supplement with this information), was there activity by this airline in this segment/market, what changes in network did this induce, demographic information
# hierarchical model
#i.e. entry until exit





def train_test_previous_presence(df,features,lagged_var_list,target="PRESENCE_IN_12AHEAD",train_yr_range=list(range(2007,2012)),test_yr_range=list(range(2012,2014)),prev_check_col = "SEG_PRESENCE",algo_object=linear_model.LogisticRegression(class_weight ='balanced') ):
    
    ## WHY NANs on ull feature set?
    Xtrain =  df[df.YEAR.isin(train_yr_range)].loc[:,features +lagged_var_list].values
    Ytrain =  df[df.YEAR.isin(train_yr_range)].loc[:,[target]].values.flatten()
    Xtest =  df[df.YEAR.isin(test_yr_range)].loc[:,features +lagged_var_list].values
    Ytest =  df[df.YEAR.isin(test_yr_range)].loc[:,[target]].values.flatten()
    
    algo_object.fit(Xtrain,Ytrain)
    # overfall fit test 
    Y_pred_test = algo_object.predict(Xtest)
    full_confmat = confusion_matrix(Ytest,Y_pred_test)
    ## MAKE SIMPLER BY INDEXING MAYBE
    Xtest_prev0 =  df[(df.YEAR.isin(test_yr_range)) & (df[prev_check_col]==0)].loc[:,features +lagged_var_list].values
    Xtest_prev1 =  df[(df.YEAR.isin(test_yr_range)) & (df[prev_check_col]==1)].loc[:,features +lagged_var_list].values
    Ytest_prev0 =  df[(df.YEAR.isin(test_yr_range)) & (df[prev_check_col]==0)].loc[:,[target]].values.flatten()
    Ytest_prev1 =  df[(df.YEAR.isin(test_yr_range)) & (df[prev_check_col]==1)].loc[:,[target]].values.flatten()
  
    Yprev0_pred_test = algo_object.predict(Xtest_prev0)
    Yprev1_pred_test = algo_object.predict(Xtest_prev1)
    
    full_confmat = confusion_matrix(Ytest,Y_pred_test)
    full_confmat_prev_check = confusion_matrix(Ytest,df[df.YEAR.isin(test_yr_range)][prev_check_col].values)
    prev0_confmat = confusion_matrix(Ytest_prev0,Yprev0_pred_test)
    prev1_confmat = confusion_matrix(Ytest_prev1,Yprev1_pred_test)
    prev0_confmat_prev_check = confusion_matrix(Ytest_prev0,np.zeros(len(Ytest_prev0)))
    prev1_confmat_prev_check = confusion_matrix(Ytest_prev1,np.ones(len(Ytest_prev1)))
   
    
    return full_confmat, full_confmat_prev_check, prev0_confmat, prev0_confmat_prev_check, prev1_confmat, prev1_confmat_prev_check
    
    
def cross_validation():
    pass

#need entry in the future window, time to entry in the future window
#Q find all changes 0-> 1 in next month, year. What did paper do again?
# time to event and  detection 

#detect if seg presence or entry/exit or any other evetn occurs at some time in future from present (generalize to time to this event), with rolling sums on rolling sums, or whatever
# plot prediction accuracy as foreward window is modified
#TEST out_col should be identical with foreward_window of 1
def event_in_future(df,event_col="SEG_PRESENCE_FORECAST",out_col = "PRESENCE_IN_12AHEAD", foreward_window=12):
    df['flipped'] = df.groupby(['UNIQUE_CARRIER','A1','A2'])[event_col].transform(np.flip,axis=0)
    df['rolling_sum'] = df.groupby(['UNIQUE_CARRIER','A1','A2'])['flipped'].transform(pd.rolling_sum,window = foreward_window, min_periods=1)
    df[out_col] =  df.groupby(['UNIQUE_CARRIER','A1','A2'])['rolling_sum'].transform(np.flip,axis=0)
    df[out_col] = np.clip(df[out_col],0,1)
    return df

def range_test():
    for w in [1,5,12,18,24]:
         df = event_in_future(df,event_col="SEG_PRESENCE_FORECAST",out_col = "PRESENCE_IN_%sAHEAD" % w, foreward_window=w)
         full_confmat, full_confmat_prev_check, prev0_confmat, prev0_confmat_prev_check, prev1_confmat, prev1_confmat_prev_check = train_test_previous_presence(df,features,lagged_var_list,target="PRESENCE_IN_%sAHEAD" % w,train_yr_range=list(range(2007,2012)),test_yr_range=list(range(2012,2014)),prev_check_col = "SEG_PRESENCE",algo_object=linear_model.LogisticRegression(class_weight ='balanced') )
         print( prev0_confmat_prev_check[1,0]-prev0_confmat[1,0] , prev0_confmat[0,1])
def forecast_as_is():
    pass


def create_exits():
    pass

 
    
def exploratory():
    pass