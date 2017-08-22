# -*- coding: utf-8 -*-
"""
@author: Reed
"""
import os
import numpy as np
import pandas as pd
from itertools import product, combinations
from datetime import datetime
import ast
import pickle
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt
import itertools
import scipy.io as sio
import time
import functools
import networkx as nx
from math import radians, cos, sin, asin, sqrt, isnan
from  sklearn import linear_model

# set paths
try: 
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data')))
    
except NameError:
    
    os.chdir("C:/Users/d29905P/Dropbox/m106_backup/m106proj/")

#helper functions to count edges and nodes
def nx_edge_count(G):
        return len(G.edges())
def nx_node_count(G):
        return len(G.nodes())
   

# optional sets of nodes test network
small_node_sets = {'ope35_sansHLN' : ['ATL', 'BOS', 'BWI', 'CLE', 'CLT', 'CVG', 'DCA', 'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'IAD', 'IAH', 'JFK', 'LAS', 'LAX', 'LGA', 'MCO', 'MDW', 'MEM', 'MIA', 'MSP', 'ORD', 'PDX', 'PHL', 'PHX', 'PIT', 'SAN', 'SEA', 'SFO', 'SLC', 'STL', 'TPA'],
'western': ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC'],
'top100_2014':['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'IAH', 'SFO', 'PHX', 'LAS', 'SEA', 'MSP', 'BOS', 'SLC', 'EWR', 'MCO', 'DTW', 'LGA', 'CLT', 'JFK', 'MDW', 'BWI', 'SAN', 'MIA', 'PHL', 'DCA', 'TPA', 'HOU', 'FLL', 'IAD', 'PDX', 'STL', 'BNA', 'HNL', 'MCI','OAK', 'DAL', 'AUS', 'SJC', 'SMF', 'RDU', 'SNA', 'MSY', 'MKE', 'SAT', 'CLE', 'IND', 'PIT', 'SJU', 'ABQ', 'CMH', 'OGG', 'OKC', 'BDL', 'ANC', 'BUR', 'JAX', 'CVG', 'ONT', 'OMA', 'TUL', 'RIC', 'ELP', 'RSW', 'BUF', 'RNO', 'PBI', 'CHS', 'LIT', 'TUS', 'MEM','SDF', 'BHM', 'LGB', 'PVD', 'KOA', 'BOI', 'GRR', 'LIH', 'ORF', 'FAT', 'XNA', 'DAY', 'MAF', 'GEG', 'MSN', 'DSM', 'COS', 'GSO', 'TYS', 'ALB', 'SAV', 'PNS', 'BTR', 'ICT', 'ROC', 'JAN', 'MHT', 'AMA','FSD', 'HPN']}
TAFM2017AirportList  = pd.read_csv("TAFM2017AirportList.csv")
#get airport time invariant location information
airport_coordinates = pd.read_csv("airports.dat",header=None)
airport_coordinates = airport_coordinates.rename(columns={4:'ORIGIN',6:'LAT',7:'LON'})
lat_dict = airport_coordinates[['ORIGIN','LAT']].set_index('ORIGIN').T.to_dict('list')
lon_dict = airport_coordinates[['ORIGIN','LON']].set_index('ORIGIN').T.to_dict('list')
port_info =pd.read_csv("t100_2016_12ports.csv")[['ORIGIN','ORIGIN_STATE_ABR','ORIGIN_CITY_MARKET_ID']].drop_duplicates()

'''
main feature construction file
'''
def nonstop_market_profile_monthly(output_file = "market_profiles/monthly_market_profile_%sm%s.csv",year = 2007, months=range(1,13), \
    t100_fn="../bts_raw_csv/T100_SEGMENTS_%s.csv",p52_fn="../bts_raw_csv/SCHEDULE_P52_%s.csv", non_directional=True, t100_avgd_fn="processed_data/t100_avgd_m%s.csv", merge_HP=True, merge_NW=True, \
    t100_summed_fn = 'processed_data/t100_summed_m%s.csv', t100_craft_avg_fn='processed_data/t100_craft_avg_m%s.csv',\
    ignore_mkts = [], merger_carriers=True,  craft_freq_cuttoff = .01,max_competitors=100,top_n_carriers = 15,\
    freq_cuttoff = .5, ms_cuttoff=.05, time_step_size = 'MONTH',yearly_regression_file = "../processed_data/market_profile_%s.csv", fs_cuttoff = .05,db1b_fn="DB1B_MARKETS_%s_Q%s.csv", rc_name = 'related_carriers_dict_%s.csv', only_big_carriers=False, airports = TAFM2017AirportList.origin.tolist() , airport_data = TAFM2017AirportList, node_feature_list = [(nx.degree_centrality,'DAILY_FREQ','DEG_CENT')], edge_feature_list = [(nx.jaccard_coefficient,'DAILY_FREQ','JACCARD_COEF')],  full_feature_list=[(nx.transitivity,'DAILY_FREQ','TRANSITIVITY'),(nx_edge_count,'DAILY_FREQ','EDGE_COUNT'),(nx_node_count,'DAILY_FREQ','NODE_COUNT')], port_info = port_info, lat_dict = lat_dict, lon_dict = lon_dict ):
    
    # defaults for debugging
    '''
    time_step_size = 'MONTH'
    top_n_carriers = 10
    merger_carriers=True
    output_file = "../processed_data/market_profile_%s.csv"
    yearly_regression_file = "../processed_data/regression_file_%s.csv"
    non_directional=True
    year = 2007
    months=range(1,13)
    t100_fn="T100_SEGMENTS_%s.csv"
    p52_fn="SCHEDULE_P52_%s.csv"
    db1b_fn="DB1B_MARKETS_%s_Q%s.csv"
    t100_avgd_fn="../processed_data/t100_avgd_m%s.csv"
    merge_HP=True 
    merge_NW = True 
    t100_summed_fn = '../processed_data/t100_summed_m%s.csv'
    t100_craft_avg_fn='../processed_data/t100_craft_avg_m%s.csv'
    ignore_mkts = []
    craft_freq_cuttoff = .01
    max_competitors=100
    freq_cuttoff = .5
    ms_cuttoff=.05
    fs_cuttoff = .05
    only_big_carriers=False
    airports = TAFM2017AirportList.origin.tolist()
    airport_data = TAFM2017AirportList
    rc_name = 'related_carriers_dict_%s.csv'
    node_feature_list = [(nx.degree_centrality,'DAILY_FREQ','DEG_CENT')] #if changed, need to update nan file
    edge_feature_list = [(nx.jaccard_coefficient,'DAILY_FREQ','JACCARD_COEF')] #if changed, need to update nan file
    full_feature_list  = [(nx.transitivity,'DAILY_FREQ','TRANSITIVITY'),(nx_edge_count,'DAILY_FREQ','EDGE_COUNT'),(nx_node_count,'DAILY_FREQ','NODE_COUNT')] #if changed, need to update nan file
    
    '''
    print("2007 data:")
    #dictionary of month to quarter
    month_to_q = {1:1,
                  2:1,
                  3:1,
                  4:2,
                  5:2,
                  6:2,
                  7:3,
                  8:3,
                  9:3,
                  10:4,
                  11:4,
                  12:4}
    #days per month
    common_year_days_month = {1:31,
                       2:28,
                       3:31,
                       4:30,
                       5:31,
                       6:30,
                       7:31,
                       8:31,
                       9:30,
                       10:31,
                       11:30,
                       12:31}
    #days per month in leap year
    leap_year_days_month = common_year_days_month.copy()
    leap_year_days_month[2] =29
    
    # is this a leap year?
    if year % 4 !=0:
        leap =False
    elif year % 100 !=0:
        leap=True
    elif year % 400 !=0:
        leap =False
    else: 
        leap = True
        
    # select day to month dict
    days_month_dict = leap_year_days_month if leap else common_year_days_month
    #states dict
    states_dict_to_num = {'PR':50,'VI':51,'TT':52}
    with open("states_abbrev.txt",'r') as infile:
        for i, line in enumerate(infile):
            states_dict_to_num[line.strip()]=i \
        
    states_dict_to_txt = {v:k for k, v in states_dict_to_num.items() }
   
    #read in revelant bts files and supplementary data files   
    t100 = pd.read_csv(t100_fn % year)
    ##p52 = pd.read_csv(p52_fn % year)

    #create bidrectional market pairs
    pairs =[sorted([pair[0],pair[1]]) for pair in product(airports,airports) if pair[0]!=pair[1] ]
    txtpairs = list(set(["_".join(pair) for pair in pairs]))
    txtpairs = [pair for pair in txtpairs if pair not in ignore_mkts]
    txtpairs_df =pd.DataFrame([sorted(p.split('_')) for p in txtpairs], columns = ['A1','A2'])
    nodes_dict_to_num = {p:i for i, p in enumerate(sorted(airports))}
    nodes_dict_to_txt = {i:p for i, p in enumerate(sorted(airports))}
    #get airports generic data
    port_info = port_info[port_info.ORIGIN.isin(airports)]    
    
    #get relevant segments within network for all market pairs
    print("creating  birdirectional  markets...")
    od = t100[['ORIGIN','DEST']].values
    od.sort(axis=1)
    od = pd.DataFrame(od,columns=["A1","A2"])
    od['BI_MARKET'] = od["A1"] + '_' + od["A2"]
    od['BI_MARKET'] = od['BI_MARKET']
    t100 = pd.concat([t100.reset_index(), od.reset_index()],axis=1)#first, create bidriectional market indicator   
    
    del od
    #get just relevant markets
    relevant_t100= t100[t100.BI_MARKET.isin(txtpairs)].reset_index()
   
   
    del t100
    #merge carrier HP under US if this is called for.
    if merge_HP and year == 2007:
        relevant_t100['UNIQUE_CARRIER']=relevant_t100['UNIQUE_CARRIER'].replace('HP','US')
    
     #merge carrier NW under DL if this is called for.
    if merge_NW and year == 2010:
        relevant_t100['UNIQUE_CARRIER']=relevant_t100['UNIQUE_CARRIER'].replace('NW','DL') # WHERE MONTH in quarter 1

    #monthly airport data
    airport_db = relevant_t100.groupby(['ORIGIN','MONTH']).aggregate({'PASSENGERS':np.sum,'DEPARTURES_PERFORMED':np.sum,'SEATS':np.sum}).reset_index()
    ##airport_db =airport_db.merge(airport_coordinates[['ORIGIN','LAT','LON']],how='left',on='ORIGIN')
    airport_db = airport_db.rename(columns={'PASSENGERS':'DEP_PASSENGERS_AIRPORT','DEPARTURES_PERFORMED':'ENPLANEMENTS','SEATS':'DEP_CAPACITY_AIRPORT'})
    #carrier airport stats
    carrier_airport_db = relevant_t100.groupby(['UNIQUE_CARRIER','ORIGIN','MONTH']).aggregate({'PASSENGERS':np.sum,'DEPARTURES_PERFORMED':np.sum,'SEATS':np.sum}).reset_index()
    carrier_airport_db = carrier_airport_db.rename(columns={'PASSENGERS':'DEP_PASSENGERS_AIRPORT','DEPARTURES_PERFORMED':'ENPLANEMENTS','SEATS':'DEP_CAPACITY_AIRPORT'})
   
        
    
    print("parsing t100 data...")
    #average relevant monthly frequency to get daily freqencies
    t100fields =['MONTH', 'QUARTER','BI_MARKET','UNIQUE_CARRIER','ORIGIN', 'DEST','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','DEPARTURES_PERFORMED','SEATS','PASSENGERS','DISTANCE','AIR_TIME']
    #monthly departures, daily seats, daily passengers, avg distance, total airtime
    t100_summed = relevant_t100[relevant_t100['MONTH'].apply( lambda x: not np.isnan(x))]
    del relevant_t100
    #convert airtime to hours
    t100_summed['AIR_HOURS']=(t100_summed['AIR_TIME']/60)
    t100_summed['FLIGHT_TIME']=t100_summed['AIR_HOURS']/t100_summed['DEPARTURES_PERFORMED']
    #get frequency per day
    days_per_month = t100_summed['MONTH'].apply(lambda row: days_month_dict[int(row)]) 
    t100_summed['DAILY_FREQ']=t100_summed['DEPARTURES_SCHEDULED']/days_per_month
    ##get average number available seats per flight
    t100_summed['SEATS_PER_FLIGHT'] = t100_summed['SEATS']/t100_summed['DEPARTURES_PERFORMED']  #CHECK NUMBERS
    #get seats per day
    t100_summed['SEATS'] = t100_summed['SEATS']/days_per_month
    t100_summed['PASSENGERS'] = t100_summed['PASSENGERS']/days_per_month
    
    #filter empty flights
    t100_summed = t100_summed[t100_summed['PASSENGERS']>0.5]
    t100_summed = t100_summed[t100_summed['DEPARTURES_SCHEDULED']>0.5] 
    #additional filters: frequency, unrealistic seats per flight 
    t100_summed = t100_summed[t100_summed['DAILY_FREQ']>=craft_freq_cuttoff]
    #check for extreme seat numbers
    if t100_summed[(t100_summed['SEATS_PER_FLIGHT']<10) | (t100_summed['SEATS_PER_FLIGHT']>500)].shape[0]>0:
        ##print("Average seat anomalies:")
        ##print(t100_summed[(t100_summed['SEATS_PER_FLIGHT']<10) | (t100_summed['SEATS_PER_FLIGHT']>500)])
        pass
    
    
    
    # calculate average airtime per carrier per directional segment across craft types, weighted by frequency of constituent aircraft types
    grpby=['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','QUARTER','MONTH']
    wavgs = weighted_average(t100_summed[grpby + ['FLIGHT_TIME','DAILY_FREQ']],'FLIGHT_TIME','DAILY_FREQ',grpby,'FLIGHT_TIME_CARRIER_WAVG')
    t100_summed = t100_summed.merge(wavgs, how='left', on=grpby)
    # calculate average seats per flight per carrier per directional segment across craft types, weighted by frequency of constituent aircraft types
    grpby=['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','QUARTER','MONTH']
    wavgs = weighted_average(t100_summed[grpby + ['SEATS_PER_FLIGHT','DAILY_FREQ']],'SEATS_PER_FLIGHT','DAILY_FREQ',grpby,'SEATS_PER_FLIGHT_CARRIER_WAVG')
    t100_summed = t100_summed.merge(wavgs, how='left', on=grpby)
     # calculate averages over craft 
   
    t100fields =['YEAR','QUARTER','MONTH','BI_MARKET','A1','A2','ORIGIN','DEST','UNIQUE_CARRIER','SEATS','DEPARTURES_SCHEDULED','DEPARTURES_PERFORMED','SEATS_PER_FLIGHT_CARRIER_WAVG','PASSENGERS','DISTANCE','DAILY_FREQ','FLIGHT_TIME_CARRIER_WAVG']
    #group by carrier, directional market, month, aggregate other fields appropriately
    t100_craft_avg = t100_summed[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','A1','A2','ORIGIN','DEST','YEAR','QUARTER','MONTH']).aggregate({'DEPARTURES_SCHEDULED':np.sum,'DEPARTURES_PERFORMED':np.sum,'SEATS':np.sum,'FLIGHT_TIME_CARRIER_WAVG':np.mean,'SEATS_PER_FLIGHT_CARRIER_WAVG':np.mean, 'PASSENGERS':np.sum,'DISTANCE':np.mean, 'DAILY_FREQ':np.sum}).reset_index()  
    #save file of t100 summed over months and averaged over craft, to check passenger equivalence between market directions
  
    del t100_summed
    
    #average values between segments sharing a bidirectional market if requested
    if non_directional:
        t100fields =['YEAR','QUARTER','MONTH','BI_MARKET','A1','A2','UNIQUE_CARRIER','SEATS','DEPARTURES_SCHEDULED','DEPARTURES_PERFORMED','SEATS_PER_FLIGHT_CARRIER_WAVG','PASSENGERS','DISTANCE','DAILY_FREQ','FLIGHT_TIME_CARRIER_WAVG']
        t100_craft_avg = t100_craft_avg[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','A1','A2','YEAR','QUARTER','MONTH']).aggregate({'DEPARTURES_SCHEDULED':np.mean,'DEPARTURES_PERFORMED':np.mean,'SEATS':np.mean,'FLIGHT_TIME_CARRIER_WAVG':np.mean,'SEATS_PER_FLIGHT_CARRIER_WAVG':np.mean, 'PASSENGERS':np.mean,'DISTANCE':np.mean, 'DAILY_FREQ':np.mean}).reset_index()  
    
    
    
    
    #remove entries below daily frequency cuttoff 
    count_pre_filter = len(t100_craft_avg)
    t100_craft_avg = t100_craft_avg[t100_craft_avg['DAILY_FREQ']>=freq_cuttoff]
    print("removed %s rows with less than daily frequency cuttoff" % (count_pre_filter-len(t100_craft_avg)))
    
    if time_step_size=='QUARTER':
        t100_craft_avg.drop('MONTH',1).groupby('UNIQUE_CARRIER','BI_MARKET','A1','A2','YEAR','QUARTER').aggregate({'DEPARTURES_SCHEDULED':'sum',related_carriers_dict_string})
    
    
    #add presence indicator
    t100_craft_avg['SEG_MONTH_PRESENCE'] = 1
    
    
    # get top carriers
    carrier_df  = t100_craft_avg[['UNIQUE_CARRIER','PASSENGERS','DAILY_FREQ']].groupby(['UNIQUE_CARRIER']).aggregate({'PASSENGERS':np.sum,'DAILY_FREQ':np.sum}).reset_index().sort_values(by=['PASSENGERS'],ascending=False)
    top_carriers = carrier_df['UNIQUE_CARRIER'][:min(top_n_carriers,len(carrier_df))].tolist()
    carrier_df.to_csv('../processed_data/carrier_annual_%s.csv' % year)
    
    #process node sizes
    print("creating airport node data...")
    # get great circle distance between all pairs
    txtpairs_df['GC_DISTANCE']=txtpairs_df.apply(lambda row: haversine(lon_dict[row.A1][0],lat_dict[row.A1][0],lon_dict[row.A2][0],lat_dict[row.A2][0]), axis=1 )   
    segs =  pd.concat([txtpairs_df]*12)
    segs['MONTH'] = np.repeat(list(range(1,13)),len(txtpairs))
    
     #get enplanements data for airports
    segs = pd.merge(segs, airport_db[['ORIGIN','ENPLANEMENTS','MONTH']], how='left',left_on=['MONTH','A1'],right_on=['MONTH','ORIGIN']).rename(columns={'ENPLANEMENTS':'ENPL_A1'})
    del segs['ORIGIN']
    segs = pd.merge(segs, airport_db[['ORIGIN','ENPLANEMENTS','MONTH']], how='left',left_on=['MONTH','A2'],right_on=['MONTH','ORIGIN']).rename(columns={'ENPLANEMENTS':'ENPL_A2'})
    del segs['ORIGIN']
    segs = segs.fillna(0)
    segs['A1_LARGER'] = (segs.ENPL_A1 >segs.ENPL_A2)
    #get larger and smaller airports
    segs['A_LARGER'] = segs.A1*segs.A1_LARGER + segs.A2*np.invert(segs.A1_LARGER)
    segs['A_SMALLER'] = segs.A2*segs.A1_LARGER + segs.A1*np.invert(segs.A1_LARGER)
    del segs['ENPL_A1'],  segs['ENPL_A2'],  segs['A1_LARGER']
    # add larger airport data
    segs = segs.merge(airport_db,how='left',left_on=['MONTH','A_LARGER'],right_on=['MONTH','ORIGIN'])
    del segs['ORIGIN']
    segs = segs.fillna(0)
    segs = segs.merge(airport_coordinates[['ORIGIN','LAT','LON']],how='left',left_on=['A_LARGER'],right_on=['ORIGIN'])
    del segs['ORIGIN']
    segs = segs.merge(port_info,how='left',left_on=['A_LARGER'],right_on=['ORIGIN'])
    del segs['ORIGIN']
    
    segs.rename(columns={col:col+'_LARGER' for col in ['DEP_PASSENGERS_AIRPORT','ENPLANEMENTS','DEP_CAPACITY_AIRPORT','LAT','LON','ORIGIN_STATE_ABR','ORIGIN_CITY_MARKET_ID']},inplace=True)
    #smaller airport data
    segs = segs.merge(airport_db,how='left',left_on=['MONTH','A_SMALLER'],right_on=['MONTH','ORIGIN'])
    del segs['ORIGIN']
    segs = segs.fillna(0)
    segs = segs.merge(airport_coordinates[['ORIGIN','LAT','LON']],how='left',left_on=['A_SMALLER'],right_on=['ORIGIN'])
    del segs['ORIGIN']
    segs = segs.merge(port_info,how='left',left_on=['A_SMALLER'],right_on=['ORIGIN'])
    del segs['ORIGIN']
    segs.rename(columns={col:col+'_SMALLER' for col in ['DEP_PASSENGERS_AIRPORT','ENPLANEMENTS','DEP_CAPACITY_AIRPORT','LAT','LON','ORIGIN_STATE_ABR','ORIGIN_CITY_MARKET_ID']},inplace=True)
    
    segs['AVG_ENPLANEMENTS_AR'] = segs[['ENPLANEMENTS_LARGER','ENPLANEMENTS_SMALLER']].mean(axis=1)
    segs['RATIO_ENPLANEMENTS_AR'] =  segs.ENPLANEMENTS_SMALLER/segs.ENPLANEMENTS_LARGER
    segs['DIFF_ENPLANEMENTS_AR'] = segs.ENPLANEMENTS_LARGER - segs.ENPLANEMENTS_LARGER
    segs.fillna(0)
    # col names text to numbers
    for col in ['ORIGIN_STATE_ABR_SMALLER','ORIGIN_STATE_ABR_LARGER']:
        segs[col] = segs[col].apply(lambda x: states_dict_to_num[x],1)
    for col in ['A1','A2','A_LARGER','A_SMALLER']:
        segs[col] = segs[col].apply(lambda x: nodes_dict_to_num[x],1)
   
    #process carrier node sizes 
    print("creating carrier airport node data...")
    print("creating airport node data...")
    # create base data frame (all segs, all months, all carriers)
    segs_cr =  pd.concat([segs[['A1','A2','MONTH']]]*min(top_n_carriers,len(carrier_df)))
    segs_cr['UNIQUE_CARRIER'] = np.repeat(top_carriers,len(segs))
    
    carrier_airport_db['ORIGIN'] = carrier_airport_db['ORIGIN'].apply(lambda x: nodes_dict_to_num[x],1)
    cols = carrier_airport_db.columns.tolist()
    cols.remove('MONTH')
    cols.remove('UNIQUE_CARRIER')
    #get enplanements data for carrier airports
    segs_cr = pd.merge(segs_cr, carrier_airport_db[['ORIGIN','UNIQUE_CARRIER','ENPLANEMENTS','MONTH']], how='left',left_on=['MONTH','A1','UNIQUE_CARRIER',],right_on=['MONTH','ORIGIN','UNIQUE_CARRIER',]).rename(columns={'ENPLANEMENTS':'ENPL_A1'})
    del segs_cr['ORIGIN']
    segs_cr = pd.merge(segs_cr, carrier_airport_db[['ORIGIN','UNIQUE_CARRIER','ENPLANEMENTS','MONTH']], how='left',left_on=['MONTH','A2','UNIQUE_CARRIER',],right_on=['MONTH','ORIGIN','UNIQUE_CARRIER',]).rename(columns={'ENPLANEMENTS':'ENPL_A2'})
    del segs_cr['ORIGIN']
    segs_cr['A1_LARGER'] = (segs_cr.ENPL_A1 >segs_cr.ENPL_A2)
    #get larger and smaller airports
    segs_cr['ACR_LARGER'] = segs_cr.A1*segs_cr.A1_LARGER + segs_cr.A2*np.invert(segs_cr.A1_LARGER)
    segs_cr['ACR_SMALLER'] = segs_cr.A2*segs_cr.A1_LARGER + segs_cr.A1*np.invert(segs_cr.A1_LARGER)
    del segs_cr['ENPL_A1'],  segs_cr['ENPL_A2'],  segs_cr['A1_LARGER']
    #larger airport data
    segs_cr = segs_cr.merge(carrier_airport_db,how='left',left_on=['MONTH','ACR_LARGER','UNIQUE_CARRIER'],right_on=['MONTH','ORIGIN','UNIQUE_CARRIER'])
    del segs_cr['ORIGIN']
    segs_cr.rename(columns={col:col+'_LARGER_ACR' for col in cols},inplace=True)
  
    #smaller airport data
    segs_cr = segs_cr.merge(carrier_airport_db,how='left',left_on=['MONTH','ACR_SMALLER','UNIQUE_CARRIER'],right_on=['MONTH','ORIGIN','UNIQUE_CARRIER'])
    del segs_cr['ORIGIN']
    segs_cr.rename(columns={col:col+'_SMALLER_ACR' for col in cols},inplace=True)
    segs_cr['AVG_ENPLANEMENTS_ACR'] = segs_cr[['ENPLANEMENTS_LARGER_ACR','ENPLANEMENTS_SMALLER_ACR']].mean(axis=1)
    segs_cr['DIFF_ENPLANEMENTS_ACR'] = segs_cr.ENPLANEMENTS_LARGER_ACR - segs_cr.ENPLANEMENTS_SMALLER_ACR
    segs_cr['RATIO_ENPLANEMENTS_ACR'] =  segs_cr.ENPLANEMENTS_SMALLER_ACR/segs_cr.ENPLANEMENTS_LARGER_ACR
    # zero one or two ports with enplanements
    segs_cr['SEG_PORTS_OCCUPIED'] =  np.clip(segs_cr.ENPLANEMENTS_SMALLER_ACR,0,1) + np.clip(segs_cr.ENPLANEMENTS_LARGER_ACR,0,1)
    segs_cr = segs_cr.fillna(0)
    
    
    #set up t100 craft avg for preiminary operations
    print("creaing segment features...")
    del t100_craft_avg['BI_MARKET']
    for col in ['A1','A2']:
         t100_craft_avg[col] = t100_craft_avg[col].apply(lambda x: nodes_dict_to_num[x],1)
    grouped = t100_craft_avg.groupby(['MONTH','A1','A2'])
        
    #market month stats
    t100_craft_avg['SEG_PLAYERS'] = grouped['UNIQUE_CARRIER'].transform('count')
    t100_craft_avg['SEG_PAX'] = grouped['PASSENGERS'].transform('sum')
    t100_craft_avg['SEG_FREQ'] = grouped['DAILY_FREQ'].transform('sum')
    t100_craft_avg['SEG_SEATS'] = grouped['SEATS'].transform('sum')
    
    
    
    #shares, load factors
    t100_craft_avg['SEG_PAX_SHARE'] =  t100_craft_avg['PASSENGERS']/t100_craft_avg['SEG_PAX']
    t100_craft_avg['SEG_PAX_SHARE_SQ'] =t100_craft_avg['SEG_PAX_SHARE']**2
    t100_craft_avg['SEG_HHI'] = (t100_craft_avg.groupby(['A1','A2','MONTH'])['SEG_PAX_SHARE_SQ']).transform('sum')
    del t100_craft_avg['SEG_PAX_SHARE_SQ']
    t100_craft_avg['SEG_FREQ_SHARE'] = t100_craft_avg['DAILY_FREQ']/t100_craft_avg['SEG_FREQ']
    t100_craft_avg['SEG_CAPACITY_SHARE'] =  t100_craft_avg['SEATS']/t100_craft_avg['SEG_SEATS']
    t100_craft_avg['SEG_LOAD_FACTOR']=t100_craft_avg['SEG_PAX']/t100_craft_avg['SEG_SEATS']
    
  
    #carrier attributes
    #read in related carriers
    related_carriers = read_related_carriers(rc_name % year)
    low_cost_carriers = ['WN','VX','SY','NK','B6','F9','G4']    
    mainline_carriers = list(related_carriers.keys())
    t100_craft_avg['IS_LOWCOST'] = np.where(t100_craft_avg['UNIQUE_CARRIER'].isin(low_cost_carriers),1,0)
    t100_craft_avg['MAIN_LINE'] = np.where(t100_craft_avg['UNIQUE_CARRIER'].isin(mainline_carriers),1,0)
    #at least one low cost or main line on segment
    grouped = t100_craft_avg.groupby(['MONTH','A1','A2'])
    t100_craft_avg['SEG_LOWCOST'] = grouped['IS_LOWCOST'].transform('sum')
    t100_craft_avg['SEG_LOWCOST']=  np.clip(t100_craft_avg['SEG_LOWCOST'],0,1)
    t100_craft_avg['SEG_MAIN_LINE'] = grouped['MAIN_LINE'].transform('sum')
    t100_craft_avg['SEG_MAIN_LINE']=  np.clip(t100_craft_avg['SEG_MAIN_LINE'],0,1)
    
    #create desired node features    
    print("creating node features...")
    for networkx_func, edge_attr, feature_name in node_feature_list:
        segs_cr = create_node_network_feature(t100_craft_avg,segs_cr,networkx_func,edge_attr,feature_name)
    
    #create desired edge features
    print("creating edge features...")
    for networkx_func, edge_attr, feature_name in edge_feature_list:
        t100_craft_avg = create_edge_network_feature(t100_craft_avg,networkx_func,edge_attr,feature_name)
    
    
    # create desired carrier network features
    print("creating full network features...")
    
    for networkx_func, edge_attr, feature_name in full_feature_list:
        t100_craft_avg = create_full_network_feature(t100_craft_avg,networkx_func,edge_attr,feature_name)
    t100_craft_avg['EDGE_NODE_RATIO'] = t100_craft_avg['EDGE_COUNT']/ t100_craft_avg['NODE_COUNT']
    t100_craft_avg['CARRIER_FLIGHTS'] = t100_craft_avg.groupby(['UNIQUE_CARRIER','MONTH'])['DAILY_FREQ'].transform('sum')
    t100_craft_avg['CARRIER_PAX'] = t100_craft_avg.groupby(['UNIQUE_CARRIER','MONTH'])['PASSENGERS'].transform('sum')
    t100_craft_avg['CARRIER_SEATS'] = t100_craft_avg.groupby(['UNIQUE_CARRIER','MONTH'])['SEATS'].transform('sum')
    t100_craft_avg['CARRIER_SEATS_FLIGHT_RATIO'] = t100_craft_avg['CARRIER_SEATS']/t100_craft_avg['CARRIER_FLIGHTS']
       
    #merge with airport and segment data
    airport_db['ORIGIN'] =  airport_db['ORIGIN'].apply(lambda x: nodes_dict_to_num[x],1)
    t100_craft_avg  = t100_craft_avg.merge(airport_db[['ORIGIN','MONTH','ENPLANEMENTS']], how='left', left_on=['A1','MONTH'], right_on=['ORIGIN','MONTH'])
    t100_craft_avg.rename(columns={"ENPLANEMENTS":'E1'}, inplace=True)
  
    t100_craft_avg  = t100_craft_avg.merge(airport_db[['ORIGIN','MONTH','ENPLANEMENTS']], how='left', left_on=['A2','MONTH'], right_on=['ORIGIN','MONTH'])
    t100_craft_avg.rename(columns={"ENPLANEMENTS":'E2'}, inplace=True)
   
    t100_craft_avg['AVG_PORT'] =   t100_craft_avg[['E1','E2']].mean(axis=1)
    
    ##t100_craft_avg  = t100_craft_avg.merge(segs_cr, how='left', on=['A1','A2','MONTH','UNIQUE_CARRIER'])
    
    #create desired carrier network features
    #average larger airport size served, average smaller airport size served, weighted by DAILY FREQ
    grpby=['UNIQUE_CARRIER','MONTH']
    wavgs = weighted_average(t100_craft_avg,'AVG_PORT','DAILY_FREQ',grpby,'CARRIER_WAVG_PORT_SIZE')
    t100_craft_avg = t100_craft_avg.merge(wavgs, how='left', on=grpby) 
    del t100_craft_avg['AVG_PORT']
    del t100_craft_avg['E1']
    del t100_craft_avg['E2']
    
    
    #read mergefile (describing how each column of t100_craft_average should be merged with segs_cr)
    '''
    col_merge_type
    0: join on
    1: carrier_segment
    2: segment
    3: carrier

    col_nan_type
    2 none
    1 nan
    0 zero
    '''
    col_types = pd.read_csv("col_types.txt",sep='\s+')
    print("merging market data...")
    #merge time data
    gby = ['MONTH']
    segs_cr = segs_cr.merge(t100_craft_avg[gby+ ['YEAR','QUARTER']].groupby(gby).aggregate(np.mean).reset_index(),how='left',on=gby)
    
     #merge segment carrier data
    gby = ['MONTH','A1','A2','UNIQUE_CARRIER']
    segs_cr = segs_cr.merge(t100_craft_avg[gby+ col_types[col_types.col_merge_type==1].col.tolist()].groupby(gby).aggregate(np.mean).reset_index(),how='left',on=gby)
     #merge segment carrier data
    gby = ['MONTH','A1','A2','UNIQUE_CARRIER']
    segs_cr = segs_cr.merge(t100_craft_avg[gby+['SEG_MONTH_PRESENCE']].groupby(gby).aggregate(np.mean).reset_index(),how='left',on=gby)
    
     #merge segment data
    gby = ['MONTH','A1','A2']
    segs_cr = segs_cr.merge(t100_craft_avg[gby+ col_types[col_types.col_merge_type==2].col.tolist()].groupby(gby).aggregate(np.mean).reset_index(),how='left',on=gby)
    
     #merge segment data
    gby = ['MONTH','UNIQUE_CARRIER']
    segs_cr = segs_cr.merge(t100_craft_avg[gby+ col_types[col_types.col_merge_type==3].col.tolist()].groupby(gby).aggregate(np.mean).reset_index(),how='left',on=gby)
    
    segs_cr.fillna(0,inplace=True)
    
    print("creating airline alliances data...")
    
    #invert related carriers
    related_carriers_invert = {}
    for key, vals in related_carriers.items():
        for val in vals:
            if val in related_carriers_invert:
                related_carriers_invert[val] = related_carriers_invert[val] + [key]
            else:
                related_carriers_invert[val] =  [key]
   
    # get cooperating carrier indicator vector for each carrier-market-month
    grouped = t100_craft_avg.groupby(['MONTH','A1','A2'])
    carriers = grouped.apply(lambda x: x['UNIQUE_CARRIER'].tolist()).reset_index()
    carriers = segs_cr.merge(carriers, how='left',on=['MONTH','A1','A2'])[0].tolist()
    carriers = [c if c==c else [] for c in carriers]    
    unique_carrier=segs_cr.UNIQUE_CARRIER.tolist()
    related_carriers_full = merge_two_dicts(related_carriers, related_carriers_invert)
    cooperators = []
    mainline_w_allies = [key for key, val in related_carriers.items() if val] + [key for key, val in related_carriers_invert.items() if val]
    for carrier, market in zip(unique_carrier,carriers):
        if (carrier not in  mainline_w_allies) or len(market)==1 :
            cooperators.append(np.array([]))
        else:
            coop_vec = [1 if c in related_carriers_full[carrier] else 0 for c in market]
            if sum(coop_vec) == 0:
                cooperators.append(np.array([]))
            else:
                cooperators.append(np.array(coop_vec)) 
                
    #do similar for merging players
    print("creating mergers data...")
    if merger_carriers:
        ushp_start = datetime.strptime("06/2005", "%m/%Y")        
        dlnw_start = datetime.strptime("10/2008", "%m/%Y")
        uaco_start = datetime.strptime("08/2010", "%m/%Y")
        wnfl_start = datetime.strptime("04/2011", "%m/%Y")
        aaus_start = datetime.strptime("11/2013", "%m/%Y")
        months=segs_cr.MONTH.tolist()
        merging_carriers = ['US','HP','DL','NW','UA','CO','WN','FL','AA']
      
        merging_partners = []
        ##merging_overlap_segment=[]
        for carrier, market, month in zip(unique_carrier,carriers,months):
            
            if (carrier not in  merging_carriers) or len(market)==1 :
                merging_partners.append(np.array([]))
                ##merging_overlap_segment.append(0)
            else:
                date = datetime.strptime("/".join([str(month).rjust(2,"0"),str(year)]), "%m/%Y")  
                coop_vec = []               
                if date >=ushp_start and date < aaus_start and carrier=='US':
                    coop_vec = [1 if c=='HP' else 0 for c in market]
                elif date >=ushp_start and carrier=='HP':
                    coop_vec = [1 if c=='US' else 0 for c in market]
                elif date >=dlnw_start and carrier=='DL':
                    coop_vec = [1 if c=='NW' else 0 for c in market]
                elif date >=dlnw_start and carrier=='NW':
                    coop_vec = [1 if c=='DL' else 0 for c in market]
                elif date >=uaco_start and carrier=='UA':
                    coop_vec = [1 if c=='CO' else 0 for c in market]
                elif date >=uaco_start and carrier=='CO':
                    coop_vec = [1 if c=='UA' else 0 for c in market]
                elif date >=wnfl_start and carrier=='WN':
                    coop_vec = [1 if c=='FL' else 0 for c in market]
                elif date >=wnfl_start and carrier=='FL':
                    coop_vec = [1 if c=='WN' else 0 for c in market]
                elif date >=aaus_start and carrier=='AA':
                    coop_vec = [1 if c=='US' else 0 for c in market]
                elif date >=aaus_start and carrier=='US':
                    coop_vec = [1 if c=='AA' else 0 for c in market]
                
                
                if sum(coop_vec) == 0:
                    merging_partners.append(np.array([]))
                    ##merging_overlap_segment.append(0)
                else:
                    merging_partners.append(np.array(coop_vec))
                    ##merging_overlap_segment.append(1)
        

    print("computing cooperative traffic...")
    # compute aggregate frequency and capacity of allied carriers by row
    segs_cr['ALLIED_FREQ']= coop_sum('DAILY_FREQ',cooperators,segs_cr,t100_craft_avg)
    segs_cr['ALLIED_SEATS']= coop_sum('SEATS',cooperators,segs_cr,t100_craft_avg)
    segs_cr['ALLIED_PAX']= coop_sum('PASSENGERS',cooperators,segs_cr,t100_craft_avg)
    segs_cr['ALLIED_PLAYERS']=[sum(r) for r in cooperators]
    
     # compute aggregate frequency and capacity of merging carriers by row
    segs_cr['MERGING_FREQ']= coop_sum('DAILY_FREQ',merging_partners,segs_cr,t100_craft_avg)
    segs_cr['MERGING_SEATS']= coop_sum('SEATS',merging_partners,segs_cr,t100_craft_avg)
    segs_cr['MERGING_PLAYERS']=[sum(r) for r in merging_partners]
    segs_cr['MERGING_PAX']= coop_sum('PASSENGERS',merging_partners,segs_cr,t100_craft_avg)
    #group to get segment-month properties
    grouped = segs_cr.groupby(['MONTH','A1','A2'])
    #merging carrier segment?
    segs_cr['MERGING_SEGMENT'] = grouped['MERGING_PLAYERS'].transform('sum')
    segs_cr['MERGING_SEGMENT']=  np.clip(segs_cr['MERGING_SEGMENT'],0,1)
    
    print("computing competitive traffic...")
    #competitor and total other behavior   
    sum_operations = grouped['SEATS'].transform('sum')
    segs_cr['OTHER_SEATS'] = (sum_operations - segs_cr['SEATS'])
    sum_operations = grouped['DAILY_FREQ'].transform('sum')
    segs_cr['OTHER_FREQ'] = (sum_operations - segs_cr['DAILY_FREQ'])
    sum_operations = grouped['PASSENGERS'].transform('sum')
    segs_cr['OTHER_PAX'] = (sum_operations - segs_cr['PASSENGERS'])
    segs_cr['OTHER_PLAYERS'] = (segs_cr['SEG_PLAYERS'] - 1)
    #subtract out cooperative/merging operations
    segs_cr['COMPETING_SEATS'] =  segs_cr['OTHER_SEATS']  -  segs_cr['ALLIED_SEATS'] -  segs_cr['MERGING_SEATS'] 
    segs_cr['COMPETING_FREQ'] =  segs_cr['OTHER_SEATS']  -  segs_cr['ALLIED_FREQ'] -  segs_cr['MERGING_FREQ'] 
    segs_cr['COMPETING_PLAYERS'] =  segs_cr['OTHER_PLAYERS']  -  segs_cr['ALLIED_PLAYERS'] -  segs_cr['MERGING_PLAYERS'] 
    segs_cr['COMPETING_PAX'] =  segs_cr['OTHER_PAX']  -  segs_cr['ALLIED_PAX'] -  segs_cr['MERGING_PAX'] 
      
    
    print("computing load factors...")
    segs_cr['OTHER_LOAD_FACTOR']=segs_cr['OTHER_PAX']/segs_cr['OTHER_SEATS']
    segs_cr['COMPETING_LOAD_FACTOR']=segs_cr['COMPETING_PAX']/segs_cr['COMPETING_SEATS']
    segs_cr['MERGING_LOAD_FACTOR']=segs_cr['MERGING_PAX']/segs_cr['MERGING_SEATS']
    segs_cr['ALLIED_LOAD_FACTOR']=segs_cr['ALLIED_PAX']/segs_cr['ALLIED_SEATS']
    # average seats per flight of other carriers
    n = grouped['SEATS_PER_FLIGHT_CARRIER_WAVG'].transform('count')
    mean = grouped['SEATS_PER_FLIGHT_CARRIER_WAVG'].transform('mean')
    segs_cr['OTHER_SEATS_PER_FLIGHT_AVG'] = (mean*n - segs_cr['SEATS_PER_FLIGHT_CARRIER_WAVG'])/(n-1)
  
   
    
   #process db1b markets
    print("parsing db1b..")
    seg_data = []
    seg_carrier_data = []
    port_data = []
    port_carrier_data = []
    for quarter in range(1,5):
       
        print("quarter %s..." % quarter)
        db1b  = pd.read_csv(db1b_fn % (year, quarter))
        db1b = db1b[['QUARTER','OPERATING_CARRIER',"MARKET_COUPONS","AIRPORT_GROUP",'MARKET_DISTANCE','BULK_FARE','MARKET_FARE','PASSENGERS','ORIGIN','DEST']]
        #filter very high/low distances and fares
        db1b = db1b.dropna()
        db1b = db1b[(db1b.MARKET_FARE>50) & (db1b.MARKET_FARE<2000)]
        db1b = db1b[(db1b.MARKET_DISTANCE<10000)]
        db1b = db1b[(db1b.BULK_FARE==0)]
        db1b = db1b[(db1b.MARKET_COUPONS<=2)]
       
        od = db1b[['ORIGIN','DEST']].values
        od.sort(axis=1)
        od = pd.DataFrame(od,columns=["A1","A2"])
        db1b = pd.concat([db1b.reset_index(),od.reset_index()],axis=1)
        del db1b['index']
        # weighted averages of fares
        grby = ['A1','A2']
        wavgs_market = weighted_average(db1b,'MARKET_FARE','PASSENGERS',grby,'AVG_MKT_FARE')
        grby = ['OPERATING_CARRIER','A1','A2']
        wavgs_cr_market = weighted_average(db1b,'MARKET_FARE','PASSENGERS',grby,'CR_AVG_MKT_FARE')
        grby = ['A1','A2']
        wavgs_nonstop_market = weighted_average(db1b[db1b.MARKET_COUPONS==1],'MARKET_FARE','PASSENGERS',grby,'AVG_MKT_FARE_0S')
        wavgs_onestop_market = weighted_average(db1b[db1b.MARKET_COUPONS==2],'MARKET_FARE','PASSENGERS',grby,'AVG_MKT_FARE_1S')
        
        #
        #get carrier and general connections
        carrier_connects = db1b[['OPERATING_CARRIER',"A1","A2",'AIRPORT_GROUP']+['PASSENGERS','MARKET_COUPONS','MARKET_DISTANCE','ORIGIN']].groupby(['OPERATING_CARRIER',"A1","A2",'AIRPORT_GROUP']).aggregate({'PASSENGERS':np.sum,'ORIGIN':'count','MARKET_DISTANCE':np.mean,'MARKET_COUPONS':lambda x: x.iloc[0]}).reset_index().rename(columns={'ORIGIN': 'ROUTE_COUNT'})
        connects = db1b[['OPERATING_CARRIER',"A1","A2",'AIRPORT_GROUP']+['PASSENGERS','MARKET_COUPONS','MARKET_DISTANCE','ORIGIN']].groupby(["A1","A2",'AIRPORT_GROUP']).aggregate({'PASSENGERS':np.sum,'ORIGIN':'count','MARKET_DISTANCE':np.mean,'MARKET_COUPONS':lambda x: x.iloc[0]}).reset_index().rename(columns={'ORIGIN': 'ROUTE_COUNT'})
        connects['COUNT'] = np.ones(len(connects))
        c0s = connects[connects.MARKET_COUPONS==1].groupby(['A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum, 'MARKET_DISTANCE':np.mean }).reset_index()
        c1s = connects[connects.MARKET_COUPONS==2].groupby(['A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum, 'MARKET_DISTANCE':np.mean }).reset_index().rename(columns=({'PASSENGERS':'PASSENGERS_CX','COUNT':'COUNT_CX', 'MARKET_DISTANCE':'MARKET_DISTANCE_CX' }))
        cs = weighted_average(connects,'MARKET_DISTANCE','PASSENGERS',['A1','A2'],'WAVG_MKT_DISTANCE')
        cs = connects.groupby(['A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum}).reset_index().merge(cs,how='left',on=['A1','A2'])
        cs = cs.merge(c1s,how='left',on=['A1','A2'])
        cs.fillna(0, inplace=True)
        cs['SEG_PAX_CXN_RATIO'] =  cs['PASSENGERS_CX']/ cs['PASSENGERS']
        del cs['PASSENGERS']
        del cs['COUNT']
        del cs['MARKET_DISTANCE_CX']
        #merge with segment fares
        cs =cs.merge(wavgs_market,how='left',on=['A1','A2'])
        cs =cs.merge(wavgs_nonstop_market,how='left',on=['A1','A2'])
        cs =cs.merge(wavgs_onestop_market,how='left',on=['A1','A2']).rename(columns={'OPERATING_CARRIER':'UNIQUE_CARRIER'})
        cs.fillna(0,inplace=True)
        cs['QUARTER'] = quarter
        #carrier connections
        carrier_connects['COUNT'] = np.ones(len(carrier_connects))
        c0s = carrier_connects[carrier_connects.MARKET_COUPONS==1].groupby(['OPERATING_CARRIER','A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum}).reset_index()
        c1s = carrier_connects[carrier_connects.MARKET_COUPONS==2].groupby(['OPERATING_CARRIER','A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum}).reset_index().rename(columns=({'PASSENGERS':'PASSENGERS_CX_CR','COUNT':'COUNT_CX_CR' }))
        cscr = carrier_connects.groupby(['OPERATING_CARRIER','A1','A2']).aggregate({'PASSENGERS':np.sum,'COUNT':np.sum}).reset_index()
        cscr = cscr.merge(c1s,how='left',on=['OPERATING_CARRIER','A1','A2'])
        cscr.fillna(0, inplace=True)
        cscr['SEG_PAX_CXN_RATIO_CR'] =  cscr['PASSENGERS_CX_CR']/ cscr['PASSENGERS']
        del cscr['PASSENGERS']
        del cscr['COUNT']
        cscr = cscr.merge(wavgs_cr_market,how='left',on=['OPERATING_CARRIER','A1','A2']).rename(columns={'OPERATING_CARRIER':'UNIQUE_CARRIER'})
        cscr.fillna(0,inplace=True)
        cscr['QUARTER'] = quarter
        
        #hubs -> get connection to destination ratio for each carrier airport
        carrier_connects['DEST'] =  carrier_connects['AIRPORT_GROUP'].apply(lambda x: x.split(':')[-1])
        carrier_connects['CXN'] =  carrier_connects['AIRPORT_GROUP'].apply(lambda x: '' if len(x)==7 else x.split(':')[1])
        cxn_db = carrier_connects[['OPERATING_CARRIER','CXN','ROUTE_COUNT']].groupby(['OPERATING_CARRIER','CXN']).aggregate({'ROUTE_COUNT':np.sum}).reset_index()
        cxn_db = cxn_db[cxn_db.CXN!='']
        dest_db = carrier_connects[['OPERATING_CARRIER','DEST','ROUTE_COUNT']].groupby(['OPERATING_CARRIER','DEST']).aggregate({'ROUTE_COUNT':np.sum}).reset_index()
        dest_db = dest_db[dest_db.DEST!=''].rename(columns={'ROUTE_COUNT':'DEST_ROUTE_COUNT','DEST':'CXN'})
        cxn_db = cxn_db.merge(dest_db,how='left',on=['OPERATING_CARRIER','CXN'])
        cxn_db=cxn_db.fillna(0)
        cxn_db['TOT_COUNT'] = cxn_db['ROUTE_COUNT'] + cxn_db['DEST_ROUTE_COUNT']
        cxn_db['CXN_RATIO'] = cxn_db['ROUTE_COUNT']/ cxn_db['TOT_COUNT']
        cxn_db= cxn_db.rename(columns={'OPERATING_CARRIER':'UNIQUE_CARRIER'})
        cxn_db = cxn_db[['UNIQUE_CARRIER','CXN','CXN_RATIO','ROUTE_COUNT']]
        cxn_db['QUARTER'] = quarter
        #get most prominent connecion number at all airports
        port_connections = cxn_db[['CXN','ROUTE_COUNT']].groupby('CXN').aggregate('max').reset_index().rename(columns={'ROUTE_COUNT':'MAX_CXN'})
        port_connections['QUARTER'] = quarter
        del cxn_db['ROUTE_COUNT']
        seg_data.append(cs)
        seg_carrier_data.append(cscr)
        port_data.append(port_connections)
        port_carrier_data.append(cxn_db)
        del db1b
    #concatenate quarterly data
    print("merging db1b...")
    seg_data = pd.concat(seg_data)
    seg_carrier_data= pd.concat(seg_carrier_data)
    port_data= pd.concat(port_data)
    port_carrier_data= pd.concat(port_carrier_data)
    seg_data = seg_data[(seg_data.A1.isin(airports)) & (seg_data.A2.isin(airports))]
    seg_carrier_data= seg_carrier_data[(seg_carrier_data.A1.isin(airports)) & (seg_carrier_data.A2.isin(airports))]
    port_data= port_data[port_data.CXN.isin(airports)]
    port_carrier_data= port_carrier_data[port_carrier_data.CXN.isin(airports)]
    #numerify port names
    for col in ['A1','A2']:
        seg_data[col] = seg_data[col].apply(lambda x: nodes_dict_to_num[x],1)
    for col in ['A1','A2']:
        seg_carrier_data[col] = seg_carrier_data[col].apply(lambda x: nodes_dict_to_num[x],1)
    for col in ['CXN']:
        port_data[col] = port_data[col].apply(lambda x: nodes_dict_to_num[x],1)
    for col in ['CXN']:
        port_carrier_data[col] = port_carrier_data[col].apply(lambda x: nodes_dict_to_num[x],1)
    
    #merge to seg_cr on a quarterly basis
    segs_cr= segs_cr.merge(seg_data,how = 'left',on = ['A1','A2','QUARTER'])   
    segs_cr =segs_cr.merge(seg_carrier_data,how = 'left',on = ['A1','A2','QUARTER','UNIQUE_CARRIER'])   
    segs_cr =segs_cr.merge(port_carrier_data,how = 'left',left_on = ['ACR_LARGER','QUARTER','UNIQUE_CARRIER'],right_on = ['CXN','QUARTER','UNIQUE_CARRIER']).rename(columns={'CXN_RATIO':'CXN_RATIO_LARGER'})  
    del segs_cr['CXN']
    segs_cr =segs_cr.merge(port_carrier_data,how = 'left',left_on = ['ACR_SMALLER','QUARTER','UNIQUE_CARRIER'],right_on = ['CXN','QUARTER','UNIQUE_CARRIER']).rename(columns={'CXN_RATIO':'CXN_RATIO_SMALLER'})  
    del segs_cr['CXN']
    segs_cr =segs_cr.merge(port_data,how = 'left',left_on = ['ACR_LARGER','QUARTER'],right_on = ['CXN','QUARTER']).rename(columns={'MAX_CXN':'MAX_CXN_LARGER'})  
    del segs_cr['CXN']
    segs_cr =segs_cr.merge(port_data,how = 'left',left_on = ['ACR_SMALLER','QUARTER'],right_on = ['CXN','QUARTER']).rename(columns={'MAX_CXN':'MAX_CXN_SMALLER'})  
    del segs_cr['CXN']
    
    
    #merge with airport data
    print("merging segment data...")
    segs_cr =segs_cr.merge(segs,how='left',on=['A1','A2','MONTH'])
    segs_cr.fillna(0, inplace=True)
    del segs
   # ADD SEGS CR AND ALLIED CITY ROUTE
     # match city market ids, any parellel? what about just one parellel?
    ##t100_craft_avg['SORTED_CITY_MARKET_ID'] = t100_craft_avg.apply(lambda row: sorted([row.ORIGIN_CITY_MARKET_ID_SMALLER,row.ORIGIN_CITY_MARKET_ID_LARGER]), axis=1)
    ##t100_craft_avg.groupby(['UNIQUE_CARRIER','MONTH','ORIGIN_CITY_MARKET_ID_SMALLER','ORIGIN_CITY_MARKET_ID_LARGER'])
    #find duplicate city id pairs: how many identical city market does this airline serve
    print("finding matching city pairs...")
    od = segs_cr[['ORIGIN_CITY_MARKET_ID_SMALLER','ORIGIN_CITY_MARKET_ID_LARGER']].values
    od.sort(axis=1)
    od = pd.DataFrame(od,columns=["OCMID1","OCMID2"])
    segs_cr = pd.concat([segs_cr.reset_index(),od.reset_index()],axis=1)
    shared_ocmid = segs_cr[['MONTH','UNIQUE_CARRIER',"OCMID1","OCMID2","SEG_MONTH_PRESENCE"]].groupby(['MONTH','UNIQUE_CARRIER',"OCMID1","OCMID2"]).aggregate(np.sum).rename(columns={'SEG_MONTH_PRESENCE':'SHARED_CITY_MARKET_ID_SEGS'}).reset_index()
    segs_cr = segs_cr.merge(shared_ocmid,how='left',on=['MONTH','UNIQUE_CARRIER',"OCMID1","OCMID2"])
    #write outputs
    print("writing outputs...")
    t100_craft_avg.to_csv(output_file % year)
    segs_cr.to_csv(yearly_regression_file % year)
    with open('../processed_data/state_numbering.txt','w') as outfile:
        for k, v in states_dict_to_txt.items():
            outfile.write(str(k) + ',' + str(v) + '\n')
    with open('../processed_data/airport_numbering.txt','w') as outfile:
        for k, v in nodes_dict_to_txt.items():
            outfile.write(str(k) + ',' + str(v) + '\n')
    return t100_craft_avg





def create_carrier_tensors():
    create_compiled_table  =True
    input_file = "processed_data/market_profile_%s.csv" 
    airports = TAFM2017AirportList.origin.tolist()
    years = range(2007,2008)
    market_files = []
    if create_compiled_table:
        for year in years:
            market_files.append(pd.read_csv(input_file % year))
    market_files = pd.concat(market_files, axis=1)        
    
      #create adjaceny matrices
    print("creating demand tensors...")
    ###IF COSTS ARE EVER FIXED, USE PLACE TO COMBINE OBSERVED WITH HYPOTHETICAL (IN ALL PLAYED MARKETS) COSTS
    t100_by_market = market_files[['YEAR','MONTH','SEG_PAX','A1','A2']].groupby(['YEAR','MONTH','A1','A2']).aggregate({'SEG_PAX':lambda x: x.iloc[0]}).reset_index()
    t100_by_market_grouped = t100_by_market.groupby(['YEAR','MONTH'])    
    demand_tensor = np.zeros([len(airports),len(airports),12*len(years)])   
    i=0
    for year in years: 
       for month in range(1,13):
            #get relevant data table
            data_time_t = t100_by_market_grouped.get_group((year,month))
            #convert to networkx undirected graph
            G=nx.from_pandas_dataframe(data_time_t, 'A1', 'A2', edge_attr=['SEG_PAX'])
            ##nx.write_edgelist(G,'netx_objs/carrier_net_demand_%s_%s.edgelist' % (year,month), data=['MARKET_TOT'])
            #convert to numpy adjacency matrix weighted by frequency
            D = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='SEG_PAX')            
            #add matrices to tensor at appropriate time step
            demand_tensor[:,:,i] = D            
            i+=1
            
            print("%s %s  demand adj mats done" % (year, month))
    np.save('demand_tensors/ts_demand',demand_tensor )
    t100_craft_avg_network_gb = market_files.groupby(['UNIQUE_CARRIER','MONTH','YEAR'])
    for carrier in market_files.UNIQUE_CARRIER.unique().tolist():
        #initialize adjaceny tensor
        adjacency_tensor = np.zeros([len(airports),len(airports),12*len(years)])
        ##print("creating adjaceny mats for %s" % carrier)
        i = 0
        for year in years: 
            for month in range(1,13):
                
                try:
                    cr_timestep_df  = t100_craft_avg_network_gb.get_group((carrier,  month, year))
                except KeyError:
                    pass
                else:
                          #convert to networkx undirected graph
                    G=nx.from_pandas_dataframe(cr_timestep_df, 'A1', 'A2', edge_attr=['DAILY_FREQ'])
                    ##nx.write_edgelist(G,'netx_objs/carrier_net_%s_%s_%s.edgelist' % (year,month,carrier), data=['DAILY_FREQ', 'FLIGHT_COST'])
                    #convert to numpy adjacency matrix weighted by frequency
                    A = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='DAILY_FREQ')                  
                    #add matrices to tensor at appropriate time step
                    adjacency_tensor[:,:,i] = A
                    
                    
                i+=1
            #next time step
            
           
                
        # save tensors
        np.save('freq_tensors/ts_freq_%s' % carrier,adjacency_tensor )
    change_class_threshold = 1 # magnitude of frequency change between months to be classed as frequency change
    carriers_of_interest = ['WN','DL','UA','AA','US']
    airports = sorted(node_sets['top100_2014'])    
    entry_dict = {}
    total_entries = 0
    entry_zero_offset = 12 # assume that if entry, airline has been out 12 months
    
    for carrier in carriers_of_interest:
        adjacency_tensor = np.load('freq_tensors/ts_freq_%s.npy' % carrier )
        # is airline ever present in market?
        #create C-ordered flattened index
        flattened_index = [(i,j) for i in airports for j in airports ]
        market_presence_mat = adjacency_tensor.sum(axis=2).astype(bool)
        #don't double count markets!
        market_presence_mat[np.tril_indices(len(airports))] =False
       
        flattened_freq = adjacency_tensor.reshape(-1, adjacency_tensor.shape[-1])
        ###flattened_cost = cost_tensor.reshape(-1, cost_tensor.shape[-1])
        flattened_market_presence_index = market_presence_mat.flatten()
        freq_series_text_index = np.array(flattened_index)[flattened_market_presence_index]
        #valued time series
        FREQ_SERIES= flattened_freq[flattened_market_presence_index]
        ###COST_SERIES = flattened_cost[flattened_market_presence_index]
       
        
        #detect entries  
        time_steps = FREQ_SERIES.shape[1]
        viable_markets_count = FREQ_SERIES.shape[0]
        entries = np.zeros([viable_markets_count,time_steps])
        changes =np.zeros([viable_markets_count,time_steps]) 
        change_classes =np.zeros([viable_markets_count,time_steps])
        for i in range(0,viable_markets_count):# for each relelvant market   
            for j in range(0,time_steps-1):            
                if j < entry_zero_offset:
                    pass
                    ##previous_freqs_sum = sum(FREQ_SERIES[i,0:j+1])
                else:
                    previous_freqs_sum =sum(FREQ_SERIES[i,j-(entry_zero_offset):j+1])
                    if previous_freqs_sum == 0 and FREQ_SERIES[i,j+1] > 1:
                        entries[i,j+1] = 1
            #additionally, calculate changes for each
            change  = np.diff(FREQ_SERIES[i,:])
            # classify changes according to cuttoff  
            change_class = np.zeros(change.shape)
            change_class[change>=change_class_threshold]=1 #one if increase
            change_class[change<=-change_class_threshold]=2
            #save change classes in matrix (assume no change from t-1 to t0)
            changes[i,1:] = change
            change_classes[i,1:] = change_class
            
        '''
        #Create plot of WN Entries
        entry_ind = entries.sum(axis=1).astype(bool)
        plt.plot(np.tile(list(range(0,105)),[FREQ_SERIES.shape[0],1])[entry_ind,:].T,FREQ_SERIES[entry_ind,:].T)
        plt.xlabel('Time Step')
        plt.ylabel('Daily Frequency')
        plt.title('124 WN markets with entry, 2007-2015')
        '''
        new_entries = entries.sum().sum()
        total_entries += new_entries
        print('%s has %s entries' % (carrier, new_entries))
        entry_dict[carrier] = {'num_entries':new_entries,'mkt_index':freq_series_text_index,'freq_mat': FREQ_SERIES, 'entry_mat':entries , 'change_mat':changes,'xclass_mat':change_classes}


'''
function to create node features with networkx package
inputs: 
    df: market data DataFrame
    df_to_merge: segment carrier DataFrame
    networkx_func: node based networkx graph theoretic function to apply
    edge_attr: edge attibute/weight of interest
    feature_name: desired name of feature (with _SMALLER/_LARGER appended for nodes with lesser/greater carrier traffic, respectively)
'''
def create_node_network_feature(df,df_to_merge,networkx_func,edge_attr,feature_name):
         #function to calculate node attribute for each node
        def apply_graph_func_node(gby, networkx_func, edge_attr):
            G = nx.from_pandas_dataframe(gby, source='A1',target='A2',edge_attr=edge_attr).to_undirected()
            network_dict =networkx_func(G)
            return pd.Series({'PORT':network_dict.keys(),'NODEATTR':network_dict.values()})
        #apply function to overall table on a month-carrier basis
        df_gb = df.groupby(['UNIQUE_CARRIER','MONTH']).apply(apply_graph_func_node, networkx_func,edge_attr)
        
        #reformulate into a proper data frame
        port_table = pd.melt(df_gb.PORT.apply(list).apply(pd.Series).reset_index(), 
             id_vars=['UNIQUE_CARRIER', 'MONTH'],
             value_name='PORT').set_index(['UNIQUE_CARRIER', 'MONTH']).drop('variable', axis=1).dropna()
        stat_table = pd.melt(df_gb.NODEATTR.apply(list).apply(pd.Series).reset_index(), 
             id_vars=['UNIQUE_CARRIER', 'MONTH'],
             value_name='NODEATTR').set_index(['UNIQUE_CARRIER', 'MONTH']).drop('variable', axis=1).dropna()
        port_stat_table = pd.concat([port_table.reset_index(),stat_table.reset_index()['NODEATTR']], axis=1)
        #merge node data to larger airports in node pair
        df_to_merge = pd.merge(df_to_merge,port_stat_table,how='left',left_on = ['MONTH','UNIQUE_CARRIER','ACR_LARGER'],right_on = ['MONTH','UNIQUE_CARRIER','PORT'])
        df_to_merge.rename(columns={'NODEATTR':feature_name + '_SMALLER'},inplace=True)
        del df_to_merge['PORT']
        #merge node data to smaller airports
        df_to_merge = pd.merge(df_to_merge,port_stat_table,how='left',left_on = ['MONTH','UNIQUE_CARRIER','ACR_SMALLER'],right_on = ['MONTH','UNIQUE_CARRIER','PORT'])
        df_to_merge.rename(columns={'NODEATTR':feature_name + '_LARGER'},inplace=True)
        del df_to_merge['PORT']
        #PERHAPS DROP NA
        return df_to_merge    
    

    
    
'''
function to create edge features with networkx package
inputs: 
    df: market data DataFrame
    networkx_func: edge based networkx graph theoretic function to apply
    edge_attr: edge attibute/weight of interest
    feature_name: desired name of feature 
'''
def create_edge_network_feature(df,networkx_func,edge_attr,feature_name):
         #function to calculate node attribute for each node
        def apply_graph_func_edge(gby, networkx_func, edge_attr):
            G = nx.from_pandas_dataframe(gby, source='A1',target='A2',edge_attr=edge_attr).to_undirected()
            network_df =pd.DataFrame(networkx_func(G, ebunch= G.edges() + list(nx.non_edges(G))))
            if len(network_df )==len(gby ):
                return pd.Series({'A1':network_df.iloc[:,0].tolist(),'A2':network_df.iloc[:,1].tolist(), 'EDGEATTR':network_df.iloc[:,2].tolist()})
            #if undefined (only one edge in network, for example), return 0
            else:
                return pd.Series({'A1':gby.A1.tolist(),'A2':gby.A2.tolist(), 'EDGEATTR':np.zeros(len(gby)).tolist()})
           
            #apply function to overall table on a month-carrier basis
        df_gb = df.groupby(['UNIQUE_CARRIER','MONTH']).apply(apply_graph_func_edge, networkx_func,edge_attr)
        
        #reformulate into a proper data frame
        node1_table = pd.melt(df_gb.A1.apply(list).apply(pd.Series).reset_index(), 
             id_vars=['UNIQUE_CARRIER', 'MONTH'],
             value_name='A1').set_index(['UNIQUE_CARRIER', 'MONTH']).drop('variable', axis=1).dropna()
        node2_table = pd.melt(df_gb.A2.apply(list).apply(pd.Series).reset_index(), 
             id_vars=['UNIQUE_CARRIER', 'MONTH'],
             value_name='A2').set_index(['UNIQUE_CARRIER', 'MONTH']).drop('variable', axis=1).dropna()
        stat_table = pd.melt(df_gb.EDGEATTR.apply(list).apply(pd.Series).reset_index(), 
             id_vars=['UNIQUE_CARRIER', 'MONTH'],
             value_name='EDGEATTR').set_index(['UNIQUE_CARRIER', 'MONTH']).drop('variable', axis=1).dropna()
        
        port_stat_table = pd.concat([node1_table.reset_index(),node2_table.reset_index()['A2'],stat_table.reset_index()['EDGEATTR']], axis=1)
        #merge node data to larger airports in node pair
        df = pd.merge(df,port_stat_table,how='left',on = ['MONTH','UNIQUE_CARRIER','A1','A2'])
        df.rename(columns={'EDGEATTR':feature_name},inplace=True)
        return df
    
    
'''
function to create feature for entire carrier network
'''
def create_full_network_feature(df,networkx_func,edge_attr,feature_name):
         #function to calculate node attribute for each node
        def apply_graph_func(gby, networkx_func, edge_attr):
            G = nx.from_pandas_dataframe(gby, source='A1',target='A2',edge_attr=edge_attr).to_undirected()
            network_val =networkx_func(G)
            if not network_val:
                network_val = 0
            return network_val
            #apply function to overall table on a month-carrier basis
        df_gb = df.groupby(['UNIQUE_CARRIER','MONTH']).apply(apply_graph_func, networkx_func,edge_attr)
        
         #merge node data to larger airports in node pair
        df = pd.merge(df,df_gb.reset_index(),how='left',on = ['MONTH','UNIQUE_CARRIER'])
        df.rename(columns={0:feature_name},inplace=True)
        return df
'''
helper function to create a bidirectional market indicator (with airports sorted by text) for origin-destination pairs
'''    
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)
    
'''
helper function to calculate weighted averages in dataframes for groups

df: original data frame
data_col: column being averaged within group
weight col: column providing weights for average
by_col: column(s) to groupby
new_col: name of new weighted average column
'''
def weighted_average(df,data_col,weight_col,by_col,new_col):
    df['_data_times_weight'] = df[data_col]*df[weight_col]
    df['_weight_where_notnull'] = df[weight_col]*pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    
    return result.reset_index().rename(columns={0:new_col})
'''
helper function get a weighed average costs and flight times across a directional market
'''
def craft_weight_avgs(gb):
    time_weighted = np.average(gb['FLIGHT_TIME'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_TIME'] = np.repeat(time_weighted,gb.shape[0])
    seats_weighted = np.average(gb['SEATS_PER_FLIGHT'], weights=gb['DAILY_FREQ'])
    gb['SEATS_PER_FLIGHT'] = np.repeat(seats_weighted,gb.shape[0])            
    return gb

'''
helper function to read related carriers file
'''
def read_related_carriers(fn):
    related_carriers = {}
    with open(fn) as infile:
        for line in infile:
            related_carriers[line.split(':')[0].strip()] =   line.split(':')[1].split()
    return related_carriers        

'''

helper function to merge dictionaries
'''
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
'''
helper function to compute total allied activity (of type sum_col) in a carrier-market-month
'''
def coop_sum(sum_col,cooperators,segs_cr,t100_craft_avg):
    grouped = t100_craft_avg.groupby(['MONTH','A1','A2'])
    sum_cols = grouped.apply(lambda x: x[sum_col].tolist()).reset_index()
    sum_cols = segs_cr.merge(sum_cols, how='left',on=['MONTH','A1','A2'])[0].tolist()
    summed_cols = []
    for coop, col in zip(cooperators, sum_cols):
        if coop.size == 0:
            summed_cols.append(0)
        else:
            summed_cols.append(np.dot(coop, col))
    return summed_cols

'''
helper function to calculate great circle distance (in mi) between 2 points given coordinates
'''
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    mi = 0.621371*km
    return mi

'''
function for creating numpy tensors of network characteristics (needs cleaning up/generalization)
'''

def create_carrier_tensors():
    create_compiled_table  =True
    input_file = "processed_data/market_profile_%s.csv" 
    airports = TAFM2017AirportList.origin.tolist()
    years = range(2007,2008)
    market_files = []
    if create_compiled_table:
        for year in years:
            market_files.append(pd.read_csv(input_file % year))
    market_files = pd.concat(market_files, axis=1)        
    
      #create adjaceny matrices
    print("creating demand tensors...")
    ###IF COSTS ARE EVER FIXED, USE PLACE TO COMBINE OBSERVED WITH HYPOTHETICAL (IN ALL PLAYED MARKETS) COSTS
    t100_by_market = market_files[['YEAR','MONTH','SEG_PAX','A1','A2']].groupby(['YEAR','MONTH','A1','A2']).aggregate({'SEG_PAX':lambda x: x.iloc[0]}).reset_index()
    t100_by_market_grouped = t100_by_market.groupby(['YEAR','MONTH'])    
    demand_tensor = np.zeros([len(airports),len(airports),12*len(years)])   
    i=0
    for year in years: 
       for month in range(1,13):
            #get relevant data table
            data_time_t = t100_by_market_grouped.get_group((year,month))
            #convert to networkx undirected graph
            G=nx.from_pandas_dataframe(data_time_t, 'A1', 'A2', edge_attr=['SEG_PAX'])
            ##nx.write_edgelist(G,'netx_objs/carrier_net_demand_%s_%s.edgelist' % (year,month), data=['MARKET_TOT'])
            #convert to numpy adjacency matrix weighted by frequency
            D = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='SEG_PAX')            
            #add matrices to tensor at appropriate time step
            demand_tensor[:,:,i] = D            
            i+=1
            
            print("%s %s  demand adj mats done" % (year, month))
    np.save('demand_tensors/ts_demand',demand_tensor )
    t100_craft_avg_network_gb = market_files.groupby(['UNIQUE_CARRIER','MONTH','YEAR'])
    for carrier in market_files.UNIQUE_CARRIER.unique().tolist():
        #initialize adjaceny tensor
        adjacency_tensor = np.zeros([len(airports),len(airports),12*len(years)])
        ##print("creating adjaceny mats for %s" % carrier)
        i = 0
        for year in years: 
            for month in range(1,13):
                
                try:
                    cr_timestep_df  = t100_craft_avg_network_gb.get_group((carrier,  month, year))
                except KeyError:
                    pass
                else:
                          #convert to networkx undirected graph
                    G=nx.from_pandas_dataframe(cr_timestep_df, 'A1', 'A2', edge_attr=['DAILY_FREQ'])
                    ##nx.write_edgelist(G,'netx_objs/carrier_net_%s_%s_%s.edgelist' % (year,month,carrier), data=['DAILY_FREQ', 'FLIGHT_COST'])
                    #convert to numpy adjacency matrix weighted by frequency
                    A = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='DAILY_FREQ')                  
                    #add matrices to tensor at appropriate time step
                    adjacency_tensor[:,:,i] = A
                    
                    
                i+=1
            #next time step
            
           
                
        # save tensors
        np.save('freq_tensors/ts_freq_%s' % carrier,adjacency_tensor )
    change_class_threshold = 1 # magnitude of frequency change between months to be classed as frequency change
    carriers_of_interest = ['WN','DL','UA','AA','US']
    airports = sorted(node_sets['top100_2014'])    
    entry_dict = {}
    total_entries = 0
    entry_zero_offset = 12 # assume that if entry, airline has been out 12 months
    
    for carrier in carriers_of_interest:
        adjacency_tensor = np.load('freq_tensors/ts_freq_%s.npy' % carrier )
        # is airline ever present in market?
        #create C-ordered flattened index
        flattened_index = [(i,j) for i in airports for j in airports ]
        market_presence_mat = adjacency_tensor.sum(axis=2).astype(bool)
        #don't double count markets!
        market_presence_mat[np.tril_indices(len(airports))] =False
       
        flattened_freq = adjacency_tensor.reshape(-1, adjacency_tensor.shape[-1])
        ###flattened_cost = cost_tensor.reshape(-1, cost_tensor.shape[-1])
        flattened_market_presence_index = market_presence_mat.flatten()
        freq_series_text_index = np.array(flattened_index)[flattened_market_presence_index]
        #valued time series
        FREQ_SERIES= flattened_freq[flattened_market_presence_index]
        ###COST_SERIES = flattened_cost[flattened_market_presence_index]
       
        
        #detect entries  
        time_steps = FREQ_SERIES.shape[1]
        viable_markets_count = FREQ_SERIES.shape[0]
        entries = np.zeros([viable_markets_count,time_steps])
        changes =np.zeros([viable_markets_count,time_steps]) 
        change_classes =np.zeros([viable_markets_count,time_steps])
        for i in range(0,viable_markets_count):# for each relelvant market   
            for j in range(0,time_steps-1):            
                if j < entry_zero_offset:
                    pass
                    ##previous_freqs_sum = sum(FREQ_SERIES[i,0:j+1])
                else:
                    previous_freqs_sum =sum(FREQ_SERIES[i,j-(entry_zero_offset):j+1])
                    if previous_freqs_sum == 0 and FREQ_SERIES[i,j+1] > 1:
                        entries[i,j+1] = 1
            #additionally, calculate changes for each
            change  = np.diff(FREQ_SERIES[i,:])
            # classify changes according to cuttoff  
            change_class = np.zeros(change.shape)
            change_class[change>=change_class_threshold]=1 #one if increase
            change_class[change<=-change_class_threshold]=2
            #save change classes in matrix (assume no change from t-1 to t0)
            changes[i,1:] = change
            change_classes[i,1:] = change_class
            
        '''
        #Create plot of WN Entries
        entry_ind = entries.sum(axis=1).astype(bool)
        plt.plot(np.tile(list(range(0,105)),[FREQ_SERIES.shape[0],1])[entry_ind,:].T,FREQ_SERIES[entry_ind,:].T)
        plt.xlabel('Time Step')
        plt.ylabel('Daily Frequency')
        plt.title('124 WN markets with entry, 2007-2015')
        '''
        new_entries = entries.sum().sum()
        total_entries += new_entries
        print('%s has %s entries' % (carrier, new_entries))
        entry_dict[carrier] = {'num_entries':new_entries,'mkt_index':freq_series_text_index,'freq_mat': FREQ_SERIES, 'entry_mat':entries , 'change_mat':changes,'xclass_mat':change_classes}



'''
old function for finding/plotting market entries based on network tensors
'''
def deprecated_time_series_analysis(node_sets):
    change_class_threshold = 1 # magnitude of frequency change between months to be classed as freuncy change
    carriers_of_interest = ['WN','DL','UA','AA','US']
    airports = sorted(node_sets['top100_2014'])    
    entry_dict = {}
    total_entries = 0
    entry_zero_offset = 12 # assume that if entry, airline has been out 12 months
    
    for carrier in carriers_of_interest:
        adjacency_tensor = np.load('freq_tensors/ts_freq_%s.npy' % carrier )
        # is airline ever present in market?
        #create C-ordered flattened index
        flattened_index = [(i,j) for i in airports for j in airports ]
        market_presence_mat = adjacency_tensor.sum(axis=2).astype(bool)
        #don't double count markets!
        market_presence_mat[np.tril_indices(len(airports))] =False
       
        flattened_freq = adjacency_tensor.reshape(-1, adjacency_tensor.shape[-1])
        ###flattened_cost = cost_tensor.reshape(-1, cost_tensor.shape[-1])
        flattened_market_presence_index = market_presence_mat.flatten()
        freq_series_text_index = np.array(flattened_index)[flattened_market_presence_index]
        #valued time series
        FREQ_SERIES= flattened_freq[flattened_market_presence_index]
        ###COST_SERIES = flattened_cost[flattened_market_presence_index]
       
        
        #detect entries  
        time_steps = FREQ_SERIES.shape[1]
        viable_markets_count = FREQ_SERIES.shape[0]
        entries = np.zeros([viable_markets_count,time_steps])
        changes =np.zeros([viable_markets_count,time_steps]) 
        change_classes =np.zeros([viable_markets_count,time_steps])
        for i in range(0,viable_markets_count):# for each relelvant market   
            for j in range(0,time_steps-1):            
                if j < entry_zero_offset:
                    pass
                    ##previous_freqs_sum = sum(FREQ_SERIES[i,0:j+1])
                else:
                    previous_freqs_sum =sum(FREQ_SERIES[i,j-(entry_zero_offset):j+1])
                    if previous_freqs_sum == 0 and FREQ_SERIES[i,j+1] > 1:
                        entries[i,j+1] = 1
            #additionally, calculate changes for each
            change  = np.diff(FREQ_SERIES[i,:])
            # classify changes according to cuttoff  
            change_class = np.zeros(change.shape)
            change_class[change>=change_class_threshold]=1 #one if increase
            change_class[change<=-change_class_threshold]=2
            #save change classes in matrix (assume no change from t-1 to t0)
            changes[i,1:] = change
            change_classes[i,1:] = change_class
            
        '''
        #Create plot of WN Entries
        entry_ind = entries.sum(axis=1).astype(bool)
        plt.plot(np.tile(list(range(0,105)),[FREQ_SERIES.shape[0],1])[entry_ind,:].T,FREQ_SERIES[entry_ind,:].T)
        plt.xlabel('Time Step')
        plt.ylabel('Daily Frequency')
        plt.title('124 WN markets with entry, 2007-2015')
        '''
        new_entries = entries.sum().sum()
        total_entries += new_entries
        print('%s has %s entries' % (carrier, new_entries))
        entry_dict[carrier] = {'num_entries':new_entries,'mkt_index':freq_series_text_index,'freq_mat': FREQ_SERIES, 'entry_mat':entries , 'change_mat':changes,'xclass_mat':change_classes}
        
    print(total_entries)
    plt.plot(entry_dict['WN']['xclass_mat'].T)
    plt.ylim([-1,3])
    
   
      
    # correlate frequencies
    mkt_index = entry_dict['WN']['mkt_index']
    corr_WN =np.corrcoef(entry_dict['WN']['freq_mat'])
    plt.hist(corr_WN.flatten())
    neg_cor = np.array(np.where(corr_WN<-.75))
    pos_cor = np.array(np.where(corr_WN>.75))
    
    np.where(corr_WN==corr_WN.min().min())    
    
    tot_airport_share = 0
    for i in range(0, neg_cor.shape[1]):
        mkt_1_ports = neg_cor[0,i]
        mkt_2_ports = neg_cor[1,i]
        shared_ports_count = len(np.intersect1d(mkt_index[mkt_1_ports],mkt_index[mkt_2_ports]))
        if shared_ports_count > 1: 
            tot_airport_share+=1
            
    #plot markets with high anti correlation
    plt.plot(entry_dict['WN']['freq_mat'][[409,297,288],:].T)
    
    tot_airport_share = 0
    for i in range(0, pos_cor.shape[1]):
        mkt_1_ports = pos_cor[0,i]
        mkt_2_ports = pos_cor[1,i]
        shared_ports_count = len(np.intersect1d(mkt_index[mkt_1_ports],mkt_index[mkt_2_ports]))
        if shared_ports_count > 1: 
            tot_airport_share+=1
    
    
    tot_airport_share = 0
    for i in range(0, pos_cor.shape[1]):
        mkt_1_ports = pos_cor[0,i]
        mkt_2_ports = pos_cor[1,i]
        shared_ports_count = len(np.intersect1d(mkt_index[mkt_1_ports],mkt_index[mkt_2_ports]))
        if shared_ports_count > 1: 
            pass
    
    #plot market entry counts
    h = entries.sum(axis=0)
    plt.plot(range(0,len(h)),h)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Entries')
    plt.title('WN entry counts per month, 2007-2015')
   
    #count increases/decreases/etc for each month
    change_class = entry_dict['WN']['xclass_mat'].astype(int)
    change_hists = []
    for i in range(0,change_class.shape[1]):
        change_hists.append(np.bincount(change_class[:,i], minlength=3))
    return None
    
'''
older function to create features, run regression

'''  
def deprecated_feature_construction(entry_dict,node_sets):
    #WANT MARKET ENTRY COLUMN AND MARKET PRESENCE
    test_time_step = 94
    carrier_of_interest = 'WN'
    training_timestep_lag = 12
    airports = sorted(node_sets['top100_2014'])
    t100ranked = pd.read_csv("full_t100ranked.csv")
    carriers = t100ranked['UNIQUE_CARRIER'].unique().tolist()
    carriers.remove(carrier_of_interest)
    freq_tensor_of_interest  = np.load('freq_tensors/ts_freq_%s.npy' % carrier_of_interest)
    #initialize competitor tensors
    other_weighted_tensor = np.zeros([freq_tensor_of_interest.shape[0],freq_tensor_of_interest.shape[1],freq_tensor_of_interest.shape[2] ])
    other_unweighted_tensor = np.zeros([freq_tensor_of_interest.shape[0],freq_tensor_of_interest.shape[1],freq_tensor_of_interest.shape[2] ])
    #combine competitor tensors    
    for other in carriers:
        ts =  np.load('freq_tensors/ts_freq_%s.npy' % other)
        other_weighted_tensor += ts
        other_unweighted_tensor += ts.astype(bool).astype(float)
    demand_ts = np.load('demand_tensors/ts_demand.npy')
    DATA_MAT = np.zeros([training_timestep_lag*freq_tensor_of_interest.shape[0]*(freq_tensor_of_interest.shape[0]-1)/2, 11  ])
    i = 0
    for timestep in range(test_time_step - training_timestep_lag,test_time_step):
        print(timestep)
        G=nx.from_numpy_matrix(freq_tensor_of_interest[:,:,timestep-1])
        deg_cent = nx.degree_centrality(G)
        for node_pair in combinations(range(0,len(airports)),2):
        
            #presence or abscence of link by airline of interest at time t
            DATA_MAT[i,0] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep].astype(bool).astype(float)
            #degree centrality of each node
            DATA_MAT[i,1] = deg_cent[node_pair[0]] + deg_cent[node_pair[1]] 
            #indicator of presence of 0, 1 or 2 nodes with non zero degree
            if deg_cent[node_pair[0]] == 0  and  deg_cent[node_pair[1]] ==0:
                DATA_MAT[i,2]  = 0.
            elif deg_cent[node_pair[0]] > 0  and  deg_cent[node_pair[1]]  > 0:
                DATA_MAT[i,2]  = 2.
            else:
                DATA_MAT[i,2]  = 1.
            #demand previous time step
            DATA_MAT[i,3] = demand_ts[node_pair[0],node_pair[1],timestep -1]
            #demand current time step
            DATA_MAT[i,4] = demand_ts[node_pair[0],node_pair[1],timestep ]
            #number of competitors on edge
            DATA_MAT[i,5] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -1 ]
            #competitor frequency  on edge
            DATA_MAT[i,6] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 1 ]
            #number of competitors on edge 2 time steps ago 
            DATA_MAT[i,7] = other_weighted_tensor[node_pair[0],node_pair[1],timestep -2 ]
            #competitor frequency  on edge
            DATA_MAT[i,8] = other_weighted_tensor[node_pair[0],node_pair[1],timestep - 2 ]
             # weight of link by carrer in previous time step
            DATA_MAT[i,9] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1]
            DATA_MAT[i,10] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1].astype(bool).astype(float)
            i+=1
    np.save('DATA_MAT.npy',DATA_MAT)
    #Create testing data
    TEST_MAT = np.zeros([freq_tensor_of_interest.shape[0]*(freq_tensor_of_interest.shape[0]-1)/2, 11 ])
    i = 0
    timestep=test_time_step    
    G=nx.from_numpy_matrix(freq_tensor_of_interest[:,:,timestep-1])
    deg_cent = nx.degree_centrality(G)
    for node_pair in combinations(range(0,len(airports)),2):
    
        #presence or abscence of link by airline of interest at time t
        TEST_MAT[i,0] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep].astype(bool).astype(float)
        #degree centrality of each node
        TEST_MAT[i,1] = deg_cent[node_pair[0]] + deg_cent[node_pair[1]] 
        #indicator of presence of 0, 1 or 2 nodes with non zero degree
        if deg_cent[node_pair[0]] == 0  and  deg_cent[node_pair[1]] ==0:
            TEST_MAT[i,2]  = 0.
        elif deg_cent[node_pair[0]] > 0  and  deg_cent[node_pair[1]]  > 0:
            TEST_MAT[i,2]  = 2.
        else:
            TEST_MAT[i,2]  = 1.
        #demand previous time step
        TEST_MAT[i,3] = demand_ts[node_pair[0],node_pair[1],timestep -1]
        #demand current time step
        TEST_MAT[i,4] = demand_ts[node_pair[0],node_pair[1],timestep ]
        #number of competitors on edge
        TEST_MAT[i,5] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -1 ]
        #frequency  on edge
        TEST_MAT[i,6] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 1 ]
        #number of competitors on edge 2 time steps ago 
        TEST_MAT[i,7] = other_weighted_tensor[node_pair[0],node_pair[1],timestep -2 ]
        #competitor frequency  on edge
        TEST_MAT[i,8] = other_weighted_tensor[node_pair[0],node_pair[1],timestep - 2 ]
        # weight of link by carrer in previous time step
        TEST_MAT[i,9] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1]
        TEST_MAT[i,10] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1].astype(bool).astype(float)
        i+=1
    np.save('TEST_MAT.npy',TEST_MAT)

        #competitor 
    #logistic regrssion    
   
    Y = DATA_MAT[:,0].flatten()
    X = DATA_MAT[:,1:]
    
    
    # create a multinomial logistic regression model with l2 regularization using sci-kit learn
    logreg = linear_model.LogisticRegression(class_weight ='auto')
    #fit model with data
    logreg.fit(X,Y)
    #get model predictions
    Ytest = TEST_MAT[:,0]
    Yhat = logreg.predict_proba(TEST_MAT[:,1:])
    Yhatlab = logreg.predict(TEST_MAT[:,1:])
    confusion_matrix(Yhatlab,Ytest)
   
       
    #extract entry mat
    entries = entry_dict[carrier_of_interest]['entry_mat']
    mkt_index = entry_dict[carrier_of_interest]['mkt_index']
    #entry markets?
    entry_mkts = mkt_index[entries[:,test_time_step].astype(bool)]
    
    full_markets =  list(combinations(range(0,len(airports)),2))
    entry_markets_nums = [(airports.index(mkt[0]),airports.index(mkt[1])) for mkt in entry_mkts]
    Yhat_entry_index  = [full_markets.index(entry) for entry in entry_markets_nums]
    entry_probs = Yhat[Yhat_entry_index]
  
     # where do non predicted fall among other non predicted?
    just_predicted_0s = np.sort(Yhat[np.invert(Yhatlab.astype(bool))][:,0] , axis =0 )
    ranks_in_unpredicted = [np.where(just_predicted_0s==entry_prob[0]) for entry_prob in entry_probs]
    
    return [ranks_in_unpredicted, logreg]

# run parser for all years

def run_parser():
    for year in range(2007,2017):
        t = nonstop_market_profile_monthly(output_file = "../processed_data/market_profile_%s.csv",year = year, months=range(1,13), \
    t100_fn="T100_SEGMENTS_%s.csv",p52_fn="SCHEDULE_P52_%s.csv", non_directional=True, t100_avgd_fn="processed_data/t100_avgd_m%s.csv", merge_HP=True, merge_NW=True, \
    t100_summed_fn = 'processed_data/t100_summed_m%s.csv', t100_craft_avg_fn='processed_data/t100_craft_avg_m%s.csv',\
    ignore_mkts = [], merger_carriers=True,  craft_freq_cuttoff = .01,max_competitors=100,top_n_carriers = 15,\
    freq_cuttoff = .5, ms_cuttoff=.05, yearly_regression_file = "../processed_data/regression_file_%s.csv",\
    fs_cuttoff = .05,db1b_fn="DB1B_MARKETS_%s_Q%s.csv", rc_name = 'related_carriers_dict_%s.csv',\
    only_big_carriers=False, airports = TAFM2017AirportList.origin.tolist() , airport_data = TAFM2017AirportList,\
    node_feature_list = [(nx.degree_centrality,'DAILY_FREQ','DEG_CENT')],\
    edge_feature_list = [(nx.jaccard_coefficient,'DAILY_FREQ','JACCARD_COEF')], \
    full_feature_list=[(nx.transitivity,'DAILY_FREQ','TRANSITIVITY'),(nx_edge_count,'DAILY_FREQ','EDGE_COUNT'),(nx_edge_count,'DAILY_FREQ','NODE_COUNT')],\
    port_info = port_info,\
    lat_dict = lat_dict,\
    lon_dict = lon_dict ,\
    time_step_size = 'MONTH')
    
 