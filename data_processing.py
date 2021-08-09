# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:48:42 2021

@author: 43739
"""

import pandas as pd

def HFDataProcess(folder = "E:/Projects/Hedge fund independency/DataInput/"):
### some dirty preprocessing for Credit Suisse Hedge Fund Index
    df_HF = pd.read_csv(folder + "Credit Suisse Hedge Fund Index.csv")
    df_HF = df_HF[df_HF.columns[:43]]
    subgroups = {} ### name mapping, drop unrelevant columns, we only need NAV
    df_HF_processed = df_HF.copy(deep = True)
    for i in range (1, 43, 3):
        subgroups[df_HF_processed[df_HF.columns[i]].values[0]] = df_HF.columns[i]
        df_HF_processed.drop(df_HF.columns[i], axis = 1, inplace = True)
        df_HF_processed.drop(df_HF.columns[i + 2], axis = 1, inplace = True)
    
    ### continue format the data
    df_HF_processed.drop(df_HF.index[:2], inplace = True)
    df_HF_processed.drop(df_HF.index[331:], inplace = True)
    df_HF_processed.index = pd.to_datetime(df_HF_processed["Unnamed: 0"])
    df_HF_processed.columns = ["str_time"] + [x for x in subgroups.keys()]
    df_HF_processed.sort_index(axis = 0, ascending = True, inplace = True)
    df_HF_processed.to_csv(folder + "HF_Index_processed.csv")
    return subgroups, df_HF_processed
### done

def SPDataProcess(folder = "E:/Projects/Hedge fund independency/DataInput/"):
    ### some dirty preprocessing for SP500 data
    df_SP_daily = pd.read_csv(folder + "SP500 Daily.csv")
    df_SP_daily.index = pd.to_datetime(df_SP_daily["Date"])
    df_SP_daily.drop(["Date"], axis = 1, inplace = True)
    df_SP_daily.sort_index(axis = 0, ascending = True, inplace = True)
    
    df_SP_weekly = pd.read_csv(folder + "SP500 Weekly.csv")
    df_SP_weekly.index = pd.to_datetime(df_SP_weekly["Date"])
    df_SP_weekly.drop(["Date"], axis = 1, inplace = True)
    df_SP_weekly.sort_index(axis = 0, ascending = True, inplace = True)
    
    df_SP_monthly = pd.read_csv(folder + "SP500 Monthly.csv")
    df_SP_monthly.index = pd.to_datetime(df_SP_monthly["Date"])
    df_SP_monthly.drop(["Date"], axis = 1, inplace = True)
    df_SP_monthly.sort_index(axis = 0, ascending = True, inplace = True)
    
    
    return df_SP_daily, df_SP_weekly, df_SP_monthly





