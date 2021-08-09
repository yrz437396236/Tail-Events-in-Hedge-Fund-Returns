# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:00:03 2021

@author: 43739
"""

import pandas as pd
import numpy as np
import math
import datetime

import data_processing

input_folder = "E:/Projects/Hedge fund independency/DataInput/"
output_folder = "E:/Projects/Hedge fund independency/DataOutput/"
subgroups, df_HF_processed = data_processing.HFDataProcess(input_folder)
df_SP_daily, df_SP_weekly, df_SP_monthly = data_processing.SPDataProcess(input_folder)


start_date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
end_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date()

df_HF_processed = df_HF_processed[(df_HF_processed.index >= pd.Timestamp(start_date)) & (df_HF_processed.index <= pd.Timestamp(end_date))]

df_SP_daily = df_SP_daily[(df_SP_daily.index >= pd.Timestamp(start_date)) & (df_SP_daily.index <= pd.Timestamp(end_date))].astype("float")
df_SP_weekly = df_SP_weekly[(df_SP_weekly.index >= pd.Timestamp(start_date)) & (df_SP_weekly.index <= pd.Timestamp(end_date))].astype("float")
df_SP_monthly = df_SP_monthly[(df_SP_monthly.index >= pd.Timestamp(start_date)) & (df_SP_monthly.index <= pd.Timestamp(end_date))].astype("float")

######################################### Independent Test (Christoffersen, 1998)   

def TwoSideChristoffersenIndependentTest(df): #X^2(1)--0.95 = 3.841
    
    VaR0975 = df.quantile(q = 0.025)
    VaR0025 = df.quantile(q = 0.975)
    N00, N01, N10, N11 = 0, 0, 0, 0
    
    for i in range(len(df) - 1):
        if  df[i] < VaR0975 or df[i] > VaR0025:
            if df[i + 1] < VaR0975 or df[i + 1] > VaR0025: 
                N11 += 1
            elif df[i + 1] >= VaR0975 and df[i + 1] <= VaR0025:
                N10 += 1
            else:
                print(df[i], "err！！")
        else:
            if df[i + 1] < VaR0975 or df[i + 1] > VaR0025:
                N01 += 1
            elif df[i + 1] >= VaR0975 and df[i + 1] <= VaR0025:
                N00 += 1
            else:
                print(df[i], "err！！")
    
    Pi, Pi0, Pi1 = 0, 0, 0
    
    Pi0 = N01 / (N00 + N01)
    Pi1 = N11 / (N10 + N11)
    Pi = (N01 + N11) / (N00 + N01 + N10 + N11)# review is wrong!!

    
    null_hypothesis = 2 * (math.log(1 - Pi)*(N00 + N10) + math.log(Pi) * (N01 + N11))
    
    alternative_hypothesis = 2 * (math.log(1 - Pi0) * N00 + math.log(Pi0) * N01 + math.log(1 - Pi1) * N10 + math.log(Pi1**N11))
    
    LR = alternative_hypothesis - null_hypothesis ## Likelihood Ratio

    return LR
######################################### End of Independent Test (Christoffersen, 1998)   

# output
df_Christoffersen = pd.DataFrame(index = ["Full Name", "LR"], columns = list(df_HF_processed.columns[1:]) + ["S&P 500 Daily", "S&P 500 Weekly", "S&P 500 Monthly"])

for i in df_HF_processed.columns[1:]:
    df = df_HF_processed[i].astype("float").diff().dropna(inplace = False)
    LR = TwoSideChristoffersenIndependentTest(df)    
    df_Christoffersen[i]["LR"] = LR
    df_Christoffersen[i]["Full Name"] = subgroups[i]
    
df_Christoffersen["S&P 500 Daily"]["LR"] = TwoSideChristoffersenIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False))
df_Christoffersen["S&P 500 Weekly"]["LR"] = TwoSideChristoffersenIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False))
df_Christoffersen["S&P 500 Monthly"]["LR"] = TwoSideChristoffersenIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False))

#X^2(1)--0.95 = 3.841 which means if the null hypo is correct, only in 5% case we will observe this number (or bigger)
#SP500 weekly :158.48698968840523
#HEDG: 118.45092613865079

df_Christoffersen.T.to_csv(output_folder + "Christoffersen_result_two_side.csv")
pd.DataFrame(subgroups, index = ["Full Name"]).T.to_csv(output_folder + "Hedge_Fund_subgroup.csv")
            
            
######################################### Independent Test (Pajhede, 2015)   

def TwoSidePajhedeIndependentTest(df, k = 1): # It ~ (It−1, ..., It−k) #X^2(1)--0.95 = 3.841
    
    def Jt_1Calculator(df, t, k, VaR0975, VaR0025):
        for i in df[t - k:t]:
            if i < VaR0975 or i > VaR0025:
                return 1
        return 0    
        
    VaR0975 = df.quantile(q = 0.025)
    VaR0025 = df.quantile(q = 0.975)
    Phi, T00, T01, T10, T11, Ps, Pe = 0, 0, 0, 0, 0, 0, 0
    for t in range(k, len(df)):
        Jt_1 = Jt_1Calculator(df, t, k, VaR0975, VaR0025)
        It = (df[t] < VaR0975).astype(int) + (df[t] > VaR0025).astype(int)
        T11 += It * Jt_1
        T01 += It * (1 - Jt_1)
        T10 += (1 - It) * Jt_1
        T00 += (1 - It) * (1 - Jt_1)

    Phi = (T01 + T11) / (T00 + T01 + T10 + T11)
    Ps = T01 / (T01 + T00)
    Pe = T11 / (T11 + T10)

    null_hypothesis = 2 * (math.log(1 - Phi) * (T00 + T10) + math.log(Phi) * (T01 + T11))
    
    alternative_hypothesis = 2 * (math.log(1 - Ps) * T00 + math.log(Ps) * T01 + math.log(1 - Pe) * T10 + math.log(Pe**T11))
       
    LR = alternative_hypothesis - null_hypothesis## Likelihood Ratio
    
    return LR
######################################### End of Independent Test (Pajhede, 2015)   

# output
df_Pajhede = pd.DataFrame(index = ["Full Name", "LR"], columns = list(df_HF_processed.columns[1:]) + ["S&P 500 Daily", "S&P 500 Weekly", "S&P 500 Monthly"])

for i in df_HF_processed.columns[1:]:
    df = df_HF_processed[i].astype("float").diff().dropna(inplace = False)
    LR = TwoSidePajhedeIndependentTest(df, k = 1)    
    df_Pajhede[i]["LR"] = LR
    df_Pajhede[i]["Full Name"] = subgroups[i]
    
df_Pajhede["S&P 500 Daily"]["LR"] = TwoSidePajhedeIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False))
df_Pajhede["S&P 500 Weekly"]["LR"] = TwoSidePajhedeIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False))
df_Pajhede["S&P 500 Monthly"]["LR"] = TwoSidePajhedeIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False))
df_Pajhede.T.to_csv(output_folder + "Pajhede_result_two_side.csv")


# finaloutput!!
df_Final = pd.DataFrame(index = ["Full Name", "Christoffersen LR (two-side)", "Pajhede LR (K=1, two-side)", "Pajhede LR (K=3, two-side)", "Pajhede LR (K=5, two-side)", "Pajhede LR (K=10, two-side)"], columns = list(df_HF_processed.columns[1:]) + ["S&P 500 Daily", "S&P 500 Weekly", "S&P 500 Monthly"])

for i in df_HF_processed.columns[1:]:
    df = df_HF_processed[i].astype("float").diff().dropna(inplace = False)
    df_Final[i]["Christoffersen LR (two-side)"] = TwoSideChristoffersenIndependentTest(df)
    df_Final[i]["Pajhede LR (K=1, two-side)"] = TwoSidePajhedeIndependentTest(df, k = 1)
    df_Final[i]["Pajhede LR (K=3, two-side)"] = TwoSidePajhedeIndependentTest(df, k = 3)
    df_Final[i]["Pajhede LR (K=5, two-side)"] = TwoSidePajhedeIndependentTest(df, k = 5)
    df_Final[i]["Pajhede LR (K=10, two-side)"] = TwoSidePajhedeIndependentTest(df, k = 10)
    df_Final[i]["Full Name"] = subgroups[i]
  
df_Final["S&P 500 Daily"]["Christoffersen LR (two-side)"] = TwoSideChristoffersenIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False))
df_Final["S&P 500 Daily"]["Pajhede LR (K=1, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False), k = 1)
df_Final["S&P 500 Daily"]["Pajhede LR (K=3, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False), k = 3)
df_Final["S&P 500 Daily"]["Pajhede LR (K=5, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False), k = 5)
df_Final["S&P 500 Daily"]["Pajhede LR (K=10, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_daily["Adj Close"].diff().dropna(inplace = False), k = 10)


df_Final["S&P 500 Weekly"]["Christoffersen LR (two-side)"] = TwoSideChristoffersenIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False))
df_Final["S&P 500 Weekly"]["Pajhede LR (K=1, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False), k = 1)
df_Final["S&P 500 Weekly"]["Pajhede LR (K=3, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False), k = 3)
df_Final["S&P 500 Weekly"]["Pajhede LR (K=5, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False), k = 5)
df_Final["S&P 500 Weekly"]["Pajhede LR (K=10, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_weekly["Adj Close"].diff().dropna(inplace = False), k = 10)


df_Final["S&P 500 Monthly"]["Christoffersen LR (two-side)"] = TwoSideChristoffersenIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False))
df_Final["S&P 500 Monthly"]["Pajhede LR (K=1, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False), k = 1)
df_Final["S&P 500 Monthly"]["Pajhede LR (K=3, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False), k = 3)
df_Final["S&P 500 Monthly"]["Pajhede LR (K=5, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False), k = 5)
df_Final["S&P 500 Monthly"]["Pajhede LR (K=10, two-side)"] = TwoSidePajhedeIndependentTest(df_SP_monthly["Adj Close"].diff().dropna(inplace = False), k = 10)

df_Final.T.to_csv(output_folder + "Combined_result_two_side.csv")