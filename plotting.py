# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:00:03 2021

@author: 43739
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math
import datetime
from statsmodels.graphics.tsaplots import plot_acf

import data_processing

input_folder = "E:/Projects/Hedge fund independency/DataInput/"
subgroups, df_HF_processed = data_processing.HFDataProcess(input_folder)
df_SP_daily, df_SP_weekly, df_SP_monthly = data_processing.SPDataProcess(input_folder)


start_date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
end_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date()

df_HF_processed = df_HF_processed[(df_HF_processed.index >= pd.Timestamp(start_date)) & (df_HF_processed.index <= pd.Timestamp(end_date))]

df_SP_daily = df_SP_daily[(df_SP_daily.index >= pd.Timestamp(start_date)) & (df_SP_daily.index <= pd.Timestamp(end_date))].astype("float")
df_SP_weekly = df_SP_weekly[(df_SP_weekly.index >= pd.Timestamp(start_date)) & (df_SP_weekly.index <= pd.Timestamp(end_date))].astype("float")
df_SP_monthly = df_SP_monthly[(df_SP_monthly.index >= pd.Timestamp(start_date)) & (df_SP_monthly.index <= pd.Timestamp(end_date))].astype("float")


######################################### Profit/Loss Plot
# PnL = df_SP_daily.diff()["Adj Close"] # return
PnL = df_SP_monthly.diff()["Adj Close"] # return
# PnL = np.log(df_SP_daily["Adj Close"] / df_SP_daily["Adj Close"].shift(1)) #log return
plt.figure(figsize = (10, 7.5), dpi = 1000)
plt.plot(PnL, label = "Daily P/L in Doller($)")
plt.axhline(PnL.quantile(q = 0.05), color = "tab:gray", linestyle = "dashed")
plt.axhline(PnL.quantile(q = 0.95), color = "tab:gray")
plt.axhline(PnL.min(), color = "tab:red", linestyle = "dashed")



# important events
# PnL.index[104]# Lehman Brothers declares bankruptcy
plt.annotate("Lehman Brothers declares bankruptcy -- 9/15/2008", 
              xy=(PnL.index[104], PnL[104]), 
              xytext =(PnL.index[0], PnL[50]-200),
              arrowprops=dict(width = 1, headwidth = 4, facecolor = 'black', shrink = 0.05))

# PnL.index[240]# WHO Director-General declared the Covid-19 outbreak a public health emergency of international concern
plt.annotate("WHO Director-General declared the Covid-19 outbreak \na public health emergency of international concern -- 1/30/2020", 
              xy=(PnL.index[240], PnL[240]-100), 
              xytext =(PnL.index[50], PnL[100]-350),
              arrowprops=dict(width = 1, headwidth = 4, facecolor = 'black', shrink = 0.05))

plt.legend(["Monthly P/L in Doller($)", "0.05 quantile (95%VaR) = " + str(PnL.quantile(q = 0.05))[:6], "0.95 quantile = " + str(PnL.quantile(q = 0.95))[:6], "Minimum = " + str(PnL.min())[:6]], loc = "upper left")
plt.xlabel("Year")
plt.ylabel("P/L in Dollor($)")
plt.title("SP500 Monthly P/L")
plt.show()


######################################### Hist

plt.figure(figsize=(10,7.5), dpi= 1000)
sns.distplot(PnL, fit=norm, kde=False, bins = 100, color="tab:blue")


plt.legend(["P/L Fitted normal","P/L Histogram"], loc = "upper left")
plt.xlabel("P/L in Dollor($)")
plt.ylabel("Frequency")
plt.title("SP500 Monthly P/L Histogram")
plt.show()


df_SP_monthly.diff()["Adj Close"].kurt()
df_SP_weekly.diff()["Adj Close"].kurt()

########################### error analysis
PnL.kurt()+3
PnL.skew()


df_HF_processed[df_HF_processed.columns[1:]].astype("float").diff().kurt()

df_HF_processed[df_HF_processed.columns[1:]].astype("float").diff().skew()


# for two subindex pass independent test but has large kurtosis, they are due to extreme large number of loss
# this shows us a fault of VaR which only care about violation but not the seriousness of violation
# that why ES sometimes perform better than VaR
df_HF_processed["HEDG_EQNTR"].astype("float").dropna().diff().kurt()

kk = df_HF_processed["HEDG_EQNTR"].astype("float").dropna().diff()

akk = kk[kk>-150].kurt()



df_HF_processed["HEDG_CVARB"].astype("float").dropna().diff().kurt()

kk = df_HF_processed["HEDG_CVARB"].astype("float").dropna().diff()

akk = kk[kk>-30].kurt()


########################### 36 month std plot

for i in df_HF_processed.columns[1:]:
    df = df_HF_processed[i].astype("float").diff().dropna(inplace = False)
    Previous12Monthstd = pd.DataFrame(columns = ["std", "kurtosis"])
    for j in range(len(df) - 11):
        std = df[j : j + 12].std()
        kurtosis = df[j : j + 11].kurt() + 3
        date = df.index[j + 11]
        Previous12Monthstd.loc[date] = [std, kurtosis]
    plt.figure(figsize = (10, 7.5), dpi = 1000)
    Previous12Monthstd.plot()
    print(i, Previous12Monthstd.std())
    plt.title(i + " yearly std")
    plt.show()
        

########################### ACF

##SP500
#one-side

VaR095 = PnL[1:].quantile(q = 0.05)

I = [1 if i < VaR095 else 0 for i in PnL[1:]]
df_I = pd.DataFrame(I, index = PnL.index[1:])

plt.figure(figsize = (10, 7.5), dpi = 1000)
plot_acf(df_I)
plt.title("SP 500 One-sided ACF")
plt.show()

#two-side
VaR0975 = PnL[1:].quantile(q = 0.025)
VaR0025 = PnL[1:].quantile(q = 0.975)

I = [1 if (i < VaR0975 or i > VaR0025) else 0 for i in PnL[1:]]
df_I = pd.DataFrame(I, index = PnL.index[1:])

plt.figure(figsize = (10, 7.5), dpi = 1000)
plot_acf(df_I)
plt.title("SP 500 Two-sided ACF")
plt.show()


for i in df_HF_processed.columns[1:]:
    ##HF
    HF = df_HF_processed[i].astype("float").diff().dropna(inplace = False)
    #one-side
    
    VaR095 = HF.quantile(q = 0.05)
    
    I = [1 if i < VaR095 else 0 for i in HF]
    df_I = pd.DataFrame(I, index = HF.index)
    
    plt.figure(figsize = (10, 7.5), dpi = 1000)
    plot_acf(df_I)
    plt.title(i + " One-sided ACF")
    plt.show()
    
    #two-side
    VaR0975 = HF.quantile(q = 0.025)
    VaR0025 = HF.quantile(q = 0.975)
    
    I = [1 if (i < VaR0975 or i > VaR0025) else 0 for i in HF]
    df_I = pd.DataFrame(I, index = HF.index)
    
    plt.figure(figsize = (10, 7.5), dpi = 1000)
    plot_acf(df_I)
    plt.title(i + " Two-sided ACF")
    plt.show()



##HF
HF = df_HF_processed["HEDG_MRARB"].astype("float").diff().dropna(inplace = False)
#one-side

VaR095 = HF.quantile(q = 0.05)

I = [1 if i < VaR095 else 0 for i in HF]
df_I = pd.DataFrame(I, index = HF.index)

plt.figure(figsize = (10, 7.5), dpi = 1000)
plot_acf(df_I)
plt.title("Event Driven Risk Arbitrage One-sided ACF")
plt.show()

#two-side
VaR0975 = HF.quantile(q = 0.025)
VaR0025 = HF.quantile(q = 0.975)

I = [1 if (i < VaR0975 or i > VaR0025) else 0 for i in HF]
df_I = pd.DataFrame(I, index = HF.index)

plt.figure(figsize = (10, 7.5), dpi = 1000)
plot_acf(df_I)
plt.title("Event Driven Risk Arbitrage Two-sided ACF")
plt.show()

