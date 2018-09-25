import pandas as pd
import numpy as np

import json

df=pd.read_csv(r'C:\Users\uverma\Downloads\all\train.csv')

df.dtypes

k=['geoNetwork','totals','trafficSource']

df.head()

[json.loads(r) for r in df.head()['device']]

[json.loads(r) for r in df.head()['geoNetwork']]

[json.loads(r) for r in df.head()["totals"]]

[json.loads(r) for r in df.head()["trafficSource"]]

RAW_TARGET_COL = "totals"
raw_target_s = df[RAW_TARGET_COL]

for id,raw_target_row in raw_target_s.sample(30).iteritems():
    print(raw_target_row)

records = []
for index, raw_target_row in raw_target_s.iteritems():
    parsed_target_row = eval(raw_target_row)
    records.append(parsed_target_row)

records

parsed_target_df = pd.DataFrame(records)

sum(parsed_target_df.transactionRevenue.isnull()) - len(parsed_target_df.transactionRevenue) 

len(parsed_target_df.transactionRevenue) 

!pip install missingno

import missingno as msno

msno.matrix(df)

msno.matrix(parsed_target_df)

def per_missing_value_count(col):
    return col.isnull().sum()*100/col.shape[0]
    

for x in  parsed_target_df.columns:
    print(  '{} has {} % of missing values'.format(x,per_missing_value_count(parsed_target_df[x])))

parsed_target_df.fillna(0,inplace=True)

import seaborn as sns
import matplotlib.pylab as plt

fig , ax = plt.subplots(1,1,figsize=(12,8))

sns.distplot(np.log(parsed_target_df.loc[:,parsed_target_df["transactionRevenue"]>0]))

import sys

import multiprocessing

multiprocessing.cpu_count()

sns.distplot(np.log(parsed_target_df[parsed_target_df.transactionRevenue.astype(float)>0].transactionRevenue))

parsed_target_df["transactionRevenue"] = parsed_target_df.transactionRevenue.astype(float)

parsed_target_df.groupby()

parsed_target_df.head()

df.head()


recordD=[]
for index , devicedetail in df["device"].iteritems():
    #recordD.append(eval(devicedetail))
    recordD.append(json.loads(devicedetail))

def dataframe_from_json_col(col):

    recordD=[]
    for index , detail in df[col].iteritems():
        #recordD.append(eval(devicedetail))
        recordD.append(json.loads(detail))
        #print("I am inside for loop")
    globals()["df_{}".format(col)]=pd.DataFrame(recordD)
    return globals()["df_{}".format(col)]
    #print("I am outside for loop")    

df["device"].iteritems()

dataframe_from_json_col('')

df_device.deviceCategory.unique()

dataframe_from_json_col("device")

recordD[0:10]

deviceDf=pd.DataFrame(recordD)

deviceDf["browserSize"].unique()
#df.name.unique()

for col in deviceDf.columns:
    print('{} = {}'.format(col,deviceDf[col].unique()))

x for x in 

ls=['browser' ,'deviceCategory','isMobile',
    'operatingSystem']

 for col in df.columns :
        eval(varname+col "=" 20)

import functools

[json.loads(r) for index, r in 
 df_trafficSource['adwordsClickInfo'].iteritems()]

[json.loads(r) for r in 

df_traficAd=pd.DataFrame([x for index,x in df_trafficSource['adwordsClickInfo'].iteritems()])

for col in df_traficAd.columns:
    print('{} = {}'.format(col,df_traficAd[col].unique()))

df12=functools.reduce(lambda x,y:pd.merge(x,y,left_index=False, right_index=False),[map(dataframe_from_json_col,ls)])

df1=pd.merge(df,df_device[ls],left_index=True,right_index=True)

[x for x in map(dataframe_from_json_col,k)]

[map(dataframe_from_json_col,ls)]



for col in df.columns:
    globals()['_{}'.format(col)]=1
    print(    globals()['_{}'.format(col)])

globals()['_{}'.format(x)]=5

_5

for col in df.columns:
    print(col)

