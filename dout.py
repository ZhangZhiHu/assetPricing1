# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  13:33
# NAME:assetPricing1-dout.py
import os
import pandas as pd
from config import *
from pandas.tseries.offsets import MonthEnd


def read_df(fn, freq,repository=DATA_PATH):
    '''
    read df from DATA_PATH

    :param fn:
    :return:
    '''
    df=pd.read_csv(os.path.join(repository, fn + '.csv'), index_col=0)
    df.index=pd.to_datetime(df.index)+MonthEnd()
    # df.index=pd.to_datetime(df.index).to_period(freq).to_timestamp(freq)
    df.index.name='t'
    return df


df1=pd.read_csv(os.path.join(DATA_PATH,'ff3.csv'),index_col=0)
df2=pd.read_csv(os.path.join(DATA_PATH,'ff3_gta.csv'),index_col=0)
d1=pd.to_datetime(df1.index[0])+MonthEnd()
d2=pd.to_datetime(df2.index[0])+MonthEnd()


# pd.Timestamp(df1.index[0])+MonthEnd()
# pd.Timestamp(df2.index[0])+MonthEnd()


df1.index=pd.to_datetime(df1.index)+MonthEnd()
df2.index=pd.to_datetime(df2.index)+MonthEnd()

d3=df1.index[0]
d4=df2.index[0]

type(d3)==type(d4)

df1.index[0]==df2.index[20]
