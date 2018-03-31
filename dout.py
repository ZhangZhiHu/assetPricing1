# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  13:33
# NAME:assetPricing1-dout.py
import os
import pandas as pd
from config import *
from pandas.tseries.offsets import MonthEnd,Day


def read_df(fn, freq,repository=DATA_PATH):
    '''
    read df from DATA_PATH

    :param fn:
    :return:
    '''
    df=pd.read_csv(os.path.join(repository, fn + '.csv'), index_col=0)
    if freq=='M':
        df.index=pd.to_datetime(df.index)+MonthEnd(0)
        # df.index=pd.to_datetime(df.index).to_period(freq).to_timestamp(freq)
    elif freq=='D':
        df.index=pd.to_datetime(df.index)

    df.index.name='t'
    return df
