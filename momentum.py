# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-10  16:10
# NAME:assetPricing1-momentum.py


from dataset import dataset
from config import *
from dout import read_df
from zht.utils.mathu import get_inter_frame
import numpy as np
import pandas as pd



def get_momentum():
    stockRetM=read_df('stockRetM','M')
    stk=stockRetM.stack()
    stk.index.names=['t','sid']

    #lagged 1 month
    d_lag={'mom':[11,9],
       'r12':[12,10],
       'r6':[6,5]}

    #nonlagged
    d_nonlag={'R12M':[12,10],
            'R9M':[9,7],
            'R6M':[6,5],
            'R3M':[3,3]}

    def _cal_cumulated_return(s):
        return np.cumprod(s+1)[-1]-1

    def _before(s,interval,min_periods):
        #for d_before,do not include return of time t
        return s.rolling(interval,min_periods=min_periods).apply(lambda s:_cal_cumulated_return(s[:-1]))

    def _upto(s,interval,min_periods):
        return s.rolling(interval,min_periods=min_periods).apply(_cal_cumulated_return)

    ss=[]
    names=[]
    for bn,bp in d_lag.items():
        ser=stk.groupby('sid').apply(lambda s:_before(s,bp[0],bp[1]))
        ss.append(ser)
        names.append(bn)

    for un,up in d_nonlag.items():
        ser=stk.groupby('sid').apply(lambda s:_upto(s,up[0],up[1]))
        ss.append(ser)
        names.append(un)

    momentum=pd.concat(ss,axis=1,keys=names)
    momentum=momentum*100

    #TODO:which type to save staked or with different files
    for col in momentum.columns:
        momentum[col].unstack().to_csv(os.path.join(DATA_PATH,col+'.csv'))


if __name__=='__main__':
    get_momentum()


