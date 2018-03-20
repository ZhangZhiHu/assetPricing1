# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-18  15:32
# NAME:assetPricing1-skewness.py

from dout import *
import numpy as np
import statsmodels.formula.api as sm


def _get_comb():
    retD = read_df('stockRetD', freq='D')
    retD = retD.stack()
    retD.index.names = ['t', 'sid']
    retD.name = 'ret'
    ff3D = read_df('ff3D', 'D')
    mktD = read_df('mktRetD', 'D')
    mktD.columns=['mkt']
    mktD['mkt_square']=mktD['mkt']**2
    combD = retD.to_frame().join(ff3D)
    combD=combD.join(mktD)

    retM = read_df('stockRetM', freq='M')
    retM = retM.stack()
    retM.index.names = ['t', 'sid']
    retM.name = 'ret'
    ff3M = read_df('ff3_gta', freq='M')
    mktM = read_df('mktRetM', 'M')
    mktM.columns=['mkt']
    mktM['mkt_square']=mktM['mkt']**2
    combM = retM.to_frame().join(ff3M)
    combM=combM.join(mktM)

    return combD,combM

def _skew(subx):
    return subx['ret'].skew()

def _coskew(subx):
    coskew=sm.ols('ret ~ mkt + mkt_square',data=subx).fit().params['mkt_square']
    return coskew

def _idioskew(subx):
    resids = sm.ols('ret ~ rp + smb + hml', data=subx).fit().resid
    idioskew = resids.skew()
    return idioskew

def _for_one_stock(x, months, history, thresh, type_func):
    '''
    calculate the indicator for one stock,and get a time series

    :param x:series or pandas DataFrame
    :param months:list,contain the months to calculate the indicators
    :param history:history length,such as 12M
    :param thresh:the mimium required observe number
    :param type_func:the function name from one of [_skew,_coskew,_idioskew]
    :return:time series
    '''
    sid=x.index.get_level_values('sid')[0]
    x=x.reset_index('sid',drop=True)
    values=[]
    for month in months:
        subx=x.loc[:month].last(history)
        subx=subx.dropna()
        if subx.shape[0]>thresh:
            values.append(type_func(subx))
        else:
            values.append(np.nan)
    print(sid)
    return pd.Series(values,index=months)

def _cal(comb, freq, dict, type_func, fn):
    values = []
    names = []
    for history, thresh in dict.items():
        days = comb.index.get_level_values('t').unique()
        months = days[days.is_month_end]
        value = comb.groupby('sid').apply(lambda df: _for_one_stock(df, months, history, thresh,type_func))
        values.append(value.T)
        names.append(freq+'_'+history)
    result = pd.concat(values, axis=0, keys=names)
    result.to_csv(os.path.join(DATA_PATH,fn+'.csv'))
    return result

def cal_skewnewss():
    dictD = {'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}
    dictM = {'12M': 10, '24M': 20, '36M': 24, '60M': 24}

    combD,combM=_get_comb()

    _cal(combD, 'D', dictD, _skew, 'skewD')
    _cal(combD, 'D', dictD, _coskew, 'coskewD')
    _cal(combD, 'D', dictD, _idioskew, 'idioskewD')

    _cal(combM, 'M', dictM, _skew, 'skewM')
    _cal(combM, 'M', dictM, _coskew, 'coskewM')
    _cal(combM, 'M', dictM, _idioskew, 'idioskewM')

if __name__=='__main__':
    cal_skewnewss()











