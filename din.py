# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-02-11  16:23
# NAME:assetPricing-din.py

import numpy as np

import pandas as pd
from config import *
from dout import read_df
import os

from zht.utils.dfu import filterDf
import statsmodels.formula.api as sm
from pandas.tseries.offsets import MonthEnd


def _readFromSrc(tbname):
    df=pd.read_csv(os.path.join(r'D:\zht\database\quantDb\sourceData\gta\data\csv',tbname+'.csv'))
    return df

def _get_df(tbname, varname, indname, colname, fn):
    '''
    get df by use pd.pivot_table

    :param tbname:table name
    :param varname:variable name in tbname
    :param indname:name in the table to be set as index of the returnd df
    :param colname:name in the table to be set as column of the returned df
    :param fn:the name of df to be saved
    :return:
    '''
    path=os.path.join(DATA_PATH, fn + '.csv')
    table=_readFromSrc(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    df.index=pd.to_datetime(df.index)
    df.to_csv(path)

def get_stockRetD():
    # get stock daily stock return
    tbname='TRD_Dalyr'
    varname='Dretwd'#考虑现金红利再投资的收益
    indname='Trddt'
    colname='Stkcd'
    fn='stockRetD'
    _get_df(tbname, varname, indname, colname, fn)

def get_mktRetD():
    # get daily market return
    newName = 'mktRetD'
    path=os.path.join(DATA_PATH, newName + '.csv')

    tbname = 'TRD_Cndalym'
    indVar = 'Trddt'

    targetVar = 'Cdretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    q1 = 'Markettype == 21'  # 21=综合A股和创业板

    df = _readFromSrc(tbname)
    df = filterDf(df, q1)

    df = df.set_index(indVar)
    df = df.sort_index()
    df = df[[targetVar]]
    del df.index.name
    df.columns = [newName]

    df.index = pd.to_datetime(df.index)
    df.to_csv(path)

def get_stockCloseD():
    #get daily stock close price
    tbname='TRD_Dalyr'
    varname='Clsprc'
    indname='Trddt'
    colname='Stkcd'
    fn='stockCloseD'
    _get_df(tbname, varname, indname, colname, fn)

def _get_rf(freq='D'):
    '''
    parse risk free rate from the database
    Args:
        freq: D (daily),W (weekly),M (monthly)

    Returns:

    '''
    dic={'D':'Nrrdaydt','W':'Nrrwkdt','M':'Nrrmtdt'}

    tname = 'TRD_Nrrate'
    src = _readFromSrc(tname)
    #NRI01=定期-整存整取-一年利率；TBC=国债票面利率,根据复利计算方法，
    # 将年度的无风险利率转化为月度数据
    src=src[src['Nrr1']=='NRI01']
    src=src.set_index('Clsdt')
    del src.index.name

    rf=src[[dic[freq]]][2:]#delete the first two rows
    rf.columns=['rf']

    rf.index=pd.to_datetime(rf.index)
    if freq in ['W','M']:
        rf=rf.resample(freq).last()

    return rf/100.0

def get_rfD():
    '''
    get daily risk free return
    :return:
    '''
    df=_get_rf(freq='D')
    df.to_csv(os.path.join(DATA_PATH, 'rfD.csv'))

def get_rfM():
    '''
    get monthly risk free return

    :return:
    '''
    df=_get_rf(freq='M')
    df.to_csv(os.path.join(DATA_PATH, 'rfM.csv'))

#market data----------monthly---------------------------------
'''
there is no filter query for market data,but for those
financial data,the filter query is needed.
'''

def get_stockRetM():
    '''
    monthly stock return with dividend

    Args:
        recal: if True,recalculate the indicator

    Returns:

    '''
    tbname = 'TRD_Mnth'
    varname='Mretwd'#考虑现金红利再投资的收益
    indname='Trdmnt'
    colname='Stkcd'
    fn='stockRetM'
    _get_df(tbname, varname, indname, colname, fn)

def get_mktRetM():
    newName='mktRetM'
    path=os.path.join(DATA_PATH, newName + '.csv')

    tbname = 'TRD_Cnmont'

    indVar = 'Trdmnt'

    targetVar = 'Cmretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    q1 = 'Markettype == 21'  # 21=综合A股和创业板

    df = _readFromSrc(tbname)
    df = filterDf(df, q1)

    df = df.set_index(indVar)
    df = df.sort_index()
    df = df[[targetVar]]
    del df.index.name
    df.columns = [newName]

    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()
    df.to_csv(path)

def get_capM():
    '''
    get stock monthly circulation market capitalization

    :return:
    '''
    tbname='TRD_Mnth'
    varname='Msmvosd' #月个股流通市值，单位 人民币
    indname='Trdmnt'
    colname='Stkcd'
    fn='mktCap'
    _get_df(tbname, varname, indname, colname, fn)


#wind data------------------------------------------------------
def _clear_data_from_wind_cg(name, freq):
    df=pd.read_csv(os.path.join(WIND_SRC_PATH,name+'.csv'),
                   index_col=0,skipfooter=3)
    df.index=pd.to_datetime(df.index).to_period(freq).to_timestamp(freq)
    df.to_csv(os.path.join(WIND_PATH,name+'.csv'))


# financial indicators-------------------------------------------
def get_bps_gta():
    tbname = 'FI_T9'
    varname = 'F091001A'
    indname = 'Accper'
    colname = 'Stkcd'
    fn = 'bps_gta'
    _get_df(tbname, varname, indname, colname, fn)

def get_bps_wind():
    '''
    from code generator by use

    w.wsd("000001.SZ,000002.SZ,000004.SZ,000005.SZ,000006.SZ", "bps", "2017-02-04", "2018-03-05", "currencyType=;Period=Q;Fill=Previous")

    :return:
    '''
    _clear_data_from_wind_cg('bps', freq='M')


#get stock close price yearly
def get_stockCloseY():
    tbname='TRD_Year'
    varname='Yclsprc'
    indname='Trdynt'
    colname='Stkcd'
    fn='stockCloseY'

    path=os.path.join(DATA_PATH, fn + '.csv')
    table=_readFromSrc(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    df.index=pd.to_datetime(df.index,format='%Y').to_period('Y').to_timestamp('Y')
    df.to_csv(path)

def get_ff3():
    '''
    from resset data

    :return:
    '''
    query1='Exchflg == 0' #所有交易所
    query2='Mktflg == A'#只考虑A股

    df=pd.read_csv(r'D:\zht\database\quantDb\sourceData\resset\data\THRFACDAT_MONTHLY.csv')
    df=df[(df['Exchflg']==0) & (df['Mktflg']=='A')]
    df=df.set_index('Date')
    df=df[['Rmrf_tmv','Smb_tmv','Hml_tmv']]#weighted with tradable capitalization
    df.columns=['rp','smb','hml']
    df.to_csv(os.path.join(DATA_PATH,'ff3.csv'))

def get_ff3_gta():
    direc=r'D:\zht\database\quantDb\sourceData\gta\data_20180314\source\csv'
    df=pd.read_csv(os.path.join(direc,'STK_MKT_ThrfacMonth.csv'),index_col=0)
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    df=df[df['MarkettypeID']=='P9709'][['TradingMonth','RiskPremium1','SMB1','HML1']]
    df.columns=['t','rp','smb','hml']
    df=df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH,'ff3_gta.csv'))


def get_ffc_gta():
    direc=r'D:\zht\database\quantDb\sourceData\gta\data_20180314\source\csv'
    df=pd.read_csv(os.path.join(direc,'STK_MKT_FivefacMonth.csv'),index_col=0)
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    #2*3 投资组合
    df=df[(df['MarkettypeID']=='P9709') & (df['Portfolios']==1)][
        ['TradingMonth','RiskPremium1','SMB1','HML1','RMW1','CMA1']]
    df.columns=['t','rp','smb','hml','rmw','cma']
    df=df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH,'ff5_gta.csv'))

def get_ffc_gta():
    direc = r'D:\zht\database\quantDb\sourceData\gta\data_20180314\source\csv'
    df = pd.read_csv(os.path.join(direc, 'STK_MKT_CarhartFourFactors.csv'), index_col=0)
    # P9709 全部A股市场包含沪深A股和创业板
    # 流通市值加权
    df = df[df['MarkettypeID'] == 'P9709'][
        ['TradingMonth', 'RiskPremium1', 'SMB1', 'HML1', 'UMD2']]
    df.columns = ['t', 'rp', 'smb', 'hml', 'mom']
    df = df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH, 'ffc_gta.csv'))

#TODO：liquidity of GTA
def get_liquidity_ps():
    direc = r'D:\zht\database\quantDb\sourceData\gta\data_20180314\source\csv'
    df = pd.read_csv(os.path.join(direc, 'Liq_PSM_M.csv'), index_col=0)
    #MarketType==21   综合A股和创业板
    # 流通市值加权，but on the page 310,Bali use total market capilization
    condition1=(df['MarketType']==21)
    condition2=(df['ST']==1)#delete the ST stocks

    df = df[condition1 & condition2][['Trdmnt','AggPS_os']]
    df.columns=['t','rm']
    df=df.set_index('t')

    df.index=pd.to_datetime(df.index).to_period('M').to_timestamp('M')


    df=df.sort_index()
    df['rm_ahead']=df['rm'].shift(1)
    df['delta_rm']=df['rm']-df['rm'].shift(1)
    df['delta_rm_ahead']=df['rm_ahead']-df['rm_ahead'].shift(1)
    #df.groupby(lambda x:x.year).apply(lambda df:df.shape[0])
    #TODO: we don't know the length of window to regress.In this place,we use the five years history
    def regr(df):
        if df.shape[0]>30:
            return sm.ols(formula='delta_rm ~ delta_rm_ahead + rm_ahead',data=df).fit().resid[0]
        else:
            return np.NaN

    window=60 # not exact 5 years
    lm=pd.Series([regr(df.loc[:month][-60:]) for month in df.index],index=df.index)
    lm.name='lm'

    ret = read_df('stockRetM', freq='M')
    rf = read_df('rfM', freq='M')
    eret = ret.sub(rf['rf'], axis=0)
    eret = eret.stack()
    eret.index.names=['t','sid']
    eret.name='eret'

    ff3_new=read_df('ff3','M')
    ff3=read_df('ff3_gta','M')
    factors=pd.concat([ff3,lm],axis=1)

    comb=eret.to_frame().join(factors)

    def _for_one_month(df):
        if df.shape[0] >=30:
            return sm.ols(formula='eret ~ rp + smb + hml + lm', data=df).fit().params['lm']
        else:
            return np.NaN

    def _get_result(df):
        thresh=30#30 month
        if df.shape[0]>thresh:
            values=[]
            sid=df.index[0]
            df = df.reset_index(level='sid', drop=True)
            months=df.index.tolist()[thresh:]
            for month in months:
                subdf=df.loc[:month][-60:]
                subdf=subdf.dropna()
                # df=df.reset_index(level='sid',drop=True).loc[:month].last(window)
                values.append(_for_one_month(subdf))
            print(sid)
            return pd.Series(values,index=months)

    result=comb.groupby('sid').apply(_get_result)
    result.unstack('sid').to_csv(os.path.join(DATA_PATH,'liqBeta.csv'))


def get_hxz4():
    '''
    D:\app\python27\zht\researchTopics\assetPricing\calFactors.py\get_hxz4Factors()

    :return:
    '''
    direc=r'E:\a\quantDb\researchTopics\assetPricing\hxz4\factor'

    fns=['rsmb','ria','rroe']

    dfs=[]
    for fn in fns:
        df=pd.read_csv(os.path.join(direc,fn+'.csv'),index_col=0)
        df.index.name='t'
        df.columns=[fn]
        dfs.append(df)
    df.head()
    comb=pd.concat(dfs,axis=1)
    comb.index=pd.to_datetime(comb.index)+MonthEnd()
    ff3=read_df('ff3_gta','M')
    comb['rp']=ff3['rp']
    comb.to_csv(os.path.join(DATA_PATH,'hxz4.csv'))



def read_src_new(tbname):
    direc=r'D:\zht\database\quantDb\sourceData\gta\data_20180314\source\csv'
    df=pd.read_csv(os.path.join(direc,tbname+'.csv'),index_col=0)
    return df

def get_ff3D():
    tbname='STK_MKT_ThrfacDay'
    df=read_src_new(tbname)
    condition1=df['MarkettypeID']=='P9707'
    # P9709 全部A股市场包含沪深A股和创业板
    # 流通市值加权
    df = df[condition1][
        ['TradingDate', 'RiskPremium1', 'SMB1', 'HML1']]
    df.columns = ['t', 'rp', 'smb', 'hml']
    df = df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH, 'ff3D.csv'))


# TODO:sets the format of the index before put them into processedcsv,with pd.to_datetime .to_period()

def run():
    get_stockRetD()
    get_mktRetD()
    get_stockCloseD()
    get_rfD()
    get_rfM()
    get_stockRetM()
    get_mktRetM()
    get_capM()
    get_bps_gta()
    get_bps_wind()
    get_stockCloseY()
    get_ff3()



