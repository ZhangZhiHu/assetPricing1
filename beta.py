# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  13:31
# NAME:assetPricing1-beta.py

from dout import *
from zht.utils.mathu import get_inter_index, reg



def usingDailyData():
    '''
    use the daily returns to calculate beta,month by month

    Returns:

    '''
    dict1={'1M':15,'3M':50,'6M':100,'12M':200,'24M':450}

    rets=read_df('stockRetD', freq='D')
    retm=read_df('mktRetD', freq='D')
    rf=read_df('rfD', freq='D')

    rets,retm,rf=get_inter_index([rets,retm,rf])

    erets=rets.sub(rf['rf'],axis=0)
    eretm=retm.sub(rf['rf'],axis=0)

    sids=erets.columns.tolist()
    months=erets.resample('M').last().index.tolist()

    cmb=erets.copy(deep=True)
    cmb['eretm']=eretm.iloc[:,0]

    for length,thresh in dict1.items():
        betadf=pd.DataFrame()
        r2df=pd.DataFrame()
        for month in months:
            tmpdf=cmb.loc[:month,:]
            for sid in sids:
                regdf=tmpdf.loc[:,[sid,'eretm']]
                regdf=regdf.last(length)
                regdf=regdf.dropna(how='any',axis=0)
                if regdf.shape[0]>thresh:
                    coefs,tvalues,r2,resids=reg(regdf)
                    betadf.loc[month.strftime('%Y-%m-%d'),sid]=coefs[1]
                    # do not use loc when the index is timestamp
                    r2df.loc[month.strftime('%Y-%m-%d'),sid]=r2
            print(month)
        betadf.to_csv(os.path.join(DATA_PATH,length+'.csv'))
        # r2df.to_csv(os.path.join(DATA_PATH,length+'_r2.csv'))

def usingMonthlyData():
    '''
    calculate betas by using monthly return,month by month.
    Returns:

    '''
    dict2={'12M':10,'24M':20,'36M':24,'60M':24}
    rets=read_df('stockRetM', freq='M')
    retm=read_df('mktRetM', freq='M')
    rf=read_df('rfM', freq='M')

    rets,retm,rf=get_inter_index([rets,retm,rf])

    erets = rets.sub(rf['rf'], axis=0)
    eretm = retm.sub(rf['rf'], axis=0)

    sids=erets.columns.tolist()
    months=erets.index.tolist()

    cmb = erets.copy(deep=True)
    cmb['eretm'] = eretm.iloc[:, 0]

    for length, thresh in dict2.items():
        betadf = pd.DataFrame()
        r2df = pd.DataFrame()
        # sids=sids[:30]#TODO
        for month in months:
            tmpdf = cmb.loc[:month, :]
            for sid in sids:
                regdf = tmpdf.loc[:, [sid, 'eretm']]
                regdf = regdf.last(length)
                regdf = regdf.dropna(how='any', axis=0)
                if regdf.shape[0] > thresh:
                    coefs, tvalues, r2, resids = reg(regdf)
                    betadf.loc[month.strftime('%Y-%m-%d'), sid] = coefs[1]
                    # do not use loc when the index is timestamp
                    r2df.loc[month.strftime('%Y-%m-%d'), sid] = r2
            print(month)

        name=str(int(int(length[:-1])/12))+'Y'
        betadf.to_csv(os.path.join(DATA_PATH, name + '.csv'))
        # r2df.to_csv(os.path.join(DATA_PATH, name + '_r2.csv'))

