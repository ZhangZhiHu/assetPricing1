# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  13:56
# NAME:assetPricing1-value.py

from config import *
from dout import read_df
from zht.utils.mathu import get_inter_frame
import numpy as np
import pandas as pd


#compare the bps of wind with bv of gta
def compare_wind_gta_bps():
    '''
    the result is different a lot!!!

    :return:
    '''
    bps_wind=read_df('bps','M',WIND_PATH)
    bps_gta=read_df('bps_gta','M',DATA_PATH)

    bps_wind.columns=[str(int(col[:-3])) for col in bps_wind.columns]
    bps_wind=bps_wind.sort_index(axis=1)

    bps_gta=bps_gta.sort_index(axis=1)

    bps_wind,bps_gta=get_inter_frame([bps_wind,bps_gta])


def get_bm():
    '''
    this function can be bookmarked as a snippet of how to manipulate date index
    in Pandas

    A little different with the book,here we use be and me for one share,
    but the data in the book is for all floating shares.However,it doesn't
    affect the bm.

    :return:
    '''
    be=read_df('bps_gta','M',DATA_PATH)
    be=be[be.index.month==12]

    me=read_df('stockCloseY','M',DATA_PATH)

    be,me=get_inter_frame([be,me])

    bm=be/me
    bm[bm<=0]=np.nan #delete those sample with bm<0
    bm=bm.shift(1,freq='6M')

    newIndex=pd.date_range(bm.index[0],bm.index[-1],freq='M')
    bm=bm.reindex(index=newIndex)
    bm=bm.fillna(method='ffill',limit=11)
    bm.to_csv(os.path.join(DATA_PATH,'bm.csv'))

    logbm=np.log(bm)
    logbm.to_csv(os.path.join(DATA_PATH,'logbm.csv'))

