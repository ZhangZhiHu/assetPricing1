# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  14:01
# NAME:assetPricing1-dataset.py

from config import *
from dout import read_df

import pandas as pd


class Factor:
    def __init__(self,name,indicators):
        self.name=name
        self.indicators=indicators


class Dataset:
    def __init__(self,factors):
        '''
        :param factors: a list of Factor()
        '''
        self.factors=factors
        self.information={factor.name:factor.indicators for factor in self.factors}
        self.data=self._combine_all_data()

    def _combine_all_data(self):
        ret = read_df('stockRetM', freq='M')
        rf = read_df('rfM', freq='M')
        eret = ret.sub(rf['rf'], axis=0)
        eret=eret.stack()
        eret.name='eret'
        #TODO: create a df to store eret

        all_indicators=[ind for l_values in self.information.values()
                        for ind in l_values]
        #TODO:
        '''
        all the data are shift forward one month except for eret,so the index denotes time t+1,
        and all the data except for the eret are from time t,only eret are from time t+1.We adjust 
        the dataset for these reasons:
        1. we will sort the indicators in time t to construct portfolios and analyse the eret in time
            t+1
        2. We need to make sure that the index for eret is corresponding to the time it was calcualted.
            If we shift back the eret in this place (rather than shift forward the other indicators),we have
            to shift forward eret again when we regress the portfolio eret on mktRetM in the function _alpha 
            in template.py
            
        
        '''
        dfs=[eret]+[read_df(ind,'M').shift(1).stack() for ind in all_indicators]
        data=pd.concat(dfs,axis=1,keys=['eret']+all_indicators)
        data.index.names=['t','sid']

        #add mktRetM
        mktRetM = read_df('mktRetM', freq='M')
        mktRetM.index.name = 't'

        data=data.join(mktRetM)#combine multiIndex dataframe with single index dataframe
        #truncate the sample
        return data[data.index.get_level_values('t').year>=1996]


    def get_by_factorname(self,name):
        return self.data[self.information[name]].copy()

    def get_by_indicators(self,indicators):
        return self.data[indicators].copy()

    def get_all_data(self):
        return self.data.copy()

BETA=Factor('beta',BETA_NAMES)
SIZE=Factor('size',SIZE_NAMES)
VALUE=Factor('value',VALUE_NAMES)
MOM=Factor('momentum',MOM_NAMES)

dataset=Dataset([BETA,SIZE,VALUE,MOM])








