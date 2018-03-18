# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-13  14:43
# NAME:assetPricing1-reversal.py
import os
from config import DATA_PATH
from dout import read_df

def get_rev():
    stockRetM=read_df('stockRetM','M')
    rev=stockRetM*100
    rev=rev.stack()
    rev.index.names=['t','sid']
    rev.to_csv(os.path.join(DATA_PATH,'rev.csv'))

if __name__=='__main__':
    get_rev()





