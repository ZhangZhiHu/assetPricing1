# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-07  13:13
# NAME:assetPricing1-config.py

import os

DATA_SRC=r'D:\zht\database\quantDb\sourceData\gta\data\csv'

PROJECT_PATH=r'D:\zht\database\quantDb\researchTopics\assetPricing1'
DATA_PATH=os.path.join(PROJECT_PATH, 'data')
BETA_PATH=os.path.join(PROJECT_PATH,'beta')
SIZE_PATH=os.path.join(PROJECT_PATH,'size')
VALUE_PATH=os.path.join(PROJECT_PATH,'value')
MOM_PATH=os.path.join(PROJECT_PATH,'momentum')


WIND_SRC_PATH=os.path.join(PROJECT_PATH,'wind_src')
WIND_PATH=os.path.join(PROJECT_PATH,'wind')


BETA_NAMES=['1M','3M','6M','12M','24M','1Y','2Y','3Y','5Y']
SIZE_NAMES=['mktCap','size','mktCap_ff','size_ff']
VALUE_NAMES=['bm','logbm']
MOM_NAMES=['mom','r12','r6','R12M','R9M','R6M','R3M']


DEFAULT_MAP={'size':'mktCap','beta':'12M','value':'bm'}


