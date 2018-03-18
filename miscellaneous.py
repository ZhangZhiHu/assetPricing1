# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-16  19:35
# NAME:assetPricing1-miscellaneous.py

from config import *

from dout import *
from zht.utils.mathu import get_inter_frame, get_inter_index


def compare_ff5_ff3_ffc():
    ff3=read_df('ff3','M')
    ff3_gta=read_df('ff3_gta','M')
    ff5=read_df('ff5_gta','M')
    ffc=read_df('ffc_gta','M')

    [ff3,ff3_gta,ff5,ffc]=get_inter_index([ff3,ff3_gta,ff5,ffc])

    ff3.cumsum().plot()
    ff3_gta.cumsum().plot()
    ff5.cumsum().plot()
    ffc.cumsum().plot()