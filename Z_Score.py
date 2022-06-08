# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:50:14 2020

@author: Administrator
"""
from sklearn import preprocessing 
import pandas as pd
from UCIdata import UCIdata

def ZScore(getdf):
    a = getdf.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    return a


df = UCIdata('iris')
print(ZScore(df))



