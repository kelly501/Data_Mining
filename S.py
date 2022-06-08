# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import time
from data1 import Dataset1
from data3 import Dataset3
from data_stu import DataStu
from UCIdata import UCIdata
from Z_Score import ZScore
from MinMax import MinMax
import matplotlib.pyplot as plt

tStart = time.time()  # 計時開始


def msw(sampledf, N, g, level):
    intlevel = int(level)
    xsum = 0
    for i in range(g):
        fliter = (sampledf["group"] == i)  # 布林
        sampledf1 = sampledf[fliter]
        sampledf1_value = sampledf1.drop(['distant', 'rank', 'group'], axis=1)  # 砍掉
        sampledf1_valuemean = sampledf1_value.mean()
        distant_array1 = np.zeros(shape=(intlevel, 1))
        for j in range(intlevel):
            distant_array1[j] = (((sampledf1_value.iloc[j] - sampledf1_valuemean) ** 2).sum())
        x = (np.sum(distant_array1))
        xsum = xsum + x
    xmean = xsum / N
    return xmean


def msb(sampledf, g, level):
    sampledfmean = sampledf.mean()
    distant_array1 = np.zeros(shape=(g, 1))
    xsum = 0
    for i in range(g):
        fliter = (sampledf["group"] == i)
        sampledf1 = sampledf[fliter]
        sampledf1_value = sampledf1.drop(['distant', 'rank', 'group'], axis=1)
        sampledf1_valuemean = sampledf1_value.mean()
        distant_array1[i] = ((sampledf1_valuemean - sampledfmean) ** 2).sum()
    x = (np.sum(distant_array1))
    xsum = xsum + x
    xmean = xsum / g
    return xmean


def Fi(sampledf, N, g, level):
    MSW = msw(sampledf, N, g, level)
    MSB = msb(sampledf, g, level)
    return MSW / MSB


# 匯入檔案
# inputpath=input("輸入檔案path:")
# outputpath=input("輸出檔案path:")
# df=pd.read_excel(r"input.xlsx")

for v in range(1):
    # N = 27
    # atti = 6
    num = 1
    g = 5
    result1 = np.zeros(shape=(num, 2))
    for t in range(num):
        # df = pd.DataFrame(Dataset(t, N, atti))
        # df = pd.read_excel("D:/documents/master_class/data_mining/Hipple/UCIdata/class2.xlsx")
        # df = pd.DataFrame(DataStu(t, N, atti))
        df = UCIdata('pid')
        N = df.shape[0]
        atti = df.shape[1]
        # df = ZScore(df)
        # df = MinMax(df)

        """
        N=input("元素個數:")
        g=input("群數:")
        N=int(N)
        g=int(g)
        """
        N = df.shape[0]

        k = df.shape[1]  # 返回行數
        # print(k)

        level = N / g
        intlevel = int(level)
        dfmean = df.mean()  # 平均值
        distant_array = np.zeros(shape=(N, 1))
        for i in range(N):
            # distant_array[i] = (df.iloc[i, 0:atti]).sum()
            distant_array[i] = (((df.iloc[i] - dfmean) ** 2).sum()) ** 0.5

        distant_df = pd.DataFrame(distant_array)
        df["distant"] = distant_df

        df.sort_values("distant", inplace=True)
        df["rank"] = df.distant.rank()
        df["group"] = pd.DataFrame(np.zeros(shape=(N, 1)))
        df.index = range(len(df))
        #print(df)

        even = np.arange(0, g, 1)
        odd = np.arange(g - 1, -1, -1)
        for i in range(intlevel):
            if i % 2 == 1:
                df.loc[g*i:g*(i+1)-1, 'group'] = odd
            else:
                df.loc[g*i:g*(i+1)-1, 'group'] = even
        if N % g != 0:
            even = np.arange(0, N % g, 1)
            odd = np.arange(g - 1, g - N % g - 1, -1)
            # print(even, odd)
            if intlevel % 2 == 1:
                df.loc[g * intlevel:g * (intlevel + 1) - 1, 'group'] = odd
            else:
                df.loc[g * intlevel:g * (intlevel + 1) - 1, 'group'] = even

        # print(df)

        Fi_value = Fi(df, N, g, level)
        # print(t, Fi_value)

        Filist = list(range(1))
        Fiarray = np.array(Filist)
        Fiarray = Fiarray.astype(float)
        Fiarray[0] = Fi_value

        # dfadd.to_excel(r"output.xlsx")
        # np.savetxt(r"output.txt",Fiarray,fmt='%.5f')

        tEnd = time.time()  # 計時結束
        # print("It cost sec", (tEnd - tStart))  # 會自動做近位
        result1[t, 0] = Fi_value
        result1[t, 1] = tEnd - tStart
        tStart = time.time()  # 計時開始
    # np.savetxt(r"output.txt", result1.sum(axis=0) / num, fmt='%.5f')
    end = result1.sum(axis=0) / num
    print(end[0], "\n", end[1], "\n", "----------------")