# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import pandas as pd
import numpy as np
from itertools import permutations
from data3 import Dataset3
from data1 import Dataset1
from UCIdata import UCIdata

tStart = time.time()#計時開始

def msw(sampledf,N,g,level):
    intlevel=int(level)
    xsum=0
    for i in range(g):
        fliter = (sampledf["group"] == i) #布林
        sampledf1=sampledf[fliter]
        sampledf1_value=sampledf1.drop(['distant','rank','level','group'], axis=1) #砍掉
        sampledf1_valuemean=sampledf1_value.mean()
        distant_array1 = np.zeros(shape=(intlevel,1))  
        for j in range(intlevel):  
            distant_array1[j]=(((sampledf1_value.iloc[j]-sampledf1_valuemean)**2).sum()) 
        x=(np.sum(distant_array1))
        xsum=xsum+x
    xmean=xsum/N
    return xmean

def msb(sampledf,g,level):
    sampledfmean=sampledf.mean()
    distant_array1 = np.zeros(shape=(g,1))
    xsum=0
    for i in range(g):
        fliter = (sampledf["group"] == i)
        sampledf1=sampledf[fliter]
        sampledf1_value=sampledf1.drop(['distant','rank','level','group'], axis=1)
        sampledf1_valuemean=sampledf1_value.mean()
        distant_array1[i]=(((sampledf1_valuemean-sampledfmean)**2).sum())
    x=(np.sum(distant_array1))    
    xsum=xsum+x
    xmean=xsum/g
    return xmean

def Fi(sampledf,N,g,level):
    MSW=msw(sampledf,N,g,level)
    MSB=msb(sampledf,g,level) 
    return MSW/MSB

def perm(n): 
    x=list(range(n)) #[0,1,2,...,n-1]  在list中的資料型別儲存的是資料的存放的地址,即指標
    y=list(permutations(x)) #排列組合 combinations和permutations返回的是對象地址
    y_array=np.array(y) #array裡面存放的都是相同的資料型別
    return y_array  #array的建立：引數既可以是list，也可以是元組
"""
#匯入檔案
inputpath=input("輸入檔案path:")
outputpath=input("輸出檔案path:")
df=pd.read_excel(r"{}.xlsx".format(inputpath))
dfmean=df.mean() #平均值

N=input("元素個數:")
g=input("群數:")
N=int(N)
g=int(g)
"""

#df=pd.read_excel(r"input.xlsx")

for v in range(1):
    num = 1
    # N = 27
    # atti = 6
    g = 3
    result1 = np.zeros(shape=(num, 2))
    for t in range(num):
        # df=pd.DataFrame(Dataset1(t, N, atti))
        # df = pd.read_excel("D:/documents/master_class/data_mining/Hipple/UCIdata/class2.xlsx")
        df = UCIdata('bupa')
        N = df.shape[0]
        atti = df.shape[1]
        # print(df)
        dfmean=df.mean() #平均值



        #到值心距離
        level=N/g #每群有幾層
        intlevel=int(level)
        distant_array = np.zeros(shape=(N,1))
        for i in range(N):
            distant_array[i]=(((df.iloc[i]-dfmean)**2).sum())**0.5 #loc:行
        #dataframe用來處理結構化資料 有列索引與欄標籤
        distant_df=pd.DataFrame(distant_array)  #以dataframe格式讀取
        df["distant"]=distant_df #於df資料加入一行distant

        #排名
        df.sort_values("distant",inplace=True) #以distant做排序
        df["rank"] = (df.distant.rank())

        #分組分層
        df["level"] = (((df["rank"]-0.1)/g)).astype(int)  # /:點數除法 浮點數轉整數 astype完成dataframe欄位型別轉換
        df["group"] = pd.DataFrame(np.zeros(shape=(N, 1)))

        x = perm(g)  # 所有可能個數
        df.index = range(len(df))
        dfadd = df.loc[0:g-1, :].copy()
        dfadd["group"] = x[0]

        # dfadd[0:g-1, "group"] = range(g)

        # print(df)
        for i in range(intlevel-1):
            temp = df.loc[(i+1)*g:(i+2)*g-1, :].copy()
            Fiposarray = np.zeros(shape=((x.shape)[0]))
            for j in range((x.shape)[0]):
                temp["group"] = x[j]
                dfadd1 = pd.concat([dfadd, temp], axis=0)
                innerN = dfadd1.shape[0]  # 列數
                innerlevel=int(innerN/g)
                Fiposarray[j] = Fi(dfadd1, innerN, g, innerlevel)  # 計算F值
            posmax = np.argwhere(Fiposarray == np.max(Fiposarray))  # 找到最大fi陣列位址
            posmaxint = int(posmax[0])
            temp["group"] = x[posmaxint]  # 最大的分配方式
            dfadd = pd.concat([dfadd, temp], axis=0)

        N = dfadd.shape[0]
        level=int(N/g)
        Fi_value=Fi(dfadd,N,g,level)  #回傳值為地址
        # print(dfadd)

        # dfempty=df.drop(df.index, inplace=False) #drop刪去 index索引 inplace:是否對原數值做修改(默認false)



        #先建立列表再轉換成陣列
        # Filist=list(range(1))
        # Fiarray=np.array(Filist)
        # Fiarray=Fiarray.astype(np.float)
        # Fiarray[0]=Fi_value
        """
        dfadd.to_excel(r"{}.xlsx".format(outputpath))
        np.savetxt(r"{}.txt".format(outputpath),Fiarray,fmt='%.5f')
        """
        # dfadd.to_excel(r"output.xlsx")
        # np.savetxt(r"output.txt",Fiarray,fmt='%.5f')

        tEnd = time.time() #計時結束

        # print ("It cost sec" ,(tEnd - tStart)) #會自動做近位
        result1[t,0]=Fi_value
        result1[t,1]=tEnd - tStart
        tStart = time.time()#計時開始
    tEnd = time.time() #計時結束
    # # np.savetxt(r"output.txt",result1,fmt='%.5f')
    end = result1.sum(axis=0)
    print(end[0], "\n", end[1], "\n", "----------------")