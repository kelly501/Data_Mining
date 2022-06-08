import pandas as pd
import numpy as np
import time
from MArray import Magic
from data1 import Dataset1
from data3 import Dataset3
from data_stu import DataStu
from UCIdata import UCIdata
from Z_Score import ZScore
from MinMax import MinMax
import matplotlib.pyplot as plt


tStart = time.time()  # 計時開始
def msw(sampledf, N, g):
    xsum = 0
    for i in range(g):
        fliter = (sampledf["group"] == i)  # 布林
        sampledf1 = sampledf[fliter]
        sampledf1_value = sampledf1.drop(['distant', 'rank', 'group'], axis=1)  # 砍掉
        #print("+++", sampledf1.shape[0])
        e = sampledf1.shape[0]
        sampledf1_valuemean = sampledf1_value.mean()
        distant_array1 = np.zeros(shape=(e, 1))
        for j in range(e):
            distant_array1[j] = (((sampledf1_value.iloc[j] - sampledf1_valuemean) ** 2).sum())
        #print("***", np.size(distant_array1))
        x = (np.sum(distant_array1))
        xsum = xsum + x
    xmean = xsum / N
    return xmean


def msb(sampledf, g):
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


def Fi(sampledf, N, g):
    MSW = msw(sampledf, N, g)
    MSB = msb(sampledf, g)
    return MSW / MSB


# 匯入檔案
# inputpath=input("輸入檔案path:")
# outputpath=input("輸出檔案path:")
# df=pd.read_excel(r"input.xlsx")

# df=pd.read_excel(r"input.xlsx")
# print(df)
for v in range(1):
    num = 10 # 跑幾次
    # N = 27  # 數據數
    # atti = 6  # 維度
    g = 5
    result1 = np.zeros(shape=(num, 2))
    for t in range(num):
        # df = pd.DataFrame(Dataset1(t, N, atti))
        # df = pd.read_excel("D:/documents/master_class/data_mining/Hipple/UCIdata/class2.xlsx")
        # df = pd.DataFrame(DataStu(t, N, atti))
        df = UCIdata('pid')
        N = df.shape[0]
        atti = df.shape[1]
        # print(df)
        """
        N=input("元素個數:")
        g=input("群數:")
        N=int(N)
        g=int(g)
        """
        # N=df.shape[0]

        k = df.shape[1]  # 返回行數
        # print(k)

        dfmean = df.mean()  # 平均值
        distant_array = np.zeros(shape=(N, 1))
        for i in range(N):
            distant_array[i]=(df.iloc[i,0:atti]).sum()
            # distant_array[i] = (((df.iloc[i] - dfmean) ** 2).sum()) ** 0.5

        distant_df = pd.DataFrame(distant_array)
        df["distant"] = distant_df

        df.sort_values("distant", inplace=True)
        df.index = range(len(df))

        # df.sort_values("distant", inplace=True)  # 以distant做排序
        df["rank"] = (df.distant.rank())
        df["group"] = pd.DataFrame(np.zeros(shape=(N, 1)))

        # print(df)
        def MASA(e):
            MA = Magic(e)
            MA.index = range(len(MA))
            # print(MA)
            M_rank = np.zeros(shape=(e**2, 1))
            for i in range(e):
                for j in range(e):
                    temp = MA.iloc[j, i] - 1
                    M_rank[temp] = i
            # print(M_rank)
            return M_rank

        if N == g**2:
            M = MASA(g)
            df.loc[:, 'group'] = M

        elif N > g**2:
            if N % (g**2) == 0:
                for i in range(N // (g**2)):
                    M = MASA(g)
                    df.loc[(g**2)*i:(g**2)*(i+1)-1, 'group'] = M
                    # print(df)
            else:
                for i in range((N // (g**2)) + 1):
                    M = MASA(g)
                    if i == (N // (g**2)):
                        df.loc[(g ** 2) * i:, 'group'] = M[0:N % (g**2)]
                    else:
                        df.loc[(g ** 2) * i:(g ** 2) * (i + 1) - 1, 'group'] = M
                    # print(df)
        elif N < g**2:
            if (g**2) % N == 0:
                df["Sgroup"] = pd.DataFrame(np.empty(shape=(N, 1)))
                dfadd = df
                dfadd = df.drop(index=df.index)
                for i in range(N//g):
                    even = np.arange(0, g, 1)
                    odd = np.arange(g-1, -1, -1)
                    if i % 2 == 1:
                        df.loc[g*i:g*(i+1)-1, 'Sgroup'] = odd
                    else:
                        df.loc[g*i:g*(i+1)-1, 'Sgroup'] = even
                e = int(g**2/N)
                k = int(N/g)
                for j in range(e):
                    event1 = df["Sgroup"] >= j*(g/e)
                    event2 = (j+1)*(g/e) > df["Sgroup"]
                    fliter = (event1 & event2)   # 布林
                    temp = df.loc[fliter]
                    M = MASA(k)
                    temp.loc[:, "group"] = M
                    dfadd = pd.concat([dfadd, temp], axis=0)
                    print(j, dfadd)
                df = dfadd.drop(['Sgroup'], axis=1)
                # print(df)
            else:
                M = MASA(g)
                df.loc[:, 'group'] = M[0:N]
                # print(df)






            # else:




        # print(df)




        Fi_value = Fi(df, N, g)
        print(t, Fi_value)

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