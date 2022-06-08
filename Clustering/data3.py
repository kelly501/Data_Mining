


#常態分布
# import matplotlib.pyplot as plt
import numpy as np


def Dataset3(t, x, y):
    # np.random.seed(None)#取消固定隨機數種子
    np.random.seed(t)
    # 產生常態分布亂數（平均值 = 0，標準差 = 1）
    a = np.random.normal(loc=0.0, scale=1, size=(x, y))  # x筆y維
    # 繪製直方圖
    # plt.hist(a, bins=100)
    # plt.show()

    # with open('data3.csv', 'w') as f:
    #     np.savetxt(f, a, delimiter=",")
    return a

