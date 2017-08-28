'''
    普通最小二乘法
    根据商品历史销量预测未来销量
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def get_data(file_name):
    "获取训练数据"
    # https://chrisalbon.com/python/pandas_dataframe_importing_csv.html
    data = pd.read_csv(file_name)
    return data

def main():
    "主函数"
    print(__doc__)
    data = get_data('data.csv')
    x = np.array(data[['week']])
    y = np.array(data['qty'])

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    plt.scatter(x, y, color='black') # 样本
    plt.plot(x, regr.predict(x), color='blue') # 预测函数
    plt.show() # 显示图形


if __name__ == '__main__':
    main()
