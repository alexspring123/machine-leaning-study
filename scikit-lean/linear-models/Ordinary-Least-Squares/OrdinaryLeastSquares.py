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
    x = np.array(data[['week']])
    y = np.array(data['qty'])
    return x, y


def create_linear_model(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return regr


def show_linear_line(x, y, model):
    plt.scatter(x, y, color='black')  # 样本
    plt.plot(x, model.predict(x), color='blue')  # 预测函数
    plt.show()  # 显示图形


def predict(model, x):
    predict_out = model.predict(x)
    print('预测结果：第', x, '周销量=', predict_out)


def main():
    "主函数"
    print(__doc__)
    x, y = get_data('data.csv')
    model = create_linear_model(x, y)

    # 预测30周销量
    predict(model, 30)

    show_linear_line(x, y, model)


if __name__ == '__main__':
    main()
