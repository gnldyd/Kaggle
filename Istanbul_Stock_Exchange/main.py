import argparse
import pandas as pd
import numpy as np
from sklearn import linear_model

def linear_regression(x, y):
    x_bias = x.copy()
    model = linear_model.LinearRegression().fit(x_bias, y)
    return np.insert(model.coef_, 0, model.intercept_), model.score(x_bias, y)


def get_r2score(y, y_pred):
    sst = float((y.sub(y.mean()) ** 2).sum())
    ssr = float((y.sub(y_pred.values) ** 2).sum())
    return 1 - ssr / sst


def normal_equation(x, y):
    x_bias = x.copy()
    x_bias.insert(0, 'bias', [1] * len(x))
    theta = np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y)
    predict = x_bias.dot(theta)
    score = get_r2score(y, predict)
    return theta, score


def run(args):
    data = pd.read_excel(args.file, header=1)
    data.rename(columns = {'ISE': 'ISE_TL', 'ISE.1': 'ISE_USD'}, inplace=True)
    for column in data.columns:
        if column is not data.columns[0]:   # except date
            x = data[data.columns.difference([data.columns[0], column])]
            y = data[[column]]
            theta1, score1 = linear_regression(x, y)
            theta2, score2 = normal_equation(x, y)

            # To do : will data visualization


if __name__ == '__main__':
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=data_url, help='import data file.')
    run(parser.parse_args())