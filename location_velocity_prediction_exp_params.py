import os
import math
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
# parse args
    parser = argparse.ArgumentParser(description='regression')
    parser.add_argument('--src', type=str, help='data path of csv')
    parser.add_argument('--loc', type=int, help='id of columns in csv')
    parser.add_argument('--min', type=int, help='min number of iteraions')
    parser.add_argument('--max', type=int, help='max number of iterations')
    parser.add_argument('--alpha', type=float, help='value of alpha')
    parser.add_argument('--beta', type=float, help='value of beta')
    parser.add_argument('--gamma', type=float, help='value of gamma')
    args = parser.parse_args()

# read data from CSV
    df = pd.read_csv(args.src)
    loc = "l" + str(args.loc)
    ## check
    print(df[loc])

    l_total = df.shape[0]
    l0 = args.min
    l3 = args.max
    ## check
    print(l0, l3, l_total)

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    ## check
    print(alpha, beta, gamma)

    dta = []
    for i in range(l_total):
        dta += [[math.exp(-alpha*i), math.exp(-beta*i), math.exp(-gamma*i)]]
    ## check
    #print(dta)

# regression
    X = []
    y = []
    for i in range(l0, l3):
        X.append(dta[i-l0]) ## timestep - l0
        y.append(df[loc][i])
    ## check
    #print(X)
    #print(y)
    reg = LinearRegression().fit(X, y)
    ## check
    print(reg.coef_)
    print(reg.intercept_)

# prediction results
    y_pre = []
    y_real = []
    for i in range(l0, l_total):
        tmp_y_pre = 0.0
        for j in range(3):
            tmp_y_pre += reg.coef_[j] * dta[i-l0][j] ## timestep - 1
        tmp_y_pre += reg.intercept_
        y_pre.append(max(tmp_y_pre, 0))
        y_real.append(df[loc][i])

# MSE
    res = 0.0
    rr = 0.0
    for i in range(l_total - l0):
        res += (y_real[i] - y_pre[i]) * (y_real[i] - y_pre[i])
        if y_real[i] >= 1.0:
            rr += abs(y_real[i] - y_pre[i]) / y_real[i]
    ## check
    print("MSE-overall: " + str(res/(l_total - l0)))
    print("rate-overall: " + str(rr/(l_total - l0)))

# MSE-after-max
    res2 = 0.0
    rr2 = 0.0
    for i in range(l3 - l0, l_total - l0):
        res2 += (y_real[i] - y_pre[i]) * (y_real[i] - y_pre[i])
        if y_real[i] >= 1.0:
            rr2 += abs(y_real[i] - y_pre[i]) / y_real[i]
    ## check
    print("MSE-after-max: " + str(res2/(l_total - l3)))
    print("rate-after-max: " + str(rr2/(l_total - l3)))

# plot
    plt.plot(range(l0, l_total), y_pre)
    plt.plot(range(l0, l_total), y_real)
    plt.show()


if __name__ == '__main__':

    main()
