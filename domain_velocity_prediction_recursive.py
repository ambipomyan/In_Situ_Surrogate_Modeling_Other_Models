import os
import math
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import random

def main():
# parse args
    parser = argparse.ArgumentParser(description='regression')
    parser.add_argument('--src', type=str, help='data path of csv')
    parser.add_argument('--loc', type=int, help='id of columns in csv')
    parser.add_argument('--min', type=int, help='min number of iteraions')
    parser.add_argument('--max', type=int, help='max number of iterations')
    # self-regression
    parser.add_argument('--start', type=int, help='location at the wave front')
    parser.add_argument('--dist', type=int, help='number of locations used for self-regression')
    parser.add_argument('--nlag', type=int, help='number of lags of number of iteration')
    # threshold
    parser.add_argument('--s', type=int, help='problem size')
    parser.add_argument('--r', type=float, help='portion of vmax')
    args = parser.parse_args()

# read data from CSV
    df = pd.read_csv(args.src)
    loc = "l" + str(args.loc)   # should be wf + 1, e.g. use 4 + 1 = 5 for test
    wf = args.start  # 4 for test
    d = args.dist    # 4 for test, including locations 1, 2, 3, and 4
    locs = []
    for i in range(d):
        tmp_loc_id = wf - i # 4, 3, 2, 1
        tmp_str = "l" + str(tmp_loc_id)
        locs.append(tmp_str)
    ## check
    print(loc)
    print(df[loc])
    print(locs)
    print(df[locs])

# self-regression
    l_total = df.shape[0]
    l0 = args.min + args.nlag
    l3 = args.max

    X = []
    y = []
    dta = df[locs].to_numpy()
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
        for j, l in enumerate(locs):
            tmp_y_pre += reg.coef_[j] * df[l][i-l0] ## timestep - 1
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
    #plt.plot(range(l0, l_total), y_pre)
    #plt.plot(range(l0, l_total), y_real)
    #plt.show()

# prediction
    locs = [6, 7, 8, 9]
    if args.s == 30:
        v_max = 1786.0 # 30
        X = [314.98, 239.01, 185.64, 147.23] #30
    if args.s == 60:
        v_max = 5053.0 # 60
        X = [890.90, 676.02, 525.09, 416.43] # 60
    if args.s == 90:
        v_max = 9283.0 # 90
        X = [1636.69, 1241.93, 964.66, 765.03] # 90        
    v_threshold = v_max * args.r

    counts = 0
    for i in range(len(X)):
        if X[i] <= v_threshold:
            counts = i
            break

    y_bar = X[-1]
    while y_bar >= v_threshold and counts < 84:
        counts += 1
        y_bar = 0.0
        for j in range(len(X)):
            y_bar += reg.coef_[j] * X[-j]
        y_bar += reg.intercept_
        for i in range(1,len(X)):
            X[i-1] = X[i]
        correct = math.exp(-0.12 - 7*args.r)
        X[-1] = y_bar*correct
        ## check
        print(X)

    res = counts + locs[0]

    print("predict: " + str(res))

if __name__ == '__main__':

    main()
