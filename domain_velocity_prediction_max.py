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
    parser.add_argument('--min', type=int, help='min number of location')
    parser.add_argument('--max', type=int, help='max number of location')
    args = parser.parse_args()

# read data from CSV
    df = pd.read_csv(args.src)
    start = args.min
    d = args.max
    locs = []
    for i in range(d):
        tmp_loc_id = i + 1
        tmp_str = "l" + str(tmp_loc_id)
        locs.append(tmp_str)
    ## check
    print(locs)
    print(df[locs])

# find max
    v_max = []
    for l in locs:
        for i in range(len(df[l])-1):
            if df[l][i+1] - df[l][i] < 0:
                v_max.append(df[l][i])
                break
    ## check
    print(v_max)
    d = len(v_max)

# self-regression
    X = np.array(range(1, start+1)).reshape(-1, 1)
    y = np.log(v_max[:start])
    ## check
    print(X)
    print(y)
    reg = LinearRegression().fit(X, y)
    ## check
    print(reg.coef_)
    print(reg.intercept_)

# prediction results
    y_pre = []
    y_real = []
    for i in range(1, d):
        tmp_y_pre = math.exp(reg.coef_[0] * i + reg.intercept_)
        y_pre.append(max(tmp_y_pre, 0))
        y_real.append(v_max[i])
    ##check
    print(y_pre)
    print(y_real)
    '''
# MSE
    res = 0.0
    rr = 0.0
    for i in range(1, d):
        res += (y_real[i] - y_pre[i]) * (y_real[i] - y_pre[i])
        if y_real[i] >= 1.0:
            rr += abs(y_real[i] - y_pre[i]) / y_real[i]
    ## check
    print("MSE-overall: " + str(res/d))
    print("rate-overall: " + str(rr/d))

# MSE-after-max
    res2 = 0.0
    rr2 = 0.0
    for i in range(start, d):
        res2 += (y_real[i] - y_pre[i]) * (y_real[i] - y_pre[i])
        if y_real[i] >= 1.0:
            rr2 += abs(y_real[i] - y_pre[i]) / y_real[i]
    ## check
    print("MSE-after-max: " + str(res2/(d-start)))
    print("rate-after-max: " + str(rr2/(d-start)))
    '''
# plot
    plt.plot(range(1, d), y_pre)
    plt.plot(range(1, d), y_real)
    plt.show()

if __name__ == '__main__':

    main()
