import os
import math
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

import torch
from torch import nn

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
    args = parser.parse_args()

## timer starts
    start = time.time()

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

## training
    net = nn.Sequential(nn.Linear(d, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.Adam(net.parameters(), lr = 0.01)

    for i in range(len(X)):
        u_t_x = torch.tensor(X[i], dtype=torch.float32)
        u_t_y = torch.tensor([y[i]], dtype=torch.float32) # using list of y[i] to avoid size mismatching warning
        trainer.zero_grad()
        l = loss(net(u_t_x), u_t_y)
        l.backward()
        trainer.step()

## timer ends
    end = time.time()
    elapsed = end - start
    print("run_time", elapsed)

# prediction results
    y_pre = []
    y_real = []
    for i in range(l0, l_total):
        tmp_x = []
        for j, l in enumerate(locs):
            tmp_x.append(df[l][i-l0])
        u_t_x = torch.tensor(tmp_x, dtype=torch.float32)
        tmp_y_pre = net(u_t_x).detach().numpy()
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
