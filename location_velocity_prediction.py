import os
import math
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def main():
# parse args
    parser = argparse.ArgumentParser(description='regression')
    parser.add_argument('--src', type=str, help='data path of csv')
    parser.add_argument('--loc', type=int, help='id of columns in csv')
    parser.add_argument('--min', type=int, help='min number of iteraions')
    parser.add_argument('--p1', type=int, help='second residue')
    parser.add_argument('--p2', type=int, help='first residue')
    parser.add_argument('--max', type=int, help='max number of iterations')
    args = parser.parse_args()

# read data from CSV
    df = pd.read_csv(args.src)
    loc = "l" + str(args.loc)
    ## check
    print(df[loc])

# obtain residues
    l_total = df.shape[0]
    l0 = args.min
    l1 = args.p1
    l2 = args.p2
    l3 = args.max
    ## check
    print(l0, l1, l2, l3, l_total)

    y3 = []
    for i in range(l2, l3):
        y3.append(df[loc][i])
    y2 = []
    for i in range(l1, l2):
        y2.append(df[loc][i])
    y1 = []
    for i in range(l0, l1):
        y1.append(df[loc][i])
    ## check
    #print(y1)
    #print(y2)
    #print(y3)

# regression C = C3*exp(-gamma*t)
    model = np.polyfit(range(l2, l3), np.log(y3), 1)
    gamma = model[0]
    C3 = model[1]
    ## check
    print("C = " + str(math.exp(C3)) + "*exp(" + str(gamma) + "*t)")
    ## validate
    y3_pre = []
    for i in range(l2, l3):
        y3_pre.append(math.exp(gamma*i + C3))
    #print(y3_pre)

# regression C = C2*exp(-beta*t) + C3*exp(-gamma*t)
    ## first residue
    y32 = []
    for i in range(l1, l2):
        y32.append(math.exp(gamma*i + C3))
    y2_res = []
    for i in range(len(y2)):
            y2_res.append(abs(y2[i] - y32[i]))
    ## regression
    model = np.polyfit(range(l1, l2), np.log(y2_res), 1)
    beta = model[0]
    C2 = model[1]
    ## check
    print("C = " + str(math.exp(C2)) + "*exp(" + str(beta) + "*t) + " + str(math.exp(C3)) + "*exp(" + str(gamma) + "*t)")
    ## validate
    y2_pre = []
    for i in range(l1, l2):
        y2_pre.append(math.exp(beta*i + C2) + math.exp(gamma*i + C3))
    #print(y2_pre)

# regression C = C1*exp(-alpha*t) + C2*exp(-beta*t) + C3*exp(-gamma*t)
    ## second residue
    y31 = []
    y21 = []
    for i in range(l0, l1):
        y31.append(math.exp(gamma*i + C3))
        y21.append(math.exp(beta*i + C2))
    y1_res = []
    for i in range(len(y1)):
            y1_res.append(abs(y1[i] - y31[i] - y21[i]))
    ## regression
    model = np.polyfit(range(l0, l1), np.log(y1_res), 1)
    alpha = model[0]
    C1 = model[1]
    ## check
    print("C = -" + str(math.exp(C1)) + "*exp(" + str(alpha) + "*t) + "  + str(math.exp(C2)) + "*exp(" + str(beta) + "*t) + " + str(math.exp(C3)) + "*exp(" + str(gamma) + "*t)")
    ## validate
    y1_pre = []
    for i in range(l0, l1):
        y1_pre.append(-math.exp(alpha*i + C1) + math.exp(beta*i + C2) + math.exp(gamma*i + C3))
    #print(y1_pre)

# prediction results
    y_pre = []
    y_real = []
    for i in range(l0, l_total):
        y_pre.append(max(-math.exp(alpha*i + C1) + math.exp(beta*i + C2) + math.exp(gamma*i + C3), 0))
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
