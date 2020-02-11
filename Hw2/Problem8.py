import random
import numpy as np
import matplotlib.pyplot as plt

n = 2000
T = 1000

def generateData():
    x = np.random.uniform(-1, 1, n)
    y = np.sign(x)
    y[y == 0] = -1
    p = np.random.uniform(0, 1, n)
    y[p >= 0.8] *= -1
    return x, y

def decision_stump(X, Y):
    theta = np.sort(X)
    num = len(theta)
    Xtmp = np.tile(X, (num, 1))
    ttmp = np.tile( np.reshape(theta, (num, 1)), (1, num) )
    ysign = np.sign(Xtmp - ttmp)
    ysign[ysign == 0] = -1
    err = np.sum(ysign != Y, axis=1)
    if np.min(err) <= num-np.max(err):
        return 1, theta[np.argmin(err)], np.min(err)/num
    else:
        return -1, theta[np.argmax(err)], (num-np.max(err))/num

def plot(diff_sum):
    plt.xlabel("E_in - E_out")
    plt.ylabel("probability density function")
    plt.title("Problem 8")
    plt.hist(diff_sum, bins=10, density=True)
    plt.savefig("Problem 8.png")


diff_sum, Ein_sum, Eout_sum = [], [], []
for _ in range(T):
    X, Y = generateData()
    theta = np.sort(X)
    s, theta, ein = decision_stump(X, Y)
    eout = 0.5+0.3*s*(((theta if theta > 0 else -theta))-1)
    Ein_sum.append(ein)
    Eout_sum.append(eout)
    diff_sum += [ein - eout]

print("Ein:", np.mean(Ein_sum))
print("Eout:", np.mean(Eout_sum))
plot(diff_sum)