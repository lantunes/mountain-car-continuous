import matplotlib.pyplot as plt
import pylab
import numpy as np

data = pylab.loadtxt('plotdata.txt', delimiter=',')
data2 = pylab.loadtxt('plotdata2.txt', delimiter=',')
data3 = pylab.loadtxt('plotdata3.txt', delimiter=',')
data4 = pylab.loadtxt('plotdata4.txt', delimiter=',')
data5 = pylab.loadtxt('plotdata5.txt', delimiter=',')
data6 = pylab.loadtxt('plotdata6.txt', delimiter=',')
data7 = pylab.loadtxt('plotdata7.txt', delimiter=',')
data8 = pylab.loadtxt('plotdata8.txt', delimiter=',')
data9 = pylab.loadtxt('plotdata9.txt', delimiter=',')
data10 = pylab.loadtxt('plotdata10.txt', delimiter=',')

mean = np.mean([data[:, 1], data2[:, 1], data3[:, 1], data4[:, 1], data5[:, 1], data6[:, 1], data7[:, 1], data8[:, 1], data9[:, 1], data10[:, 1]], axis=0)
sigma = np.sqrt(np.var([data[:, 1], data2[:, 1], data3[:, 1], data4[:, 1], data5[:, 1], data6[:, 1], data7[:, 1], data8[:, 1], data9[:, 1], data10[:, 1]], axis=0))

sigma_upper = mean + sigma
sigma_lower = mean - sigma

pylab.errorbar(data[:, 0], mean, sigma, ecolor='red', errorevery=10, elinewidth=0.8)

plt.xlabel("iteration", size=15)
plt.ylabel("total reward", size=15)
plt.grid(True)

plt.show()
