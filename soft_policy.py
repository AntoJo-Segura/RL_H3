from problem import Problem
import numpy as np
import matplotlib.pyplot as plt


problem = Problem()

M = 50
l_rates = np.logspace(.01, 10, num=M)
reward = dict()

for m in range(M):
    print('iteration ',m,' of ', M)
    eta = l_rates[m]
    reward[eta] = problem.softPolicyIteration(eta)

plt.xscale('log')
plt.xlabel('learning rate')
plt.ylabel('reward')
plt.plot(reward.keys(),reward.values())
plt.savefig("eta_reward.jpg")