import numpy as np
import math

class Problem():
    def __init__(self):
        self.N = 100
        self.K = 100
        A = 2
        self.A = A
        p = .5
        q = [.51, .6]
        c = [0, .01]
        self.gamma = .9
        T = 10**4
        N = self.N

        r = np.zeros((N,A))
        for x in range(N):
            for a in range(A):
                r[x,a] = - (x/N) ** 2 - c[a]

        P = np.zeros((N,N,A))
        for x in range(N):
            for a in range(A):
                P[x,x,a] = (1-p) * (1-q[a]) + p * q[a]
                if x+1 < N: P[x,x+1,a] = p*(1-q[a])
                if x-1 >= 0: P[x,x-1,a] = (1-p) * q[a]
                P[0,1,a] = p
                P[0,0,a] = 1-p
                P[N-1, N-2, a] = q[a]
                P[N-1, N-1, a] = 1-q[a]

        phi_pwl = np.zeros((N, 2*(N//5)))
        for x in range(N):
            for i in range(2*(N//5)):
                if x in range(5*(i-1), 5*i):
                    if i <= N//5: phi_pwl[x][i] = 1
                    else: phi_pwl[x][i] = 1*((x-5*(i-1))/5)


        def lstd(pi, x):
            size = phi_pwl.shape[1]

            A_B = np.zeros((size, size))
            B_T = np.zeros(size)

            sum_rt = 0
            for _ in range(T):
                a = np.argmax(pi[x])
                next_x = np.random.choice(np.arange(self.N), p=P[x,:,a])
                A_B += phi_pwl[x].reshape(size,1) * (phi_pwl[x]-self.gamma*phi_pwl[next_x])
                B_T += phi_pwl[x] * r[x,a]
                x = next_x
                sum_rt += r[x,a]

            if np.linalg.det(A_B) == 0: A_B += 1e-5 * np.eye(size)
            theta = np.linalg.solve(A_B, B_T)
            V = np.zeros(self.N)
            for x in range(self.N): V[x] = np.dot(theta.T, phi_pwl[x])
            
            # Here we generate the new x for the next iter of softPolicyIteration
            a = np.argmax(pi[x])
            x = np.random.choice(np.arange(self.N), p=P[x,:,a])
            return V, x, sum_rt

        self.lstd = lstd


        def EQ( x, a, V):
            if x == 0: 
                return r[x,a] + self.gamma*p*(V[x]) + self.gamma*p*(V[x+1])
            elif x == self.N-1: 
                return r[x,a] + self.gamma*(q[a]*V[x-1] + (1-q[a])*V[x])
            else:
                return r[x,a] + self.gamma*(1-p)*(q[a]*V[x-1]+(1-q[a])*V[x]) + \
                    self.gamma*p*(q[a]*V[x] + (1-q[a])*V[x+1])

        self.EQ = EQ

        def softPolicyIteration(eta):
            pi = np.zeros((self.N, self.A))
            pi[:,:] = 1 / self.A
            x = self.N-1
            Rm = 0 
            for _ in range(self.K):
                V, x, sum_rt = self.lstd(pi, x)
                Rm += sum_rt
                for x in range(self.N):
                    maxQE= 0
                    normalitation_cte = 0
                    for a in range(self.A):
                        maxQE = max(maxQE, self.EQ(x,a,V) ) 
                        normalitation_cte += pi[x,a]
                    for a in range(self.A): 
                        pi[x,a] *= math.exp(eta * self.EQ(x,a,V) )
                        pi[x,a] -= maxQE #prof sugestion: for large eta
                        if(normalitation_cte != 0):
                            pi[x,a] /= normalitation_cte
                
            return Rm
            
        self.softPolicyIteration = softPolicyIteration

