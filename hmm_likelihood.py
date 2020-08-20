# 参考サイト：https://qiita.com/amber_kshz/items/dadc6cfe67d3b80edeb1
import numpy as np
from matplotlib import pyplot as plt

class HMM:
    def __init__(self, K):
        # それぞれの変数はPRMLに対応している。
        self.K = K
        self.pi = None
        self.A = None
        self.phi = None

    def _init_params(self, pi=None, A=None, phi=None, seed_pi=None, seed_A=None, seed_phi=None):
        # 初期値
        self.pi =  pi if (pi is not None) else np.random.RandomState(seed=seed_pi).dirichlet(alpha=np.ones(self.K))
        self.A = A if (A is not None) else np.random.RandomState(seed=seed_A).dirichlet(alpha=np.ones(self.K), size=self.K)
        self.phi = phi if (phi is not None) else np.random.RandomState(seed=seed_phi).rand(self.K)  # emissoin probability dependent

    def _calc_pmatrix(self, X):
        pmatrix = (self.phi**X) * ((1-self.phi)**(1-X)) # emissoin probability dependent
        return pmatrix

    # 前向きアルゴリズム
    def _forward(self, pmatrix):
        '''
        N: 系列の長さ
        alpha：PRML式(13.36)で表される値
        c: alphaの正規化項
        tmp: alphaの計算。正規化はされていないところに注意。
        　　  PRML式(13.37)で表される。厳密には正規化してないので(13.37)ではない。
        '''
        N = len(pmatrix)
        alpha = np.zeros((N, self.K))
        c = np.zeros(N)
        # alphaの初期値の計算
        tmp = self.pi * pmatrix[0]
        c[0] = np.sum(tmp)
        alpha[0] = tmp / c[0]

        for n in range(1, N, 1):
            tmp = pmatrix[n] * ( (self.A).T @  alpha[n-1] )
            c[n] = np.sum(tmp)
            alpha[n] = tmp / c[n]
        return alpha, c

    # 後ろ向きアルゴリズム
    def _backward(self, pmatrix, c):
        '''
        beta: PRML式(13.38)に対応プログラムでは変数betaは正規化されていないので注意。
        '''
        N = len(pmatrix)
        beta = np.zeros((N, self.K))
        beta[N - 1] = np.ones(self.K)
        for n in range(N-2, -1, -1):
            # PRML式(13.38)に対応。
            beta[n] = self.A @ ( beta[n+1] * pmatrix[n+1] ) / c[n+1]
        return beta
         
    def _estep(self, pmatrix, alpha, beta, c):
        '''
        gamma： PRML式(13.33)
        xi: PRML式(13.43)
        '''
        gamma = alpha * beta
        xi = np.roll(alpha, shift=1, axis=0).reshape(N, self.K, 1) * np.einsum( "jk,nk->njk", self.A, pmatrix * beta) / np.reshape( c, (N, 1,1))
        return gamma, xi

    def _mstep(self, X, gamma, xi):
        self.pi = gamma[0] / np.sum(gamma[0])
        xitmp = np.sum(xi[1:], axis=0)
        self.A = xitmp / np.reshape(np.sum(xitmp, axis=1) , (self.K, 1))
        self.phi = (gamma.T @ X[:,0])  / np.sum(gamma, axis=0)

    
    def fit(self, X, max_iter=1000, tol=1e-3, **kwargs):
        self._init_params(**kwargs)
        log_likelihood = -np.inf
        for i in range(max_iter):
            pmatrix = self._calc_pmatrix(X)
            alpha, c = self._forward(pmatrix)
            beta = self._backward(pmatrix, c)
            gamma, xi = self._estep(pmatrix, alpha, beta, c)
            self._mstep(X, gamma, xi)

            log_likelihood_prev = log_likelihood
            log_likelihood = np.sum(np.log(c))
            if abs(log_likelihood - log_likelihood_prev) < tol:
                break
        print(f"The number of iteration : {i}")
        print(f"Converged : {i < max_iter - 1}")
        print(f"log likelihood : {log_likelihood}")

    def predict_proba(self, X):
        '''
        Calculates and returns the probability that latent variables 
        corresponding to the input X are in each class.
        '''
        pmatrix = self._calc_pmatrix(X)
        alpha, c = self._forward(pmatrix)
        beta = self._backward(pmatrix, c)
        gamma = alpha * beta
        return gamma

    def predict(self, X):
        '''
        Calculates and returns which classes the latent variables 
        corresponding to the input X are in.
        '''
        pred = self.predict_proba(X).argmax(axis=1)
        return pred

    
N = 200 # the number of data

# Here we consider a heavily biased coin.
mu0 = 0.1 
mu1 = 0.8

tp = 0.03 # transition probability

rv_cointoss = np.random.RandomState(seed=0).rand(N)
rv_transition = np.random.RandomState(seed=1).rand(N)

X = np.zeros((N, 1))
states = np.zeros(N)
current_state = 0
for n in range(N):
    states[n] = current_state
    if rv_cointoss[n] < mu0*(int(not(current_state))) + mu1*current_state:
        X[n][0] = 1.0
    if rv_transition[n] < tp:
        current_state = int(not(current_state))

# fit model
hmm = HMM(K=2)
hmm.fit(X, seed_pi=0, seed_A=1, seed_phi=2)

print(f"pi : {hmm.pi}")
print(f"A : {hmm.A}")
print(f"phi : {hmm.phi}")

plt.figure(figsize=(12,6))
plt.plot(states, '.-', label='ground truth latent variable')
plt.plot(hmm.predict_proba(X)[:,0], '.-', label='predicted probability of state 0')
plt.plot(0.3*X[:,0]+0.35,'.-',label='observation')
plt.savefig("results/hmm_cointoss.png")