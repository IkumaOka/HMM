# 参考サイト：https://qiita.com/amber_kshz/items/dadc6cfe67d3b80edeb1

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

    def _forward(self, pmatrix):
        N = len(pmatrix)
        alpha = np.zeros((N, self.K))
        c = np.zeros(N)
        tmp = self.pi * pmatrix[0]
        c[0] = np.sum(tmp)
        alpha[0] = tmp / c[0]

        for n in range(1, N, 1):
            tmp = pmatrix[n] * ( (self.A).T @  alpha[n-1] )
            c[n] = np.sum(tmp)
            alpha[n] = tmp / c[n]
        return alpha, c
        