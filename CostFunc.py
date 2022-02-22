import numpy as np

# regularized linear regression cost function
class QuadraticCostFunc:
    def __init__(self, A, y, gamma):
        self.A = A
        self.y = y
        self.ga = gamma
        self.nclient = len(y)
        self.nsample = sum([len(y[i]) for i in range(self.nclient)])
        _, self.ndim = self.A[0].shape

    def local_func(self, x, i):
        x = x.reshape(-1, 1)
        Ai = self.A[i]
        ni, _ = Ai.shape
        yi = self.y[i].reshape((ni, 1))
    
        return 0.5 * (np.linalg.norm(Ai.dot(x) - yi)**2 / self.nsample + self.ga * np.linalg.norm(x)**2 / self.nclient)

    def local_grad(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        yi = self.y[i].reshape((ni, 1))
    
        return Ai.T.dot(Ai.dot(x) - yi) / self.nsample + self.ga * x / self.nclient

    def local_hess(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape

        return Ai.T.dot(Ai) / self.nsample + self.ga * np.identity(self.ndim) / self.nclient

    def global_func(self, x):
        f = 0
        for i in range(self.nclient):
            f = f + self.local_func(x, i)

        return f
    
    def global_grad(self, x):
        g = np.zeros([self.ndim,1])
        for i in range(self.nclient):
            g = g + self.local_grad(x, i)
        return g


# regularized logistic regression cost function
class LogisticCostFunc:
    def __init__(self, A, y, gamma):
        self.A = A
        self.y = y
        self.ga = gamma
        self.nclient = len(y)
        self.nsample = sum([len(y[i]) for i in range(self.nclient)])
        _, self.ndim = self.A[0].shape

    def sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1 + np.exp(-z))

    def local_func(self, x, i):
        x = x.reshape(-1, 1)
        Ai = self.A[i]
        ni, _ = Ai.shape
        yi = self.y[i].reshape((ni, 1))
        hi = self.sigmoid(Ai.dot(x))
        reg = 0.5 * self.ga * np.linalg.norm(x)**2

        #fi = (- yi.T.dot(np.log(hi)) - (1 - yi).T.dot(np.log(1-hi))) / self.nsample + reg / self.nclient
        fi = (- yi.T.dot(np.log(hi)) - (1 - yi).T.dot(np.log(1-hi))) / ni + reg / ni
    
        return fi[0][0]

    def local_grad(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        yi = self.y[i].reshape((ni, 1))
        hi = self.sigmoid(Ai.dot(x))
        reg = self.ga * x

        #gi = Ai.T.dot(hi - yi) / self.nsample + reg / self.nclient
        gi = Ai.T.dot(hi - yi) / ni + reg / ni
    
        return gi

    def local_hess(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        hi = self.sigmoid(Ai.dot(x))
        S = np.diag(hi.T[0]) * np.diag((1-hi).T[0])
        #Hi = Ai.T.dot(S).dot(Ai) / self.nsample + self.ga * np.identity(self.ndim) / self.nclient
        Hi = Ai.T.dot(S).dot(Ai) / ni + self.ga * np.identity(self.ndim) / ni

        return Hi

    def global_func(self, x):
        s = 0
        for i in range(self.nclient):
            s = s + self.local_func(x, i)

        return s
    
    def global_grad(self, x):
        s = np.zeros([self.ndim,1])
        for i in range(self.nclient):
            s = s + self.local_grad(x, i)
        return s