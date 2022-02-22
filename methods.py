import numpy as np
import random
#import matplotlib.pyplot as plt

# define train_param / tune_param
class TrainParam:
    def __init__(self, alpha1 = None, beta1 = None, alpha2 = None, beta2 = None, mu = None, K = None, client_gradient = None, client_Newton = None, initial_x = None):
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.mu = mu
        self.K = K
        self.client_gradient = client_gradient
        self.client_Newton = client_Newton
        self.initial_x = initial_x


# alpha1, beta1, etc are ranges of parameters to be tuned 
class TuneParam:
    def __init__(self, alpha1_range = None, beta1_range = None, alpha2_range = None, beta2_range = None, mu_range = None, K = None, client_gradient = None, client_Newton = None, initial_x = None):
        self.alpha1_range = alpha1_range
        self.beta1_range = beta1_range
        self.alpha2_range = alpha2_range
        self.beta2_range = beta2_range
        self.mu_range = mu_range
        self.K = K
        self.client_gradient = client_gradient # clients doing second-order updates in FedHybrid
        self.client_Newton = client_Newton # clients doing first-order updates in FedHybrid
        self.initial_x = initial_x


# Algorithms
class FedAvg:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star

    def train(self, param):
        alpha1, K = 2**param.alpha1, param.K 
        # verbose = param.verbose

        x = param.initial_x
        fn = []
  
        for k in range(K):
          
            x = x - alpha1 * self.func.global_grad(x)

            fn.append(self.func.global_func(x))

            # if verbose > 0:
            #     print("current loss: %f" % fn[-1])
  
            if fn[-1] > 1e20: # diverge
                break
  
            if np.log(fn[-1] - self.fn_star) < -20: # converge to the optimal point
                break

        return np.array(fn) - self.fn_star, k, x 

    def tune(self, param):
        tune_fedavg = []
        a_range = param.alpha1_range
        for a in a_range:
            fn_fedavg, k_fedavg, x_last = self.train(param = TrainParam(alpha1 = a, K = param.K, initial_x=param.initial_x))
            if fn_fedavg[-1] < 100:
                print('a=', a, 'fn_fedavg_last=', fn_fedavg[-1], 'k_fedavg=', k_fedavg, 'x_last', x_last)
                tune_fedavg.append([a, k_fedavg, fn_fedavg])
            print('a', a)
        
        return tune_fedavg


class GIANT:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star

    def train(self, param):
        alpha1, K = 2**param.alpha1, param.K 

        x = param.initial_x
        fn = []
  
        for k in range(K):
            p = np.zeros((self.ndim, 1)) # the update direction
            for i in range(self.nclient):
                p = p + np.linalg.inv(
                    self.func.local_hess(x, i)
                    ).dot(self.func.global_grad(x))
          
            x = x - alpha1 * p
        
            fn.append(self.func.global_func(x))
            # two rounds of communications
            fn.append(self.func.global_func(x))
  
            if fn[-1] > 1e10:
                break
  
            if np.log(fn[-1] - self.fn_star) < -20:
                break
        return np.array(fn) - self.fn_star, k, x  

    def tune(self, param):
        tune_giant = []
        a_range = param.alpha1_range
        for a in a_range:
            fn_giant, k_giant, x_last = self.train(param = TrainParam(alpha1 = a, K = param.K, initial_x=param.initial_x))
            if fn_giant[-1] < 100:
                print('a=', a, 'fn_giant_last=', fn_giant[-1], 'k_giant=', k_giant, 'x_last', x_last)
                tune_giant.append([a, k_giant, fn_giant])
            print('a=', a)
        
        return tune_giant


class DiSCO:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star

    def train(self, param):
        alpha1, K = param.alpha1, param.K

        x = param.initial_x
        fn = []
    
        mu = 10000
        L = 0
        totalIte = 0
        
        for k in range(K):
    
            H = [[] for i in range(self.nclient)]
            for i in range(self.nclient):
                H[i] = self.func.local_hess(x, i)
                mu = min(mu, min(np.linalg.eigvals(H[i])))
                L = max(mu, max(np.linalg.eigvals(H[i])))
            H_sum = np.sum(H, axis = 0)
                
            g = 0
            for i in range(self.nclient):
                g += self.func.local_grad(x, i)
            eps = 1/20*(mu/L)**0.5*np.linalg.norm(g)
            
            P = H[0] + 0.01 * np.identity(self.ndim)
            innerIte = 1
            v = np.zeros([self.ndim,1])
            r = g
            s = np.linalg.inv(P)@r
            u = s
            
            while np.linalg.norm(r) > eps:
                a = r.T@s/(u.T@H_sum@u)
                v_old = np.copy(v)
                v = v + a*u
                r1 = r - a*H_sum@u
                s1 = np.linalg.inv(P)@r1
                b = r1.T@s1/(r.T@s)
                r = r1
                s = s1
                u_old = np.copy(u)
                u = s + b*u
                innerIte += 1
    
                fn.append(self.func.global_func(x))
    
            totalIte += innerIte

            if alpha1 is None: # if alpha1 is None, then we run the DiSCO (damped Newton given in the paper)
                delta = np.sqrt(v.T@H_sum@v_old + a*v.T@H_sum@u_old)
                x = x - v/(1+delta)

            else: # if alpha1 is not None, then we run the practical DiSCO with constant stepsize
                x = x - 2**alpha1*v
    
            fn.append(self.func.global_func(x))

            if fn[-1] > 1e40:
                break
    
            if np.log(fn[-1] - self.fn_star) < -20:
                break
                
        return np.array(fn) - self.fn_star, totalIte, x

    def tune(self, param):
        tune_disco = []
        a_range = param.alpha1_range
        for a in a_range:
            fn_disco, k_disco, x_last = self.train(param = TrainParam(alpha1 = a, K = param.K, initial_x=param.initial_x))
            if fn_disco[-1] < 1:
                print('a=', a, 'fn_disco_last=', fn_disco[-1], 'k_disco=', k_disco, 'x_last', x_last)
                tune_disco.append([a, k_disco, fn_disco])
        
        return tune_disco  


# FedHybrid, when nsecond=0, it's FedH_G; when nsecond=nclient, it's FedH_N
class FedHybrid:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star

    def train(self, param):
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K

        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1])
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []

        for k in range(K):
            for i in param.client_gradient: 
                g = self.func.local_grad(x[i], i) # local gradient
    
                new_x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
            
                dual[i] = dual[i] + beta1 * (x0 - x[i])
    
                x[i] = new_x[i]

            for i in param.client_Newton: 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
    
                new_x[i] = x[i] - alpha2 * np.asmatrix(np.linalg.inv(H)) * (g - dual[i] + mu * (x[i] - x0))
            
                dual[i] = dual[i] + beta2 * np.asmatrix(H) * (x0 - x[i])
    
                x[i] = new_x[i]
                
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
    
            if fn[-1] > 1e10:
                break
            
            if np.log(fn[-1] - self.fn_star) < -20:
                break
    
        return np.array(fn) - self.fn_star, k, x    

    def tune(self, param):
        tune_fedh = []
        a_range, b_range, a2_range, b2_range, mu_range = param.alpha1_range, param.beta1_range, param.alpha2_range, param.beta2_range, param.mu_range
        for a in a_range:
            for b in b_range:
                for a2 in a2_range:
                    for b2 in b2_range:
                        for u in mu_range:
                            fn_fedh, k_fedh, _ = self.train(param = TrainParam(alpha1 = a, beta1 = b, alpha2 = a2, beta2 = b2, mu = u, K = param.K, 
                                                            client_gradient= param.client_gradient, client_Newton = param.client_Newton, initial_x=param.initial_x))
                            if fn_fedh[-1] < 0.01: #and k_fedh < param.K-1:
                                print('alpha=', a, 'beta=', b, 'alpha2=', a2, 'beta2=', b2, 'mu=', u, 'fn_fedh_last=', fn_fedh[-1], 'k_fedh=', k_fedh)
                                tune_fedh.append([a, b, a2, b2, u, k_fedh, fn_fedh])
                                #plt.plot(np.log(fn_fedh))
                                #plt.savefig('Logistic_Synthetic/tune/FedH' + str(len(param.client_Newton)) + '/tune' + str(a) + '_' + str(b) + '_' + str(a2) + '_' + str(b2) + '_' + str(u) + '.pdf')
                                #plt.clf()
            print('a=', a)
        
        return tune_fedh

 
# Federated PN_DG (Second-order primal and first-order dual)
class PN_DG:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star

    def train(self, param):
        beta1, alpha2, mu, K = 10**param.beta1, 10**param.alpha2, 10**param.mu, param.K

        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1])
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        
        for k in range(K):
            for i in range(self.nclient): 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
    
                new_x[i] = x[i] - alpha2 * np.asmatrix(np.linalg.inv(H), dtype='float') * (g - dual[i] + mu * (x[i] - x0))
            
                dual[i] = dual[i] + beta1 * (x0 - x[i])
    
                x[i] = new_x[i]
                
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
    
            if fn[-1] > 1e20:
                break
            
            if np.log(fn[-1] - self.fn_star) < -20:
                break
    
        return np.array(fn) - self.fn_star, k, x    

    def tune(self, param):
        tune_pn_dg = []
        a_range, b_range, mu_range = param.alpha2_range, param.beta1_range, param.mu_range
        for a in a_range:
            for b in b_range:
                for u in mu_range:
                    fn_pn_dg, k_pn_dg, _ = self.train(param = TrainParam(alpha2 = a, beta1 = b, mu = u, K = param.K, initial_x = param.initial_x))
                    if fn_pn_dg[-1] < 1:
                        print('alpha=', a, 'beta=', b, 'mu=', u, 'fn_pn_dg_last=', fn_pn_dg[-1], 'k_pn_dg=', k_pn_dg)
                        tune_pn_dg.append([a, b, u, k_pn_dg, fn_pn_dg])
                        #plt.plot(np.log(fn_pn_dg))
                        #plt.savefig('Quadratic_Synthetic/LargeLocalCond/tune/PN_DG' + '/tune' + str(a) + '_' + str(b) + '_' + str(u) + '.pdf')
                        #plt.clf()
                print('b=', b)
        
        return tune_pn_dg