from SMKBO.bo import SpectralBO
from SMKBO.test_func import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#%% test1
f = Func2C(lamda=1e-6, normalize=False)
max_iters = 21
bo = SpectralBO(problem_type=f.problem_type, cat_vertices=f.config, cont_lb=f.lb, cont_ub=f.ub,
           cat_dims=f.categorical_dims, cont_dims=f.continuous_dims,
                 continuous_kern_type='smk', n_Cauchy=5, n_Gaussian=4, n_init=20, acq_func='ei',
                 noise_variance=None, ard=True)

for i in range(max_iters):
    x_next = bo.optim.suggest()
    y_next = f.compute(x_next, normalize=f.normalize)
    bo.optim.observe(x_next, y_next)
    if f.normalize:
        Y = np.array(bo.optim.smkbo.fX) * f.std + f.mean
    else:
        Y = np.array(bo.optim.smkbo.fX)
    if Y[:i].shape[0]:
        argmin = np.argmin(Y[:i])
        print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
              % (i, x_next, float(Y[-1]), Y[:i][argmin]))

#%% test2
f = Func3C(lamda=1e-6, normalize=False)
max_iters = 21
bo = SpectralBO(problem_type=f.problem_type, cat_vertices=f.config, cont_lb=f.lb, cont_ub=f.ub,
           cat_dims=f.categorical_dims, cont_dims=f.continuous_dims,
                 continuous_kern_type='mat52', n_init=20, acq_func='ei',
                 noise_variance=None, ard=True)

for i in range(max_iters):
    x_next = bo.optim.suggest()
    y_next = f.compute(x_next, normalize=f.normalize)
    bo.optim.observe(x_next, y_next)
    if f.normalize:
        Y = np.array(bo.optim.smkbo.fX) * f.std + f.mean
    else:
        Y = np.array(bo.optim.smkbo.fX)
    if Y[:i].shape[0]:
        argmin = np.argmin(Y[:i])
        print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
              % (i, x_next, float(Y[-1]), Y[:i][argmin]))

#%% test3
f = QUBO()
max_iters = 21
bo = SpectralBO(problem_type=f.problem_type, cat_vertices=f.config,  n_init=20, acq_func='ei',
                 noise_variance=None, ard=True)

for i in range(max_iters):
    x_next = bo.optim.suggest()
    y_next = f.compute(x_next)
    bo.optim.observe(x_next, y_next)
    Y = np.array(bo.optim.smkbo.fX)
    if Y[:i].shape[0]:
        argmin = np.argmin(Y[:i])
        print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
              % (i, x_next, float(Y[-1]), Y[:i][argmin]))

#%% test4
f = Hartmann3()
max_iters = 21
bo = SpectralBO(problem_type=f.problem_type, cont_lb=f.lb, cont_ub=f.ub,
                continuous_kern_type='smk', n_Cauchy=9, n_Gaussian=0, n_init=20, acq_func='ucb',
                 noise_variance=None, ard=True)

for i in range(max_iters):
    x_next = bo.optim.suggest()
    y_next = f.compute(x_next)
    bo.optim.observe(x_next, y_next)
    Y = np.array(bo.optim.fX)
    if Y[:i].shape[0]:
        argmin = np.argmin(Y[:i ])
        print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
              % (i, x_next, float(Y[-1]), Y[:i][argmin]))
