# SMKBO: Spectral Mixture Kernel for Bayesian Optimization

This repo contains implementation of the SMKBO algorithm for mixed input variables.

### Installation

#### Dependencies

```
Python>=3.9
gpytorch>=1.14
pytorch==2.5.1
botorch>=0.14.0
numpy==1.16.4
```

#### Option 1: Installing the latest release

The latest release of SMKBO is easily installed via `pip`:

```bash
pip install smkbo
```


#### Option 2: Editable install

If you want to contribute to SMKBO, you will want to install editably so that you can change files and have the changes reflected in your local install.

```bash
git clone https://github.com/YyyyyyiZ/SMKBO.git
cd SMKBO
pip install -e .
```



### Getting Started

Here's a quick run down of the main components of a Bayesian optimization loop.

1. Import packages

   ```python
   from SMKBO.test_func import *
   from SMKBO.bo import SpectralBO
   ```



2. Define your optimization problem

   **Option 2a: Mixed problem**

   ```python
   f = Func2C(lamda=1e-6, normalize=False)
   max_iters = 30
   batch_size = 1
   bo = SpectralBO(problem_type=f.problem_type, cat_vertices=f.config, cont_lb=f.lb, cont_ub=f.ub,
                   cat_dims=f.categorical_dims, cont_dims=f.continuous_dims, continuous_kern_type='smk', 
                   n_Cauchy=5, n_Gaussian=4, n_init=20, acq_func='ei', noise_variance=None, ard=True)
   ```

   

   **Option 2b: Categorical problem**

   ```python
   f = QUBO()
   max_iters = 23
   batch_size = 1
   bo = SpectralBO(problem_type=f.problem_type, cat_vertices=f.config,  n_init=20, acq_func='ei', noise_variance=None, ard=True)
   ```

   

   **Option 2c: Continuous problem**

   ```python
   f = Hartmann3()
   max_iters = 23
   batch_size = 1
   bo = SpectralBO(problem_type=f.problem_type, cont_lb=f.lb, cont_ub=f.ub,
                   continuous_kern_type='smk', n_Cauchy=9, n_Gaussian=0, n_init=20, acq_func='ucb',
                    noise_variance=None, ard=True)
   ```

   

3. Optimization loop

   ```python
   for i in range(max_iters):
       x_next = bo.optim.suggest(batchsize=batch_size)
       y_next = f.compute(x_next)
       bo.optim.observe(x_next, y_next)
       Y = np.array(bo.optim.fX)
       if Y[:i].shape[0]:
           argmin = np.argmin(Y[:i * batch_size])
           print('Iter %d, Last X %s; \n fX:  %.4f. fX_best: %.4f'
                 % (i, x_next, float(Y[-1]), Y[:i * batch_size][argmin]))
   ```

   

### Customize a New Optimization Problem

1. Go to `./test_func/base.py` for the implementation of the base class. A new problem should be a derived class that properly implements all the required methods of the base class. 

2. For different types of optimization problem, see examples in `./test_func/mixed.py`, `./test_func/categorical.py`, `./test_func/continuous.py`, respectively

3. Edit the appropriate imports and other specifications and run from `test.py` as usual.

   

### Citing SMKBO

If you use SMKBO, please cite the following paper:

```tex
@misc{zhang2025spectralmixturekernelsbayesian,
      title={Spectral Mixture Kernels for Bayesian Optimization}, 
      author={Yi Zhang and Cheng Hua},
      year={2025},
      eprint={2505.17393},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17393}, 
}
```

### License

SMKBO is MIT licensed, as found in the [LICENSE](https://github.com/YyyyyyiZ/SMKBO/blob/main/LICENSE) file.



### Acknowledgements

This code repository uses materials from the following public repositories. The authors thank the respective repository maintainers

1. CASMOPOLITAN: Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C. &amp; Osborne, M.A.. (2021). Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. Proceedings of the 38th International Conference on Machine Learning (ICML). Code repo: https://github.com/xingchenwan/Casmopolitan.
2. BoTorch: M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33 (NIPS 33).  Code repo: https://github.com/pytorch/botorch.
