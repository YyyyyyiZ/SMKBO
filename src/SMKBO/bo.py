from SMKBO.cas.optimizer import Optimizer
from SMKBO.cas.optimizer_mixed import MixedOptimizer
from SMKBO.cas.optimizer_cont import OptimizerCont

class SpectralBO:
    def __init__(self, problem_type='mixed', cat_vertices=None, cont_lb=None, cont_ub=None, cat_dims=None, cont_dims=None,
                 continuous_kern_type='smk', n_Cauchy=5, n_Gaussian=4, n_init=20, acq_func='ei',
                 noise_variance=None, ard=True):
        self.problem_type = problem_type
        self.cat_vertices = cat_vertices
        self.cont_lb = cont_lb
        self.cont_ub = cont_ub
        self.cat_dims = cat_dims
        self.cont_dims = cont_dims
        self.continuous_kern_type = continuous_kern_type
        self.n_Cauchy = n_Cauchy
        self.n_Gaussian = n_Gaussian
        self.n_init = n_init
        self.acq_func = acq_func
        self.ard = ard
        self.noise_variance = noise_variance
        self._sanity_check()

        if problem_type == 'mixed':
            kwargs = {"continuous_kern_type": self.continuous_kern_type, "num_mixtures1": self.n_Cauchy,
                      "num_mixtures2": self.n_Gaussian}
            self.optim = MixedOptimizer(self.cat_vertices, self.cont_lb, self.cont_ub, self.cont_dims, self.cat_dims,
                                   n_init=self.n_init, use_ard=self.ard, acq=self.acq_func, kernel_type='mixed',
                                   noise_variance=self.noise_variance, **kwargs)
        elif problem_type == 'categorical':
            self.optim = Optimizer(self.cat_vertices, n_init=self.n_init, use_ard=self.ard, acq=self.acq_func,
                              kernel_type='transformed_overlap', noise_variance=self.noise_variance)
        else:
            self.optim=OptimizerCont(lb=self.cont_lb, ub=self.cont_ub, continuous_kern_type=self.continuous_kern_type,
                                     n_Cauchy=self.n_Cauchy, n_Gaussian=self.n_Gaussian, acq=self.acq_func, n_init=self.n_init)

    def _sanity_check(self):
        assert self.problem_type in ['categorical', 'continuous', 'mixed'], 'Unknown problem type ' + str(self.problem_type)
        assert self.continuous_kern_type in ['mat52', 'rbf', 'smk'], 'Unknown continuous kernel choice ' + str(self.continuous_kern_type)
        assert self.acq_func in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(self.acq_func)
