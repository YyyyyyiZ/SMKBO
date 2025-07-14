import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, qKnowledgeGradient
from botorch.optim import optimize_acqf

from SMKBO.spectral.gp_regression import SingleTaskGP


class OptimizerCont:
    def __init__(self, lb, ub, continuous_kern_type='smk', n_Cauchy=5, n_Gaussian=4, acq='ei', n_init=20):
        self.lb = lb
        self.ub = ub
        self.bounds = torch.stack((self.lb, self.ub))
        self.dim = len(self.lb)
        self.continuous_kern_type = continuous_kern_type
        self.n_Cauchy = n_Cauchy
        self.n_Gaussian = n_Gaussian
        self.acq = acq
        self.n_init = n_init
        self.X = None
        self.fX =None
        self.best_init_y = None

    def _create_initial_points(self):
        init_x = torch.rand(self.n_init, self.dim)
        init_x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * init_x
        return init_x

    def observe(self, x_next, y_next):
        self.X = torch.cat([self.X, x_next]) if self.X is not None else x_next
        self.fX = torch.cat([self.fX, y_next]) if self.fX is not None else y_next
        self.best_init_y = self.fX.min().item()

    def suggest(self, batch_size=1):
        if self.X is None:
            x_next = self._create_initial_points()
        else:
            x_next = self._get_next_points(batch_size=batch_size)
        return x_next


    def _get_next_points(self, batch_size):
        noise_prior = GammaPrior(1.1, 0.5)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=[],
            noise_constraint=GreaterThan(
                # 0.000005,  # minimum observation noise assumed in the GP model
                0.0001,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        single_model = SingleTaskGP(self.X, self.fX, likelihood=likelihood, covar_module=self.continuous_kern_type,
                                    n_mixture=self.n_Gaussian+self.n_Gaussian, n_mixture1=self.n_Cauchy, n_mixture2=self.n_Gaussian)
        mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_mll(mll)
        if self.acq == 'ei':
            acq_function = ExpectedImprovement(model=single_model, best_f=self.best_init_y, maximize=False)
        elif self.acq == 'pi':
            acq_function = ProbabilityOfImprovement(model=single_model, best_f=self.best_init_y, maximize=False)
        elif self.acq == 'ucb':
            acq_function = UpperConfidenceBound(model=single_model, beta=0.5, maximize=False)
        else:
            raise ValueError('Acquisition function not identified')
        candidates, _ = optimize_acqf(acq_function=acq_function, bounds=self.bounds, q=batch_size, num_restarts=100,
                                      raw_samples=512, options={"batch_limit": 5, "maxiter": 100})
        return candidates