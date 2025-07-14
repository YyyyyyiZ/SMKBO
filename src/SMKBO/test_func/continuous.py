import torch
from .base import TestFunction

class Branin(TestFunction):
    """
    Details: http://www.sfu.ca/~ssurjano/branin.html
    Global optimum: :math:`f(-\\pi,12.275)=0.397887`
    """
    problem_type = 'continuous'
    def __init__(self, normalize=False):
        super(Branin, self).__init__(normalize)
        self.min = 0.397887
        self.minimum = torch.tensor([-torch.pi, 12.275])
        self.dim = 2
        self.lb = -3.0 * torch.ones(2)
        self.ub = 3.0 * torch.ones(2)
        self.int_var = torch.tensor([])
        self.cont_var = torch.arange(0, 2)
        self.info = (
            "2-dimensional Branin function \nGlobal minimum: "
            + "f(-pi, 12.275) = 0.397887"
        )

    def compute(self, x, normalize=False):
        x1 = x[:, 0]
        x2 = x[:, 1]

        t = 1 / (8 * torch.pi)
        s = 10
        r = 6
        c = 5 / torch.pi
        b = 5.1 / (4 * torch.pi ** 2)
        a = 1

        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * torch.cos(x1)
        f = term1 + term2 + s

        return f.unsqueeze(1)


class Hartmann3(TestFunction):
    """
    Details: http://www.sfu.ca/~ssurjano/hart3.html
    Global optimum: :math:`f(0.114614,0.555649,0.852547)=-3.86278`

    """
    problem_type = 'continuous'
    def __init__(self, normalize=False):
        super(Hartmann3, self).__init__(normalize)
        self.dim = 3
        self.lb = torch.zeros(3)
        self.ub = torch.ones(3)
        self.int_var = torch.tensor([])
        self.cont_var = torch.arange(0, 3)
        self.min = -3.86278
        self.minimum = torch.tensor([0.114614, 0.555649, 0.852547])
        self.info = "3-dimensional Hartmann function \nGlobal maximum: f(0.114614,0.555649,0.852547) = -3.86278"

    def compute(self, x, normalize=False):
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        A = torch.tensor([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        P = torch.tensor([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ]) * 1e-4

        f = torch.zeros((x.shape[0], 1))
        for i in range(4):
            tmp = torch.zeros_like(f)
            for j in range(3):
                tmp += (A[i, j] * (x[:, j] - P[i, j]) ** 2).unsqueeze(1)
            f += alpha[i] * torch.exp(-tmp)
        return -f

class Hartmann6(TestFunction):
    """
    Details: http://www.sfu.ca/~ssurjano/hart6.html
    Global optimum: :math:f(0.201,0.150,0.476,0.275,0.311,0.657)=-3.322
    """
    problem_type = 'continuous'
    def __init__(self, normalize=False):
        super(Hartmann6, self).__init__(normalize)
        self.min = -3.32237
        self.minimum = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        self.dim = 6
        self.lb = torch.zeros(6)
        self.ub = torch.ones(6)
        self.int_var = torch.tensor([])
        self.cont_var = torch.arange(0, 6)
        self.info = (
            "6-dimensional Hartmann function \nGlobal optimum: "
            + "f(0.2016,0.15001,0.47687,0.27533,0.31165,0.657) = -3.3223"
        )

    def compute(self, x, normalize=False):
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        A = torch.tensor(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        P = 1e-4 * torch.tensor(
            [
                [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
                [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
            ]
        )
        result = torch.zeros(x.shape[0])
        for ii in range(4):
            inner = torch.sum(A[ii, :] * (x - P[ii, :]) ** 2, dim=1)
            result += alpha[ii] * torch.exp(-inner)
        return -result[:, None]


