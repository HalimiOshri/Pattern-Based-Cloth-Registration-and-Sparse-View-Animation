import numpy as np
from sympy.core.symbol import symbols
from sympy.solvers.solveset import nonlinsolve
from sympy import simplify, solve
from scipy.optimize import fsolve

def calc_square_3d(square_corners_2d):
    coeff = np.matmul(square_corners_2d, np.transpose(square_corners_2d)) + 1e6
    coeff = coeff.astype(np.float32)
    b = 2.68

    def func(x):
        return [coeff[0, 0] * x[0] ** 2 + coeff[1, 1] * x[1] ** 2 - 2 * coeff[0, 1] * x[0] * x[1] - b ** 2 + 1e19 * (x[0] < 0) + 1e19 * (x[1] < 0),
              coeff[1, 1] * x[1] ** 2 + coeff[2, 2] * x[2] ** 2 - 2 * coeff[1, 2] * x[1] * x[2] - b ** 2 + 1e19 * (x[1] < 0) + 1e19 * (x[2] < 0),
              coeff[2, 2] * x[2] ** 2 + coeff[3, 3] * x[3] ** 2 - 2 * coeff[2, 3] * x[2] * x[3] - b ** 2 + 1e19 * (x[2] < 0) + 1e19 * (x[3] < 0),
              coeff[3, 3] * x[3] ** 2 + coeff[0, 0] * x[0] ** 2 - 2 * coeff[3, 0] * x[3] * x[0] - b ** 2 + 1e19 * (x[3] < 0) + 1e19 * (x[0] < 0)]

    res = fsolve(func, [10, 10, 10, 10])
    pass