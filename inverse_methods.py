# Copyright 2025 Emil Beurer
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
import numpy as np
import cvxpy as cp
import scipy.sparse as sparse
from dipoles import get_divergence_matrix


def mce(leadfield_matrix, measurements, lam):
    """
    Computes the MCE solution (with positivity constraint) for the given problem. Makes use of the CVXPY modeling language and the MOSEK optimization package as backend.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param lam: The regularization parameter lambda.
    :return prob: A cvxpy problem object.
    :return x: The solution x as a cvxpy variable object. The solution array can be extracted via x.value.
    """
    n = int(leadfield_matrix.shape[1])

    # Define variables
    if measurements.ndim == 2:
        T = measurements.shape[1]
        x = cp.Variable((n,T))
    else:
        x = cp.Variable(n)

    # Define objective
    objective = cp.sum_squares(measurements-leadfield_matrix @ x) + lam * cp.sum(cp.abs(x))
    
    # Solve
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective)
    prob.solve(solver="MOSEK", mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'})
    return prob, x


def constant_mce(leadfield_matrix, measurements, lam):
    """
    Computes the cMCE solution (with positivity constraint) for the given problem. Makes use of the CVXPY modeling language and the MOSEK optimization package as backend.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param lam: The regularization parameter lambda.
    :return prob: A cvxpy problem object.
    :return x: The solution x as a cvxpy variable object. The solution array can be extracted via x.value.
    """
    n = int(leadfield_matrix.shape[1])
    
    # Define variables
    if measurements.ndim == 2:
        T = measurements.shape[1]
        x = cp.Variable((n,1))
    else:
        x = cp.Variable(n)

    # Define objective
    objective = cp.sum_squares(measurements-leadfield_matrix @ (x @ np.ones((1, T)))) + lam * T * cp.sum(cp.abs(x))
    
    # Solve
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective)
    prob.solve(solver="MOSEK", mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'})
    return prob, x


def norm_str(leadfield_matrix, measurements, lam, mu):
    """
    Computes the STR solution (with positivity constraint) for the given problem. Makes use of the CVXPY modeling language and the MOSEK optimization package as backend.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :return prob: A cvxpy problem object.
    :return x: The solution x as a cvxpy variable object. The solution array can be extracted via x.value.
    """
    n = int(leadfield_matrix.shape[1])
    T = measurements.shape[1]

    # Define variables
    x = cp.Variable((n,T))
    B = sparse.diags([-np.ones(n*(T-1)),np.ones(n*(T-1))], [0,n], (n*(T-1),n*T))

    # Define objective
    objective = cp.sum_squares(measurements-leadfield_matrix @ x) + lam * cp.sum(cp.abs(x)) + mu * cp.sum_squares(B @ cp.vec(x, "F"))
    
    # Solve
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective)
    prob.solve(solver="MOSEK", mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'})
    return prob, x


def emd_str(leadfield_matrix, measurements, connections, lam, mu, divergence_matrix=None, verbose=False):
    """
    Computes the (balanced) EMD solution for the given problem. Makes use of the CVXPY modeling language and the MOSEK optimization package as backend.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param connections: The connection list.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :return prob: A cvxpy problem object.
    :return x: The solution x as a cvxpy variable object. The solution array can be extracted via x.value.
    :return s: The EMD flow s as a cvxpy variable object. The array can be extracted via s.value.
    """
    n = leadfield_matrix.shape[1]
    m = connections.shape[0]
    T = measurements.shape[1]

    # Define variables
    x = cp.Variable((n,T), nonneg=True)
    s = cp.Variable((m,T-1))
    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections,n)

    # Define constraints and objective
    constraints = [divergence_matrix @ s == x[:,:T-1] - x[:,1:]]
    objective = cp.sum_squares(measurements-leadfield_matrix @ x) + lam * cp.sum(x) + mu * cp.sum(cp.multiply(cp.abs(s),np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1)))
    
    # Solve
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="MOSEK", mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'}, verbose=verbose)
    return prob, x, s


def unbalanced_emd_str(leadfield_matrix, measurements, connections, lam, mu, divergence_matrix=None, verbose=False, growth_penalty=50, standard_solver=False):
    """
    Computes the uEMD solution for the given problem. Makes use of the CVXPY modeling language and the MOSEK optimization package as backend.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param connections: The connection list.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :param growth_penalty: The penalty parameter tau for mass destruction/creation.
    :return prob: A cvxpy problem object.
    :return x: The solution x as a cvxpy variable object. The solution array can be extracted via x.value.
    :return s: The EMD flow s as a cvxpy variable object. The array can be extracted via s.value.
    """
    n = leadfield_matrix.shape[1]
    m = connections.shape[0]
    T = measurements.shape[1]
    # print(n)
    # Define variables
    x = cp.Variable((n,T), nonneg=True)
    # F = [cp.Variable((m,2), nonneg=True) for i in range(T-1)]
    s = cp.Variable((m,T-1))
    r = cp.Variable((n,T-1))
    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections,n)

    # Define constraints
    constraints = [divergence_matrix @ s - x[:,:T-1] + x[:,1:] == r]
    objective = cp.sum_squares(measurements-leadfield_matrix @ x) + lam * cp.sum(x) + mu * (cp.sum(cp.multiply(cp.abs(s),np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1))) + growth_penalty * cp.sum(cp.abs(r)))

    
    # Define objective
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective, constraints)
    if standard_solver:
        prob.solve(verbose=verbose)
    else:
        prob.solve(solver="MOSEK", mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'}, verbose=verbose)
    # print(prob.status)
    return prob, x, s