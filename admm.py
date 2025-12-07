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
from scipy.linalg import cho_factor, cho_solve
from dipoles import get_divergence_matrix
from tqdm import tqdm
import scipy.sparse as sparse
import cvxpy as cp
import time


def admm(leadfield_matrix, measurements, connections, lam, mu, rho=1, epsilon=1e-3, max_iterations=500, divergence_matrix=None, reference_solution=None):
    """
    Our own ADMM formulation for the MCE with EMD temporal regularization.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param connections: The connection list.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :param rho: The ADMM penalty parameter
    :param epsilon: A stopping criterion.
    :return prob: A cvxpy problem object.
    """
    n = leadfield_matrix.shape[1]
    m = connections.shape[0]
    T = measurements.shape[1]

    j = np.zeros((n,T))
    x = np.zeros((n,T))
    s = np.zeros(m*(T-1))

    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections, n)

    A = sparse.block_diag([-sparse.eye(n*T)]+[divergence_matrix for i in range(T-1)], format="csr")
    B = sparse.bmat([[sparse.eye(n*T)],[sparse.diags([-np.ones(n*(T-1)),np.ones(n*(T-1))], [0,n], (n*(T-1),n*T))]], format="csr")

    u = np.zeros(2*n*T-n)

    # Precomputations for x
    prox_x_decomposition = cho_factor(np.eye(n)+2*(leadfield_matrix.T@leadfield_matrix)/rho)
    prox_x_summand = 2/rho * (leadfield_matrix.T @ measurements)

    # Problem setup for j
    j_var = cp.Variable(j.flatten("F").shape, nonneg=True)
    j_rhs = cp.Parameter(2*n*T-n)
    j_objective = cp.Minimize(rho/2 * cp.sum_squares(B @ j_var - j_rhs) + lam * cp.sum(j_var))
    j_prob = cp.Problem(j_objective)

    primal_residual_norm = np.zeros(max_iterations)
    dual_residual_norm = np.zeros(max_iterations)
    relative_error = np.zeros(max_iterations)
    cost = np.zeros((max_iterations,3))

    time_x = 0
    time_s = 0
    time_j = 0

    for i in tqdm(range(max_iterations)):
        # Store previous j for dual residual calculation
        j_prev = j.copy()

        # Update x
        start = time.time()
        x = cho_solve(prox_x_decomposition, j+u[:n*T].reshape(j.shape, order="F")+prox_x_summand)
        time_x += time.time()-start

        # Update s
        start = time.time()
        s_var = cp.Variable(s.shape)
        objective = rho/2 * cp.sum_squares(A[n*T:,n*T:] @ s_var + u[n*T:] + B[n*T:,:] @ j.flatten("F")) + mu * cp.sum(cp.multiply(cp.abs(s_var),np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1).reshape(s.shape, order="F")))
        s_objective = cp.Minimize(objective)
        s_prob = cp.Problem(s_objective)
        s_prob.solve(solver="MOSEK", ignore_dpp=True, mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'})
        s = s_var.value
        time_s += time.time()-start

        # Update j
        start = time.time()
        j_rhs.value = -u - A @ np.append(x.flatten("F"), s)
        j_prob.solve(solver="MOSEK", ignore_dpp=True, mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'})
        j = j_var.value.reshape(j.shape, order="F")
        time_j += time.time()-start

        # Update u
        primal_residual = A @ np.append(x.flatten("F"), s) + B @ j.flatten("F")
        u = u + primal_residual

        primal_residual_norm[i] = np.linalg.norm(primal_residual)
        dual_residual_norm[i] = rho * np.linalg.norm(A.T @ (B @ (j.flatten("F") - j_prev.flatten("F"))))
        if reference_solution is not None:
            relative_error[i] = np.linalg.norm(reference_solution-j) / np.linalg.norm(reference_solution)
        cost[i,0] = np.linalg.norm(measurements-leadfield_matrix @ x)**2
        cost[i,1] = lam * np.sum(j)
        cost[i,2] = mu * np.sum(np.abs(s.reshape(m,T-1, order="F") * np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1)))

        if primal_residual_norm[i] < epsilon and dual_residual_norm[i] < epsilon:
            print(f"Converged after {i} iterations.")
            primal_residual_norm = primal_residual_norm[:i+1]
            dual_residual_norm = dual_residual_norm[:i+1]
            relative_error = relative_error[:i+1]
            cost = cost[:i+1,:]
            break

    print(f"time_x: {time_x}, time_s: {time_s}, time_j: {time_j}")

    if np.linalg.norm(j)<1e-16:
        print("x is 0 :(")
    return j, primal_residual_norm, dual_residual_norm, relative_error, cost, [x, s]


def unbalanced_admm(leadfield_matrix, measurements, connections, lam, mu, rho=1, epsilon=1e-3, max_iterations=500, divergence_matrix=None, reference_solution=None, growth_penalty=20, iteration_numbers=None):
    """
    Our own ADMM formulation for the MCE with uEMD temporal regularization.

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param connections: The connection list.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :param rho: The ADMM penalty parameter
    :param epsilon: A stopping criterion.
    :param growth_penalty: The penalty parameter tau for mass destruction/creation.
    :return prob: A cvxpy problem object.
    """
    n = leadfield_matrix.shape[1]
    m = connections.shape[0]
    T = measurements.shape[1]

    j = np.zeros((n,T))
    x = np.zeros((n,T))
    s = np.zeros(m*(T-1))
    r = np.zeros(n*(T-1))

    if iteration_numbers is not None:
        out = []

    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections, n)

    A = sparse.bmat([[-sparse.eye(n*T), None, None],[None, sparse.block_diag([divergence_matrix for i in range(T-1)]), -sparse.eye(n*(T-1))]], format="csr")
    B = sparse.bmat([[sparse.eye(n*T)],[sparse.diags([-np.ones(n*(T-1)),np.ones(n*(T-1))], [0,n], (n*(T-1),n*T))]], format="csr")

    u = np.zeros(2*n*T-n)

    # Precomputations for x
    prox_x_decomposition = cho_factor(np.eye(n)+2*(leadfield_matrix.T@leadfield_matrix)/rho)
    prox_x_summand = 2/rho * (leadfield_matrix.T @ measurements)

    # Problem setup for s and r
    s_var = cp.Variable(s.shape)
    r_var = cp.Variable(r.shape)
    sr_rhs = cp.Parameter(n*T-n)
    sr_objective = cp.Minimize(rho/2 * cp.sum_squares(A[n*T:(n*T+n*(T-1)),n*T:(n*T+m*(T-1))] @ s_var - r_var - sr_rhs) + mu * (cp.sum(cp.abs(cp.multiply(s_var,np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1).reshape(s.shape, order="F")))) + growth_penalty * cp.sum(cp.abs(r_var))))
    s_prob = cp.Problem(sr_objective)

    # Problem setup for j
    j_var = cp.Variable(j.flatten("F").shape, nonneg=True)
    j_rhs = cp.Parameter(2*n*T-n)
    j_objective = cp.Minimize(rho/2 * cp.sum_squares(B @ j_var - j_rhs) + lam * cp.sum(j_var))
    j_prob = cp.Problem(j_objective)

    primal_residual_norm = np.zeros(max_iterations)
    dual_residual_norm = np.zeros(max_iterations)
    relative_error = np.zeros(max_iterations)
    cost = np.zeros((max_iterations,3))

    time_x = 0
    time_s = 0
    time_j = 0

    for i in tqdm(range(max_iterations)):
        # Store previous j for dual residual calculation
        j_prev = j.copy()

        try:
            # Update x
            start = time.time()
            x = cho_solve(prox_x_decomposition, j+u[:n*T].reshape(j.shape, order="F")+prox_x_summand)
            time_x += time.time()-start

            # Update s
            start = time.time()
            sr_rhs.value = -u[n*T:] - B[n*T:,:] @ j.flatten("F")
            s_prob.solve(solver="MOSEK", ignore_dpp=True)
            s = s_var.value
            r = r_var.value
            time_s += time.time()-start

            # Update j
            start = time.time()
            j_rhs.value = -u - A @ np.concatenate((x.flatten("F"), s, r))
            j_prob.solve(solver="MOSEK", ignore_dpp=True)
            j = j_var.value.reshape(j.shape, order="F")
            time_j += time.time()-start

            # Update u
            primal_residual = A @ np.concatenate((x.flatten("F"), s, r)) + B @ j.flatten("F")
            u = u + primal_residual
        except:
            print(f"Error after {i} iterations.")
            print(f"time_x: {time_x}, time_s: {time_s}, time_j: {time_j}")
            if np.linalg.norm(j)<1e-16:
                print("x is 0 :(")
            if iteration_numbers is not None:
                for k in range(len(iteration_numbers)-len(out)):
                    out.append(np.zeros(j.shape))
                return np.array(out), primal_residual_norm, dual_residual_norm, relative_error, cost
            else:
                return j, primal_residual_norm, dual_residual_norm, relative_error, cost


        primal_residual_norm[i] = np.linalg.norm(primal_residual)
        dual_residual_norm[i] = rho * np.linalg.norm(A.T @ (B @ (j.flatten("F") - j_prev.flatten("F"))))
        if reference_solution is not None:
            relative_error[i] = np.linalg.norm(reference_solution-j) / np.linalg.norm(reference_solution)
        cost[i,0] = np.linalg.norm(measurements-leadfield_matrix @ x)**2
        cost[i,1] = lam * np.sum(j)
        cost[i,2] = mu * (np.sum(np.abs(s.reshape(m,T-1, order="F") * np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1))) + np.sum(np.abs(r)))

        if iteration_numbers is not None and i+1 in iteration_numbers:
            out.append(j)

        if primal_residual_norm[i] < epsilon and dual_residual_norm[i] < epsilon:
            print(f"Converged after {i} iterations.")
            primal_residual_norm = primal_residual_norm[:i+1]
            dual_residual_norm = dual_residual_norm[:i+1]
            relative_error = relative_error[:i+1]
            cost = cost[:i+1,:]
            break

    print(f"time_x: {time_x}, time_s: {time_s}, time_j: {time_j}")

    if np.linalg.norm(j)<1e-16:
        print("x is 0 :(")
    if iteration_numbers is not None:
        return np.array(out), primal_residual_norm, dual_residual_norm, relative_error, cost
    else:
        return j, primal_residual_norm, dual_residual_norm, relative_error, cost
    

def paper_admm(leadfield_matrix, measurements, connections, lam, mu, rho=1, epsilon=1e-3, max_iterations=500, initial_values=None, early_stopping=10, unbalanced=False, divergence_matrix=None, reference_solution=None):
    """
    Implements an ADMM algorithm very similar to the one from [John Lee, Nicholas P. Bertrand, and Christopher J. Rozell. “Unbalanced Optimal Transport Regularization for Imaging Problems”. In: IEEE Transactions on Computational Imaging 6 (2020), pp. 1219–1232. doi: 10.1109/tci.2020.3012954].

    :param leadfield_matrix: The leadfield matrix for the problem.
    :param measurements: The measurements, with each column corresponding to the measurement at one time point.
    :param connections: The connection list.
    :param lam: The spatial regularization parameter lambda.
    :param mu: The temporal regularization parameter mu.
    :param rho: The ADMM penalty parameter
    :param epsilon: A stopping criterion.
    :param early_stopping: After how many steps the inner iteration will be stopped.
    :return prob: A cvxpy problem object.
    """
    n = leadfield_matrix.shape[1]
    m = connections.shape[0]
    T = measurements.shape[1]

    j = np.zeros((n,T))
    x = np.zeros((n,T))
    z = np.zeros((n,T)) # Last column will remain zero
    w = np.zeros((n,T)) # First column will remain zero

    a = np.zeros((n,T))
    b = np.zeros((n,T))
    c = np.zeros((n,T))

    # Precomputation for quickly calculating proximal operator of ||Phi-Lx||
    prox_x_decomposition = cho_factor(np.eye(n)+2*(leadfield_matrix.T@leadfield_matrix)/rho)
    prox_x_summand = 2/rho * (leadfield_matrix.T @ measurements)

    # Precomputation and warm-start variables for proximal operator of EMD(x1,x2)
    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections, n)
    warm_s = np.zeros((m,T-1))
    warm_a = np.zeros((n,T-1))

    primal_residual_norm = np.zeros(max_iterations)
    dual_residual_norm = np.zeros(max_iterations)
    relative_error = np.zeros(max_iterations)
    cost = np.zeros((max_iterations,3))

    for i in tqdm(range(max_iterations)):
        # Store previous values for x,z,w for dual residual calculation
        x_prev = x.copy()
        z_prev = z.copy()
        w_prev = w.copy()
        # Proximal operator for j
        np.clip((x+a+z+b+w+c)/3-lam/3/rho, a_min=0, a_max=None, out=j)
        # Proximal operator for x
        x = cho_solve(prox_x_decomposition, j-a+prox_x_summand)
        # Proximal operator for z and w
        if unbalanced:
            warm_r = np.zeros(warm_a.shape)
            z[:,:T-1], w[:,1:], warm_s, warm_r, warm_a = prox_unbalanced_emd(j[:,:T-1]-b[:,:T-1], j[:,1:]-c[:,1:], connections, mu/rho, 1e-3, 1, early_stopping, warm_s, warm_r, warm_a, divergence_matrix=divergence_matrix, growth_penalty=20)
        else:
            z[:,:T-1], w[:,1:], warm_s, warm_a = prox_emd(j[:,:T-1]-b[:,:T-1], j[:,1:]-c[:,1:], connections, mu/rho, 1e-3, 1, early_stopping, warm_s, warm_a, divergence_matrix=divergence_matrix)
        # Dual variable updates
        a += x - j
        b[:, :T-1] += z[:, :T-1] - j[:, :T-1]
        c[:, 1:] += w[:, 1:] - j[:, 1:]

        primal_residual_norm[i] = np.sqrt(np.sum((j-x)**2)+np.sum((j[:,0:-1]-z[:,0:-1])**2)+np.sum((j[:,1:]-w[:,1:])**2))
        dual_residual_norm[i] = rho * np.linalg.norm(x+z+w-x_prev-z_prev-w_prev)/3
        if reference_solution is not None:
            relative_error[i] = np.linalg.norm(reference_solution-j) / np.linalg.norm(reference_solution)
        cost[i,0] = np.linalg.norm(measurements-leadfield_matrix @ x)**2
        cost[i,1] = lam * np.sum(j)
        cost[i,2] = mu * np.sum(np.abs(warm_s * np.repeat(connections[:,2].reshape(-1,1), T-1, axis=1)))

        if primal_residual_norm[i] < epsilon and dual_residual_norm[i] < epsilon:
            print(f"Converged after {i} iterations.")
            primal_residual_norm = primal_residual_norm[:i+1]
            dual_residual_norm = dual_residual_norm[:i+1]
            relative_error = relative_error[:i+1]
            cost = cost[:i+1,:]
            break

    if np.linalg.norm(j)<1e-16:
        print("x is 0 :(")
    return j, primal_residual_norm, dual_residual_norm, relative_error, cost, [x, warm_s]


def prox_emd(v1: np.ndarray, v2: np.ndarray, connections: np.ndarray, lam=1, tau1=1, tau2=1, max_iterations=100, s_init=None, a_init=None, divergence_matrix=None):
    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections, v1.shape[0])
    if (s_init is None) and (v1.ndim==2):
        s_prev = np.zeros((connections.shape[0], v1.shape[1]))
    elif (s_init is None) and (v1.ndim==1):
        s_prev = np.zeros(connections.shape[0])
    else:
        s_prev = s_init.copy()
    if a_init is None:
        a_prev = np.zeros(v1.shape)
    else:
        a_prev = a_init.copy()
    x1_prev = v1.copy()
    x2_prev = v2.copy()
    x1 = np.zeros(x1_prev.shape)
    x2 = np.zeros(x2_prev.shape)
    s = np.zeros(s_prev.shape)
    a = np.zeros(a_prev.shape)
    for i in range(max_iterations):
        # Proximal operator for x1 and x2
        np.clip(tau1/(lam+tau1)*v1+lam/(lam+tau1)*(x1_prev+tau1*a), a_min=0, a_max=None, out=x1)
        np.clip(tau1/(lam+tau1)*v2+lam/(lam+tau1)*(x2_prev-tau1*a), a_min=0, a_max=None, out=x2)
        # Proximal operator for s
        with np.errstate(divide='ignore'):
            data = s_prev-tau1*(np.take(a, connections[:,0].astype(int), axis=0)-np.take(a, connections[:,1].astype(int), axis=0))
            s = data * np.clip(1-tau1*(connections[:,2]/np.abs(data).T).T, a_min=0, a_max=None)
        # Dual variable update
        a = a_prev + tau2 * (divergence_matrix @ s - (2*x1-x1_prev) + (2*x2-x2_prev))
        # residual = np.sum((s-s_prev)**2)/tau1 + np.sum((a-a_prev)**2)/tau2 + 2*np.sum((a-a_prev)*(divergence_matrix @ (s-s_prev)))
        np.copyto(x1_prev, x1)
        np.copyto(x2_prev, x2)
        np.copyto(s_prev, s)
        np.copyto(a_prev, a)
    # print(residual)
    return x1, x2, s, a


def prox_unbalanced_emd(v1: np.ndarray, v2: np.ndarray, connections: np.ndarray, lam=1, tau1=1, tau2=1, max_iterations=100, s_init=None, r_init=None, a_init=None, divergence_matrix=None, growth_penalty=1):
    if divergence_matrix is None:
        divergence_matrix = get_divergence_matrix(connections, v1.shape[0])
    if (s_init is None) and (v1.ndim==2):
        s_prev = np.zeros((connections.shape[0], v1.shape[1]))
    elif (s_init is None) and (v1.ndim==1):
        s_prev = np.zeros(connections.shape[0])
    else:
        s_prev = s_init.copy()
    if a_init is None:
        a_prev = np.zeros(v1.shape)
    else:
        a_prev = a_init.copy()
    if r_init is None:
        r_prev = np.zeros(v1.shape)
    else:
        r_prev = r_init.copy()
    x1_prev = v1.copy()
    x2_prev = v2.copy()
    x1 = np.zeros(x1_prev.shape)
    x2 = np.zeros(x2_prev.shape)
    s = np.zeros(s_prev.shape)
    r = np.zeros(r_prev.shape)
    a = np.zeros(a_prev.shape)
    for i in range(max_iterations):
        # Proximal operator for x1 and x2
        np.clip(tau1/(lam+tau1)*v1+lam/(lam+tau1)*(x1_prev+tau1*a), a_min=0, a_max=None, out=x1)
        np.clip(tau1/(lam+tau1)*v2+lam/(lam+tau1)*(x2_prev-tau1*a), a_min=0, a_max=None, out=x2)
        # Proximal operator for s and r
        with np.errstate(divide='ignore'):
            data = s_prev-tau1*(np.take(a, connections[:,0].astype(int), axis=0)-np.take(a, connections[:,1].astype(int), axis=0))
            s = data * np.clip(1-tau1*(connections[:,2]/np.abs(data).T).T, a_min=0, a_max=None)
            data = r_prev+tau1*a_prev
            r = data * np.clip(1-tau1*growth_penalty/np.abs(data), a_min=0, a_max=None)
        # Dual variable update
        a = a_prev + tau2 * (divergence_matrix @ s - (2*x1-x1_prev) + (2*x2-x2_prev) - (2*r-r_prev))
        # residual = np.sum((s-s_prev)**2)/tau1 + np.sum((a-a_prev)**2)/tau2 + 2*np.sum((a-a_prev)*(divergence_matrix @ (s-s_prev)))
        np.copyto(x1_prev, x1)
        np.copyto(x2_prev, x2)
        np.copyto(s_prev, s)
        np.copyto(r_prev, r)
        np.copyto(a_prev, a)
    # print(residual)
    return x1, x2, s, r, a
