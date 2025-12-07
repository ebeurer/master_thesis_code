# Several scripts have to be run before this one: dipoles.py, example_problem_setup.py, morozov_test.py, morozov_test_long.py
import os
import pickle
from tqdm import tqdm
from inverse_problem import InverseProblem
from configuration import Configuration
from dipoles import Dipoles, get_divergence_matrix
import numpy as np
from admm import unbalanced_admm


n_sources = 5000
problem_type = "long"
noise_level_subset = [0,3,6,9,12]
mu_values = [0.5e-4, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2, 10]
# Take best mu values from MOSEK computations:
if problem_type == "short":
    indices = [2,3,11,9,9]
    mu = [mu_values[index] for index in indices]
elif problem_type == "long":
    indices = [5,6,7,8,8]
    mu = [mu_values[index] for index in indices]
iteration_numbers = [2,4,6,8,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,450,500,600,700,800,900,1000]
seed = 1
version = "_g50"


with open("inverse_code/inversion_methods/small_example/lambda_experiments/"+problem_type+f"_range_{str(n_sources)}_{seed}.pkl", "rb") as file:
    lambdas, noise_levels = pickle.load(file)
    lambdas = np.array(lambdas)
    noise_levels = [noise_levels[index] for index in noise_level_subset]
    if problem_type == "short":
        lambdas = lambdas[noise_level_subset, -1]


config = Configuration("inverse_code/inversion_methods/small_example/configs.ini")
config.output_folder = "inverse_code/inversion_methods/small_example"
config.input_folder = "inverse_code/inversion_methods/small_example"

# Create source space for reconstruction
grid = Dipoles.from_txt(f"inverse_code/inversion_methods/text_files/half_grid_{str(n_sources)}.txt")

# Import leadfield matrices
leadfield = np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_eeg_leadfield_analytical.npy"))
leadfield = np.append(leadfield, np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_meg_leadfield_analytical.npy")), axis=0)

# Import inverse problem
with open("inverse_code/inversion_methods/small_example/"+problem_type+"_range_problem.pkl", "rb") as file:
    inverse_problem = pickle.load(file)

# Get connection list
connections = grid.short_connection_list(5, cutoff=10)
if problem_type == "long":
    point1 = np.argmin(np.linalg.norm(grid.dipoles[:,:3]-inverse_problem.dipoles.dipoles[0,:3],axis=1))
    point2 = np.argmin(np.linalg.norm(grid.dipoles[:,:3]-inverse_problem.dipoles.dipoles[1,:3],axis=1))
    connections = np.append(connections,[[point1, point2, 5]],axis=0)

divergence_matrix = get_divergence_matrix(connections, leadfield.shape[1])

admm_solutions = np.zeros((len(iteration_numbers), leadfield.shape[1], inverse_problem.measurements.shape[1], len(noise_levels)))
admm_scores = np.zeros((len(iteration_numbers), len(noise_levels)))
relative_errors = np.zeros((iteration_numbers[-1], len(noise_levels)))

primal_residuals = np.zeros((iteration_numbers[-1], len(noise_levels)))
dual_residuals = np.zeros((iteration_numbers[-1], len(noise_levels)))

for i in tqdm(range(len(noise_levels))):
    inverse_problem.add_noise(noise_levels[i], seed)
    try:
        with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/mu_{problem_type}_range_{str(n_sources)}_{seed}{version}.pkl", "rb") as file:
            reference_solution = np.array(pickle.load(file)[0])[:,:,i,indices[i]]
    except:
        reference_solution = None
        print("Couldn't load reference solution.")
    admm_solutions[:,:,:,i], primal_residuals[:,i], dual_residuals[:,i], relative_errors[:,i], _ = unbalanced_admm(leadfield, inverse_problem.measurements, connections, lambdas[i], mu[i], rho=0.5, max_iterations=iteration_numbers[-1], growth_penalty=50, divergence_matrix=divergence_matrix, iteration_numbers=iteration_numbers, reference_solution=reference_solution)
    for k in range(len(iteration_numbers)):
        try:
            if problem_type == "short":
                time_indices = [0]
            elif problem_type == "long":
                time_indices = [5]
            admm_scores[k,i] = np.mean(inverse_problem.evaluate_solution_emd(grid.dipoles[:,:3], admm_solutions[k,:,:,i], time_indices=time_indices, mute=True))
        except AssertionError:
            print(f"Solution was 0 at iteration {iteration_numbers[k]}.")
            admm_scores[k,i] = np.inf


with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/admm_{problem_type}_range_{str(n_sources)}_{seed}{version}.pkl", "wb") as file:
    pickle.dump([admm_solutions.tolist(), admm_scores.tolist(), primal_residuals.tolist(), dual_residuals.tolist(), relative_errors.tolist(), iteration_numbers, noise_level_subset], file)