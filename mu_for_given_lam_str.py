# Several scripts have to be run before this one: dipoles.py, example_problem_setup.py, morozov_test.py, morozov_test_long.py
import os
import pickle
from tqdm import tqdm
from inverse_problem import InverseProblem
from configuration import Configuration
from dipoles import Dipoles
import numpy as np
from inverse_methods import mce, constant_mce, norm_str


n_sources = 5000
problem_type = "short"
noise_level_subset = [0,3,6,9,12]
mu_values = [0.5e-4, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2, 10]
recompute_benchmarks = False
seed = 1
version = "_str"


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

# If needed, compute mce and cmce benchmarks
if (not os.path.isfile(f"inverse_code/inversion_methods/small_example/lambda_experiments/bench_{problem_type}_range_{str(n_sources)}_{seed}.pkl")) or (recompute_benchmarks==True):
    mce_solutions = np.zeros((leadfield.shape[1], inverse_problem.measurements.shape[1], len(noise_levels)))
    cmce_solutions = np.zeros(mce_solutions.shape)
    mce_scores = np.zeros(len(noise_levels))
    cmce_scores = np.zeros(len(noise_levels))

    for i in tqdm(range(len(noise_levels))):
        inverse_problem.add_noise(noise_levels[i], seed)
        try:
            mce_solutions[:,:,i] = mce(leadfield, inverse_problem.measurements, lambdas[i])[1].value
            cmce_solutions[:,:,i] = constant_mce(leadfield, inverse_problem.measurements, lambdas[i])[1].value.reshape(-1,1) @ np.ones((1,inverse_problem.measurements.shape[1]))
        except:
            mce_solutions[:,:,i] = np.inf
            cmce_solutions[:,:,i] = np.inf
            print(f"Benchmark minimizer failed for noise level {noise_levels[i]}.")
        try:
            if problem_type == "short":
                time_indices = [0]
            elif problem_type == "long":
                time_indices = [5]
            mce_scores[i] = inverse_problem.evaluate_solution_emd(grid.dipoles[:,:3], mce_solutions[:,:,i], time_indices=time_indices, mute=True)[0]
            cmce_scores[i] = inverse_problem.evaluate_solution_emd(grid.dipoles[:,:3], cmce_solutions[:,:,i], time_indices=time_indices, mute=True)[0]
        except:
            mce_scores[i] = np.inf
            cmce_scores[i] = np.inf
            print(f"Benchmark evaluation failed for noise level {noise_levels[i]}.")
    with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/bench_{problem_type}_range_{str(n_sources)}_{seed}.pkl", "wb") as file:
        pickle.dump([mce_solutions.tolist(), mce_scores.tolist(), cmce_solutions.tolist(), cmce_scores.tolist(), mu_values, noise_level_subset], file)

# Compute norm_str solutions and emd scores
str_solutions = np.zeros((leadfield.shape[1], inverse_problem.measurements.shape[1], len(noise_levels), len(mu_values)))
str_scores = np.zeros((len(noise_levels), len(mu_values)))
for i in tqdm(range(len(noise_levels))):
    inverse_problem.add_noise(noise_levels[i], seed)
    for j in tqdm(range(len(mu_values))):
        try:
            str_solutions[:, :, i, j] = norm_str(leadfield, inverse_problem.measurements, lambdas[i], mu_values[j])[1].value
        except:
            str_solutions[:, :, i, j] = np.inf
            print(f"Minimizer failed for noise level {noise_levels[i]}, mu value {mu_values[j]}.")
        try:
            if problem_type == "short":
                time_indices = [0]
            elif problem_type == "long":
                time_indices = [5]
            str_scores[i, j] = inverse_problem.evaluate_solution_emd(grid.dipoles[:,:3], str_solutions[:,:,i,j], time_indices=time_indices, mute=True)[0]
        except:
            str_scores[i, j] = np.inf
            print(f"Evaluation failed for noise level {noise_levels[i]}, mu value {mu_values[j]}.")

# Save everything to file
with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/mu_{problem_type}_range_{str(n_sources)}_{seed}{version}.pkl", "wb") as file:
    pickle.dump([str_solutions.tolist(), str_scores.tolist(), mu_values, noise_level_subset], file)