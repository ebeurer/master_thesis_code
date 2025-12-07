# Several scripts have to be run before this one: dipoles.py, example_problem_setup.py, morozov_test.py, morozov_test_long.py
import os
import pickle
from inverse_problem import InverseProblem
from configuration import Configuration
from dipoles import Dipoles, get_divergence_matrix
import numpy as np
import time
from admm import unbalanced_admm


repeats = 3
mute = False
n_sources = 5000

noise_level = 0.14
mu = 0.2
lam_long = 0.1076
lam_short = 0.02213


config = Configuration("inverse_code/inversion_methods/small_example/configs.ini")
config.output_folder = "inverse_code/inversion_methods/small_example"
config.input_folder = "inverse_code/inversion_methods/small_example"

# Create source space for reconstruction
grid = Dipoles.from_txt(f"inverse_code/inversion_methods/text_files/half_grid_{str(n_sources)}.txt")

# Import leadfield matrices
leadfield = np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_eeg_leadfield_analytical.npy"))
leadfield = np.append(leadfield, np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_meg_leadfield_analytical.npy")), axis=0)

# Import inverse problem
with open("inverse_code/inversion_methods/small_example/long_range_problem.pkl", "rb") as file:
    inverse_long = pickle.load(file)

with open("inverse_code/inversion_methods/small_example/short_range_problem.pkl", "rb") as file:
    inverse_short = pickle.load(file)

connections = grid.short_connection_list(5, cutoff=10)
connections_long = connections.copy()

point1 = np.argmin(np.linalg.norm(grid.dipoles[:,:3]-inverse_long.dipoles.dipoles[0,:3],axis=1))
point2 = np.argmin(np.linalg.norm(grid.dipoles[:,:3]-inverse_long.dipoles.dipoles[1,:3],axis=1))
connections_long = np.append(connections,[[point1, point2, 5]],axis=0)

divergence_matrix = get_divergence_matrix(connections, leadfield.shape[1])
divergence_long = get_divergence_matrix(connections_long, leadfield.shape[1])

admm_times_short = np.zeros(repeats)
admm_times_long = np.zeros(repeats)

for i in range(repeats):
    start = time.time()
    _, _, _, _, _ = unbalanced_admm(leadfield, inverse_short.measurements, connections, lam_short, mu, rho=0.5, max_iterations=100, divergence_matrix=divergence_matrix, growth_penalty=50)
    admm_times_short[i] = time.time()-start
    if not mute:
        print(f"ADMM took {admm_times_short[i]:.1f} seconds for 100 iterations.")
    start = time.time()
    _, _, _, _, _ = unbalanced_admm(leadfield, inverse_long.measurements, connections_long, lam_long, mu, rho=0.5, max_iterations=100, divergence_matrix=divergence_long, growth_penalty=50)
    admm_times_long[i] = time.time()-start
    if not mute:
        print(f"ADMM took {admm_times_long[i]:.1f} seconds for 100 iterations.")

with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/time_admm_{str(n_sources)}_{repeats}.pkl", "wb") as file:
    pickle.dump([admm_times_short.tolist(), admm_times_long.tolist()], file)