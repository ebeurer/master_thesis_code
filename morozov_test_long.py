# Several scripts have to be run before this one: dipoles.py, example_problem_setup.py
import os
import pickle
from tqdm import tqdm
from inverse_problem import InverseProblem, mce_discrepancy_lambda
from configuration import Configuration
from dipoles import Dipoles
import numpy as np


n_sources = 5000
problem_name = "long_range_problem.pkl"
noise_levels = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
seed = 1


config = Configuration("inverse_code/inversion_methods/small_example/configs.ini")
config.output_folder = "inverse_code/inversion_methods/small_example"
config.input_folder = "inverse_code/inversion_methods/small_example"

# Create source space for reconstruction
grid = Dipoles.from_txt(f"inverse_code/inversion_methods/text_files/half_grid_{str(n_sources)}.txt")

# Import leadfield matrices
leadfield = np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_eeg_leadfield_analytical.npy"))
leadfield = np.append(leadfield, np.load(os.path.join(config.output_folder, "leadfieldmatrices", f"half_angle_cort_{str(n_sources)}_meg_leadfield_analytical.npy")), axis=0)

# Import inverse problem
with open("inverse_code/inversion_methods/small_example/"+problem_name, "rb") as file:
    inverse_problem = pickle.load(file)

lambdas = np.zeros(len(noise_levels))
for i in tqdm(range(len(noise_levels))):
    inverse_problem.add_noise(noise_levels[i], seed)
    try:
        lambdas[i] = mce_discrepancy_lambda(leadfield, inverse_problem.measurements[:,-1], inverse_problem.noise_variance).root
    except:
        lambdas[i] = np.inf

print(f"{np.sum(np.isinf(lambdas))} errors...")
with open(f"inverse_code/inversion_methods/small_example/lambda_experiments/long_range_{str(n_sources)}_{seed}.pkl", "wb") as file:
    pickle.dump([lambdas.tolist(), noise_levels], file)
