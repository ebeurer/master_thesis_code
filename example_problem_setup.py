import pickle
from dipoles import Dipoles
from inverse_problem import InverseProblem
from configuration import Configuration
import numpy as np


config = Configuration("inverse_code/inversion_methods/small_example/configs.ini")
config.output_folder = "inverse_code/inversion_methods/small_example"
config.input_folder = "inverse_code/inversion_methods/small_example"

## Short range problem
n_timesteps = 5
source_dipoles = Dipoles.moving_along_cortex(config,5,np.pi/28,n_timesteps)

# Create inverse problem file for later import
inv = InverseProblem()
inv.dipole_setup(source_dipoles, config).timeline_setup(np.eye(n_timesteps))
with open("inverse_code/inversion_methods/small_example/short_range_problem.pkl", "wb") as file:
    # Duneuro objects are removed because otherwise problem can't be pickled
    inv.electrodes = None
    inv.coils = None
    inv.projections = None
    pickle.dump(inv, file)

## Long range problem
n_timesteps = 16

# Create dipole locations and directions
offset = 1
amplitude = 5
frequency = 28
theta1 = np.pi/4
theta2 = np.pi/4 + np.pi/frequency
phi1 = np.pi/2
phi2 = 3*np.pi/2
dipoles = np.zeros((2,6))
r1 = config.radii[-1] - (amplitude+offset) - 2.5 + 5*np.cos(frequency*theta1)
r2 = config.radii[-1] - (amplitude+offset) - 2.5 + 5*np.cos(frequency*theta2)
dipoles[0, :3] = [r1*np.sin(theta1)*np.cos(phi1), r1*np.sin(theta1)*np.sin(phi1), r1*np.cos(theta1)]
dipoles[1, :3] = [r2*np.sin(theta2)*np.cos(phi2), r2*np.sin(theta2)*np.sin(phi2), r2*np.cos(theta2)]
dipoles[0, 3:] = -np.cross([(-frequency*amplitude*np.sin(frequency*theta1)*np.sin(theta1)*np.cos(phi1)+r1*np.cos(theta1)*np.cos(phi1)), (-frequency*amplitude*np.sin(frequency*theta1)*np.sin(theta1)*np.sin(phi1)+r1*np.cos(theta1)*np.sin(phi1)), (-frequency*amplitude*np.sin(frequency*theta1)*np.cos(theta1)-r1*np.sin(theta1))],[(-r1*np.sin(theta1)*np.sin(phi1)), (r1*np.sin(theta1)*np.cos(phi1)), 0])
dipoles[1, 3:] = -np.cross([(-frequency*amplitude*np.sin(frequency*theta2)*np.sin(theta2)*np.cos(phi2)+r2*np.cos(theta2)*np.cos(phi2)), (-frequency*amplitude*np.sin(frequency*theta2)*np.sin(theta2)*np.sin(phi2)+r2*np.cos(theta2)*np.sin(phi2)), (-frequency*amplitude*np.sin(frequency*theta2)*np.cos(theta2)-r2*np.sin(theta2))],[(-r2*np.sin(theta2)*np.sin(phi2)), (r2*np.sin(theta2)*np.cos(phi2)), 0])
dipoles[:,3:] /= np.linalg.norm(dipoles[:,3:], axis=1).reshape(-1,1)
dipoles[:,:3] += config.sphere_center

sources = Dipoles(dipoles)

# Create amplitudes
start1 = 0
start2 = 5
end1 = 13
end2 = 16
amplitude1 = 0.8
amplitude2 = 1
curve1 = lambda t: amplitude1*np.sin(np.pi*(t-start1+1)/(end1-start1+1))
curve2 = lambda t: amplitude2*np.sin(np.pi*(t-start2)/(end2-start2)/2)

timesteps = range(n_timesteps)
active_dipoles = np.zeros((2,n_timesteps))
active_dipoles[0,start1:end1] = curve1(np.array(range(start1,end1)))
active_dipoles[1,start2:end2] = curve2(np.array(range(start2,end2)))

# Create inverse problem file for later import
inverse_problem = InverseProblem().dipole_setup(sources, config).timeline_setup(active_dipoles)
with open("inverse_code/inversion_methods/small_example/long_range_problem.pkl", "wb") as file:
        # Duneuro objects are removed because otherwise problem can't be pickled
        inverse_problem.electrodes = None
        inverse_problem.coils = None
        inverse_problem.projections = None
        pickle.dump(inverse_problem, file)
