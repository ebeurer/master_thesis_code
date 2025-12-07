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
from dipoles import Dipoles
from configuration import Configuration
from forwardmodel import ForwardModel
import numpy as np
import sys
from ot import dist, emd2
from scipy.optimize import root_scalar


class InverseProblem:
    """
    An object containing the true dipoles and the corresponding measurements over time. First all dipoles relevant to the problem are passed via dipole_setup, then timeline_setup specifies when each dipole is active, and with what magnitude.
    """
    def __init__(self):
        """
        Initialization of all attributes.
        """
        self.dipoles = None
        self.dipole_responses = None
        self.active_dipoles = None
        self.time_steps = None
        self.noiseless_measurements = None
        self.measurements = None
        self.noise_variance = None


    def dipole_setup(self, dipoles: Dipoles, config: Configuration, solve_analytical=True):
        """
        Adds all dipoles that are relevant to the problem and computes their leadfields. Currently only analytical solution is implemented.
        """
        self.dipoles = dipoles
        fw_model = ForwardModel(config)
        self.electrodes = fw_model.electrodes
        self.coils = fw_model.coils
        self.projections = fw_model.projections
        if solve_analytical:
            self.dipole_responses = fw_model.analytical_eeg_leadfield(self.dipoles)
            self.dipole_responses = np.append(self.dipole_responses, fw_model.analytical_meg_leadfield(dipoles), axis=0)
        else:
            raise Exception("Oops, numerical solution not yet implemented")
        return self
        

    def timeline_setup(self, active_dipoles: np.ndarray, time_steps: np.ndarray=None):
        """
        After dipole_setup has been run, specifies when each dipole is active, and with what magnitude.

        :param active_dipoles: Every row corresponds to a dipole and every column to a time step.
        """
        if self.dipole_responses is None:
            raise Exception("Dipole setup needs to be done first.")
        self.noiseless_measurements = self.dipole_responses @ active_dipoles
        self.measurements = self.noiseless_measurements.copy()
        self.active_dipoles = active_dipoles
        if time_steps is None:
            self.time_steps = np.arange(active_dipoles.shape[1])
        else:
            self.time_steps = np.array(time_steps)
        return self
    

    def add_noise(self, noise_level, seed=None):
        """
        Adds noise at the specified noise level. The noiseless measurements are kept, and repeated calling doesn't lead to more noise.
        """
        if self.measurements is None:
            raise Exception("No measurements computed yet.")
        rng = np.random.default_rng(seed)
        standard_deviation = noise_level * np.max(np.abs(self.noiseless_measurements))
        self.measurements = self.noiseless_measurements + rng.normal(scale=standard_deviation, size=self.measurements.shape)
        self.noise_variance = standard_deviation ** 2
        return self
    

    def sources_vtk_writer(self, duneuro_path, file_path):
        """
        Writes the sources to a .vtk file.
        """
        if self.active_dipoles is None:
            raise Exception("Need active dipoles to write timeline.")
        sys.path.append(duneuro_path)
        import duneuropy as dp
        for t in range(self.active_dipoles.shape[1]):
            dipole_array = self.dipoles.dipoles[self.active_dipoles[:,t].astype(bool),:]
            dipole_array[:,3:] *= self.active_dipoles[self.active_dipoles[:,t].astype(bool),t].reshape(-1,1)
            writer = dp.PointVTKWriter3d([dp.FieldVector3D(dipole_array[i, :3]) for i in range(dipole_array.shape[0])], True)
            writer.addVectorData("moment", [dp.FieldVector3D(dipole_array[i, 3:]) for i in range(dipole_array.shape[0])])
            writer.write(file_path+"_"+str(t))


    def eeg_measurements_vtk_writer(self, duneuro_path, file_path):
        """
        Writes the simulated EEG measurements to a .vtk file.
        """
        if self.measurements is None:
            raise Exception("Need the measurements to write them.")
        sys.path.append(duneuro_path)
        import duneuropy as dp
        for t in range(self.measurements.shape[1]):
            writer = dp.PointVTKWriter3d(self.electrodes, True)
            writer.addScalarData("potential", self.measurements[:len(self.electrodes),t])
            writer.write(file_path+"_"+str(t))
    

    def meg_measurements_vtk_writer(self, duneuro_path, file_path):
        """
        Writes the simulated MEG measurements to a .vtk file.
        """
        if self.measurements is None:
            raise Exception("Need the measurements to write them.")
        sys.path.append(duneuro_path)
        import duneuropy as dp
        for t in range(self.measurements.shape[1]):
            writer = dp.PointVTKWriter3d(self.coils, True)
            writer.addScalarData("mag_field", self.measurements[len(self.electrodes):,t])
            writer.write(file_path+"_"+str(t))
    

    def evaluate_solution_emd(self, grid_positions, predicted_coefficients, mute=False, time_indices=None):
        """
        Calculates EMD score between the true sources and some predicted solution at one or more time points.

        :param grid_positions: The grid on which the predicted solution has been calculated.
        :param predicted_coefficients: Each column contains the predicted coefficient for the corresponding grid positions at one time point.
        :param time_indices: Specify in case not every time point is of interest.
        """
        if time_indices is None:
            time_indices = range(self.time_steps.shape[0])

        emds = np.zeros(len(time_indices))
        for i in range(len(time_indices)):
            emds[i] = earth_movers_distance(grid_positions, np.abs(predicted_coefficients[:,time_indices[i]]), self.dipoles.dipoles[:,:3], np.abs(self.active_dipoles[:,time_indices[i]]))
        if not mute:
            print(f"Average EMD over time: {emds.mean()}\nMaximum EMD: {emds.max()}\nMinimum EMD: {emds.min()}")
        return emds
    

    def evaluate_solution_dist(self, grid_positions, predicted_coefficients, mute=False):
        """
        Calculates dipole localization error (DLE) between the true sources and some predicted solution at all time points. In case of several sources being active at the same time this may not always make sense.

        :param grid_positions: The grid on which the predicted solution has been calculated.
        :param predicted_coefficients: Each column contains the predicted coefficient for the corresponding grid positions at one time point.
        """
        distances = np.zeros(self.time_steps.shape[0])
        for i in range(distances.shape[0]):
            distances[i] = np.linalg.norm(grid_positions[np.argmax(predicted_coefficients[:,i]),:3]-self.dipoles.dipoles[np.argmax(np.abs(self.active_dipoles[:,i])),:3])
        if not mute:
            print(f"Average distance over time: {distances.mean()}\nMaximum distance: {distances.max()}\nMinimum distance: {distances.min()}")
        return distances
    

    def evaluate_solution_residual(self, leadfield, predicted_coefficients):
        """
        Calculates how well the measurements are explained by a predicted solution at every time point.

        :param grid_positions: The grid on which the predicted solution has been calculated.
        :param predicted_coefficients: Each column contains the predicted coefficient for the corresponding grid positions at one time point.
        """
        residuals = np.linalg.norm(self.measurements - leadfield @ predicted_coefficients, axis=0)**2
        print(f"Average residual over time: {residuals.mean()}\nMaximum residual: {residuals.max()}\nMinimum residual: {residuals.min()}")
        return residuals
    

    def best_possible_distances(self, grid_positions):
        """
        Calculates the best DLE that is possible for the chosen grid at every time point.

        :param grid_positions: The grid on which the predicted solution has been calculated.
        """
        distances = np.zeros(self.time_steps.shape[0])
        for i in range(distances.shape[0]):
            distances[i] = np.min(np.linalg.norm(grid_positions-self.dipoles.dipoles[np.argmax(np.abs(self.active_dipoles[:,i])),:3], axis=1))
        print(f"Best possible average distance: {distances.mean()}\nBest possible maximum distance: {distances.max()}\nBest possible minimum distance: {distances.min()}")


def earth_movers_distance(positions_a, coefficients_a, positions_b, coefficients_b):
    """
    Calculates the EMD score for two discrete measures on the R^3. Uses the Python Optimal Transport toolbox.

    :param positions_a: The support of the first measure. Each row represents one point.
    :param coefficients_a: The coefficients of the first measure.
    :param positions_b: The support of the second measure. Each row represents one point.
    :param coefficients_b: The coefficients of the second measure.
    """
    M = dist(positions_a, positions_b, metric='euclidean')
    return emd2(coefficients_a/np.sum(coefficients_a), coefficients_b/np.sum(coefficients_b), M)


def mce_discrepancy_lambda(leadfield, measurement, noise_variance):
    """
    Computes the regularization parameter lambda for the MCE via Morozov's discrepancy principle. Beware that unlike for MNE, the problem isn't always well posed, and thus the routine might fail.

    :param leadfield: The leadfield matrix for the problem.
    :param measurement: The measurement at one time point.
    :param noise_variance: The variance of the Gaussian noise which has been added to the measurements
    :return lam: The regularization parameter lambda, as a RootResults object. A number can be extracted via lam.root.
    """
    func = lambda delt: np.linalg.norm(measurement - leadfield @ mce_opt_solver(leadfield, measurement, delt)[1].value)**2 - leadfield.shape[0] * noise_variance
    if np.linalg.norm(measurement)**2 < leadfield.shape[0] * noise_variance:
        print("Measurements are dominated by noise")
    lam = root_scalar(func, x0=1e-2, x1=1, bracket=(1e-8, 1e1), rtol=1e-3)
    return lam
