# Heavily modified from https://zenodo.org/records/12575669
from configuration import Configuration
import sys
import os
import numpy as np
import time
import json
from dipoles import Dipoles


class ForwardModel:
    """
    Implements the DUNEuro python interface with a config file in the style of the local subtraction approach paper.
    """
    def __init__(self, config: Configuration):
        """
        :param config:
            A configuration object with all the relevant parameters.
        """
        self.config = config
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp
        if self.config.numerical_simulation_needed:
            self.driver = dp.MEEGDriver3d(self.config.driver_config)
        if self.config.do_eeg:
            self._set_electrodes()
        if self.config.do_meg:
            self._set_coils_and_projections()

    def eeg_transfer_uptodate(self):
        """
        Returns True if the cached eeg transfer matrix is up-to-date or False if it needs to be recomputed.
        """
        eeg_dependencies_change_time = max(os.path.getctime(os.path.join(self.config.input_folder, 'sensors', 'electrodes.txt')), \
            os.path.getctime(os.path.join(self.config.input_folder, 'volume_conductor', 'conductivities.txt')), \
            os.path.getctime(os.path.join(self.config.input_folder, 'volume_conductor', 'mesh.msh')))
    
        update_needed = (not os.path.exists(os.path.join(self.config.output_folder, 'transfermatrices', 'eeg_transfer_matrix.npy'))) or \
            (os.path.getctime(os.path.join(self.config.output_folder, 'transfermatrices', 'eeg_transfer_matrix.npy')) < eeg_dependencies_change_time)
        
        if not update_needed:
            self.eeg_transfer_matrix = np.load(os.path.join(self.config.output_folder, 'transfermatrices', 'eeg_transfer_matrix.npy'))

        return not update_needed
    

    def meg_transfer_uptodate(self):
        """
        Returns True if the cached meg transfer matrix is up-to-date or False if it needs to be recomputed.
        """
        meg_dependencies_change_time = max(os.path.getctime(os.path.join(self.config.input_folder, 'sensors', 'coils.txt')), \
            os.path.getctime(os.path.join(self.config.input_folder, 'sensors', 'projections.txt')), \
            os.path.getctime(os.path.join(self.config.input_folder, 'volume_conductor', 'conductivities.txt')), \
            os.path.getctime(os.path.join(self.config.input_folder, 'volume_conductor', 'mesh.msh')))
    
        update_needed = (not os.path.exists(os.path.join(self.config.output_folder, 'transfermatrices', 'meg_transfer_matrix.npy'))) or \
            (os.path.getctime(os.path.join(self.config.output_folder, 'transfermatrices', 'meg_transfer_matrix.npy')) < meg_dependencies_change_time)
        
        if not update_needed:
            self.meg_transfer_matrix = np.load(os.path.join(self.config.output_folder, 'transfermatrices', 'meg_transfer_matrix.npy'))
        
        return not update_needed
    

    def _set_electrodes(self):
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp
        print(f"Setting electrodes")
        self.electrodes = dp.read_field_vectors_3d(os.path.join(self.config.input_folder, 'sensors', 'electrodes.txt'))
        if self.config.numerical_simulation_needed:
            self.driver.setElectrodes(self.electrodes, self.config.electrode_config)
        print(f"Electrodes set")

    
    def _set_coils_and_projections(self):
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp
        print(f"Setting coils and projections")
        self.coils = dp.read_field_vectors_3d(os.path.join(self.config.input_folder, 'sensors', 'coils.txt'))
        self.projections = dp.read_projections_3d(os.path.join(self.config.input_folder, 'sensors', 'projections.txt'))
        if self.config.numerical_simulation_needed:
            self.driver.setCoilsAndProjections(self.coils, self.projections)
        print(f"Coils and projections set")


    def compute_eeg_transfer(self):
        """
        Computes the eeg transfer matrix for the specified configuration and saves it under transfermatrices in the specified output folder.
        """
        if not self.config.numerical_simulation_needed:
            raise Exception("Forward model not initialized for numerical simulation.")
        print(f"Computing EEG transfer matrix")
        # compute transfer matrix
        start_time = time.time()
        transfer_matrix_raw, computation_information = self.driver.computeEEGTransferMatrix(self.config.transfer_configs)
        self.eeg_transfer_matrix = np.array(transfer_matrix_raw)
        print(f"Computing the EEG transfer matrix took {time.time() - start_time} seconds")
        print(f"EEG transfer matrix computed")
        print(f"Saving EEG transfer matrix")
        np.save(os.path.join(self.config.output_folder, 'transfermatrices', 'eeg_transfer_matrix.npy'), self.eeg_transfer_matrix)
        with open(os.path.join(self.config.output_folder, 'transfermatrices', 'eeg_transfer_matrix_computation_information.json'), 'w') as outfile:
            outfile.write(json.dumps(computation_information))
        print(f"Transfer matrix saved")
        return self.eeg_transfer_matrix


    def compute_meg_transfer(self):
        """
        Computes the meg transfer matrix for the specified configuration and saves it under transfermatrices in the specified output folder.
        """
        if not self.config.numerical_simulation_needed:
            raise Exception("Forward model not initialized for numerical simulation.")
        print(f"Computing MEG transfer matrix")
        start_time = time.time()
        transfer_matrix_raw, computation_information = self.driver.computeMEGTransferMatrix(self.config.transfer_configs)
        self.meg_transfer_matrix = np.array(transfer_matrix_raw)
        print(f"Computing the MEG transfer matrix took {time.time() - start_time} seconds")
        print(f"Transfer matrix computed")
        print(f"Saving transfer matrix")
        np.save(os.path.join(self.config.output_folder, 'transfermatrices', 'meg_transfer_matrix.npy'), self.meg_transfer_matrix)
        with open(os.path.join(self.config.output_folder, 'transfermatrices', 'meg_transfer_matrix_computation_information.json'), 'w') as outfile:
            outfile.write(json.dumps(computation_information))
        print(f"Transfer matrix saved")
        return self.meg_transfer_matrix
    

    def analytical_eeg_solution(self):
        """
        Computes analytical eeg solutions for all dipole sources inside the input folder specified in config.
        """
        if not hasattr(self, "electrodes"):
            raise Exception("Electrodes haven't been set.")
        sys.path.append(self.config.simbiosphere)
        import simbiopy as sp
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp

        print("Computing analytical EEG solutions")
        electrodes_simbio = [np.array(electrode).tolist() for electrode in self.electrodes]
        #radial dipoles
        print(f"Computing solutions for radial dipoles")
        radial_dipole_filenames = os.listdir(os.path.join(self.config.input_folder, 'dipoles', 'radial'))
        for filename in radial_dipole_filenames:
            dipoles = dp.read_dipoles_3d(os.path.join(self.config.input_folder, 'dipoles', 'radial', filename))
            with open(os.path.join(self.config.output_folder, 'results', 'eeg', 'radial', 'analytical_solution', f'analytical_solution_{filename}'), mode='w') as outputfile:
                for count, dipole in enumerate(dipoles):
                    analytical_solution = sp.analytic_solution(self.config.radii, self.config.sphere_center, self.config.conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
                    mean = sum(analytical_solution) / float(len(analytical_solution))
                    analytical_solution = [x - mean for x in analytical_solution]
                    outputfile.write(" ".join([str(x) for x in analytical_solution]))
                    if count < len(dipoles)  - 1:
                        outputfile.write("\n")
        # tangential dipoles
        print(f"Computing solutions for tangential dipoles")
        tangential_dipole_filenames = os.listdir(os.path.join(self.config.input_folder, 'dipoles', 'tangential'))
        for filename in tangential_dipole_filenames:
            dipoles = dp.read_dipoles_3d(os.path.join(config.input_folder, 'dipoles', 'tangential', filename))
            with open(os.path.join(self.config.output_folder, 'results', 'eeg', 'tangential', 'analytical_solution', f'analytical_solution_{filename}'), mode='w') as outputfile:
                for count, dipole in enumerate(dipoles):
                    analytical_solution = sp.analytic_solution(self.config.radii, self.config.sphere_center, self.config.conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
                    mean = sum(analytical_solution) / float(len(analytical_solution))
                    analytical_solution = [x - mean for x in analytical_solution]
                    outputfile.write(" ".join([str(x) for x in analytical_solution]))
                    if count < len(dipoles)  - 1:
                        outputfile.write("\n")
        print(f"Analytical solutions computed")


    def analytical_eeg_leadfield(self, dipoles: Dipoles):
        """
        Computes analytical eeg lead field matrix for the given dipole sources.
        """
        if not hasattr(self, "electrodes"):
            raise Exception("Electrodes haven't been set.")
        sys.path.append(self.config.simbiosphere)
        import simbiopy as sp
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp

        print("Computing analytical EEG solutions")
        electrodes_simbio = [np.array(electrode).tolist() for electrode in self.electrodes]
        analytical_leadfield = np.zeros((len(electrodes_simbio),dipoles.dipoles.shape[0]))
        for count, dipole in enumerate(dipoles.get_dune_dipoles(self.config.duneuro_path)):
            analytical_solution = sp.analytic_solution(self.config.radii, self.config.sphere_center, self.config.conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
            mean = sum(analytical_solution) / float(len(analytical_solution))
            analytical_leadfield[:,count] = [x - mean for x in analytical_solution]
        self.eeg_leadfield_analytical = analytical_leadfield
        print(f"Analytical solutions computed")
        return self.eeg_leadfield_analytical


    def analytical_meg_solution(self):
        """
        Computes analytical meg solutions for all tangential dipole sources inside the input folder specified in config.
        """
        if not hasattr(self, "coils") or not hasattr(self, "projections"):
            raise Exception("Coils and/or projections haven't been set.")
        sys.path.append(self.config.duneuro_analytic_solution)
        import duneuroAnalyticSolutionPy as dp_analytic
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp

        print("Computing analytical MEG solutions")
        MEGSolver = dp_analytic.AnalyticSolutionMEG(dp.FieldVector3D(self.config.sphere_center))
        tangential_dipole_filenames = os.listdir(os.path.join(self.config.input_folder, 'dipoles', 'tangential'))
        for filename in tangential_dipole_filenames:
            dipoles = dp.read_dipoles_3d(os.path.join(self.config.input_folder, 'dipoles', 'tangential', filename))
            with open(os.path.join(self.config.output_folder, 'results', 'meg', 'analytical_solution', f'analytical_solution_{filename}'), mode='w') as outputfile:
                for count, dipole in enumerate(dipoles):
                    MEGSolver.bind(dipole)
                    analytic_vals = []
                    for coil_index, coil in enumerate(self.coils):
                        for projection in self.projections[coil_index]:
                            analytic_vals.append(str(MEGSolver.totalField(coil, projection)))
                    outputfile.write(' '.join(analytic_vals))
                    if count < len(dipoles) -1:
                        outputfile.write("\n")
        print(f"Analytical solutions computed")


    def analytical_meg_leadfield(self, dipoles: Dipoles):
        """
        Computes analytical meg lead field matrix for the given dipole sources.
        """
        if not hasattr(self, "coils") or not hasattr(self, "projections"):
            raise Exception("Coils and/or projections haven't been set.")
        sys.path.append(self.config.duneuro_analytic_solution)
        import duneuroAnalyticSolutionPy as dp_analytic
        sys.path.append(self.config.duneuro_path)
        import duneuropy as dp

        print("Computing analytical MEG solutions")
        MEGSolver = dp_analytic.AnalyticSolutionMEG(dp.FieldVector3D(self.config.sphere_center))
        analytical_leadfield = np.zeros((len(self.coils),dipoles.dipoles.shape[0]))
        for count, dipole in enumerate(dipoles.get_dune_dipoles(self.config.duneuro_path)):
            MEGSolver.bind(dipole)
            analytic_vals = []
            for coil_index, coil in enumerate(self.coils):
                for projection in self.projections[coil_index]:
                    analytic_vals.append(str(MEGSolver.totalField(coil, projection)))
            analytical_leadfield[:, count] = analytic_vals
        self.meg_leadfield_analytical = analytical_leadfield
        print(f"Analytical solutions computed")
        return self.meg_leadfield_analytical


    def eeg_leadfield_matrix(self, dipoles):
        """
        Computes eeg leadfield matrix for the dipoles given as list of DUNEuro dipole objects.
        """
        if not self.config.numerical_simulation_needed:
            raise Exception("Forward model not initialized for numerical simulation.")
        if not hasattr(self, "eeg_transfer_matrix"):
            raise Exception("Transfer matrix needs to be computed before the leadfield matrix.")
        leadfield, _ = self.driver.applyEEGTransfer(self.eeg_transfer_matrix, dipoles, self.config.driver_config)
        self.eeg_leadfield = np.array(leadfield).T
        return self.eeg_leadfield
    

    def meg_leadfield_matrix(self, dipoles):
        """
        Computes meg leadfield matrix for the dipoles given as list of DUNEuro dipole objects.
        """
        if not self.config.numerical_simulation_needed:
            raise Exception("Forward model not initialized for numerical simulation.")
        if not hasattr(self, "meg_transfer_matrix"):
            raise Exception("Transfer matrix needs to be computed before the leadfield matrix.")
        leadfield, _ = self.driver.applyMEGTransfer(self.meg_transfer_matrix, dipoles, self.config.driver_config)
        self.meg_leadfield = np.array(leadfield).T
        return self.meg_leadfield
    

    def save_leadfields(self, prefix=""):
        """
        Saves the computed leadfields for loading at a later time.
        """
        if hasattr(self, "eeg_leadfield_analytical"):
            np.save(os.path.join(self.config.output_folder, "leadfieldmatrices", prefix+"eeg_leadfield_analytical.npy"), self.eeg_leadfield_analytical)
        if hasattr(self, "meg_leadfield_analytical"):
            np.save(os.path.join(self.config.output_folder, "leadfieldmatrices", prefix+"meg_leadfield_analytical.npy"), self.meg_leadfield_analytical)
        if hasattr(self, "eeg_leadfield"):
            np.save(os.path.join(self.config.output_folder, "leadfieldmatrices", prefix+"eeg_leadfield.npy"), self.eeg_leadfield_analytical)
        if hasattr(self, "meg_leadfield"):
            np.save(os.path.join(self.config.output_folder, "leadfieldmatrices", prefix+"meg_leadfield.npy"), self.meg_leadfield_analytical)


if __name__ == "__main__":
    from configuration import create_output_folder
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


    config = Configuration("/home/emil/Uni/Masterthesis/local_subtraction_paper_code/multilayer_sphere_validation_study/configs.ini")
    config.output_folder = "/home/emil/Uni/Masterthesis/inverse_code"

    sys.path.append(config.duneuro_path)
    import duneuropy as dp

    create_output_folder(config.output_folder)

    fw_model = ForwardModel(config)

    print(fw_model.eeg_transfer_uptodate())

    if config.do_eeg and not fw_model.eeg_transfer_uptodate():
        fw_model.compute_eeg_transfer()

    if config.do_meg and not fw_model.meg_transfer_uptodate():
        fw_model.compute_meg_transfer()

    # compute numerical eeg solutions
    if config.do_eeg and config.numerical_simulation_needed:
        print("Computing numerical EEG solutions")
        transfer_matrix = np.load(os.path.join(config.output_folder, 'transfermatrices', 'eeg_transfer_matrix.npy'))
        # radial dipoles
        print("Simulating radial dipoles")
        radial_dipole_filenames = os.listdir(os.path.join(config.input_folder, 'dipoles', 'radial'))
        for filename in radial_dipole_filenames:
                dipoles = dp.read_dipoles_3d(os.path.join(config.input_folder, 'dipoles', 'radial', filename))
                for source_model in config.source_models_for_numerical_simulation:
                    config.driver_config['source_model'] = config.source_model_config_database[source_model]
                    start_time = time.time()
                    numerical_solution, computation_information = fw_model.driver.applyEEGTransfer(transfer_matrix, dipoles, config.driver_config)
                    print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
                    np.savetxt(os.path.join(config.output_folder, 'results', 'eeg', 'radial', 'numerical_solution', f'{source_model}_numerical_solution_{filename}'), np.array(numerical_solution), fmt='%.18f', delimiter = ' ')
        # tangential dipoles
        print("Simulating tangential dipoles")
        tangential_dipole_filenames = os.listdir(os.path.join(config.input_folder, 'dipoles', 'tangential'))
        for filename in tangential_dipole_filenames:
                dipoles = dp.read_dipoles_3d(os.path.join(config.input_folder, 'dipoles', 'tangential', filename))
                for source_model in config.source_models_for_numerical_simulation:
                    config.driver_config['source_model'] = config.source_model_config_database[source_model]
                    start_time = time.time()
                    numerical_solution, computation_information = fw_model.driver.applyEEGTransfer(transfer_matrix, dipoles, config.driver_config)
                    print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
                    np.savetxt(os.path.join(config.output_folder, 'results', 'eeg', 'tangential', 'numerical_solution', f'{source_model}_numerical_solution_{filename}'), np.array(numerical_solution), fmt='%.18f', delimiter = ' ')
        print("Numerical EEG solutions computed")
    
    # compute numerical meg solutions
    if config.do_meg and config.numerical_simulation_needed:
        print("Computing numerical MEG solutions")
        transfer_matrix = np.load(os.path.join(config.output_folder, 'transfermatrices', 'meg_transfer_matrix.npy'))
        # tangential dipoles
        print("Simulating tangential dipoles")
        tangential_dipole_filenames = os.listdir(os.path.join(config.input_folder, 'dipoles', 'tangential'))
        for filename in tangential_dipole_filenames:
                dipoles = dp.read_dipoles_3d(os.path.join(config.input_folder, 'dipoles', 'tangential', filename))
                for source_model in config.source_models_for_numerical_simulation:
                    config.driver_config['source_model'] = config.source_model_config_database[source_model]
                    config.driver_config['post_process'] = False
                    start_time = time.time()
                    numerical_solution, computation_information = fw_model.driver.applyMEGTransfer(transfer_matrix, dipoles, config.driver_config)
                    print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
                    np.savetxt(os.path.join(config.output_folder, 'results', 'meg', 'numerical_solution', f'{source_model}_numerical_solution_{filename}'), np.array(numerical_solution), fmt='%.18f', delimiter = ' ')
        print("Numerical MEG solutions computed")

    fw_model.analytical_eeg_solution()
    fw_model.analytical_meg_solution()


    #########################################
    # compare numerical and analytical solutions
    #########################################

    # first define error measures

    # relative error
    # params  : 
    #     - numerical_solution  : 1-dimensional numpy array
    #     - analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
    def relative_error(numerical_solution, analytical_solution):
        assert len(numerical_solution) == len(analytical_solution)
        return np.linalg.norm(numerical_solution - analytical_solution) / np.linalg.norm(analytical_solution)

    # rdm error
    # params  :
    #     - numerical_solution  : 1-dimensional numpy array
    #     - analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
    def rdm(numerical_solution, analytical_solution):
        assert len(numerical_solution) == len(analytical_solution)
        return np.linalg.norm(numerical_solution/np.linalg.norm(numerical_solution) - analytical_solution/np.linalg.norm(analytical_solution))

    # lnMAG
    # params  :
    #     - numerical_solution  : 1-dimensional numpy array
    #     - analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
    def lnMAG(numerical_solution, analytical_solution):
        assert len(numerical_solution) == len(analytical_solution)
        return np.log(np.linalg.norm(numerical_solution)/np.linalg.norm(analytical_solution))

    # compare analytical solutions and numerical solutions inside folder and write the results to files
    # params  :
    #   - source_model          : source model for which we want to compute comparisons
    #   - basefolder            : basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
    #   - orientation_tag       : 'radial' or 'tangential'
    def create_comparisons(source_model, basefolder, orientation_tag):
        basename_files = os.listdir(os.path.join(config.input_folder, 'dipoles', orientation_tag))
        local_dataframes = [None] * len(basename_files)
        current_offset = 0
        for count, basename in enumerate(basename_files):
                analytical_solutions = np.loadtxt(os.path.join(basefolder, 'analytical_solution', f'analytical_solution_{basename}'))
                numerical_solutions = np.loadtxt(os.path.join(basefolder, 'numerical_solution', f'{source_model}_numerical_solution_{basename}'))
                assert analytical_solutions.shape == numerical_solutions.shape
                relative_errors = [relative_error(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
                rdms = [rdm(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
                lnMAGs = [lnMAG(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
                
                # write error measures to file
                with open(os.path.join(basefolder, 'relative_error', f'{source_model}_relative_error_{basename}'), mode='w') as outputfile:
                    outputfile.write('\n'.join([str(val) for val in relative_errors]))
                with open(os.path.join(basefolder, 'rdm', f'{source_model}_rdm_{basename}'), mode='w') as outputfile:
                    outputfile.write('\n'.join([str(val) for val in rdms]))
                with open(os.path.join(basefolder, 'lnMAG', f'{source_model}_lnMAG_{basename}'), mode='w') as outputfile:
                    outputfile.write('\n'.join([str(val) for val in lnMAGs]))
                
                #extract eccentricity from file name. For this we assume the filename to contain the substring "ecc_0.x_", where x is some number, e.g. 1 or 95
                base_index = basename.find('ecc_0.')
                base_offset = len('ecc_')
                end_offset = basename[base_index+base_offset:].find('_')
                eccentricity = basename[base_index+base_offset:base_index+base_offset+end_offset]
                
                local_dataframes[count] = pd.DataFrame({'source_model' : source_model, 'ecc' : eccentricity, 'relative_error' : relative_errors, 'rdm' : rdms, 'lnMAGS' : lnMAGs}, index=range(current_offset, current_offset + len(analytical_solutions)))
                current_offset += len(analytical_solutions)
            
        total_dataframe = pd.concat(local_dataframes)
        total_dataframe.to_csv(os.path.join(basefolder, 'dataframes', f'{source_model}_dataframe.csv'))
        
        return

    # we now iterate over all source models and compare their respective numerical solutions with the analytical solution
    if config.comparisons_needed:
        print("Comparing numerical and analytical solutions")
        for source_model in config.source_models_for_comparison:
                print(f"Performing comparisons for {source_model} source model")
                # eeg
                if config.do_eeg:
                    print("Comparing EEG solutions")
                    
                    # radial
                    print("Comparing radial solutions")
                    create_comparisons(source_model, os.path.join(config.output_folder, 'results', 'eeg', 'radial'), 'radial')
                    
                    # tangential
                    print("Comparing tangential solutions")
                    create_comparisons(source_model, os.path.join(config.output_folder, 'results', 'eeg', 'tangential'), 'tangential')
                
                # meg
                if config.do_meg:
                    print("Comparing MEG solutions")
                    
                    #tangential
                    print("Comparing tangential solutions")
                    create_comparisons(source_model, os.path.join(config.output_folder, 'results', 'meg'), 'tangential')
                
                print(f"Comparisons for {source_model} source model finished")
        print("All comparisons finished")
    else:
        print("No comparisons need to be computed")

    #########################################
    # create boxplots
    #########################################

    # create pandas dataframe for later creation of boxplots via seaborn. For the structure of the constructed dataframe look at the parameter description of the function "create_boxplots"
    # params  :
    #   - eccentricity_selection      : list of strings, where each string represents an eccentricity to include in the boxplot, e.g. ['0.7', '0.8', '0.9']
    #   - source_model_selection      : list of strings, where each string represents a source model to include in the boxplot, e.g. ['venant', 'local_subtraction']
    #   - basefolder                  : basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
    #   - orientation_tag             : 'radial' or 'tangential'
    def create_dataframe(eccentricity_selection, source_model_selection, basefolder, orientation_tag):
        number_of_eccentricities = len(eccentricity_selection)
        
        # we first create a dataframe for every source model. These will be concatenated later on
        number_of_source_models = len(config.source_models_for_boxplot)
        source_model_data_frames = [None] * number_of_source_models
        
        for i, source_model in enumerate(config.source_models_for_boxplot):
                local_data_frames = [None] * number_of_eccentricities
                for j, eccentricity in enumerate(eccentricity_selection):
                    relative_errors = np.loadtxt(os.path.join(basefolder, 'relative_error', f'{source_model}_relative_error_ecc_{eccentricity}_{orientation_tag}.txt'))
                    rdms = np.loadtxt(os.path.join(basefolder, 'rdm', f'{source_model}_rdm_ecc_{eccentricity}_{orientation_tag}.txt'))
                    lnMAGs = np.loadtxt(os.path.join(basefolder, 'lnMAG', f'{source_model}_lnMAG_ecc_{eccentricity}_{orientation_tag}.txt'))
                    local_data_frames[j] = pd.DataFrame({'source_model' : source_model, 'ecc' : eccentricity, 'relative_error' : relative_errors, 'rdm' : rdms, 'lnMAG' : lnMAGs})
                source_model_data_frames[i] = pd.concat(local_data_frames)
            
        total_dataframe = pd.concat(source_model_data_frames)
        
        return total_dataframe


    # create boxplots for relative_error, rdm and lnMAG for different source models
    # params :
    #     - dataframe       : pandas dataframe containing the information to be plotted inside the boxplot. The structure of this dataframe is supposed to be
    #                         
    #                           source_model          ecc         relative_error          rdm         lnMAG
    #
    #                       0   partial_integration   0.1         0.04                    0.03        -0.001
    #                                                                 .
    #                                                                 .
    #                                                                 .
    #     - basefolder      : basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
    #     - modality_tag    : 'eeg' or 'meg'
    #     - orientation_tag : 'radial' or 'tangential'
    def create_boxplots(dataframe, basefolder, modality_tag, orientation_tag):
        # relative error
        fig_relative_error, ax_relative_error = plt.subplots()
        ax_relative_error.set_ylim(0, 0.1)
        sns.boxplot(x='ecc', y='relative_error', hue='source_model', data=dataframe, ax=ax_relative_error)
        plt.savefig(os.path.join(basefolder, 'boxplots', f'relative_error_{modality_tag}_{orientation_tag}_{"_".join(config.source_models_for_boxplot)}_{"_".join(eccentricity_selection)}.png'))
        plt.clf()
        
        # rdm
        fig_rdm, ax_rdm = plt.subplots()
        ax_rdm.set_ylim(0, 0.1)
        sns.boxplot(x='ecc', y='rdm', hue='source_model', data=dataframe, ax=ax_rdm)
        plt.savefig(os.path.join(basefolder, 'boxplots', f'rdm_{modality_tag}_{orientation_tag}_{"_".join(config.source_models_for_boxplot)}_{"_".join(eccentricity_selection)}.png'))
        plt.clf()
        
        # lnMAG
        fig_lnMAG, ax_lnMAG = plt.subplots()
        ax_lnMAG.set_ylim(-0.1, 0.1)
        sns.boxplot(x='ecc', y='lnMAG', hue='source_model', data=dataframe, ax=ax_lnMAG)
        plt.savefig(os.path.join(basefolder, 'boxplots', f'lnMAG_{modality_tag}_{orientation_tag}_{"_".join(config.source_models_for_boxplot)}_{"_".join(eccentricity_selection)}.png'))
        plt.clf()
        
        return

    if config.boxplots_needed:
        print("Creating boxplots")
        # eeg
        if config.do_eeg:
                for eccentricity_selection in config.eccentricity_sets:
                    # radial
                    dataframe_eeg_radial = create_dataframe(eccentricity_selection, config.source_models_for_boxplot, os.path.join(config.output_folder, 'results', 'eeg', 'radial'), 'radial')
                    create_boxplots(dataframe_eeg_radial, os.path.join(config.output_folder, 'results', 'eeg', 'radial'), 'eeg', 'radial')
                    
                    # tangential
                    dataframe_eeg_tangential = create_dataframe(eccentricity_selection, config.source_models_for_boxplot, os.path.join(config.output_folder, 'results', 'eeg', 'tangential'), 'tangential')
                    create_boxplots(dataframe_eeg_tangential, os.path.join(config.output_folder, 'results', 'eeg', 'tangential'), 'eeg', 'tangential')

        # meg
        if config.do_meg:
                for eccentricity_selection in config.eccentricity_sets:
                    #tangential
                    dataframe_meg_tangential = create_dataframe(eccentricity_selection, config.source_models_for_boxplot, os.path.join(config.output_folder, 'results', 'meg'), 'tangential')
                    create_boxplots(dataframe_meg_tangential, os.path.join(config.output_folder, 'results', 'meg'), 'meg', 'tangential')
        print("Boxplots created")
    else:
        print("No boxplots need to be created")

    print("The program didn't crash!")
