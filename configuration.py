import numpy as np
import os
from pathlib import Path


class Configuration:
    """
    Reads a configuration file in the format of the example from the local subtraction method paper.
    """
    def __init__(self, config_path):
        import configparser
        configs = configparser.ConfigParser()
        configs.optionxform = str
        configs.read(config_path)

        # read library paths
        self.duneuro_path = configs.get('libraries', 'duneuro')
        self.duneuro_analytic_solution = configs.get('libraries', 'duneuro_analytic_solution')
        self.simbiosphere = configs.get('libraries', 'simbiosphere')

        #read io information
        self.input_folder = configs.get('io', 'input_folder')
        self.output_folder = configs.get('io', 'output_folder')

        #read sphere parameters
        self.radii = [float(x) for x in configs.get('multilayer_sphere', 'radii').split()]
        self.sphere_center = [float(x) for x in configs.get('multilayer_sphere', 'center').split()]
        self.conductivities = [float(x) for x in configs.get('multilayer_sphere', 'conductivities').split()]
        self.eccentricities = np.array([float(x) for x in configs.get('multilayer_sphere', 'eccentricities').split()])

        # set io information
        self.filename_grid = os.path.join(self.input_folder, 'volume_conductor', 'mesh.msh')
        self.filename_tensors = os.path.join(self.input_folder, 'volume_conductor', 'conductivities.txt')
        self.volume_conductor = {'grid.filename' : self.filename_grid, 'tensors.filename' : self.filename_tensors, 'refine_brain' : False, 'refine_skin' : True}
        self.folder_dipoles = os.path.join(self.input_folder, 'dipoles')

        # read driver config
        self.driver_config = self._read_section_into_dict(configs, 'driver_config')
        self.driver_config['volume_conductor'] = self.volume_conductor
        self.driver_config['solver'] = self._read_section_into_dict(configs, 'solver_config')

        # read electrode config
        self.electrode_config = self._read_section_into_dict(configs, 'electrode_config')

        # read meg config
        self.driver_config['meg']  = self._read_section_into_dict(configs, 'meg_config')

        # read source models
        self.source_model_list = configs.get('source_models', 'types').split()
        self.source_model_config_database = {}
        for source_model in self.source_model_list:
            self.source_model_config_database[source_model] = self._read_section_into_dict(configs, f"{source_model}_config")
        # print("Source models to validate:")
        # for source_model in self.source_model_list:
        #     print(source_model)

        self.source_models_for_numerical_simulation = []
        # for source_model in self.source_model_list:
        #     if configs[f"{source_model}_config"].getboolean('skip_numerical_simulation', False):
        #         print(f"Skipping numerical simulation for {source_model} source model")
        #     else:
        #         print(f"Performing numerical simulation for {source_model} source model")
        #         self.source_models_for_numerical_simulation.append(source_model)

        self.numerical_simulation_needed = len(self.source_models_for_numerical_simulation) > 0

        self.source_models_for_comparison = []
        # for source_model in self.source_model_list:
        #     if configs[f"{source_model}_config"].getboolean('skip_comparison', False):
        #         print(f"Skipping comparison of numerical and analytical solutions for {source_model} source model")
        #     else:
        #         print(f"Performing comparison of numerical and analytical solutions for {source_model} source model")
        #         self.source_models_for_comparison.append(source_model)

        self.comparisons_needed = len(self.source_models_for_comparison) > 0

        self.source_models_for_boxplot = []
        # for source_model in self.source_model_list:
        #     if configs[f"{source_model}_config"].getboolean('skip_boxplot', False):
        #         print(f"Skipping {source_model} source model for boxplot")
        #     else:
        #         print(f"Showing {source_model} in result boxplots")
        #         self.source_models_for_boxplot.append(source_model)

        # read transfer matrix configs
        self.transfer_configs = {}
        self.transfer_configs['solver'] = self._read_section_into_dict(configs, 'transfer_config.solver')
        if 'grainSize' in self.transfer_configs['solver']:
            self.transfer_configs['grainSize'] = self.transfer_configs['solver']['grainSize']
        if 'numberOfThreads' in self.transfer_configs['solver']:
            self.transfer_configs['numberOfThreads'] = self.transfer_configs['solver']['numberOfThreads']
        self.force_transfer_recomputation = configs['transfer_config.solver'].getboolean('force_recomputation', False)

        # read testing parameters
        self.do_eeg = configs['testing_parameters'].getboolean('do_eeg')
        self.do_meg = configs['testing_parameters'].getboolean('do_meg')

        self.skip_analytical_eeg = configs['testing_parameters'].getboolean('skip_analytical_eeg')
        self.skip_analytical_meg = configs['testing_parameters'].getboolean('skip_analytical_meg')

        # read boxplot configs
        self.eccentricity_sets = [selection.strip().split(' ') for selection in configs.get('boxplot_config', 'eccentricities_per_boxplot').split('|')]

        self.boxplots_needed = len(self.eccentricity_sets) > 0 and len(self.source_models_for_boxplot) > 0
    

    def _read_section_into_dict(self, configs, section_name):
        dict_storage = {}
        for key, value in configs.items(section_name):
            dict_storage[key] = value
        return dict_storage
    

def create_output_folder(output_folder):
    """
    Creates output folder structure for a forward model run with several different sources for eeg and meg.
    """
    Path(output_folder).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'transfermatrices')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential')).mkdir(exist_ok=True)

    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'analytical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'numerical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'relative_error')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'rdm')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'lnMAG')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'dataframes')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'radial', 'boxplots')).mkdir(exist_ok=True)

    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'analytical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'numerical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'relative_error')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'rdm')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'lnMAG')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'dataframes')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'eeg', 'tangential', 'boxplots')).mkdir(exist_ok=True)

    Path(os.path.join(output_folder, 'results', 'meg', 'analytical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'numerical_solution')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'relative_error')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'rdm')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'lnMAG')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'dataframes')).mkdir(exist_ok=True)
    Path(os.path.join(output_folder, 'results', 'meg', 'boxplots')).mkdir(exist_ok=True)
