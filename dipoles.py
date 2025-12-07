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
from configuration import Configuration
import sys
from scipy.spatial import KDTree
import scipy.sparse as sparse


class Dipoles:
    """
    For handling different dipole formats (text file, numpy array, list of DUNEuro dipoles). Includes routines for grid creation, and for creating the dipoles for a simple reconstruction problem.
    """
    def __init__(self, dipoles):
        """
        Generic initializer.
        
        :param dipoles: A 2d array containing in every row the x,y,z coordinates and optionally the x,y,z moment for a dipole.
        """
        self.dipoles = dipoles
        if self.dipoles.shape[1]==3:
            self.locations_only = True
        elif self.dipoles.shape[1]==6:
            self.locations_only = False
        else:
            raise Exception("Dipoles have wrong format.")


    @classmethod
    def from_txt(cls, dipole_path):
        """
        Construct Dipoles object from text file.

        :param dipole_path: Should lead to a txt file containing coordinates, optionally together with directions (same format as for the read_dipoles_3d DUNEuro method).
        """
        return cls(np.loadtxt(dipole_path))


    @classmethod
    def regular_grid(cls, config: Configuration, resolution, shell_thickness=None):
        """
        Construct Dipoles object containing a regular grid for a spherical head model.

        :param config: Configuration object containing geometrical information about the spherical model.
        :param resolution: Determines how far dipoles are apart in the grid.
        :param shell_thickness: If none, the whole brain compartment is filled with dipoles, otherwise only a shell with the given thickness, emulating grey and white matter.
        """
        center = config.sphere_center
        radius = config.radii[-1]
        x = np.arange(center[0]-radius-resolution,center[0]+radius+2*resolution,resolution)
        y = np.arange(center[1]-radius-resolution,center[1]+radius+2*resolution,resolution)
        z = np.arange(center[2]-radius-resolution,center[2]+radius+2*resolution,resolution)
        grid = np.vstack(list(map(np.ravel,np.meshgrid(x,y,z)))).T
        grid = grid[np.linalg.norm(grid-center,axis=1)<=radius,:]
        if shell_thickness:
            grid = grid[np.linalg.norm(grid-center,axis=1)>=radius-shell_thickness,:]
        print("Number of positions: "+str(grid.shape[0]))
        return cls(grid).add_canonical_directions()
    

    @classmethod
    def angle_based_cortex_grid(cls, config: Configuration, n_dipoles, thickness):
        """
        Construct Dipoles object containing a grid which emulates the cortical surface for a spherical head model.

        :param config: Configuration object containing geometrical information about the spherical model.
        :param n_dipoles: A rough orientation how many sources the grid should contain. It won't contain exactly that number, usually slightly more.
        :param thickness: How thick is the cortex assumed to be?
        """
        offset = 1
        amplitude = 5
        frequency = 28
        center = config.sphere_center
        radius = config.radii[-1]
        factor = 1
        n_theta = int(np.sqrt(factor*n_dipoles))
        combinations = []
        for theta in np.linspace(1e-3, np.pi/2, n_theta):
            for phi in np.linspace(0, 2*np.pi, int(n_theta*2*np.sin(theta)/factor)):
                combinations.append([theta,phi])
        combinations = np.array(combinations)
        # theta = np.linspace(1e-3, np.pi/2, n_theta)
        # phi = np.linspace(0, np.pi, int(2*n_theta/factor)+1)
        # combinations = np.vstack(list(map(np.ravel,np.meshgrid(theta,phi)))).T
        radii = radius- (amplitude+offset) - thickness/2 + amplitude * np.cos(frequency*combinations[:,0])
        grid = np.concatenate(((radii*np.sin(combinations[:,0])*np.cos(combinations[:,1])).reshape(-1,1), (radii*np.sin(combinations[:,0])*np.sin(combinations[:,1])).reshape(-1,1), (radii*np.cos(combinations[:,0])).reshape(-1,1)), axis=1)
        print("Number of positions: "+str(grid.shape[0]))
        theta = combinations[:,0]
        phi = combinations[:,1]
        grid = np.append(grid,-np.cross(np.concatenate(((-frequency*amplitude*np.sin(frequency*theta)*np.sin(theta)*np.cos(phi)+radii*np.cos(theta)*np.cos(phi)).reshape(-1,1), (-frequency*amplitude*np.sin(frequency*theta)*np.sin(theta)*np.sin(phi)+radii*np.cos(theta)*np.sin(phi)).reshape(-1,1), (-frequency*amplitude*np.sin(frequency*theta)*np.cos(theta)-radii*np.sin(theta)).reshape(-1,1)),axis=1),np.concatenate(((-radii*np.sin(theta)*np.sin(phi)).reshape(-1,1), (radii*np.sin(theta)*np.cos(phi)).reshape(-1,1), np.zeros(theta.shape).reshape(-1,1)),axis=1), axis=1),axis=1)
        grid[:,3:] /= np.linalg.norm(grid[:,3:], axis=1).reshape(-1,1)
        grid[:,:3] += center
        return cls(grid)
    

    def cutoff_connection_list(self, cutoff=10):
        """
        Create a connection list for this Dipole object, weighted according to the Euclidean distance, only adding connections that are below the cutoff.

        :param cutoff: Connections between dipoles with a greater Euclidean distance won't be added.
        """
        connections = []
        tree = KDTree(self.dipoles)
        neighbors = tree.query_ball_tree(tree, cutoff)
        for i in range(self.dipoles.shape[0]):
            for j in neighbors[i]:
                if j > i:
                    connections.append([i,j,np.linalg.norm(self.dipoles[i,:]-self.dipoles[j,:])])
        return np.array(connections)
    

    def neighbor_connection_list(self, n_neighbors=5, cutoff=10):
        """
        Create a connection list for this Dipole object, with every dipole having connections to its nearest neighbors.

        :param n_neighbors: Connections to (at most, see cutoff) the n_neighbors nearest neighbors will be added.
        :param cutoff: Connections between dipoles with a greater Euclidean distance won't be added.
        """
        connections = []
        tree = KDTree(self.dipoles)
        for i in range(self.dipoles.shape[0]):
            distance, index = tree.query(self.dipoles[i,:], n_neighbors+1, distance_upper_bound=cutoff)
            if distance.shape[0]<=1:
                raise Exception(f"Cutoff too small, point {i} has no connections.")
            for j in range(index.shape[0]):
                if index[j] > i and distance[j] < np.inf:
                    connections.append([i, index[j], distance[j]])
        return np.array(connections)
        

    @classmethod
    def moving_along_cortex(cls, config: Configuration, thickness, angle, n_steps):
        """
        Create dipole sources that are moving along the cortical surface as defined in angle_based_cortex_grid.
        """
        offset = 1
        amplitude = 5
        frequency = 28
        center = config.sphere_center
        radius = config.radii[-1]
        phi = np.pi/2
        theta = np.linspace(np.pi/4, np.pi/4+angle, n_steps)
        r = radius - (amplitude+offset) - 0.5*thickness + amplitude*np.cos(28*theta)

        sources = np.zeros((n_steps, 3))
        sources[:,1] = r * np.sin(theta)
        sources[:,2] = r * np.cos(theta)
        sources = np.append(sources,-np.cross(np.concatenate(((-frequency*amplitude*np.sin(frequency*theta)*np.sin(theta)*np.cos(phi)+r*np.cos(theta)*np.cos(phi)).reshape(-1,1), (-frequency*amplitude*np.sin(frequency*theta)*np.sin(theta)*np.sin(phi)+r*np.cos(theta)*np.sin(phi)).reshape(-1,1), (-frequency*amplitude*np.sin(frequency*theta)*np.cos(theta)-r*np.sin(theta)).reshape(-1,1)),axis=1),np.concatenate(((-r*np.sin(theta)*np.sin(phi)).reshape(-1,1), (r*np.sin(theta)*np.cos(phi)).reshape(-1,1), np.zeros(theta.shape).reshape(-1,1)),axis=1), axis=1),axis=1)
        sources[:,3:] /= np.linalg.norm(sources[:,3:], axis=1).reshape(-1,1)
        sources[:,:3] += center
        return cls(sources)
    

    def strip_directions(self):
        """
        Strips the directions from the dipole array, so that only locations remain.
        """
        if self.locations_only:
            print("Directions already not included.")
            return
        self.dipoles = self.dipoles[:,:3]


    def add_canonical_directions(self):
        """Adds the canonical basis directions (1,0,0), (0,1,0) and (0,0,1) to every dipole position. Strips existing dipole moments"""
        if not self.locations_only:
            raise Exception("Dipole directions haven't been stripped.")
        self.dipoles = np.append(np.repeat(self.dipoles,3,axis=0), np.tile([[1,0,0],[0,1,0],[0,0,1]], [self.dipoles.shape[0],1]), axis=1)
        return self
    

    def get_positions(self):
        """
        If at every position there are 3 dipoles (pointing to the 3 canonical directions), use this function to get only the positions, with every position occuring once.
        """
        return self.dipoles[::3,:3]


    def save_txt(self, file_path):
        """
        Saves the dipoles to a txt file.
        """
        np.savetxt(file_path, self.dipoles)


    def get_dune_dipoles(self, duneuro_path):
        """
        Converts dipoles to DUNEuro dipole format.
        """
        sys.path.append(duneuro_path)
        import duneuropy as dp
        dune_dipoles = []
        for i in range(self.dipoles.shape[0]):
            dune_dipoles.append(dp.Dipole3d(self.dipoles[i,:]))
        return dune_dipoles
    

    def positions_vtk_writer(self, duneuro_path, file_path=None):
        """
        Writes the dipole positions to a .vtk file. Expects position triplicates with unit vectors.
        """
        sys.path.append(duneuro_path)
        import duneuropy as dp
        writer = dp.PointVTKWriter3d([dp.FieldVector3D(self.get_positions()[i, :]) for i in range(int(self.dipoles.shape[0]/3))], True)
        if file_path is not None:
            writer.write(file_path)
        return writer
    

    def vectors_vtk_writer(self, duneuro_path, file_path=None):
        """
        Writes the dipole positions and their moments to a .vtk file.
        """
        sys.path.append(duneuro_path)
        import duneuropy as dp
        writer = dp.PointVTKWriter3d([dp.FieldVector3D(self.dipoles[i, :3]) for i in range(self.dipoles.shape[0])], True)
        writer.addVectorData("moment", [dp.FieldVector3D(self.dipoles[i, 3:]) for i in range(self.dipoles.shape[0])])
        if file_path is not None:
            writer.write(file_path)
        return writer
    

    def vtk_writer_add_current_density(self, distribution, duneuro_path, file_path=None, writer=None, suffix=""):
        """
        Writes the dipole positions and a given current density to a .vtk file.
        """
        sys.path.append(duneuro_path)
        import duneuropy as dp
        if writer is None:
            writer = self.vectors_vtk_writer(config.duneuro_path)
        for t in range(distribution.shape[1]):
            writer.addScalarData("current_density_"+suffix+str(t), distribution[:,t])
        if file_path is not None:
            writer.write(file_path)
        return writer


def get_divergence_matrix(connections, m):
    """
    Calculates the sparse matrix corresponding to the divergence operator of a given connection list.

    :param connections: A connection list.
    :param m: The number of source positions.
    """
    indices = np.array([])
    indptr = [0]
    data = np.array([])
    for i in range(m):
        plus_locations = np.argwhere((connections[:,0]==i)).flatten()
        indices = np.append(indices, plus_locations)
        data = np.append(data, np.ones(plus_locations.shape))
        minus_locations = np.argwhere((connections[:,1]==i)).flatten()
        indices = np.append(indices, minus_locations)
        data = np.append(data, -np.ones(minus_locations.shape))
        indptr.append(indptr[i]+plus_locations.shape[0]+minus_locations.shape[0])
    return sparse.csr_array((data,indices,indptr), shape=(m, connections.shape[0]))


if __name__ == "__main__":
    from configuration import create_output_folder
    from forwardmodel import ForwardModel
    import time


    n_sources = 5000

    config = Configuration("/home/emil/Uni/Masterthesis/local_subtraction_paper_code/multilayer_sphere_validation_study/configs.ini")
    config.output_folder = "/home/emil/Uni/Masterthesis/inverse_code/inversion_methods/small_example"
    config.input_folder = "/home/emil/Uni/Masterthesis/inverse_code/inversion_methods/small_example"

    create_output_folder(config.output_folder)

    dipoles = Dipoles.angle_based_cortex_grid(config,n_sources,5)

    dipoles.save_txt(f"/home/emil/Uni/Masterthesis/inverse_code/inversion_methods/text_files/half_grid_{n_sources}.txt") 

    fw_model = ForwardModel(config)
    start = time.time()
    fw_model.analytical_eeg_leadfield(dipoles)
    print("EEG: "+str(time.time()-start))
    start = time.time()
    fw_model.analytical_meg_leadfield(dipoles)
    print("MEG: "+str(time.time()-start))

    fw_model.save_leadfields("half_angle_cort_5000_")
