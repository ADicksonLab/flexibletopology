import os
import numpy as np
#import h5py
import pickle as pkl
import simtk.unit as unit
from collections import defaultdict

MAX_ATOM_NUMS = 300

FIELDS = ['time', 'ml_forces', 'ml_potentialEnergy', 'ml_velosities',
          'ml_coordinates']


class H5Reporter(object):

    GLOBAL_VARIABLES = ['charge_g', 'sigma_g', 'epsilon_g', 'lambda_g']

    def __init__(self, traj_file_path, reportInterval=100, groups=30,
                 num_ghosts=3, time=True, forces=True, potentialEnergy=True,
                 velocities=True, coordinates=True, global_variables=True):
        self.traj_file_path = traj_file_path
        self._h5 = None
        self._reportInterval = reportInterval
        self._groups = groups
        self.num_ghosts = num_ghosts
        self._is_intialized = False
        self._time = bool(time)
        self._forces = bool(forces)
        self._potentialEnerg = bool(potentialEnergy)
        self._velosities = bool(velocities)
        self._coordinates = bool(coordinates)
        self._global_variables = bool(global_variables)

    def _initialize(self, simulation):

        self.h5 = h5py.File(self.traj_file_path, 'w')

        if self._time:
            self.h5.create_dataset('time', (0, ), maxshape=(None, ))

        if self._forces:
            self.h5.create_dataset('forces', (0, 0, 0),
                                   maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._potentialEnerg:
            self.h5.create_dataset('potentialEnergy',
                                   (0, ), maxshape=(None, ))

        if self._velosities:
            self.h5.create_dataset('velosities', (0, 0, 0),
                                   maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._coordinates:
            self.h5.create_dataset('coordinates', (0, 0, 0),
                                   maxshape=(None, MAX_ATOM_NUMS, 3))
        if self._global_variables:
            for variable_name in self.GLOBAL_VARIABLES:
                for gh_idx in range(self.num_ghosts):
                    self.h5.create_dataset(f'/global_variables/{gh_idx}/{variable_name}', (0, ),
                                           maxshape=(None, ))

    # Modified from openmm hdf5.py script
    def _extend_traj_field(self, field_name, field_data):
        """Add one new frames worth of data to the end of an existing
        contiguous (non-sparse)trajectory field.

        Parameters
        ----------

        field_name : str
            Field name
        field_data : numpy.array
            The frames of data to add.

        """

        field = self.h5[field_name]

        # of datase new frames
        n_new_frames = 1

        # check the field to make sure it is not empty
        if all([i == 0 for i in field.shape]):

            feature_dims = field_data.shape
            field.resize((n_new_frames, *feature_dims))

            # set the new data to this
            field[0:, ...] = field_data

        else:
            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field.resize((field.shape[0] + n_new_frames, *field_data.shape))
            # add the new data
            field[-n_new_frames:, ...] = field_data

    def describeNextReport(self, simulation):

        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, self._coordinates, self._velosities, self._forces, self._potentialEnerg)

    def report(self, simulation, state):

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True

        ml_state = simulation.context.getState(getForces=True, getEnergy=True,
                                               getPositions=True, getVelocities=True,
                                               groups={self.groups})

        state = simulation.context.getState(getPositions=True)

        if self._time:
            time = ml_state.getTime().value_in_unit(unit.picosecond)
            self._extend_traj_field('time', np.array(time))

        if self._coordinates:
            coordinates = state.getPositions(asNumpy=True)
            self._extend_traj_field('coordinates', coordinates)

        if self._forces:
            forces = ml_state.getForces(asNumpy=True)
            self._extend_traj_field('forces', forces)

        if self._potentialEnerg:
            potentialEnergy = ml_state.getPotentialEnergy(
            ).value_in_unit(unit.kilocalories_per_mole)
            self._extend_traj_field(
                'potentialEnergy', np.array(potentialEnergy))

        if self._velosities:
            velocities = ml_state.getVelocities(asNumpy=True)
            self._extend_traj_field('velosities', velocities)

        if slef._global_variables:
            for variable_name in self.GLOBAL_VARIABLES:
                gvalues = []
                for gh_idx in range(self.num_ghosts):
                    gvalues.append(simulation.context.getParameter(
                        f'{variable_name}{gh_idx}'))
                self._extend_traj_field(f'/global_variables/{gh_idx}/{variable_name}',
                                        np.array(gvalues))

        self.h5.flush()

    def close(self):
        "Close the underlying trajectory file"
        self.h5.close()


class GlobalVariablesReporter(object):
    def __init__(self, file_path, reportInterval, num_ghosts=3):
        self.file_path = file_path
        self._reportInterval = reportInterval
        self.num_ghosts = num_ghosts
        self.gvalues = defaultdict(list)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, True, False)

    def report(self, simulation, state):
        for i in range(self.num_ghosts):
            self.gvalues[f'charge_g{i}'].append(
                simulation.context.getParameter(f'charge_g{i}'))
            self.gvalues[f'sigma_g{i}'].append(
                simulation.context.getParameter(f'sigma_g{i}'))
            self.gvalues[f'epsilon_g{i}'].append(
                simulation.context.getParameter(f'epsilon_g{i}'))
            self.gvalues[f'lambda_g{i}'].append(
                simulation.context.getParameter(f'lambda_g{i}'))

        with open(self.file_path, 'wb') as wfile:
            pkl.dump(self.gvalues, wfile)
