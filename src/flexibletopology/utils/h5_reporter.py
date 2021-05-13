import os
import numpy as np
import h5py

import simtk.unit as unit


MAX_ATOM_NUMS = 300

FIELDS = ['time', 'ml_forces', 'ml_potentialEnergy', 'ml_velosities',
          'ml_coordinates']

class MLForceReporter(object):
    def __init__(self, traj_file_path, reportInterval, time=True, forces=True,
                 potentialEnergy=True, velocities=True, coordinates=True, groups):
        self.traj_file_path = traj_file_path
        self._h5 = None
        self._reportInterval = reportInterval
        self._is_intialized = False
        self._time = bool(time)
        self._forces = bool(forces)
        self._potentialEnerg = bool(potentialEnergy)
        self._velosities = bool(velocities)
        self._coordinates = bool(coordinates)
        self._groups = groups

    def _initialize(self, simulation):

        self.h5 = h5py.File(self.traj_file_path, 'w')

        if self._time:
            self.h5.create_dataset('time', (0,), maxshape=(None,))

        if self._forces:
            self.h5.create_dataset('ml_forces', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._potentialEnerg:
            self.h5.create_dataset('ml_potentialEnergy', (0,), maxshape=(None,))

        if self._velosities:
            self.h5.create_dataset('ml_velosities', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._coordinates:
            self.h5.create_dataset('ml_coordinates', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

    #Modified from openmm hdf5.py script
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

        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, self._coordinates, self._velosities, self._forces, self._potentialEnerg)

    def report(self, simulation, state):

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True


        ml_state = simulation.context.getState(getForces=True, getEnergy=True,
                                            getPositions=True, getVelocities = True,
                                               groups=self.groups)
        if self._coordinates:
            coordinates = ml_state.getPositions(asNumpy=True)
            self._extend_traj_field('ml_coordinates', coordinates)

        if self._time:
            time =  ml_state.getTime().value_in_unit(unit.picosecond)
            self._extend_traj_field('time', np.array(time))

        if self._forces:
            forces = ml_state.getForces(asNumpy=True)
            self._extend_traj_field('ml_forces', forces)

        if self._potentialEnerg:
            potentialEnergy = ml_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            self._extend_traj_field('ml_potentialEnergy', np.array(potentialEnergy))

        if self._velosities:
            velocities = ml_state.getVelocities(asNumpy=True)
            self._extend_traj_field('ml_velosities', velocities)

        self.h5.flush()

    def close(self):
        "Close the underlying trajectory file"
        self.h5.close()
