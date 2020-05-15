"""
This is a reporter for CustomForces from OpenMM. It creates an
HDF5 files with the total forces for the system (forces) and the force
created by the CustomForce (exforce). Under "def report", "groups" is
the way the group number for you CustomForce is defined. This must be
defined in your CustomForce script as well (as an interger). In this
script the force is given a group number via
"0b00000000000000000000010000000000" which is read from right to left,
starting at 0 (up to 32, so 0-31). The value here is defined as group
10. This is OpenMM's way of assigning a value to a force group. Groups
0-6 are assigned to existing forces and should not be used.

"""

import os
import numpy as np
import h5py

import simtk.unit as unit


MAX_ATOM_NUMS = 300

FIELDS = ['time', 'nn_forces', 'nn_potentialEnergy', 'nn_velocities',
          'nn_coordinates']

class NNForceReporter(object):
    def __init__(self, traj_file_path, reportInterval, time=True, forces=True,
                 potentialEnergy=True, velocities=True, coordinates=True):
        self.traj_file_path = traj_file_path
        self._h5 = None
        self._reportInterval = reportInterval
        self._is_intialized = False
        self._time = bool(time)
        self._forces = bool(forces)
        self._potentialEnerg = bool(potentialEnergy)
        self._velocities = bool(velocities)
        self._coordinates = bool(coordinates)

    def _initialize(self, simulation):

        self.h5 = h5py.File(self.traj_file_path, 'w')

        if self._time:
            self.h5.create_dataset('time', (0,), maxshape=(None,))

        if self._forces:
            self.h5.create_dataset('nn_forces', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._potentialEnerg:
            self.h5.create_dataset('nn_potentialEnergy', (0,), maxshape=(None,))

        if self._velocities:
            self.h5.create_dataset('nn_velocities', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

        if self._coordinates:
            self.h5.create_dataset('nn_coordinates', (0, 0, 0), maxshape=(None, MAX_ATOM_NUMS, 3))

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
        return (steps, self._coordinates, self._velocities, self._forces, self._potentialEnerg)

    def report(self, simulation, state):

        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True


        # this assigns the CF to group 10
        nn_state = simulation.context.getState(getForces=True, getEnergy=True,
                                            getPositions=True, getVelocities = True,
                                            groups=0b00000000000000000000010000000000)
        if self._coordinates:
            coordinates = nn_state.getPositions(asNumpy=True)
            self._extend_traj_field('nn_coordinates', coordinates)

        if self._time:
            time =  nn_state.getTime().value_in_unit(unit.picosecond)
            self._extend_traj_field('time', np.array(time))

        if self._forces:
            forces = nn_state.getForces(asNumpy=True)
            self._extend_traj_field('nn_forces', forces)

        if self._potentialEnerg:
            potentialEnergy = nn_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            self._extend_traj_field('nn_potentialEnergy', np.array(potentialEnergy))

        if self._velocities:
            velocities = nn_state.getVelocities(asNumpy=True)
            self._extend_traj_field('nn_velocities', velocities)

        self.h5.flush()

    def close(self):
        "Close the underlying trajectory file"
        self.h5.close()
