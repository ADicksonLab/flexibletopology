import sys
import os
import os.path as osp
import pickle as pkl
import numpy as np
import mdtraj as mdj

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from openmm_testsystems.testsystems import LennardJonesFluid
from sys import stdout
import time


#from flexibletopology.mlmodels.AniGSGraph import AniGSGraph
import nnforce
from nnforce_reporter import NNForceReporter

omm.Platform.loadPluginsFromDirectory('/usr/local/openmm/lib/plugins/')


IDX_START = 177
IDX_END = 117
inputs_path = 'inputs'
outputs_path = f'outputs/S{IDX_START}_E{IDX_END}'
NNMODEL_NAME = 'ani_gsg_model.pt'
DATA_FILE = f'S{IDX_START}_E{IDX_END}_W4_110.pkl'
NNFORCESCALE = 2000.0

#MD simulations settings
NUM_ATOMS = 8
PRESSURE = 80 * unit.atmospheres
TEMPERATURE = 1.0 * unit.kelvin
FRICTION_COEFFICIENT = 1.0 / unit.picosecond
STEP_SIZE = 0.001 * unit.picoseconds
STEPS = 100000
REPORT_STEPS = 100
EPSILON = 0.0*unit.kilocalories_per_mole
#Set input and output files name

PDB = 'traj8.pdb'
SIM_TRAJ = f'ljfluid_{IDX_START}to{IDX_END}.dcd'
EXNNFORCE_REPORTER = 'anigsgforces_traj.h5'


def read_data(data_file_name):

    dataset_path = osp.join(inputs_path, data_file_name)

    with open(dataset_path, 'rb') as pklf:
        data = pkl.load(pklf)


    return data['initial_coords'], data['initial_signals'], data['target_features']


if __name__=='__main__':



    fluid = LennardJonesFluid(nparticles=NUM_ATOMS, epsilon=EPSILON)

    integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)
    system, positions = fluid.system, fluid.positions

    #Set periodic box vectors (nm)
    fluid.system.setDefaultPeriodicBoxVectors([4,0,0],
                                              [0,4,0],
                                              [0,0,4])
    system = fluid.system
    omm_topology = fluid.topology

    #load the nnforce model
    positions, signals, target_features = read_data(DATA_FILE)

    ex_nnforce = nnforce.PyTorchForce(file=osp.join(inputs_path, NNMODEL_NAME), initialSignals=signals,
                                    targetFeatures=target_features, scale=NNFORCESCALE)

    ex_nnforce.setForceGroup(10)
    system.addForce(ex_nnforce)


    simulation = omma.Simulation(topology=fluid.topology,
                                 system=system,
                                 integrator=integrator)

    #move to the center of box
    for p in positions:
        p += [2,2,2]


    print("Simulations Starts")
    print(f"Steps: {STEPS}")
    begin = time.time()
    simulation.context.setPositions(positions)


    # Adding reporters
    # simulation.reporters.append(omma.StateDataReporter(stdout, REPORT_STEPS,
    #                                                    step=True,
    #                                                    potentialEnergy=True,
    #                                                    temperature=True))

    if not osp.exists(outputs_path):
        os.makedirs(outputs_path)

    simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(outputs_path, SIM_TRAJ),
                                                          REPORT_STEPS))
    simulation.reporters.append(NNForceReporter(osp.join(outputs_path, EXNNFORCE_REPORTER),
                                                 reportInterval=REPORT_STEPS))
    simulation.step(STEPS)

    # apply PBC to the saved trajectory
    pdb = mdj.load_pdb(osp.join(inputs_path, PDB))
    traj = mdj.load_dcd(osp.join(outputs_path, SIM_TRAJ), top=pdb.top)
    traj = traj.center_coordinates()
    traj.save_dcd(osp.join(outputs_path, SIM_TRAJ))
    print("Simulations Ends")
    end = time.time()
    print(f"Run time = {np.round(end - begin, 3)}s")
    simulation_time = round((STEP_SIZE * STEPS).value_in_unit(unit.nanoseconds),
                            2)
    print(f"Simulation time: {simulation_time}ns")
