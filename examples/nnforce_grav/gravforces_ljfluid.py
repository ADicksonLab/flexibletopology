import sys
import os
import os.path as osp
import numpy as np
import mdtraj as mdj

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from openmm_testsystems.testsystems import LennardJonesFluid
from sys import stdout
import time
import nnforce
from flexibletopology.nnforce_reporter import NNForceReporter

omm.Platform.loadPluginsFromDirectory('/usr/local/openmm/lib/plugins/')


inputs_path = 'inputs'
outputs_path ='outputs'
NNMODEL_NAME = 'grav_model.pt'
NNFORCESCALE = 1.0
NSIG = 4

#MD simulations settings
NUM_ATOMS = 8
PRESSURE = 1*unit.atmospheres
TEMPERATURE = 300.0 *unit.kelvin
FRICTION_COEFFICIENT = 1.0/unit.picosecond
STEP_SIZE = 0.002*unit.picoseconds
STEPS = 10000
REPORT_STEPS = 10

#Set input and output files name
PDB = 'traj8.pdb'
SIM_TRAJ = 'ljfluid_traj.dcd'
EXNNFORCE_REPORTER = 'gravforces_traj.h5'

if __name__=='__main__':



    fluid = LennardJonesFluid(nparticles=NUM_ATOMS, epsilon=0.0*unit.kilocalories_per_mole)

    integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)
    system, positions = fluid.system, fluid.positions

    #Set periodic box vectors (nm)
    fluid.system.setDefaultPeriodicBoxVectors([4,0,0],
                                              [0,4,0],
                                              [0,0,4])
    system = fluid.system
    omm_topology = fluid.topology

    #load the nnforce model
    positions = np.random.random((NUM_ATOMS,3))*2.0 + 2.0  # middle of box
    signals = np.random.random((NUM_ATOMS,NSIG))
    target_features = np.zeros((NUM_ATOMS,1))

    ex_nnforce = nnforce.PyTorchForce(file=osp.join(inputs_path, NNMODEL_NAME), initialSignals=signals,
                                      targetFeatures=target_features, scale=NNFORCESCALE)

    ex_nnforce.setForceGroup(10)
    system.addForce(ex_nnforce)


    simulation = omma.Simulation(topology=fluid.topology,
                                 system=system,
                                 integrator=integrator)

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
    print(f"Simulation time = {np.round(end - begin, 3)}s")
