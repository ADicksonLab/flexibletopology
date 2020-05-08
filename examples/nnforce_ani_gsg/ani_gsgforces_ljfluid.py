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
import torch
import torchani

from flexibletopology.mlmodels.GSGraph import GSGraph
import nnforce
from nnforce_reporter import NNForceReporter

omm.Platform.loadPluginsFromDirectory('/usr/local/openmm/lib/plugins/')



inputs_path = 'inputs'
outputs_path ='outputs'
NNMODEL_NAME = 'ani_gsg_model.pt'
DATASET_NAME = 'openchem_3D_8_110.pkl'
IDX_START = 117
IDX_END = 177
NNFORCESCALE = 1000.0

#MD simulations settings
NUM_ATOMS = 8
PRESSURE = 80*unit.atmospheres
TEMPERATURE = 1.0 *unit.kelvin
FRICTION_COEFFICIENT = 1.0/unit.picosecond
STEP_SIZE = 0.001*unit.picoseconds
STEPS = 1000
REPORT_STEPS = 100

#Set input and output files name
PDB = 'traj8.pdb'
SIM_TRAJ = 'ljfluid_traj.dcd'
EXNNFORCE_REPORTER = 'gsgforces_traj.h5'

TORCHANI_PATH = os.path.dirname(osp.realpath(torchani.__file__))
TORCHANI_PARAMS_FILE = '../torchani/resources/ani-1ccx_8x/rHCNO-5.2R_16-3.5A_a4-8.params'



def read_data(dataset_name, idx_start, idx_end):

    dataset_path = osp.join(inputs_path, dataset_name)

    with open(dataset_path, 'rb') as pklf:
        data = pkl.load(pklf)

    target_features = np.copy(data['gaff_features_notype'][idx_end])
    signals = np.copy(data['gaff_signals_notype'][idx_end])
    tmp = signals[2]
    signals[2] = signals[6]
    signals[6] = tmp

    #convert to nm
    positions = np.copy(data['coords'][idx_start]) / 10

    return positions, signals, target_features

def ANI_AEV(coordinates):

    coordinates = torch.from_numpy(coordinates).to(torch.float32).unsqueeze(0)
    num_atoms = coordinates.shape[1]
    # Consider all atoms as carbon C=6
    atom_types = ''
    for i in range(num_atoms):
        atom_types+='C'

    #create signals from TorchANI model
    const_file = osp.join(TORCHANI_PATH, TORCHANI_PARAMS_FILE)
    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)

    species = consts.species_to_tensor(atom_types).unsqueeze(0)
    _, aev_signals = aev_computer((species, coordinates))

    return aev_signals.squeeze(0)

def GSG_features(coordinates, atomistic_signals, aev_signals):
    aev_signals = aev_signals.double()
    atomistic_signals = torch.from_numpy(atomistic_signals)
    coordinates = torch.from_numpy(coordinates).double()
    #set the GSG parameters
    wavelet_num_steps = 8
    radial_cutoff = 7.5
    scf_flags= (True, True, False)

    #construct the Torch GSG model
    model = GSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)
    device = torch.device('cpu')
    model.to(device)
    model.double()

    signals = torch.cat((aev_signals.squeeze(0), atomistic_signals), 1)
    target_features = model(coordinates, signals)

    return signals.numpy(), target_features.numpy()


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
    positions, signals, _ = read_data(DATASET_NAME, IDX_START, IDX_END)
    aev_signals = ANI_AEV(positions)
    signals, target_features = GSG_features(positions, signals, aev_signals)


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
    print(f"Simulation time = {np.round(end - begin, 3)}s")
