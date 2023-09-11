# Living pharmacophore example

This folder contains files and instructions for creating distributions
of ghost particles in a ligand binding site.  These can be thought of
as "living examplars" or "living pharmacophores" as they are created
through self-consistent modeling of protein, solvent and pharmacophore
degrees of freedom.

## Installation and setup

Since this never uses the typical Flexible Topology structure restraint
implemented by MLForce, setup is much simpler than a typical FT simulation.
Aside from openmm and `flexibletopology` (this repo), you only need
`openmmcontinuityforce`.

### Create a conda environment and install `openmm` and `mdtraj`
If you don't already have a working environment with CONDA, I recommend
setting it up at this point:
```
conda create -n flextop python=3.11
conda activate flextop
conda install -c conda-forge openmm mdtraj
```

### Installation of `flexibletopology`
If you have not already installed `flexibletopology` (this repo) then do so
now.

```
git clone https://github.com/ADicksonLab/flexibletopology.git
cd flexibletopology
pip install .
```

### Installation of `openmmcontinuityforce`

This is a plugin found [here](https://github.com/alexrd/openmmcontinuityforce).
Installation instructions are given in the github README.  To summarize, you will
need to clone the repo and build it with cmake, following the instructions provided
(making sure to set the right openmm directory!).  You might need to install `cmake`,
which you can do with conda (`conda install cmake`).

Another important note is that if you would like to run this in a GPU-enabled environment
(recommended!) then be sure to build this with `BUILD_CUDA_LIB` set to `ON`

## Running the example

To run the example, run the `build_minimize_heat_noMLF.py` script with arguments as follows:
```
python build_minimize_heat_noMLF.py $n_ghosts $run_idx $OPENMMDIR $OUTPUTFOLDER
```
where `$n_ghosts` is the number of ghost particles you want, `$run_idx` is the index of this run, and `$OPENMMDIR` is the directory where
you installed the `openmmcontinuityforce` plugin. `$OUTPUTFOLDER` is a base folder where you would like your outputs to be written.
The script will automatically make subdirectories for your different runs within `$OUTPUTFOLDER`.

This runs a simulation in the pocket of the BRD1 bromodomain (files are in the `inputs` directory). To run a simulation in a different system,
adjust the section of the `build_minimize_heat_noMLF.py` script that is labelled: "System-specific information".

## Visualizing the output

The attributes are saved in separate HDF5 files for each of the heating steps.  The final heating step is the longest, which is at 300 K.
The corresponding `dcd` files save the coordinates of the protein, solvent and ghost particles.  One way to visualize this is using the VMD program.
To prepare these outputs for visualization, run the following script:
```
python prepare_viz.py $n_ghosts $run_idx $OUTPUTFOLDER
```
which will concatenate your `dcd` files and attributes for all of the heating files.
The resulting files `all_heating.dcd` and `all_attrs.txt` can be copied to a local machine for visualization, along with a pdb of the system (e.g. `minimized_pos.pdb`).

Using the pdb file, load the `dcd` into VMD.  Then source the file `flextop.tcl` into the TKconsole.  It is found in the flexibletopology repo in the `vis_utils` folder.

To visualize the trajectory with the attributes, enter the following commands to the TKconsole:
```
set mol 0

ft_setup_mol_reps $mol
ft_load_attributes $mol ATTR_FILE_PATH
```
where ATTR_FILE_PATH is the path to the `all_attrs.txt` on your machine.

You can then navigate to a specific frame with:
```
ft_goto $frame $mol
```
or step through the trajectory with:
```
ft_step $step_size $mol
```

Note that stepping through with the slider will change the atomic positions but will not update the attributes.
