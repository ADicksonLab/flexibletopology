import h5py
import mdtraj as mdj
import glob
import os.path as osp
import sys

work_folder = sys.argv[1]

# concatenate trajectories
pdb = mdj.load_pdb(osp.join(work_folder,'minimized_pos.pdb'))

n_files = len(glob.glob(osp.join(work_folder,'heating*.dcd')))

traj_all = []
for i in range(n_files):
    traj = mdj.load_dcd(osp.join(work_folder,f'heating{i}.dcd'),pdb.top)
    traj_all.append(traj)

n_ghosts = len(pdb.top.select('resname gho'))
    
big_traj = mdj.join(traj_all)
big_traj.save_dcd(osp.join(work_folder,'all_heating.dcd'))

# concatenate attributes
attr_names = ['charge','sigma','epsilon','lambda']

with open(osp.join(work_folder,'all_attr.txt'),'w') as f:
    for i in range(n_files):
        f2 = open(osp.join(work_folder,f'attr{i}.txt'),'w')
        h5 = h5py.File(osp.join(work_folder,f'traj{i}.h5'),'r')
        n_cycles = len(h5['global_variables']['0']['charge'])
        for j in range(n_cycles):
            for k in range(n_ghosts):
                attrs = [h5['global_variables'][f'{k}'][attr][j] for attr in attr_names]
                print(*attrs, file=f)
                print(*attrs, file=f2)
        f2.close()
                
