import os.path as osp
import numpy as np
import mdtraj as mdj

def build_mdtraj_top(N):
    new_top = mdj.Topology()
    chain = new_top.add_chain()
    
    for i in range(N):
        res = new_top.add_residue('gho',chain)
        _ = new_top.add_atom(f'Ar{i}',mdj.element.Element.getBySymbol('Ar'),res)

    return new_top

def extend_mdtraj_top(top,N):
    chain = top.add_chain()
    
    for i in range(N):
        res = top.add_residue('gho',chain)
        _ = top.add_atom(f'Ar{i}',mdj.element.Element.getBySymbol('Ar'),res)

    return top

def merge_pdbs(system_pdb, ligand_pdb, resname_to_replace, out_dir, ligand_id, solvent_name):
    """ Replace atoms with a specific residue name in the system PDB file 
    using coordinates and atom names from a ligand PDB.

    Args:
    system_pdb (str): Path to the system PDB.
    ligand_pdb (str): Path to the ligand PDB corresponding to the generated system_pdb.
    resname_to_replace (str): Resname in the syatem_pdb to be replaced.
    out_dir (str): Output dir to save the new PDB.
    ligand_id (str): Ligand ID to be used in output PDB naming.
    solvent_name (str): Solvent name to be used in output PDB naming.

    Returns:
        None.
    """

    system_traj = mdj.load(system_pdb)
    ligand_traj = mdj.load(ligand_pdb)

    # Get indices of all atoms except resname_to_replace
    keep_indices = [atom.index for atom in system_traj.top.atoms if atom.residue.name.lower() != resname_to_replace]
    cleaned_traj = system_traj.atom_slice(keep_indices)

    top_cleaned, xyz_cleaned, time_cleaned, ul_cleaned, ua_cleaned = cleaned_traj.top, cleaned_traj.xyz, cleaned_traj.time, cleaned_traj.unitcell_lengths, cleaned_traj.unitcell_angles
    top_lig, xyz_lig = ligand_traj.top, ligand_traj.xyz

    top_merged = top_cleaned.join(top_lig)
    xyz_merged = np.concatenate((xyz_cleaned, xyz_lig), axis=1)

    merged_pdb = mdj.Trajectory(xyz=xyz_merged, topology=top_merged, time=time_cleaned, unitcell_lengths=ul_cleaned, unitcell_angles=ua_cleaned) 
    merged_pdb.save(osp.join(out_dir, f'{ligand_id}_{solvent_name}.pdb'))
