import openmm.app as omma
import numpy as np

def _n_less_than(idx,idx_list):
    """Returns the number of elements of idx_list 
    that are less than idx.  Useful for renumbering
    PSF files upon deleting atoms."""
    n = 0
    for i in idx_list:
        if i < idx:
            n += 1
    return n

def removeFromPSF(psf,idxs_to_remove,verbose=False):
    n = len(psf.atom_list)

    removed_residues = []
    # make a list of residues that are now removed
    new_res_list = []
    
    for r in psf.residue_list:
        new_at_list = []
        for a in r.atoms:
            if a.idx not in idxs_to_remove:
                a.idx -= _n_less_than(a.idx,idxs_to_remove)
                new_at_list.append(a)
            else:
                a.idx = -1  # mark it as deleted
        r.atoms = new_at_list
        if len(new_at_list) > 0:
            r.idx -= _n_less_than(r.idx,removed_residues)
            new_res_list.append(r)
        else:
            if verbose: print(f"Removing {r} from residue list")
            removed_residues.append(r.idx)
            
    # remove atoms from atom list
    new_atom_list = []
    for a in psf.atom_list:
        if a.idx != -1:
            new_atom_list.append(a)
        else:
            if verbose: print(f"Removing {a} from atom list")

    psf.atom_list = new_atom_list

    # remove associated bonds from bond list; renumber
    new_bond_list = []
    for b in psf.bond_list:
        if b.atom1.idx != -1 and b.atom2.idx != -1:
            new_bond_list.append(b)
        else:
            if verbose: print(f"Removing {b} from bond list")

    psf.bond_list = new_bond_list

    # remove associated angles from angle list; renumber
    new_angle_list = []
    for a in psf.angle_list:
        if a.atom1.idx != -1 and a.atom2.idx != -1 and a.atom3.idx != -1:
            new_angle_list.append(a)
        else:
            if verbose: print(f"Removing {a} from angle list")

    psf.angle_list = new_angle_list

    # remove associated cmap terms from cmap list; renumber
    new_cmap_list = []
    for c in psf.cmap_list:
        if c.atom1.idx != -1 and c.atom2.idx != -1 and c.atom3.idx != -1 and c.atom4.idx != -1 and c.atom5.idx != -1:
            new_cmap_list.append(c)
        else:
            if verbose: print(f"Removing {c} from cmap list")

    psf.cmap_list = new_cmap_list
    
    # remove associated dihedral terms from dihedral list; renumber
    new_dihedral_list = []
    for d in psf.dihedral_list:
        if d.atom1.idx != -1 and d.atom2.idx != -1 and d.atom3.idx != -1 and d.atom4.idx != -1:
            new_dihedral_list.append(d)
        else:
            if verbose: print(f"Removing {d} from dihedral list")

    psf.dihedral_list = new_dihedral_list

    # remove associated improper terms from improper list; renumber
    new_improper_list = []
    for d in psf.improper_list:
        if d.atom1.idx != -1 and d.atom2.idx != -1 and d.atom3.idx != -1 and d.atom4.idx != -1:
            new_improper_list.append(d)
        else:
            if verbose: print(f"Removing {d} from improper list")

    psf.improper_list = new_improper_list
        
    return psf

def removeFromPDB(pdb,to_delete):
    n_atoms = pdb.n_atoms
    to_keep = [i for i in range(n_atoms) if i not in to_delete]

    return pdb.atom_slice(to_keep)

def findCloseWaterAtoms(pdb,centroid,n_waters):

    # sort waters by their proximity to the centroid
    water_oxy = pdb.top.select('water and name O')
    dists = np.sqrt(np.sum(np.square(pdb.xyz[0][water_oxy] - centroid),axis=1))
    sort_water_idxs = dists.argsort()

    # find out the indices of everything in them
    to_delete = []
    for i in range(n_waters):
        to_delete += list(pdb.top.select(f'resid {pdb.top.atom(water_oxy[sort_water_idxs[i]]).residue.index}'))

    return to_delete
