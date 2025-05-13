import numpy as np
import mdtraj as mdj

from flexibletopology.utils.mdtrajutils import build_mdtraj_top


ATTR_NAMES = ['charge','sigma','epsilon','lambda']

class Cloud(object):
    """
    A container for:
    
    - positions
    - attributes

    with utilities for reading/writing and analysis
    """

    def __init__(self,positions,attributes):
        """
        Inputs: 
        positions - array-like, shape (n_frames, n_atoms,3) or (n_atoms,3)
        attributes - list of dict (needs keys for 'charge', 'sigma', 'epsilon', 'lambda')
                                  values are np.array, shape (n_atoms),
                     or array-like, shape (n_frames, n_atoms, 4)
        """

        self.pos = np.array(positions)
        if len(self.pos.shape) == 2:
            self.pos = self.pos.reshape((1,self.pos.shape[0],self.pos.shape[1]))
            
        assert self.pos.shape[2] == 3, f"positions have incorrect shape, should be (n_atoms,3), passed {self.pos.shape}"
        self.n_atoms = self.pos.shape[1]
        self.n_frames = self.pos.shape[0]
        
        if type(attributes) is dict:
            self.attr = np.zeros((1,self.n_atoms,4))
            for i,attr in enumerate(ATTR_NAMES):
                assert attr in attributes, f"attributes dict must contain '{attr}'"
                assert len(attributes[attr]) == self.n_atoms, f"Num atoms mismatch! ({len(attributes[attr])} for {attr} != {self.n_atoms})"
                self.attr[0,:,i] = np.array(attributes[attr])
        elif type(attributes[0]) is dict:
            assert len(attributes) == self.n_frames, f"Num frames mismatch ({self.n_frames} for positions and {len(attributes)} for attributes"
            self.attr = np.zeros((self.n_frames,self.n_atoms,4))
            for f_idx in range(self.n_frames):
                for i,attr in enumerate(ATTR_NAMES):
                    assert attr in attributes[f_idx], f"attributes dict must contain '{attr}'"
                    assert len(attributes[f_idx][attr]) == self.n_atoms, f"Num atoms mismatch! ({len(attributes[f_idx][attr])} for {attr} != {self.n_atoms})"
                    self.attr[f_idx,:,i] = np.array(attributes[f_idx][attr])
        else:
            self.attr = np.array(attributes)
            if len(self.attr.shape) == 2:
                self.attr = self.attr.reshape((1,self.attr.shape[0],self.attr.shape[1]))
            assert self.attr.shape[1] == self.n_atoms, f"attributes have incorrect shape ({self.attr.shape} != (n_frames,{self.n_atoms},4)"
            assert self.attr.shape[2] == 4, f"attributes have incorrect shape ({self.attr.shape} != (n_frames,{self.n_atoms},4)"

    def write_frame_attr(self,fname,frame_idx=0):
        """
        Inputs:
        fname - str
        frame_idx - int (default = 0)
        """

        assert frame_idx < self.n_frames, "invalid frame index!"

        np.savetxt(fname,self.attr[frame_idx])

    def write_attr(self,fname):
        """
        Inputs:
        fname - str
        """

        np.savetxt(fname,self.attr.reshape((self.attr.shape[0]*self.attr.shape[1],self.attr.shape[2])))

    def write_pdb(self,fname,frame_idx=0):
        """
        Inputs:
        fname - str
        frame_idx - int (default = 0)
        """

        top = build_mdtraj_top(self.n_atoms)
        traj = mdj.Trajectory(self.pos[frame_idx],top)

        traj.save_pdb(fname)

    def write_dcd(self,fname):
        """
        Inputs:
        fname - str
        """

        top = build_mdtraj_top(self.n_atoms)
        traj = mdj.Trajectory(self.pos,top)

        traj.save_dcd(fname)
        
            
    
