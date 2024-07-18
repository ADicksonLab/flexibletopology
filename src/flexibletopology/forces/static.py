import openmm as omm
import openmm.unit as unit
import numpy as np

def add_ghosts_to_nb_forces(system, n_ghosts, n_part_system):
        
    nb_forces = []
    cnb_forces = []

    exclusion_list = []

    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            force = _modify_nb_force(force, n_ghosts)
            exclusion_list=[]
            for i in range(force.getNumExceptions()):
                particles = (force.getExceptionParameters(i)[0],force.getExceptionParameters(i)[1])
                exclusion_list.append(particles)


        if force.__class__.__name__ == 'CustomNonbondedForce':
            if force.getEnergyFunction() == '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)':
                force = _modify_charmm_cnb_force(force, n_ghosts)
            else:
                force = _modify_other_cnb_force(force, n_ghosts)

    return system, exclusion_list

def _modify_nb_force(force, n_ghosts):
    for idx in range(n_ghosts):
        force.addParticle(0.0, #charge
                          1.0, #sigma (nm)
                          0.0) #epsilon (kJ/mol)
    
    return force

def _modify_charmm_cnb_force(force, n_ghosts):

    # loop over the two tabulated functions
    for fn_idx in range(2):
        tfunc = force.getTabulatedFunction(fn_idx)

        # get the parameters
        func_params = tfunc.getFunctionParameters()

        # modify them
        n_types = func_params[0]
        assert func_params[0] == func_params[1], "Unexpected behavior in CHARMM custom non-bonded force"

        tab_values = np.array(func_params[2]).reshape(func_params[0],func_params[1])

        func_params[0] += 1
        func_params[1] += 1
        new_tab_values = np.pad(tab_values,(0,1))
        func_params[2] = tuple(new_tab_values.flatten())

        # set the parameters
        tfunc.setFunctionParameters(*func_params)

    for idx in range(n_ghosts):
        force.addParticle([float(n_types)])
        
    return force

def _modify_other_cnb_force(force, n_ghosts):

    for idx in range(n_ghosts):
        force.addParticle([0.0])
    force.addInteractionGroup(set(range(n_part_system)),
                              set(range(n_part_system)))

    return force

def build_protein_restraint_force(positions,prot_idxs,bb_idxs,box):

    posresPROT = omm.CustomExternalForce('f*(px^2+py^2+pz^2); \
    px=min(dx, boxlx-dx); \
    py=min(dy, boxly-dy); \
    pz=min(dz, boxlz-dz); \
    dx=abs(x-x0); \
    dy=abs(y-y0); \
    dz=abs(z-z0);')
    posresPROT.addGlobalParameter('boxlx',box[0])
    posresPROT.addGlobalParameter('boxly',box[1])
    posresPROT.addGlobalParameter('boxlz',box[2])
    posresPROT.addPerParticleParameter('f')
    posresPROT.addPerParticleParameter('x0')
    posresPROT.addPerParticleParameter('y0')
    posresPROT.addPerParticleParameter('z0')

    for at_idx in prot_idxs:
        if at_idx in bb_idxs:
            f = 400.
        else:
            f = 40.
        xpos  = positions[at_idx].value_in_unit(unit.nanometers)[0]
        ypos  = positions[at_idx].value_in_unit(unit.nanometers)[1]
        zpos  = positions[at_idx].value_in_unit(unit.nanometers)[2]
        posresPROT.addParticle(at_idx, [f, xpos, ypos, zpos])

    return posresPROT

def add_pars_to_force(force, initial_attr):
    n_ghosts = len(initial_attr['charge'])
    for gh_idx in range(n_ghosts):
        force.addGlobalParameter(f'charge_g{gh_idx}', initial_attr['charge'][gh_idx])
        force.addGlobalParameter(f'sigma_g{gh_idx}', initial_attr['sigma'][gh_idx])
        force.addGlobalParameter(f'epsilon_g{gh_idx}', initial_attr['epsilon'][gh_idx])
        force.addGlobalParameter(f'lambda_g{gh_idx}', initial_attr['lambda'][gh_idx])
        
        # adding the del(signal)s [needed in the integrator]
        force.addEnergyParameterDerivative(f'charge_g{gh_idx}')
        force.addEnergyParameterDerivative(f'sigma_g{gh_idx}')
        force.addEnergyParameterDerivative(f'epsilon_g{gh_idx}')
        force.addEnergyParameterDerivative(f'lambda_g{gh_idx}')
        
    return force

def add_custom_cbf_com(system, group_num, ghost_particle_idxs, center_of_mass, initial_attr):

    cbf = omm.CustomCentroidBondForce(1, "0.5*k*step(d - d0)*(d - d0)^2; d = sqrt((x1-com_x)^2 + (y1-com_y)^2 + (z1-com_z)^2)")
    cbf.addGlobalParameter('k', 1000)
    cbf.addGlobalParameter('d0', 0.9)
        
    cbf.addGlobalParameter('com_x', center_of_mass[0])
    cbf.addGlobalParameter('com_y', center_of_mass[1])
    cbf.addGlobalParameter('com_z', center_of_mass[2]) 
        
    for gh_idx in range(len(ghost_particle_idxs)):
        gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
        cbf.addBond([gh_grp_idx])

    cbf.setForceGroup(group_num)

    cbf = add_pars_to_force(cbf, initial_attr)
    system.addForce(cbf)
        
    return system

def add_custom_cbf(system, group_num, ghost_particle_idxs, anchor_idxs, initial_attr):

    cbf = omm.CustomCentroidBondForce(2, "0.5*k*step(distance(g1,g2) - d0)*(distance(g1,g2) - d0)^2")
    cbf.addGlobalParameter('k', 1000) 
    cbf.addGlobalParameter('d0', 0.9) 
        
    anchor_grp_idx = cbf.addGroup(anchor_idxs)
    for gh_idx in range(len(ghost_particle_idxs)):
        gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
        cbf.addBond([anchor_grp_idx, gh_grp_idx])

    cbf.setForceGroup(group_num)

    cbf = add_pars_to_force(cbf, initial_attr)
    system.addForce(cbf)
        
    return system
