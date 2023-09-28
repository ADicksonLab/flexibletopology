import openmm as omm

def add_gs_force(system,
                 n_ghosts=None,
                 n_part_system=None,
                 group_num=None,
                 initial_attr=None,
                 sys_attr=None,
                 nb_exclusion_list=None):

    for gh_idx in range(n_ghosts):
        energy_function = f'lambda_g{gh_idx}*(4.0*epstot*(sor12-sor6)+138.935456*q1*charge_g{gh_idx}/r);'
        energy_function += 'sor12 = sor6^2; sor6 = (sigtot/r)^6;'
        energy_function += f'sigtot = 0.5*(sigma1+sigma_g{gh_idx}); epstot = sqrt(epsilon1*epsilon_g{gh_idx})'
        gs_force = omm.CustomNonbondedForce(energy_function)

        gs_force.addPerParticleParameter('q')
        gs_force.addPerParticleParameter('sigma')
        gs_force.addPerParticleParameter('epsilon')
            
        # set to initial values
        gs_force.addGlobalParameter(f'charge_g{gh_idx}', initial_attr['charge'][gh_idx])
        gs_force.addGlobalParameter(f'sigma_g{gh_idx}', initial_attr['sigma'][gh_idx])
        gs_force.addGlobalParameter(f'epsilon_g{gh_idx}', initial_attr['epsilon'][gh_idx])
        gs_force.addGlobalParameter(f'lambda_g{gh_idx}', initial_attr['lambda'][gh_idx])
        gs_force.addGlobalParameter(f'fcharge_g{gh_idx}', 0.0)
        gs_force.addGlobalParameter(f'fsigma_g{gh_idx}', 0.0)
        gs_force.addGlobalParameter(f'fepsilon_g{gh_idx}', 0.0)
        gs_force.addGlobalParameter(f'flambda_g{gh_idx}', 0.0)

        gs_force.addGlobalParameter(f'assignment_g{gh_idx}', 0)

        # adding the del(signal)s [needed in the integrator]
        gs_force.addEnergyParameterDerivative(f'lambda_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'charge_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'sigma_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'epsilon_g{gh_idx}')
        
        # adding the systems params to the force
        for p_idx in range(n_part_system):
            gs_force.addParticle(
                [sys_attr['charge'][p_idx], sys_attr['sigma'][p_idx], sys_attr['epsilon'][p_idx]])

        # for each force term you need to add ALL the particles even
        # though we only use one of them!
        for p_idx in range(n_ghosts):
            gs_force.addParticle(
                [1.0, 0.0, 1.0]) # add ghosts using neutral parameters that won't affect the force at all

        # interaction between ghost and system    
        gs_force.addInteractionGroup(set(range(n_part_system)),
                                     set([n_part_system + gh_idx]))


        for j in range(len(nb_exclusion_list)):
            gs_force.addExclusion(nb_exclusion_list[j][0], nb_exclusion_list[j][1])

        # set force parameters
        gs_force.setForceGroup(group_num)
        gs_force.setNonbondedMethod(gs_force.CutoffPeriodic)
        gs_force.setCutoffDistance(1.0)
        system.addForce(gs_force)
        
    return system

def add_ghosts_to_nb_forces(system, n_ghosts, n_part_system):
        
    nb_forces = []
    cnb_forces = []

    exclusion_list = []

    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            for idx in range(n_ghosts):
                force.addParticle(0.0, #charge
                                     0.2, #sigma (nm)  (minimum distance between ghosts)  [not used currently.  add exceptions to use this for g-g interactions?]
                                     0.0) #epsilon (kJ/mol)
            exclusion_list=[]
            for i in range(force.getNumExceptions()):
                particles = (force.getExceptionParameters(i)[0],force.getExceptionParameters(i)[1])
                exclusion_list.append(particles)


        if force.__class__.__name__ == 'CustomNonbondedForce':
            for idx in range(n_ghosts):
                force.addParticle([0.0])
            force.addInteractionGroup(set(range(n_part_system)),
                                      set(range(n_part_system)))


    return system, exclusion_list
