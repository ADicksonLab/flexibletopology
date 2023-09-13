import openmm as omm

def add_gs_force(system,
                 n_ghosts=None,
                 n_part_system=None,
                 group_num=None,
                 initial_signals=None,
                 sys_params=None,
                 nb_exclusion_list=None):

    for gh_idx in range(n_ghosts):
        energy_function = f'lambda_g{gh_idx}*epsilon*(sor12-sor6)+138.935456*lambda_g{gh_idx}*charge1*charge2*charge_g{gh_idx}/r;'
        energy_function += 'sor12 = sor6^2; sor6 = (sigma/r)^6;'
        energy_function += f'sigma = 0.5*(sigma1+sigma2+sigma_g{gh_idx}); epsilon = sqrt(epsilon1*epsilon2*epsilon_g{gh_idx})'
        gs_force = omm.CustomNonbondedForce(energy_function)

        gs_force.addPerParticleParameter('charge')
        gs_force.addPerParticleParameter('sigma')
        gs_force.addPerParticleParameter('epsilon')
            
        # set to initial values
        gs_force.addGlobalParameter(f'charge_g{gh_idx}', initial_signals[gh_idx, 0])
        gs_force.addGlobalParameter(f'sigma_g{gh_idx}', initial_signals[gh_idx, 1])
        gs_force.addGlobalParameter(f'epsilon_g{gh_idx}', initial_signals[gh_idx, 2])
        gs_force.addGlobalParameter(f'lambda_g{gh_idx}', initial_signals[gh_idx, 3])
        gs_force.addGlobalParameter(f'assignment_g{gh_idx}', 0)

        # adding the del(signal)s [needed in the integrator]
        gs_force.addEnergyParameterDerivative(f'lambda_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'charge_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'sigma_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'epsilon_g{gh_idx}')
        
        # adding the systems params to the force
        for p_idx in range(n_part_system):
            gs_force.addParticle(
                [sys_charge[p_idx], sys_sigma[p_idx], sys_epsilon[p_idx]])

        # for each force term you need to add ALL the particles even
        # though we only use one of them!
        for p_idx in range(n_ghosts):
            gs_force.addParticle(
                [1.0, 0.0, 1.0]) # add ghosts using neutral parameters that won't affect the force at all

        # interaction between ghost and system    
        gs_force.addInteractionGroup(set(range(n_part_system)),
                                     set([n_part_system + gh_idx]))


        for j in range(len(exclusion_list)):
            gs_force.addExclusion(exclusion_list[j][0], exclusion_list[j][1])

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
            print(f"Non-bonded force: {force.getNumExceptions()} exceptions")
            for i in range(force.getNumExceptions()):
                particles = (nb_force.getExceptionParameters(i)[0],nb_force.getExceptionParameters(i)[1])
                exclusion_list.append(particles)


        if force.__class__.__name__ == 'CustomNonbondedForce':
            for idx in range(n_ghosts):
                force.addParticle([0.0])
            force.addInteractionGroup(set(range(n_part_system)),
                                      set(range(n_part_system)))

            print(f"Custom non-bonded force: {force.getNumExclusions()} exclusions")


        return system, exclusion_list
