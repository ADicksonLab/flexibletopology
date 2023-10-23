import openmm as omm

def add_attr_force(system,
                   n_ghosts=None,
                   group_num=None,
                   initial_attr=None):

    eps_terms = ''
    charge_terms = ''
    fsig_eqns = ''
    for gh_idx in range(n_ghosts):
        eps_terms += f'A*fsig_{gh_idx}*(epsilon_g{gh_idx}-0.25)^2' 
        charge_terms += f'B*fsig_{gh_idx}*(charge_g{gh_idx}-0.25)^2'
        fsig_eqns += f'fsig_{gh_idx} = 1/(1+exp(100*(sigma_g{gh_idx}-0.3))); '
        if gh_idx < n_ghosts-1:
            eps_terms += ' + '
            charge_terms += ' + '
    energy_function = eps_terms + ' + ' + charge_terms + '; ' + fsig_eqns
        
    attr_force = omm.CustomCVForce(energy_function)

    attr_force.addGlobalParameter('A', 941.5)
    attr_force.addGlobalParameter('B', 275.3)

    for gh_idx in range(n_ghosts):
        # set to initial values
        attr_force.addGlobalParameter(f'charge_g{gh_idx}', initial_attr['charge'][gh_idx])
        attr_force.addGlobalParameter(f'sigma_g{gh_idx}', initial_attr['sigma'][gh_idx])
        attr_force.addGlobalParameter(f'epsilon_g{gh_idx}', initial_attr['epsilon'][gh_idx])

        # adding the del(signal)s [needed in the integrator]
        attr_force.addEnergyParameterDerivative(f'charge_g{gh_idx}')
        attr_force.addEnergyParameterDerivative(f'sigma_g{gh_idx}')
        attr_force.addEnergyParameterDerivative(f'epsilon_g{gh_idx}')
        
    # set force parameters
    attr_force.setForceGroup(group_num)
    system.addForce(attr_force)
        
    return system

    
