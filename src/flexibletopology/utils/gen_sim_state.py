from copy import copy
import numpy as np

try:
    import openmm.app as omma
    import openmm as omm
    import openmm.unit as unit
except ModuleNotFoundError:
    raise ModuleNotFoundError("OpenMM has not been installed, which this runner requires.")

GET_STATE_KWARG_DEFAULTS = (('getPositions', True),
                            ('getVelocities', True),
                            ('getForces', True),
                            ('getEnergy', True),
                            ('getParameters', True),
                            ('getParameterDerivatives', True),
                            ('enforcePeriodicBox', True),)

def gen_param_state(positions, system, integrator, parameters, temperature, n_ghosts, PLATFORM, getState_kwargs=None):
    """Convenience function for generating an omm.State object.
    Parameters
    ----------
    positions : arraylike of float
        The positions for the system you want to set
    system : openmm.app.System object
    integrator : openmm.Integrator object
    Returns
    -------
    sim_state : openmm.State object
    """

    # handle the getState_kwargs
    tmp_getState_kwargs = getState_kwargs

    # start with the defaults
    getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
    getState_kwargs['enforcePeriodicBox'] = False
    # if there were customizations use them
    if tmp_getState_kwargs is not None:
        getState_kwargs.update(tmp_getState_kwargs)

    # generate a throwaway context, using the reference platform so we
    # don't screw up other platform stuff later in the same process
    if PLATFORM == 'CUDA':
        print("Using CUDA platform..")
        platform = omm.Platform.getPlatformByName('CUDA')
    elif PLATFORM == 'OpenCL':
        print("Using OpenCL platform..")
        platform = omm.Platform.getPlatformByName('OpenCL')
        prop = dict(OpenCLPrecision='double')
    else:
        print("Using Reference platform..")
        prop = {}
        platform = omm.Platform.getPlatformByName('Reference')

    context = omm.Context(system, copy(integrator), platform)

    # set the positions
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)  

    for i in range(n_ghosts):
        context.setParameter(f"charge_g{i}", parameters[i][0])
        context.setParameter(f"sigma_g{i}", parameters[i][1])
        context.setParameter(f"epsilon_g{i}", parameters[i][2])
        context.setParameter(f"lambda_g{i}", parameters[i][3])        

    # then just retrieve it as a state using the default kwargs
    sim_state = context.getState(**getState_kwargs)

    return sim_state
