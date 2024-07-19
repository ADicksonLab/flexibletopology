from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from openmm import unit
import numpy as np

class AttrMinimizer(object):
    """
    A class to minimize ghost particle simulations according to their attributes,
    which are saved within an OpenMM context as global variables.

    This is valuable as the standard simulation.minimize function does not
    minimize the attributes, only the positions.

    Internally, the dependent variables are represented by x, which stores the 
    attributes in the following order:

    x[0] - charge[0]
    x[1] - sigma[0]
    x[2] - epsilon[0]
    x[3] - lambda[0]
    x[4] - charge[1]
    ...

    """
    def __init__(self, simulation, n_ghosts, bounds, charge_sum=0.0, max_charge_var=0.08):
        self.simulation = simulation

        self.n_ghosts = n_ghosts
        
        lower_bounds = [bounds['charge'][0],bounds['sigma'][0],bounds['epsilon'][0],bounds['lambda'][0]] * n_ghosts
        upper_bounds = [bounds['charge'][1],bounds['sigma'][1],bounds['epsilon'][1],bounds['lambda'][1]] * n_ghosts
        self.bounds = Bounds(lower_bounds,upper_bounds)

        self.constraints = []
        if charge_sum is not None:
            sum_charge_vector = [1,0,0,0] * n_ghosts
            self.constraints.append(LinearConstraint([sum_charge_vector],[charge_sum],[charge_sum]))

        if max_charge_var is not None:
            def cons_f(x):
                sum_sq = 0.0
                for i in range(n_ghosts):
                    sum_sq += x[4*i]*x[4*i]
                sum_sq /= n_ghosts
                return [sum_sq]
            def cons_J(x):
                jvec = []
                for i in range(n_ghosts):
                    for j in range(4):
                        if j == 0:
                            jvec.append(2*x[4*i]/n_ghosts)
                        else:
                            jvec.append(0)
                return [jvec]

            self.constraints.append(NonlinearConstraint(cons_f, 0, max_charge_var*max_charge_var, jac=cons_J, hess='2-point'))

    def get_attributes(self):
        pars = self.simulation.context.getParameters()

        x = np.zeros((4*self.n_ghosts))
        for i in range(self.n_ghosts):
            x[4*i] = pars[f'charge_g{i}']
            x[4*i + 1] = pars[f'sigma_g{i}']
            x[4*i + 2] = pars[f'epsilon_g{i}']
            x[4*i + 3] = pars[f'lambda_g{i}']
            
        return x

    def set_attributes(self, x):
        for i in range(self.n_ghosts):
            self.simulation.context.setParameter(f'charge_g{i}',x[4*i])
            self.simulation.context.setParameter(f'sigma_g{i}',x[4*i + 1])
            self.simulation.context.setParameter(f'epsilon_g{i}',x[4*i + 2])
            self.simulation.context.setParameter(f'lambda_g{i}',x[4*i + 3])
                
        return
    
    def _energy(self, x):
        self.set_attributes(x)
        return self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def _energy_derivs(self, x):
        self.set_attributes(x)
        par_derivs = self.simulation.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()

        v = np.zeros((4*self.n_ghosts))
        for i in range(self.n_ghosts):
            v[4*i] = par_derivs[f'charge_g{i}']
            v[4*i + 1] = par_derivs[f'sigma_g{i}']
            v[4*i + 2] = par_derivs[f'epsilon_g{i}']
            v[4*i + 3] = par_derivs[f'lambda_g{i}']

        return v

    def attr_minimize(self,method='trust-constr', options={'verbose':1}):
        res = minimize(self._energy, self.get_attributes(), method=method, jac=self._energy_derivs, hess='2-point',
                       constraints=self.constraints, options=options, bounds=self.bounds)

        self.set_attributes(res.x)

        return res.x

