import numpy as np
import simtk.openmm.app as omma
import simtk.openmm.openmm as omm
from simtk import unit
import math

class CustomLPIntegrator(omm.CustomIntegrator):
    def __init__(self, n_ghosts, timestep=1.0 * unit.femtoseconds,
                 nu_lambda=10, nu_charge=10,
                 nu_sigma=10, nu_epsilon=10, bounds=None):

        super(CustomLPIntegrator, self).__init__(timestep)
        if bounds is None:
            bounds = {}
            bounds['lambda'] = (0.0,1.0)
            bounds['charge'] = (-1.0,1.0)
            bounds['sigma'] = (0.07,0.50)
            bounds['epsilon'] = (0.15,1.0)

        global_parameters = ['charge', 'sigma', 'epsilon', 'lambda']

        # initialize
        self.addPerDofVariable("x0", 0)

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)
                self.addGlobalVariable(f"v{parameter_name}_g{idx}", 0.0)

        self.addGlobalVariable("nu_charge", nu_charge)
        self.addGlobalVariable("nu_sigma", nu_sigma)
        self.addGlobalVariable("nu_epsilon", nu_epsilon)
        self.addGlobalVariable("nu_lambda", nu_lambda)

        self.addUpdateContextState()
        # calcuate new positions and velocities
        self.addComputePerDof("x0", "x")
        self.addComputePerDof("v", "v+dt*f/m")

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-x0)/dt")
        # parameters
        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"v{parameter_name}_g{idx}",
                                      f"v{parameter_name}_g{idx}+dt*f{parameter_name}_g{idx}/nu_{parameter_name}")

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx}+dt*v{parameter_name}_g{idx},{bounds[parameter_name][1]}),{bounds[parameter_name][0]})")

        #self.addUpdateContextState()


class CustomVerletIntegrator(omm.CustomIntegrator):
    def __init__(self, n_ghosts, timestep=1.0 * unit.femtoseconds,
                 nu_lambda=10, nu_charge=10, nu_sigma=10,
                 nu_epsilon=10, bounds=None):

        super(CustomVerletIntegrator, self).__init__(timestep)

        if bounds is None:
            bounds = {}
            bounds['lambda'] = (0.0,1.0)
            bounds['charge'] = (-1.0,1.0)
            bounds['sigma'] = (0.07,0.50)
            bounds['epsilon'] = (0.15,1.0)

        global_parameters = ['charge', 'sigma', 'epsilon', 'lambda']

        # variable initialization
        self.addPerDofVariable("x1", 0)

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)
                self.addGlobalVariable(f"v{parameter_name}_g{idx}", 0.0)

        self.addGlobalVariable("nu_charge", nu_charge)
        self.addGlobalVariable("nu_sigma", nu_sigma)
        self.addGlobalVariable("nu_epsilon", nu_epsilon)
        self.addGlobalVariable("nu_lambda", nu_lambda)

        self.addUpdateContextState()

        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        for idx in range(n_ghosts):
            for parameter_name in global_parameters:
                self.addComputeGlobal(f"v{parameter_name}_g{idx}",
                                      f"v{parameter_name}_g{idx}+0.5*dt+f{parameter_name}_g{idx}/nu_{parameter_name}")

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx}+dt*v{parameter_name}_g{idx},{bounds[parameter_name][1]}),{bounds[parameter_name][0]})")


class CustomLGIntegrator(omm.CustomIntegrator):
    def __init__(self,n_ghosts, temperature, 
            friction_coeff, timestep, 
            bounds=None, **nu):

        super(CustomLGIntegrator, self).__init__(timestep)

        if bounds is None:
            bounds = {}
            bounds['lambda'] = (0.0,1.0)
            bounds['charge'] = (-0.5,0.5)
            bounds['sigma'] = (0.2,0.3)
            bounds['epsilon'] = (0.05,1.2)
        
        global_parameters = ['lambda', 'charge', 'sigma', 'epsilon']

        if nu is None:
            nu = {}
            nu['lambda'] = 1.0
            nu['charge'] = 1.0
            nu['sigma'] =  1.0
            nu['epsilon'] = 1.0

        ## variable initialization
        #self.addPerDofVariable("x1", 0)

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)
        self.addGlobalVariable("kT", (0.008314463*temperature))
        #self.addGlobalVariable("m_lambda", 100.0)
        #self.addGlobalVariable("m_charge", 100.0)
        #self.addGlobalVariable("m_sigma", 100.0)
        #self.addGlobalVariable("m_epsilon", 100.0)

        #starting the integration algorithm
        #self.addUpdateContextState()
        #self.addComputePerDof("v", "v+0.5*dt*f/m")
        #self.addComputePerDof("x", "x+dt*v")
        #self.addComputePerDof("x1", "x")
        #self.addConstrainPositions()
        #self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        #self.addConstrainVelocities()

## Add a part for Langevin integrator for the molecules in the system!
        self.addGlobalVariable("a", math.exp(-friction_coeff*timestep));
        self.addGlobalVariable("b", math.sqrt(1- math.exp(-2*friction_coeff*timestep)));
        #self.addGlobalVariable("kT", kB*temperature);
        self.addPerDofVariable("x1", 0);
        self.addUpdateContextState();
        self.addComputePerDof("v", "v + dt*f/m");
        self.addConstrainVelocities();
        self.addComputePerDof("x", "x + 0.5*dt*v");
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian");
        self.addComputePerDof("x", "x + 0.5*dt*v");
        self.addComputePerDof("x1", "x");
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + (x-x1)/dt");

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        for idx in range(n_ghosts):
            for parameter_name in global_parameters:
        #        self.addComputeGlobal(f"v{parameter_name}_g{idx}",
        #                              f"v{parameter_name}_g{idx}+0.5*dt+f{parameter_name}_g{idx}/nu_{parameter_name}")
#                self.addComputeGlobal(f"{parameter_name}_g{idx}",
#                                      f"max(min({parameter_name}_g{idx} + dt*((1.0/({nu[parameter_name]}*m_{parameter_name})*f{parameter_name}_g{idx}) + sqrt(2*kT/({nu[parameter_name]}*m_{parameter_name}))*gaussian*1),{bounds[parameter_name][1]}),{bounds[parameter_name][0]})")

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx} + dt*((1.0/({nu[parameter_name]})*f{parameter_name}_g{idx}) + sqrt(2*kT/({nu[parameter_name]}))*gaussian),{bounds[parameter_name][1]}),{bounds[parameter_name][0]})")
