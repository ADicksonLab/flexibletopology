import numpy as np
import openmm.app as omma
import openmm.openmm as omm
from openmm import unit
import math

class CustomHybridIntegrator(omm.CustomIntegrator):

    GLOBAL_PARAMETERS = ['charge', 'sigma', 'epsilon', 'lambda']

    def __init__(self, n_ghosts, temperature, friction_coeff, timestep,
                 attr_fric_coeffs=None, attr_bounds=None):

        super(CustomHybridIntegrator, self).__init__(timestep)

        assert attr_fric_coeffs is not None, "Coefficients must be given."
        assert attr_bounds is not None, "Parameter bounds must be given."

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)

        # check on this boltzmann constant (kJ/mol/K)
        self.addGlobalVariable("kT", (0.008314463*temperature))

        # Add a part for Langevin integrator for the molecules in the system!
        self.addGlobalVariable("a", math.exp(-friction_coeff*timestep))
        self.addGlobalVariable("b", math.sqrt(
            1 - math.exp(-2*friction_coeff*timestep)))
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

        for idx in range(n_ghosts):
            for parameter_name in self.GLOBAL_PARAMETERS:
                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx}"
                                      f"+dt*f{parameter_name}_g{idx}/{attr_fric_coeffs[parameter_name]}"
                                      f"+sqrt(2*kT*dt/{attr_fric_coeffs[parameter_name]})*gaussian,"
                                      f"{attr_bounds[parameter_name][1]}),{attr_bounds[parameter_name][0]})")


class CustomHybridIntegratorConstCharge(omm.CustomIntegrator):

    GLOBAL_PARAMETERS = ['charge', 'sigma', 'epsilon', 'lambda']

    def __init__(self, n_ghosts, temperature, friction_coeff, timestep,
                 attr_fric_coeffs=None, attr_bounds=None, const_charge=0.):

        super(CustomHybridIntegratorConstCharge, self).__init__(timestep)

        assert attr_fric_coeffs is not None, "Coefficients must be given."
        assert attr_bounds is not None, "Parameter bounds must be given."

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)

        # define a global variable for the total charge
        self.addGlobalVariable("tot_charge", 0.0)

        # check on this boltzmann constant (kJ/mol/K)
        self.addGlobalVariable("kT", (0.008314463*temperature))

        # Add a part for Langevin integrator for the molecules in the system!
        self.addGlobalVariable("a", math.exp(-friction_coeff*timestep))
        self.addGlobalVariable("b", math.sqrt(
            1 - math.exp(-2*friction_coeff*timestep)))
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

        for idx in range(n_ghosts):
            for parameter_name in self.GLOBAL_PARAMETERS:

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx}"
                                      f"+dt*(f{parameter_name}_g{idx}/{attr_fric_coeffs[parameter_name]}"
                                      f"+sqrt(2*kT/(dt*{attr_fric_coeffs[parameter_name]}))*gaussian),"
                                      f"{attr_bounds[parameter_name][1]}),{attr_bounds[parameter_name][0]})")

        compute_tc_string = ""
        for idx in range(n_ghosts):
            compute_tc_string += f"charge_g{idx}"
            if idx != n_ghosts - 1:
                compute_tc_string += " + "
        self.addComputeGlobal("tot_charge", compute_tc_string)

        for idx in range(n_ghosts):
            self.addComputeGlobal(f"charge_g{idx}",
                                  f"charge_g{idx} - (tot_charge - {const_charge})/{n_ghosts}")


class CustomHybridIntegratorRestrictedChargeVariance(omm.CustomIntegrator):

    GLOBAL_PARAMETERS = ['charge', 'sigma', 'epsilon', 'lambda']

    def __init__(self, n_ghosts, temperature, friction_coeff, timestep,
                 attr_fric_coeffs=None, attr_bounds=None, const_charge = 0., max_charge_variance=0.08):

        super(CustomHybridIntegratorRestrictedChargeVariance, self).__init__(timestep)

        assert attr_fric_coeffs is not None, "Coefficients must be given."
        assert attr_bounds is not None, "Parameter bounds must be given."

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 0.0)

        # define a global variable for the charge variance
        self.addGlobalVariable("tot_charge", 0.0)
        self.addGlobalVariable("chargeVariance", 0.0)
        self.addGlobalVariable("factor", 1.0)

        # check on this boltzmann constant (kJ/mol/K)
        self.addGlobalVariable("kT", (0.008314463*temperature))

        # Add a part for Langevin integrator for the molecules in the system!
        self.addGlobalVariable("a", math.exp(-friction_coeff*timestep))
        self.addGlobalVariable("b", math.sqrt(
            1 - math.exp(-2*friction_coeff*timestep)))
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()

        for parameter_name in self.GLOBAL_PARAMETERS:
            for idx in range(n_ghosts):
                self.addComputeGlobal(f"f{parameter_name}_g{idx}",
                                      f"-deriv(energy, {parameter_name}_g{idx})")

        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

        for idx in range(n_ghosts):
            for parameter_name in self.GLOBAL_PARAMETERS:
                
                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"max(min({parameter_name}_g{idx}"
                                      f"+dt*(f{parameter_name}_g{idx}/{attr_fric_coeffs[parameter_name]}"
                                      f"+sqrt(2*kT/(dt*{attr_fric_coeffs[parameter_name]}))*gaussian),"
                                      f"{attr_bounds[parameter_name][1]}),{attr_bounds[parameter_name][0]})")

        compute_tc_string = ""
        for idx in range(n_ghosts):
            compute_tc_string += f"charge_g{idx}"
            if idx != n_ghosts - 1:
                compute_tc_string += " + "
        self.addComputeGlobal("tot_charge", compute_tc_string)

        for idx in range(n_ghosts):
            self.addComputeGlobal(f"charge_g{idx}",
                                 f"charge_g{idx} - (tot_charge - {const_charge})/{n_ghosts}")


        compute_chargeVariance_string = ""
        for idx in range(n_ghosts):
            compute_chargeVariance_string += f"charge_g{idx} * charge_g{idx}"
            if idx != n_ghosts - 1:
                compute_chargeVariance_string += " + "
        self.addComputeGlobal("chargeVariance", compute_chargeVariance_string )
        self.addComputeGlobal("chargeVariance" , f"chargeVariance / {n_ghosts}")

        self.beginIfBlock(f"chargeVariance > {max_charge_variance}")
        self.addComputeGlobal("factor", f"sqrt(chargeVariance / {max_charge_variance})")
        for idx in range(n_ghosts):
            self.addComputeGlobal(f"charge_g{idx}", f"charge_g{idx} / factor" ) 
        self.endBlock() 

class CustomDiffIntegratorNoAttr(omm.CustomIntegrator):

    def __init__(self, n_ghosts, temperature, friction_coeff, timestep, difftime_decay_rate):

        super(CustomDiffIntegratorNoAttr, self).__init__(timestep)

        # check on this boltzmann constant (kJ/mol/K)
        self.addGlobalVariable("kT", (0.008314463*temperature))

        # Add a part for Langevin integrator for the molecules in the system!
        self.addGlobalVariable("a", math.exp(-friction_coeff*timestep))
        self.addGlobalVariable("b", math.sqrt(1 - math.exp(-2*friction_coeff*timestep)))
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 2*(x-x1)/dt")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 2*(x-x1)/dt")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")

        self.addComputeGlobal("diffTime",f"diffTime - {difftime_decay_rate}")
        self.beginIfBlock(f"diffTime < 0.0")
        self.addComputeGlobal("diffTime", "0.0")
        self.endBlock() 

        
