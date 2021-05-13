import numpy as np
import simtk.openmm.app as omma
import simtk.openmm.openmm as omm
from simtk import unit


class CustomLPIntegrator(omm.CustomIntegrator):
    def __init__(self, n_ghosts, timestep=1.0 * unit.femtoseconds,
                 m_lambda=10, m_charge=10,
                 m_sigma=10, m_epsilon=10):

        super(CustomLPIntegrator, self).__init__(timestep)
        lambda_min = 0.0
        lambda_max = 1.0
        charge_min = -1.0
        charge_max = 1.0
        sigma_min = 0.07
        sigma_max = 0.50
        epsilon_min = 0.15
        epsilon_max = 1.0

        global_parameters = ['charge', 'sigma', 'epsilon', 'lambda']

        # initialize
        self.addPerDofVariable("x0", 0)

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)
                self.addGlobalVariable(f"v{parameter_name}_g{idx}", 0.0)

        self.addGlobalVariable("m_charge", m_charge)
        self.addGlobalVariable("m_sigma", m_sigma)
        self.addGlobalVariable("m_epsilon", m_epsilon)
        self.addGlobalVariable("m_lambda", m_lambda)

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
                                      f"v{parameter_name}_g{idx}+dt*f{parameter_name}_g{idx}/m_{parameter_name}")

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"{parameter_name}_g{idx}+dt*v{parameter_name}_g{idx}")

                # self.addComputeGlobal(f"{parameter_name}_g{idx}",
                #                       f"max(min({parameter_name}_g{idx}+dt*v{parameter_name}_g{idx}"
                #                       f",{parameter_name}_max),{parameter_name}_min)")

        self.addUpdateContextState()


class CustomVerletIntegrator(omm.CustomIntegrator):
    def __init__(self, n_ghosts, timestep=1.0 * unit.femtoseconds,
                 m_lambda=10, m_charge=10, m_sigma=10,
                 m_epsilon=10):

        super(CustomVerletIntegrator, self).__init__(timestep)

        lambda_min = 0.0
        lambda_max = 1.0
        charge_min = -1.0
        charge_max = 1.0
        sigma_min = 0.02
        sigma_max = 0.20
        epsilon_min = 0.2
        epsilon_max = 10.0
        global_parameters = ['charge', 'sigma', 'epsilon', 'lambda']

        # variable initialization
        self.addPerDofVariable("x1", 0)

        for parameter_name in global_parameters:
            for idx in range(n_ghosts):
                self.addGlobalVariable(f"f{parameter_name}_g{idx}", 1.0)
                self.addGlobalVariable(f"v{parameter_name}_g{idx}", 0.0)

        self.addGlobalVariable("m_charge", m_charge)
        self.addGlobalVariable("m_sigma", m_sigma)
        self.addGlobalVariable("m_epsilon", m_epsilon)
        self.addGlobalVariable("m_lambda", m_lambda)

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
                                      f"v{parameter_name}_g{idx}+0.5*dt+f{parameter_name}_g{idx}/m_{parameter_name}")

                self.addComputeGlobal(f"{parameter_name}_g{idx}",
                                      f"{parameter_name}_g{idx}+dt*v{parameter_name}_g{idx}")

                # self.addComputeGlobal(f"{parameter_name}_g{idx}",
                #                       f"max(min({parameter_name}_g{idx}+dt*v{parameter_name}_g{idx},"
                #                       f"{parameter_name}_max),{parameter_name}_min)")


# class CustomVerletIntegrator(omm.CustomIntegrator):
#     def __init__(self, n_ghosts, timestep=1.0 * unit.femtoseconds,
#                  m_lambda=10, m_charge=10, m_sigma=10,
#                  m_epsilon=10):

#         super(CustomVerletIntegrator, self).__init__(timestep)

#         lambda_min = 0.0
#         lambda_max = 1.0
#         charge_min = -1.0
#         charge_max = 1.0
#         sigma_min = 0.02
#         sigma_max = 0.20
#         epsilon_min = 0.2
#         epsilon_max = 10.0

#         self.addGlobalVariable("m_lambda", m_lambda)
#         self.addGlobalVariable("m_charge", m_charge)
#         self.addGlobalVariable("m_sigma", m_sigma)
#         self.addGlobalVariable("m_epsilon", m_epsilon)

#         # update velocities and positions
#         self.addPerDofVariable("x1", 0)

#         self.addUpdateContextState()
#         self.addComputePerDof("v", "v+0.5*dt*f/m")
#         self.addComputePerDof("x", "x+dt*v")
#         self.addComputePerDof("x1", "x")
#         self.addConstrainPositions()
#         self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
#         self.addConstrainVelocities()

#         for idx in range(n_ghosts):
#             self.addGlobalVariable(f"flambda_g{idx}", 0.0)
#             self.addGlobalVariable(f"vlambda_g{idx}", 0.0)

#             self.addGlobalVariable(f"fcharge_g{idx}", 0.0)
#             self.addGlobalVariable(f"vcharge_g{idx}", 0.0)

#             self.addGlobalVariable(f"fsigma_g{idx}", 0.0)
#             self.addGlobalVariable(f"vsigma_g{idx}", 0.0)

#             self.addGlobalVariable(f"fepsilon_g{idx}", 0.0)
#             self.addGlobalVariable(f"vepsilon_g{idx}", 0.0)

#         self.addUpdateContextState()

#         for idx in range(n_ghosts):
#             #si = str(i)
#             # compute forces on dynamical variables using derivatives from custom forces
#             self.addComputeGlobal(
#                 f"flambda_g{idx}", f"-deriv(energy, lambda_g{idx})")
#             self.addComputeGlobal(
#                 f"fcharge_g{idx}", f"-deriv(energy, charge_g{idx})")
#             self.addComputeGlobal(
#                 f"fsigma_g{idx}", f"-deriv(energy, sigma_g{idx})")
#             self.addComputeGlobal(
#                 f"fepsilon_g{idx}", f"-deriv(energy, epsilon_g{idx})")

#             # use forces to update velocities
#             self.addComputeGlobal(f"vlambda_g{idx}",
#                                   f"vlambda_g{idx}+0.5*dt*flambda_g{idx}/m_lambda")
#             self.addComputeGlobal(f"vcharge_g{idx}",
#                                   f"vcharge_g{idx}+0.5*dt*fcharge_g{idx}/m_charge")
#             self.addComputeGlobal(f"vsigma_g{idx}",
#                                   "vsigma_g{idx}+0.5*dt*fsigma_g{idx}/m_sigma")
#             self.addComputeGlobal(f"vepsilon_g{idx}",
#                                   f"vepsilon_g{idx}+0.5*dt*fepsilon_g{idx}/m_epsilon")

#             # use velocities to update dynamical variables
#             self.addComputeGlobal(f"lambda_g{idx}",
#                                   f"max(min(lambda_g{idx}+dt*vlambda_g{idx},{lambda_max}),{lambda_min})")
#             self.addComputeGlobal(f"charge_g{idx}",
#                                   f"max(min(charge_g{idx}+dt*vcharge_g{idx},{charge_max}),{charge_min})")
#             self.addComputeGlobal(f"sigma_g{idx}",
#                                   f"max(min(sigma_g{idx}+dt*vsigma_g{idx},{sigma_max}),{sigma_min})")
#             self.addComputeGlobal(f"epsilon_g{idx}",
#                                   f"max(min(epsilon_g{idx}+dt*vepsilon_g{idx},{epsilon_max}),{epsilon_min})")

#         # self.addUpdateContextState()

#         for i in range(n_ghosts):
#             # recompute forces using the new "positions" of the dynamical variables
#             self.addComputeGlobal(
#                 f"flambda_g{idx}", f"-deriv(energy, lambda_g{idx})")
#             self.addComputeGlobal(
#                 f"fcharge_g{idx}", f"-deriv(energy, charge_g{idx})")
#             self.addComputeGlobal(
#                 f"fsigma_g{idx}", f"-deriv(energy, sigma_g{idx})")
#             self.addComputeGlobal(
#                 f"fepsilon_g{idx}", f"-deriv(energy, epsilon_g{idx})")

#             self.addComputeGlobal(f"vlambda_g{idx}",
#                                   f"vlambda_g{idx}+0.5*dt*flambda_g{idx}/m_lambda")
#             self.addComputeGlobal(f"vcharge_g{idx}",
#                                   f"vcharge_g{idx}+0.5*dt*fcharge_g{idx}/m_charge")
#             self.addComputeGlobal(f"vsigma_g{idx}",
#                                   f"vsigma_g{idx}+0.5*dt*fsigma_g{idx}/m_sigma")
#             self.addComputeGlobal(f"vepsilon_g{idx}",
#                                   f"vepsilon_g{idx}+0.5*dt*fepsilon_g{idx}/m_epsilon")

#         self.addUpdateContextState()
