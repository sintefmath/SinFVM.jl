#
import sys
print(sys.path)
import numpy as np
# from matplotlib import animation, rc
# from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec

import os
import datetime
import sys
# import pycuda.driver as cuda
from tqdm import tqdm

# Import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common, WindStress
# Import initial condition and bathymetry generating functions:
from SWESimulators.BathymetryAndICs import *


class BumpSimulation:

    def __init__(self):
        print(sys.version)
        # zself.gpu_ctx = Common.CUDAContext()  # cuda.Context.get_current()
        # get_ipython().run_line_magic('cuda_context_handler', 'gpu_ctx')

        # Set initial conditions common to all simulators
        self.sim_args = {
            "gpu_ctx": None,
            "nx": 256, "ny": 256,
            "dx": 200.0, "dy": 200.0,
            "dt": 1,
            "g": 9.81,
            "f": 0.0012,
            "coriolis_beta": 1.0e-6,
            "r": 0.0
        }

        ghosts = np.array([2, 2, 2, 2])  # north, east, south, west
        dataShape = (self.sim_args["ny"] + ghosts[0]+ghosts[2],
                     self.sim_args["nx"] + ghosts[1]+ghosts[3])

        self.H = np.ones(
            (dataShape[0]+1, dataShape[1]+1), dtype=np.float32) * 60.0
        self.eta0 = np.zeros(dataShape, dtype=np.float32)
        self.u0 = np.zeros(dataShape, dtype=np.float32)
        self.v0 = np.zeros(dataShape, dtype=np.float32)

        # Create bump in to lower left of domain for testing
        addCentralBump(self.eta0, self.sim_args["nx"], self.sim_args["ny"],
                       self.sim_args["dx"], self.sim_args["dy"], ghosts)

        # Initialize simulator
        ctcs_args = {"H": self.H, "eta0": self.eta0,
                     "hu0": self.u0, "hv0": self.v0, "use_rk2": True}
        # self.sim = KP07.KP07(**ctcs_args, **self.sim_args, write_netcdf=False)

    def step(self, steps, stepsize):
        for i in range(steps):
            self.sim.step(stepsize)
        return self.sim.download(interior_domain_only=True)

# simulation = BumpSimulation()
# simulation.step(1, 10.0)