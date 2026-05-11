"""MPI wrapper: generates the matplotlib font cache once per node
before the simulation script imports matplotlib.

Usage: mpiexec -np N python mpl_wrapper.py <script.py> [args...]
"""
import sys
import os
from mpi4py import MPI

# Point matplotlib cache to node-local fast storage
mpl_config = os.path.join(os.environ["PBS_JOBFS"], "matplotlib")
os.environ["MPLCONFIGDIR"] = mpl_config

# Split into node-local communicators
node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

# Local rank 0 on each node generates the font cache
if node_comm.rank == 0:
    os.makedirs(mpl_config, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# Everyone waits for the cache to be ready
node_comm.Barrier()
node_comm.Free()

# Run the actual simulation script
import runpy
sys.argv[:] = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
