import sys

sys.path.append("/work/lechevaa/PycharmProjects/IMPES/Yads")

import os
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = ["coucou", 1, [1, 2], {"hi": 0}]
else:
    data = None
data = comm.scatter(data, root=0)
print(rank, data)
