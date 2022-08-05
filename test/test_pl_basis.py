# -*- coding: utf-8 -*-

import multiprocessing as mp
#from multiprocessing import Process, Queue, Pool
from mpi4py import MPI

import os, time
import numpy as np


def task(pid, x, return_dict):
    print('task-'+str(pid), os.getpid())
    t=time.time()
    res = 0
    for i in range(x):
        res += i
    t=time.time()-t
    print('task-'+str(pid), os.getpid(), t)
    #return res
    return_dict[pid] = res


class parallel_obj():
    def __init__(self, n):
        self.n = n
        self.pool = mp.Pool(n)
        
    def compute(self, v):
        return v*v
        
    def run(self):
        out = []
        out = self.pool.map(self.compute, range(5))
        #for i in range(4):
        #    o = self.pool.map(self.compute, range(5))
        #    out.push(o)
        return out

def test_parallel():
    manager = mp.Manager()
    return_dict = manager.dict()
    
    p1=mp.Process(target=task, args=(1, 10000000, return_dict))
    p2=mp.Process(target=task, args=(2, 10000000, return_dict))
    print('main',os.getpid())
    t=time.time()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    t=time.time()-t
    print('main',os.getpid(), t)
    print(return_dict)


def mpi_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"size={size}, rank={rank}")
    
    if rank == 0:
        data = range(10)
        comm.send(data, dest=1, tag=11)
        comm.isend(data, dest=1, tag=11)
        print("process {} immediate send {}...".format(rank, data))
    else:
        data = comm.recv(source=0, tag=11)
        print("process {} recv {}...".format(rank, data))
        data = comm.recv(source=0, tag=11)
        print("process {} recv {}...".format(rank, data))

def mpi_bcast():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num = np.random.randint(0,100)
    print(f"rank={rank}, size={size}, num={num}")
    r = comm.bcast((rank, num), 0)
    print(f"rank-{rank}, bcast, num={num}, r={r}")
    time.sleep(1)
    print("group 2:")
    print(f"  rank={rank}, size={size}, num={num}")
    r = comm.bcast((rank, num), 1)
    print(f"  rank-{rank}, bcast, num={num}, r={r}")


def mpi_alltoall():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num = np.random.randint(1,100)
    buffer = [num for i in range(size)]
    # buffer[i] stores the data sent for rank-i
    print(f"rank={rank}, size={size}, num={num}")
    r = comm.alltoall(buffer)
    print(f"rank={rank}, buffer={buffer}, r={r}")
        


if __name__ == "__main__":
    #test_parallel()
    
    #a=parallel_obj(3)
    #print(a.run())
    
    #mpi_test()
    #mpi_bcast()
    mpi_alltoall()
    
    
    