# -*- coding: utf-8 -*-
from mpi4py import MPI


class Network:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def send(self, data, dst, tag=0):
        self.comm.send(data, dest=dst, tag=tag)

    def isend(self, data, dst, tag=0):
        self.comm.isend(data, dest=dst, tag=tag)

    def recv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        s = MPI.Status()
        d = self.comm.recv(source=source, tag=tag, status=s)
        return s.Get_source(), s.Get_tag(), d

    def irecv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        s = MPI.Status()
        if self.comm.iprobe(source=source, tag=tag, status=s):
            return self.recv(s.Get_source(), s.Get_tag())
        else:
            return None


    def broadcast(self, data, root):
        '''
        Send data to all workers.
        '''
        r = self.comm.bcast(data, root)
        return r

    def scatter(self, data, root):
        '''
        Send data[i] to worker i.
        <data> is a list of length self.size.
        Return the received value.
        Pair with gather.
        '''
        r = self.comm.scatter(data, root)
        return r

    def gather(self, data, root):
        '''
        Receive data from all workers.
        <data> is a value.
        Return the received list of length self.size.
        Pair with scatter.
        '''
        r = self.comm.gather(data, root)
        return r

    def alltoall(self, data):
        '''
        Scatter data to all other workers.
        And then gather received values of all workers (including itself) as a list
        Equivalent to scatter + gather.
        '''
        r = self.comm.alltoall([data for _ in range(self.size)])
        return r

    def alltoallw(self, data):
        '''
        <data> is a list of length self.size
        It sends data[i] to worker i.
        Return a list of received values from all workers (including itself)
        '''
        r = self.comm.alltoall(data)
        return r

    def barrier(self):
        self.comm.barrier()

