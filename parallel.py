# -*- coding: utf-8 -*-

import multiprocessing as mp
#from multiprocessing import Process, Queue, Pool
import os, time


def task(pid, x):
    print('task-'+str(pid), os.getpid())
    t=time.time()
    res = 0
    for i in range(x):
        res += i
    t=time.time()-t
    print('task-'+str(pid), os.getpid(), t)
    return res

class parallel_run():
    def __init__(self, n):
        self.n = n
        self.pool = mp.Pool(n)
        
    def compute(self, v):
        return v*v
        
    def run(self):
        out = []
        for i in range(4):
            o = self.pool.map(lambda x: self.compute, range(5))
            out.push(o)
        return out


if __name__ == "__main__":
    p1=mp.Process(target=task, args=(1, 10000000,))
    p2=mp.Process(target=task, args=(2, 10000000,))
    t=time.time()
    p1.start()
    p2.start()
    print("xxxx")
    p1.join()
    p2.join()
    t=time.time()-t
    print('main',os.getpid(), t)
    