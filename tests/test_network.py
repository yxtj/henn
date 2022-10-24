# -*- coding: utf-8 -*-

from network import Network

def main():
    net = Network()
    rank = net.rank
    size = net.size
    print(f"rank={net.rank}, size={net.size}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    num=100 + rank
    net.send(num, (rank+1)%size)
    s,t,d = net.recv()
    print(f"rank={net.rank}, send:{num}, recv:{d}, source:{s}, tag:{t}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    num=200 + rank
    net.send(num, (rank+2)%size)
    s,t,d = net.recv()
    print(f"rank={net.rank}, send:{num}, recv:{d}, source:{s}, tag:{t}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    r = net.broadcast(301, 0)
    print(f"rank={net.rank}, broadcast from 0 result: {r}", flush=True)
    r = net.broadcast(402, 1)
    print(f"rank={net.rank}, broadcast from 1 result: {r}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    r = net.scatter([500+i for i in range(size)], 0)
    print(f"rank={net.rank}, scatter from 0 result: {r}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    num = 600+rank
    r = net.alltoall(num)
    print(f"rank={net.rank}, all-to-all send: {num} result: {r}", flush=True)

    net.barrier()
    if rank == 0:
        print("--------", flush=True)
    data = [700+rank*10+i for i in range(size)]
    r = net.alltoallw(data)
    print(f"rank={net.rank}, all-to-all-w send: {data} result: {r}", flush=True)


if __name__ == '__main__':
    main()
