import torch as th
import os, socket
from cm import dist_util
from mpi4py import MPI
import blobfile as bf
import torch.distributed as dist

GPUS_PER_NODE = 2

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    print(f"CUDA_VISIBLE_DEVICES: {MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}", flush = True)
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    print(f"backend: {backend}", flush = True)
    print(f"hostname: {hostname}", flush = True)

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}", flush = True)
    print(f"RANK: {os.environ['RANK']}", flush = True)
    print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}", flush = True)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    print(f"MASTER_PORT: {str(port)}", flush = True)

    dist.init_process_group(backend=backend, init_method="env://")

def main():

    setup_dist()

    path = '../../weights/celebahq/training_ckpts/openai-2024-06-10-20-18-10-701815/model040000.pt'

    if dist.get_rank() == 0:

        chunk_size = 2**30  # MPI has a relatively small size limit
        if MPI.COMM_WORLD.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
            num_chunks = len(data) // chunk_size

            if len(data) % chunk_size:
                num_chunks += 1
            MPI.COMM_WORLD.bcast(num_chunks)

            for i in range(0, len(data), chunk_size):
                print(f'rank 0: broadcasting {i}:{i + chunk_size}', flush = True)
                MPI.COMM_WORLD.bcast(data[i : i + chunk_size])

        else:
            num_chunks = MPI.COMM_WORLD.bcast(None)
            data = bytes()
            for _ in range(num_chunks):
                print(f'rank 1: broadcasting', flush = True)
                data += MPI.COMM_WORLD.bcast(None)

        print('supposedly loading data', flush = True)


if __name__ == "__main__":
    main()