import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedContext:
    def __init__(self):
        self.is_distributed = int(os.environ.get("RANK", -1)) != -1

        if self.is_distributed:
            self._setup_distributed()
        else:
            self._setup_single_gpu()

    def _setup_distributed(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.master_process = self.rank == 0

        self.device = torch.device(
            f"cuda:{self.local_rank}"
        )

        torch.cuda.set_device(self.device)

        dist.init_process_group(
            backend="nccl"
        )

    def _setup_single_gpu(self):
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.master_process = True

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def wrap_model(self, model):
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
            )

        return model
    
    def unwrap_model(self, model):
        if self.is_distributed:
            return model.module

        return model
    
    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def cleanup(self):
        if self.is_distributed:
            dist.destroy_process_group()