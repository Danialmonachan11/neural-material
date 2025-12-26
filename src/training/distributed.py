"""
Distributed training utilities for multi-GPU and cluster training.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import Optional


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> tuple:
    """
    Initialize distributed training.

    Args:
        backend: distributed backend ('nccl', 'gloo')
        init_method: initialization method URL
    Returns:
        (rank, world_size) tuple
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")

    # Get rank and world size from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Initialize process group
        if init_method is None:
            init_method = "env://"

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        # Set device
        torch.cuda.set_device(local_rank)

        print(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        print("Running in single-GPU mode")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized():
        dist.barrier()


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: model to wrap
        device_id: GPU device ID
        find_unused_parameters: whether to find unused parameters
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        return model

    model = model.to(device_id)
    model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )

    return model


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dict across all processes.

    Args:
        input_dict: dict of tensors to reduce
        average: whether to average (vs sum)
    Returns:
        reduced dict
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


class DistributedSampler(torch.utils.data.Sampler):
    """
    Distributed sampler for data loading.

    Ensures each process gets different data samples.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # Deterministic shuffling
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
