import os
import shutil
import torch
from omegaconf import OmegaConf
from multiprocessing.pool import ThreadPool
from torch.distributed.tensor import DTensor
from safetensors.torch import load_file, save_file


def _chunk(tensor, num_chunks):
    chunks = torch.chunk(tensor, num_chunks, dim=0)
    chunks += tuple(
        torch.empty(
            *[0, *tensor.shape[1:]], dtype=tensor.dtype, device=tensor.device
        )
        for _ in range(num_chunks - len(chunks))
    )
    return chunks


def _load_distributed_checkpoint(
    path, num_threads=0, map_location=None, weights_only=False
):
    state_dict_shard_paths = [
        os.path.join(path, file_name) for file_name in os.listdir(path)
        if file_name.endswith('.pt')
    ]

    def distributed_load(thread_num):
        thread_state_dict_shards = []
        for state_dict_shard_path in state_dict_shard_paths[thread_num::max(1, num_threads)]:
            state_dict_shard_number = int(
                state_dict_shard_path.split('/')[-1].replace('.pt', '')
            )
            state_dict_shard = load(
                state_dict_shard_path,
                map_location=map_location, weights_only=weights_only
            )
            thread_state_dict_shards.append(
                (state_dict_shard_number, state_dict_shard)
            )
        return thread_state_dict_shards

    state_dict_shards = []
    if num_threads == 0:
        state_dict_shards += distributed_load(0)
    else:
        with ThreadPool(processes=num_threads) as pool:
            for thread_state_dict_shards in pool.map(
                distributed_load, list(range(num_threads))
            ):
                state_dict_shards += thread_state_dict_shards
    state_dict_shards.sort()
    state_dict_shards = list(map(lambda x: x[1], state_dict_shards))
    return state_dict_shards


def _save_distributed_checkpoint(state_dict_shards, path, num_threads=0):
    state_dict_shards = list(enumerate(state_dict_shards))

    def distributed_save(thread_num):
        for state_dict_shard_number, state_dict_shard in state_dict_shards[thread_num::max(1, num_threads)]:
            state_dict_shard_path = os.path.join(
                path, f'{state_dict_shard_number}.pt'
            )
            save(state_dict_shard, state_dict_shard_path)

    if num_threads == 0:
        distributed_save(0)
    else:
        with ThreadPool(processes=num_threads) as pool:
            pool.map(distributed_save, list(range(num_threads)))


def load(path, map_location=None, weights_only=False):
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def save(data, path):
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    torch.save(data, path)


def gather_distributed_model_state_dict(
    path, num_threads=0, map_location=None, weights_only=False
):
    state_dict_shards = _load_distributed_checkpoint(
        path, num_threads, map_location, weights_only
    )

    state_dict = {}
    for state_dict_shard in state_dict_shards:
        for key, value in state_dict_shard.items():
            state_dict.setdefault(key, []).append(value)

    for key, value in state_dict.items():
        state_dict[key] = torch.cat(value)
    return state_dict


def scatter_distributed_model_state_dict(
    state_dict, path, num_shards, num_threads=0
):
    state_dict_shards = [{} for _ in range(num_shards)]
    for key, value in state_dict.items():
        shards = _chunk(value, num_shards)
        for i, shard in enumerate(shards):
            state_dict_shards[i][key] = shard.clone()

    _save_distributed_checkpoint(state_dict_shards, path, num_threads)


def reshard_distributed_model_state_dict(
    source_path, target_path, num_shards, num_threads=0
):
    state_dict = gather_distributed_model_state_dict(
        source_path, num_threads
    )
    scatter_distributed_model_state_dict(
        state_dict, target_path, num_shards, num_threads
    )


def load_fsdp_base_model(model, path):
    local_rank = int(os.environ["LOCAL_RANK"])
    if path.endswith('.pt'):
        state_dict = load(path)
    elif path.endswith('.safetensors'):
        state_dict = load_file(path)
    for key, value in model.state_dict().items():
        if any(subkey in key for subkey in ["lora_A", "lora_B"]):
            continue
        key_base = key.replace("base_model.model.", "").replace(".base_layer", "")
        state_dict[key] = DTensor.from_local(
            state_dict.pop(key_base).to(local_rank),
            device_mesh=value.device_mesh,
            placements=value.placements,
            shape=value.shape,
            stride=value.stride()
        )
    model.load_state_dict(state_dict, strict=False)


def load_fsdp_adapter_model(model, path):
    local_rank = int(os.environ["LOCAL_RANK"])
    if path.endswith('.pt'):
        state_dict = load(path)
    elif path.endswith('.safetensors'):
        state_dict = load_file(path)
    for key, value in model.state_dict().items():
        if not any(subkey in key for subkey in ["lora_A", "lora_B"]):
            continue
        state_dict[key] = DTensor.from_local(
            state_dict[key].to(local_rank),
            device_mesh=value.device_mesh,
            placements=value.placements,
            shape=value.shape,
            stride=value.stride()
        )
    model.load_state_dict(state_dict, strict=False)


def save_fsdp_adapter_model(
    model, path,
    nan_check=True,
    raise_exception=False,
    local_shards=None
):
    torch.cuda.empty_cache()
    rank = int(os.environ["RANK"])
    world_size = local_shards if local_shards > 0 else int(os.environ["WORLD_SIZE"])
    state_dict = model.state_dict()
    local_state_dict = {}
    nan_keys = []

    for key, value in state_dict.items():
        if not any(subkey in key for subkey in ["lora_A", "lora_B"]):
            continue
        local_value = value.to_local()
        if nan_check:
            has_nan = torch.isnan(local_value).any()
            if has_nan:
                nan_keys.append(key)
        local_state_dict[key] = local_value.cpu()

    if nan_check:
        object_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            object_list, nan_keys,
            group=torch.distributed.new_group(ranks=list(range(world_size)))
        )
        nan_keys = set(sum(object_list, []))
        if len(nan_keys) > 0:
            msg = f"Model weights became NaN in {nan_keys}. Checkpoint won't be saved."
            if raise_exception:
                raise RuntimeError(msg)
            elif rank == 0:
                print(msg)
            return False

    save(local_state_dict, path)
    return True


def save_adapter_model(
        model,
        path,
        nan_check=True,
        raise_exception=False,
        local_shards=None,
        trigger=None,
        del_fsdp_dir=True
):
    rank = int(os.environ["RANK"])
    temp_dir = os.path.join(path, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    save_fsdp_adapter_model(
        model, os.path.join(temp_dir, f"{rank}.pt"), 
        nan_check, raise_exception, local_shards
    )
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    if rank == 0:
        state_dict = gather_distributed_model_state_dict(temp_dir)
        for key in state_dict:
            if "time_embeddings" not in key and "modulation" not in key:
                state_dict[key] = state_dict[key].to(dtype=torch.bfloat16)
        save_file(
            state_dict,
            os.path.join(path, "model.safetensors"),
            {"trigger": trigger} if trigger is not None else {}
        )
        if del_fsdp_dir:
            shutil.rmtree(temp_dir)
    torch.distributed.barrier()
    return True


def load_fsdp_optimizer(optimizer, path):
    state_dict = load(path)
    optimizer.load_state_dict(state_dict)


def save_fsdp_optimizer(optimizer, path):
    save(optimizer.state_dict(), path)


def load_trainer_adapter_checkpoint(conf, dit, optimizer, lr_scheduler):
    rank = int(os.environ["RANK"])
    local_shards = conf.trainer.device_meshs.dit.fsdp_mesh
    if local_shards > 0:
        rank = rank % local_shards

    save_dir = conf.trainer.checkpoint_path
    model_path = os.path.join(save_dir, f"model/{rank}.pt")
    optimizer_path = os.path.join(save_dir, f"optimizer/{rank}.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")

    load_fsdp_adapter_model(dit, model_path)
    load_fsdp_optimizer(optimizer, optimizer_path)
    lr_scheduler.load_state_dict(load(scheduler_path, weights_only=False))
    return dit, optimizer, lr_scheduler


def save_trainer_adapter_checkpoint(conf, dit, optimizer, lr_scheduler):
    rank = int(os.environ["RANK"])
    local_shards = conf.trainer.device_meshs.dit.fsdp_mesh

    save_dir = conf.trainer.checkpoint_path
    model_path = os.path.join(save_dir, "model/")
    optimizer_path = os.path.join(save_dir, "optimizer/")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")

    # NOTE: is_saved is always same on all ranks
    is_saved = save_adapter_model(dit, model_path, local_shards=local_shards)
    if is_saved:
        save_fsdp_optimizer(
            optimizer, os.path.join(optimizer_path, f"{rank}.pt")
        )
        if rank == 0:
            save(lr_scheduler.state_dict(), scheduler_path)
            OmegaConf.save(conf, conf.common.conf_path)
    else:
        if rank == 0:
            print("Skip saving optimizer, loader and scheduler state due to error in the first save_fsdp_model")
