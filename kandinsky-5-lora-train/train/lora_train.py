import os
import sys
import gc
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed._composable import checkpoint
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.optimization import get_constant_schedule_with_warmup
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars

from .checkpoint import (
    load_fsdp_base_model,
    load_trainer_adapter_checkpoint,
    save_trainer_adapter_checkpoint,
    scatter_distributed_model_state_dict,
)

from .train_utils import (
    add_visual_cond,
    DeviceMeshs,
    get_rope_pos,
    get_autoreg_timesteps,
    get_sparse_params,
    preprocess_batch,
)

from .logger import ModelLogger, AllRankTensorBoardLogger
from .utils import prepare_folders
from .datasets.lora_dataloader import get_dataloader

sys.path.insert(
    0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../kandinsky5")
)

from kandinsky.models.vae import build_vae
from kandinsky.models.dit import TransformerDecoderBlock, get_dit
from kandinsky.models.text_embedders import get_text_embedder

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def train(
    conf, dit, optimizer, lr_scheduler, train_dataloader, vae, logger,
    text_embedder=None
):
    rank, local_rank, world_size \
        = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if getattr(conf.trainer, 'checkpoint_path', None) is not None:
        torch.distributed.barrier()
        dit, optimizer, lr_scheduler = load_trainer_adapter_checkpoint(
            conf, dit, optimizer, lr_scheduler
        )
        torch.cuda.empty_cache()

    step = lr_scheduler.state_dict()['last_epoch']

    step_times = []
    with train_dataloader:
        for batch in train_dataloader:
            start_time = time.perf_counter()

            batch_embeds, cu_seqlens, attention_mask  \
                = preprocess_batch(dit, batch, local_rank, text_embedder=text_embedder)
            visual_rope_pos, text_rope_pos = get_rope_pos(dit, batch_embeds, cu_seqlens)

            latent_visual = batch_embeds["visual"][..., :dit.in_visual_dim] * vae.config.scaling_factor
            text_embeds, pooled_text_embed = batch_embeds["text_embeds"], batch_embeds["pooled_embed"]
            visual_cu_seqlens, text_cu_seqlens = cu_seqlens["visual"], cu_seqlens["text_embeds"]
            sparse_params = get_sparse_params(dit, latent_visual, cu_seqlens["visual_rope"])

            if dit.attention.chunk:
                timesteps = get_autoreg_timesteps(dit, cu_seqlens["visual_rope"])
            else:
                timesteps = torch.sigmoid(torch.randn(visual_cu_seqlens.shape[0] - 1, device=local_rank))
            scheduler_scale = conf.trainer.params.scheduler_scale
            timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
            sampling_timesteps = timesteps.repeat_interleave(torch.diff(visual_cu_seqlens)).reshape(-1, *((1,) * len(latent_visual.shape[1:])))

            noise = torch.randn_like(latent_visual)
            x_t = (1.0 - sampling_timesteps) * latent_visual + sampling_timesteps * noise
            velocity = noise - latent_visual

            if hasattr(conf.trainer.params, 'visual_cond_prob') and dit.visual_cond:
                visual_cond_prob = conf.trainer.params.visual_cond_prob
                x_t = add_visual_cond(x_t, latent_visual, cu_seqlens["visual_rope"], visual_cond_prob)

            if dit.instruct_type == 'channel':
                instruct_visual = batch_embeds["visual"][..., dit.in_visual_dim: -1] * vae.config.scaling_factor
                instruct_mask = batch_embeds["visual"][..., -1:]
                x_t = torch.cat([x_t, instruct_visual, instruct_mask], -1)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                pred_velocity = dit(
                    x_t, text_embeds, pooled_text_embed, 1000 * timesteps,
                    visual_rope_pos, text_rope_pos,
                    scale_factor=conf.trainer.params.scale_factor,
                    sparse_params=sparse_params,
                    attention_mask=attention_mask,
                )
                loss = F.mse_loss(pred_velocity, velocity)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dit.parameters(), max_norm=conf.optimizer.max_norm, norm_type=2.0
            )
            optimizer.step()
            lr_scheduler.step()

            step = lr_scheduler.state_dict()['last_epoch']

            step_time = torch.tensor(
                time.perf_counter() - start_time, device=local_rank
            )
            dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
            step_times.append(step_time.cpu().item())
            if step % conf.logger.log_interval == 0:
                log_loss = loss.detach() / world_size
                dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                step_time = sum(step_times) / len(step_times)
                step_times = []

                log_dict = {
                    "train_status/train_loss": log_loss.cpu().item(),
                    "train_status/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                # log_dict.update(ModelLogger.log_model_status(dit))
                log_dict.update(ModelLogger.log_system_status(step_time))
                metrics = convert_tensors_to_scalars(log_dict)
                if getattr(conf, 'debug', False) and getattr(conf.debug, 'tb_all_rank', False):
                    logger.log_metrics_all(metrics=metrics, step=step)
                else:
                    logger.log_metrics(metrics=metrics, step=step)

                if rank == 0:
                    print(f"Step {step} | Loss: {log_loss:.4f} | Step Time {step_time:.4f} s")
                gc.collect()
                torch.cuda.empty_cache()

            for checkpoint_name, checkpoint_type in zip([f"step_{step}", "last"], ["regular_save_interval", "last_save_interval"]):
                if step % conf.checkpoint[checkpoint_type] == 0:
                    torch.distributed.barrier()
                    local_shards = conf.trainer.device_meshs.dit.fsdp_mesh
                    if (local_shards < 0 or rank < local_shards):
                        conf.trainer.checkpoint_path = os.path.join(
                            conf.checkpoint.root_dir, checkpoint_name
                        )
                        save_trainer_adapter_checkpoint(
                            conf, dit, optimizer, lr_scheduler
                        )
                    torch.cuda.empty_cache()


def run(conf):
    torch.set_float32_matmul_precision("medium")
    world_size = conf.trainer.params.num_nodes * conf.trainer.params.devices

    with DeviceMeshs(conf.trainer.device_meshs, world_size) as device_meshs:
        rank, local_rank, world_size \
            = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16
        )

        if rank == 0:
            prepare_folders(conf)
            print("CONFIG PATH:", conf.common.conf_path)

        tb_logger = AllRankTensorBoardLogger(
            **conf.logger.tensorboard, sub_dir=f"rank_{rank:04d}"
        )
        train_dataloader = get_dataloader(
            conf.data, rank, world_size,
        )
        with torch.device("cpu"):
            dit = get_dit(conf.dit.dit_params)
            dit.attention = conf.dit.attention
            lora_config = LoraConfig(**conf.trainer.lora)
            dit = get_peft_model(dit, lora_config)

        for module in dit.modules():
            if isinstance(module, TransformerDecoderBlock):
                checkpoint(module)
                fully_shard(
                    module, mesh=device_meshs.dit['model'], mp_policy=mp_policy
                )
        fully_shard(dit, mesh=device_meshs.dit['model'], mp_policy=mp_policy)

        if (
            getattr(conf.dit, 'checkpoint_path', None) is not None and
            getattr(conf.trainer, 'checkpoint_path', None) is None
        ):
            if world_size > 1:
                local_shards = conf.trainer.device_meshs.dit.fsdp_mesh
                if local_shards < 0:
                    local_rank = rank
                else:
                    local_rank = rank % local_shards

                if os.path.isfile(conf.dit.checkpoint_path): 
                    state_dict_path = conf.dit.checkpoint_path

                    if state_dict_path.endswith('.pt'):
                        state_dict = torch.load(
                            state_dict_path, map_location='cpu'
                        )

                    elif state_dict_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(state_dict_path)

                    target_path = os.path.join(
                        os.path.dirname(state_dict_path),
                        f'reshard_{world_size}gpus'
                    )
                    os.makedirs(target_path, exist_ok=True)
                    if local_rank == 0:
                        scatter_distributed_model_state_dict(
                            state_dict, target_path, world_size, os.cpu_count()
                        )
                    torch.distributed.barrier()
                    model_path = os.path.join(target_path, f"{local_rank}.pt")

                elif os.path.isdir(conf.dit.checkpoint_path):
                    model_path = os.path.join(
                        conf.dit.checkpoint_path, f"{local_rank}.pt"
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Checkpoint file not found: {model_path}.\
You should provide dir for fsdp sharded checkpoint or path to non-shareded checkpoint")       
                else:
                    raise FileNotFoundError(
                        f"Checkpoint file not found: {conf.dit.checkpoint_path}.\
You should provide dir for fsdp sharded checkpoint or path to non-shareded checkpoint")
            else:
                model_path = conf.dit.checkpoint_path
            if rank == 0:
                print(f"Loading checkpoint from: {model_path}")
            load_fsdp_base_model(dit, model_path)

        if rank == 0:
            conf.model_size \
                = f"{sum(p.numel() for p in dit.parameters()) / 1e9:.1f}B"
            print("MODEL SIZE:", conf.model_size)

        optimizer = torch.optim.AdamW(
            dit.parameters(), **conf.optimizer.params
        )
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, **conf.scheduler.params
        )

        torch.distributed.barrier()
        vae = build_vae(conf.vae)
        vae = vae.eval()

        if train_dataloader.need_text_encoder():
            text_embedder = get_text_embedder(
                conf.text_embedder, device=f'cuda:{local_rank}',
                quantized_qwen=False, text_token_padding=True
            )
        else:
            text_embedder = None

        train(
            conf,
            dit,
            optimizer,
            lr_scheduler,
            train_dataloader,
            vae,
            tb_logger,
            text_embedder=text_embedder
        )
