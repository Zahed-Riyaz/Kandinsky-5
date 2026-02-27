import os
import glob
import json
import torch
import numpy as np
from collections import Counter
from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
    RandomSampler
)

from .utils import GroupedBatchSampler, LoopWrapper


class LoraDataset(Dataset):
    def __init__(self, conf):
        self.conf = conf
        latents_paths = set(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(conf.latents_dir, "*.pt"))
        )
        text_embeds_paths = set(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(conf.text_embeds_dir, "*.pt"))
        )

        common = sorted(
            latents_paths & text_embeds_paths, key=lambda x: (len(x), x)
        )

        if not common:
            raise RuntimeError(
                f"No matching <idx>.pt files in both dirs:\n"
                f"  latents_dir={conf.latents_dir}\n"
                f"  text_embeds_dir={conf.text_embeds_dir}"
            )

        self.indices = list(common)
        self.null_text = torch.load(conf.uncond_embed, map_location="cpu")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        text = ''
        if hasattr(self.conf, 'text_dir'):
            text_path = os.path.join(self.conf.text_dir, f"{idx}.txt")
            with open(text_path, "r") as f:
                text = f.read()
        latent_path = os.path.join(self.conf.latents_dir, f"{idx}.pt")
        text_embed_path = os.path.join(self.conf.text_embeds_dir, f"{idx}.pt")
        text_embed_dict = torch.load(text_embed_path, map_location="cpu")

        # Unconditional sample
        null_text = np.random.random() < self.conf.uncond_prob
        if null_text:
            text_embed_dict.update(**self.null_text)
            text = ''

        # Add batch dim
        for k in text_embed_dict.keys():
            if (
                (
                    k == 'text_embeds' and 
                    len(text_embed_dict[k].shape) == 2
                ) or
                (
                    k in ['pooled_embed', 'attention_mask'] and
                    len(text_embed_dict[k].shape) == 1
                )
            ):
                text_embed_dict[k] = text_embed_dict[k].unsqueeze(0)

        sample = {
            "text": text,
            **text_embed_dict,
            "visual": torch.load(latent_path, map_location="cpu"),
        }
        # for k in sample:
        #     if k == 'text':
        #         print(sample[k])
        #     else:
        #         print(f'Dataset {k}: {sample[k].shape}')

        return sample


def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [sample[k] for sample in batch]
    return out


def get_dataloader(conf, rank=0, world_size=1, *args, **kwargs):
    dataset = LoraDataset(conf)
    aspect_ratios = []
    need_text_encoder = False
    for x in dataset:
        aspect_ratios.append((x["visual"].shape[-2] * 8, x["visual"].shape[-3] * 8))
        if 'text_embeds' not in x.keys(): need_text_encoder = True
    group_indices = [f"r{width}_{height}" for width, height in aspect_ratios]
    if rank == 0:
        counts = Counter(group_indices)
        print("Aspect ratios distribution:")
        print(json.dumps(counts, indent=2))

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    else:
        sampler = RandomSampler(dataset,)
    batch_sampler = GroupedBatchSampler(
        sampler,
        group_indices,
        batch_size=conf.max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )
    dataloader = LoopWrapper(dataloader, batch_sampler)
    dataloader.need_text_encoder = lambda: need_text_encoder
    return dataloader
