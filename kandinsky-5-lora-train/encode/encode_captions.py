import os
import sys

import glob
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(
    0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../kandinsky5")
)

from kandinsky.models.text_embedders import get_text_embedder


def main(args):
    conf = OmegaConf.load(args.config)
    text_embedder = get_text_embedder(
        conf.text_embedder, device=f"cuda:{args.device}"
    )
    caption_paths = glob.glob(
        os.path.join(args.images_captions_dir, "*.txt")
    )
    os.makedirs(
        args.save_text_embeds_dir, exist_ok=True
    )
    for caption_path in tqdm(caption_paths, desc="Encoding captions"):
        save_name = os.path.splitext(os.path.basename(caption_path))[0]
        save_path = os.path.join(args.save_text_embeds_dir, f"{save_name}.pt")
        with open(caption_path, "r") as file:
            caption = file.read().strip()
        text_embed, text_cu_seqlen, attention_mask \
            = text_embedder.encode([caption])
        out_dict = text_embed
        out_dict['text_cu_seqlen'] = text_cu_seqlen
        out_dict['attention_mask'] = attention_mask

        torch.save(out_dict, save_path)

    # null embed
    text_embed, text_cu_seqlen, attention_mask = text_embedder.encode([''])
    out_dict = text_embed
    out_dict['text_cu_seqlen'] = text_cu_seqlen
    out_dict['attention_mask'] = attention_mask
    save_path = os.path.join(args.save_text_embeds_dir, "null.pt")
    torch.save(out_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_captions_dir", type=str, required=True)
    parser.add_argument("--save_text_embeds_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    main(args)
