import os
import sys
import glob
import torch
import argparse
import gc
from PIL import Image
from tqdm import tqdm
import numpy as np
from math import sqrt
from omegaconf import OmegaConf
import torchvision.transforms.functional as F

# Suppress PIL decompression bomb errors for huge LAION images
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(
    0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "../kandinsky5")
)
from kandinsky.models.vae import build_vae

sys.path.insert(
    0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
)
from train.datasets.constants import RESOLUTIONS

def encode_image(image_path, vae, conf):
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # Target area
        max_area = 1024 * 1024 if conf.vae.name == "flux" else 512 * 512
        
        # Calculate scale
        k = sqrt(max_area / (w * h))
        
        # FORCED MULTIPLE OF 16 (The Fix)
        # We use (dimension * k // 16) * 16 to ensure it is always divisible by 16
        new_w = int(max(round(w * k / 16), 1) * 16)
        new_h = int(max(round(h * k / 16), 1) * 16)
        
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)
        
        # ... rest of the encoding logic ...
        pixel_values = F.pil_to_tensor(image).unsqueeze(0) / 127.5 - 1.0
        
        with torch.no_grad():
            if conf.vae.name == "hunyuan":
                pixel_values = pixel_values.to(device=vae.device, dtype=torch.float16)
                # Video models expect [B, C, T, H, W]. For images, T=1.
                pixel_values = pixel_values.transpose(0, 1).unsqueeze(0) 
                latent = vae.encode(pixel_values).latent_dist.sample()
                # Save as [T, H, W, C]
                latent = latent.squeeze(0).permute(1, 2, 3, 0).cpu()
                return latent
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main(args):
    conf = OmegaConf.load(args.config)
    print(f"Loading VAE: {conf.vae.name}...")
    vae = build_vae(conf.vae)
    vae = vae.eval().to(args.device)

    # Robust globbing for all image types
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG"]
    media = []
    for ext in exts:
        media.extend(glob.glob(os.path.join(args.images_captions_dir, ext)))
    
    # Add videos
    media.extend(glob.glob(os.path.join(args.images_captions_dir, "*.mp4")))

    print(f"Found {len(media)} files to process.")
    os.makedirs(args.save_latents_dir, exist_ok=True)

    for path in tqdm(media, desc="Encoding"):
        # Skip if already exists to allow resuming
        save_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(args.save_latents_dir, f"{save_name}.pt")
        if os.path.exists(save_path):
            continue

        if path.endswith(".mp4"):
            # If you have video issues, you can disable video here or use a similar resize fix
            continue 
        else:
            latent = encode_image(path, vae, conf)

        if latent is not None:
            torch.save(latent, save_path)
            
        # FORCE CLEANUP after every image
        del latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_captions_dir", type=str, required=True)
    parser.add_argument("--save_latents_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    main(args)