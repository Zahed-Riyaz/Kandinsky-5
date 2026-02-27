import os
import sys
import argparse
import torch
import gc
from datasets import load_dataset, load_dataset_builder

# --- STEP 1: ARGUMENT PARSING ---
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument("--percentage", type=float, default=None, help="Percentage of dataset to process")
custom_parser.add_argument("--limit", type=int, default=10, help="Hard count of samples to process")
custom_parser.add_argument("--dataset", type=str, default="laion/relaion2B-en-research-safe")
custom_args, remaining_argv = custom_parser.parse_known_args()

# Clean sys.argv for the official 'test' import
sys.argv = [sys.argv[0]] + remaining_argv

# --- STEP 2: INITIALIZE OFFICIAL CODE & ENV ---
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import test 

def run_percentage_inference():
    official_args = test.parse_args()
    test.disable_warnings()
    test.set_seed(official_args.seed)

    device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}
    mode = test.get_generation_mode(official_args.config)

    # 3. Initialize Pipeline
    print(f"==> Initializing Official {mode.upper()} Pipeline...")
    if mode in ["t2i", "i2i"]:
        pipe = test.get_image_pipeline(
            device_map=device_map, resolution=1024,
            conf_path=official_args.config, offload=official_args.offload,
            magcache=official_args.magcache, quantized_qwen=official_args.qwen_quantization,
            attention_engine=official_args.attention_engine, mode=mode,
        )
    else:
        pipe = test.get_video_pipeline(
            device_map=device_map, conf_path=official_args.config,
            offload=official_args.offload, magcache=official_args.magcache,
            quantized_qwen=official_args.qwen_quantization,
            attention_engine=official_args.attention_engine, mode=mode,
        )
        pipe = test.get_distributed_pipeline(pipe, official_args.tp_size, mode=mode)

    # --- STEP 4: CALCULATE LIMIT & CONFIRM ---
    print(f"==> Inspecting dataset: {custom_args.dataset}...")
    
    if custom_args.percentage is not None:
        try:
            known_sizes = {"laion/relaion2B-en-research-safe": 2322161611}
            total_samples = known_sizes.get(custom_args.dataset)
            if total_samples is None:
                builder = load_dataset_builder(custom_args.dataset)
                total_samples = builder.info.splits['train'].num_examples
            limit = max(1, int((custom_args.percentage / 100) * total_samples))
        except:
            limit = custom_args.limit
    else:
        limit = custom_args.limit

    # PRE-GENERATION SUCCESS MESSAGE
    print("\n" + "="*60)
    print(f" SUCCESS: CONFIGURATION VALIDATED")
    print(f" Mode:          {mode.upper()}")
    print(f" Dataset:       {custom_args.dataset}")
    print(f" Target Limit:  {limit} samples")
    print(f" Action:        The script will STOP automatically after {limit} items.")
    print("="*60 + "\n")

    # --- STEP 5: LOAD DATASET ---
    dataset = load_dataset(custom_args.dataset, split="train", streaming=True)
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

# --- STEP 6: INFERENCE LOOP ---
    count = 0
    for item in dataset:
        if count >= limit:
            print(f"\n==> Reached limit of {limit}. Stopping as requested.")
            break
            
        prompt = item.get("TEXT") or item.get("text") or item.get("caption")
        if not prompt: continue

        clean_name = "".join(x for x in str(prompt)[:30] if x.isalnum() or x in "._- ")
        ext = ".mp4" if "v" in mode else ".png"
        save_path = os.path.join(out_dir, f"gen_{count}_{clean_name}{ext}")

        print(f"[{count+1}/{limit}] Generating: {save_path}")

        try:
            with torch.inference_mode():
                if mode == "t2i":
                    pipe(str(prompt),
                         width=official_args.width, height=official_args.height,
                         num_steps=official_args.sample_steps,
                         guidance_weight=official_args.guidance_weight,
                         scheduler_scale=official_args.scheduler_scale,
                         expand_prompts=official_args.expand_prompt,
                         save_path=save_path, seed=official_args.seed)
                
                elif mode == "i2i":
                    # I2I: Removed width/height (uses input image size)
                    pipe(str(prompt), 
                         image=official_args.image,
                         num_steps=official_args.sample_steps, 
                         guidance_weight=official_args.guidance_weight,
                         scheduler_scale=official_args.scheduler_scale, 
                         expand_prompts=official_args.expand_prompt,
                         save_path=save_path, seed=official_args.seed)
                
                elif mode == "t2v":
                    pipe(str(prompt), 
                         time_length=official_args.video_duration,
                         width=official_args.width, height=official_args.height,
                         num_steps=official_args.sample_steps, 
                         guidance_weight=official_args.guidance_weight,
                         scheduler_scale=official_args.scheduler_scale, 
                         expand_prompts=official_args.expand_prompt,
                         save_path=save_path, seed=official_args.seed)
                
                elif mode == "i2v":
                    # I2V: Removed width/height (uses input image size)
                    pipe(str(prompt), 
                         image=official_args.image, 
                         time_length=official_args.video_duration,
                         num_steps=official_args.sample_steps, 
                         guidance_weight=official_args.guidance_weight,
                         scheduler_scale=official_args.scheduler_scale, 
                         expand_prompts=official_args.expand_prompt,
                         save_path=save_path, seed=official_args.seed)
            
            count += 1
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on sample {count}: {e}")
            torch.cuda.empty_cache()

    print(f"\nDone! Successfully processed {count} items.")

if __name__ == "__main__":
    run_percentage_inference()