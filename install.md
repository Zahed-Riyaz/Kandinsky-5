# 1. Finetune LoRA

git clone https://github.com/kandinskylab/kandinsky-5-lora-train.git  
cd kandinsky-5-lora-train  
git submodule update --init --remote  
cd kandinsky5  

# Download download_captions.py from drive folder  
python download_captions.py --output_dir "./captions_dir" --limit 50  

# Update ‘checkpoint_path’ in configs/model/hunyuan.yaml to “./kandinsky5/weights/vae”  
# In encode.sh, update IMAGES_CAPTIONS_DIR="/lambda/nfs/us-east-1/zahed/kandinsky-5-lora-train/captions_dir"  
bash encode/encode.sh  

# Update devices value to devices: 1 in configs/trainer/lora_image.yaml  
# Add num_workers:0 in data: for configs/data/lora_video_dataloader.yaml  

# Files to replace :  
# Replace configs/trainer/lora_video.yaml with in drive folder lora_video.yaml  
# Replace encode/encode_images.py with encode_images.py in drive folder  
# Replace train/lora_train.py with lora_train.py in drive folder  
# Replace train.sh with drive folder train.sh  

bash train.sh  


# 2. Inference

git clone https://github.com/kandinskylab/kandinsky-5.git  
cd kandinsky-5/  

conda create -n kandinsky5 python=3.10 -y  
conda activate kandinsky5  

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  

awk '
BEGIN { added_click=0 }
/^torch==/ {
    sub(/==/, ">=");
}
/^torchaudio==/ {
    print
    next
}
/^click/ {
    added_click=1
}
{ print }
END {
    if (!added_click) print "click>=7.0"
}
' requirements.txt | pip install -r /dev/stdin  

pip install -U diffusers transformers accelerate torchcodec lightning tensorboard peft datasets pyarrow  

python download_models.py  

# Download inference.py from drive folder  
# Commands for various configs :  

python inference.py --config configs/k5_lite_t2v_5s_sft_sd.yaml --limit 5 --offload --qwen_quantization --width 512 --height 512 --video_duration 5 --sample_steps 50 --magcache  

python inference.py --config configs/k5_lite_t2v_10s_sft_sd.yaml --limit 5 --offload --qwen_quantization --width 512 --height 512 --video_duration 10 --sample_steps 50 —magcache  

python inference.py --config configs/k5_lite_t2v_5s_distil_sd.yaml --limit 5 --offload --qwen_quantization --width 512 --height 512 --video_duration 5 --sample_steps 16  

python inference.py --config configs/k5_lite_i2v_5s_sft_sd.yaml --limit 5 --offload --qwen_quantization --width 512 --height 512 --video_duration 5 --sample_steps 50 --image "./assets/test_image.jpg" --magcache  

python inference.py --config configs/k5_pro_t2v_10s_sft_hd.yaml --limit 5 --offload --qwen_quantization --width 1024 --height 1024 --video_duration 10 --sample_steps 50  

python inference.py --config configs/k5_pro_i2v_5s_sft_hd.yaml --limit 5 --offload --qwen_quantization --width 1024 --height 1024 --video_duration 5 --sample_steps 50 --image “./assets/test_image.jpg"  

python inference.py --config configs/k5_pro_t2v_10s_sft_sd.yaml --limit 5 --offload --qwen_quantization --width 512 --height 512 --video_duration 10 --sample_steps 50  