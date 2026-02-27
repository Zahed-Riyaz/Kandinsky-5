# Kandinsky LoRA Training

This README provides a quick and practical guide for preparing data, configuring training, and running LoRA fine-tuning for Kandinsky models.

---

## 🚀 1. Clone the Repository and Submodule

After clone this repo don't forget do:
```bash
git submodule update --init --remote
```

---

## 📥 2. Download Models

Download all required pretrained models with `kandinsky5/download_models.py` and place them into:

```
kandinsky5/weights
```


---

## 🎬 3. Prepare Your Data

Prepare a directory containing pairs:

* `*.mp4` **or** `*.png`
* `*.txt` — caption for the same sample

Then:

1. Open `encode/encode.sh`
2. Set correct local paths for input data and output directories
3. Run:

```bash
bash encode/encode.sh
```

This will generate:

* `cache/latents_image/`
* `cache/text_embeds/`

---

## ⚙️ 4. Configure Training

### Choose a config:

* **T2I** → `configs/lora_image.yaml`
* **T2V / I2V** → `configs/lora_video.yaml`

Update in the selected config:

* `experiment_dir`
* `log_dir`
* `checkpoint_dir`

Then edit dataloader configs: `configs/data/lora_*_dataloader.yaml`.

Set:

* `latents_dir` → path to latents from Step 3
* `text_embeds_dir` → path to text embeds from Step 3
* `uncond_embed` → `text_embeds_dir` + `/null.pt`

---

## 🧩 5. GPU & LoRA Setup

Edit:

```
configs/trainer/lora*.yaml
```

Configure:

* `devices` → number of GPUs
* Optional: LoRA architecture parameters

---

## ▶️ 6. Start Training

Choose the correct config inside `train.sh`:

* `configs/lora_video.yaml` for T2V / I2V
* `configs/lora_image.yaml` for T2I

Correct `--nproc_per_node` on your number of GPUs and then run:

```bash
bash train.sh
```

**Note:** FSDP is enabled by default.
