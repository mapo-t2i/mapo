# Margin-aware Preference Optimization for Aligning Diffusion Models without Reference (MaPO)

This repository provides the official PyTorch implementation for MaPO. 

<div align="center">
<img src="assets/mapo_overview.png" width=750/>
</div><br>

_By: Jiwoo Hong<sup>\*</sup>, Sayak Paul<sup>\*</sup>, Noah Lee, Kashif Rasul, James Thorne, Jongheon Jeong_
<br>_(<small><sup>*</sup> indicates equal contribution</small>)_

For the paper, models, datasets, etc., please visit the [project website](TODO).

**Contents**:

* [Running MaPO training](#running-mapo-training)
* [Models and Datasets](#models-and-datasets) 
* [Inference](#inference)
* [Citation](#citation)

## Running MaPO training

### Hardware requirements

We ran our experiments on a node of 8 H100s (80GB). But `train.py` can run on a single GPU having at least 40GB VRAM. 

### Environment

Create a Python virtual environment with your favorite package manager. 

After activating the environment, install PyTorch. We recommend following the [official website](https://pytorch.org/) for this. 

Finally, install the other requirements from `requirements.txt`. 

### Steps to run the code

We performed our experiments on the [`yuvalkirstain/pickapic_v2`](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) dataset which is 335 GB in size. However, here's another smaller version of the dataset that can be used for debugging -- [`kashif/pickascore`](https://huggingface.co/datasets/kashif/pickascore).

When using `yuvalkirstain/pickapic_v2`, be sure to specify the `--dataset_split_name` CLI arg as `train`.

Below is an example training command for a single-GPU run:

```bash
accelerate launch train.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="mapo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --seed="0" 
```

### Misc

<details>
<summary>To run on multiple GPUs, specify the `--multi_gpu` option:</summary>

```bash
accelerate launch --multi_gpu train.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="mapo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --seed="0" 
```
</details><br>

<details>
<summary>To run intermediate validation runs (i.e., generate samples for model assessment), add the following things:</summary>

```diff
+  --run_validation --validation_steps=50 \
+  --report_to="wandb"
```

This will additionally, log the generated results and other metrics to Weights and Biases. This requires you to install the `wandb` Python package. 

Other option for an experiment logger is `tensorboard`. 
</details><br>

To push the intermediate checkpoints and the final checkoint to the Hugging Face Hub platform, pass the `--push_to_hub` option. Note that you need to be authenticated to use your Hugging Face Hub account for this. 

**Notes on evaluation**:

* If you are constrained by GPU memory, then we encourage you to conduct LoRA-based MaPO training. You could repurpose the Diffusion DPO with LoRA code from [here](https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/train_diffusion_dpo_sdxl.py) for this. 
* For evaluation with metrics like Aesthetic Scoring, HPS v2.1, and Pickscore, we followed the respective official codebases.

For visual quantitative results, please refer to the [project website](TODO).

## Models and Datasets

All the models and datasets of our work can be found via our Hugging Face Hub organization: https://huggingface.co/mapo-t2i/.

## Inference

```python
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
import torch 

sdxl_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae_id = "madebyollin/sdxl-vae-fp16-fix"
unet_id = "mapo-t2i/mapo-beta"

vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(unet_id, torch_dtype=torch.float16)
pipeline = DiffusionPipeline.from_pretrained(sdxl_id, vae=vae, unet=unet, torch_dtype=torch.float16).to("cuda")

prompt = "A lion with eagle wings coming out of the sea , digital Art, Greg rutkowski, Trending artstation, cinematographic, hyperrealistic"
image = pipeline(prompt=prompt, num_inference_steps=30).images[0]
```

## Citation

```bibtex
@misc{todo,
    title={Margin-aware Preference Optimization for Aligning Diffusion Models without Reference}, 
    author={Jiwoo Hong and Sayak Paul and Noah Lee and Kashif Rasuland James Thorne and Jongheon Jeong},
    year={2024},
    eprint={todo},
    archivePrefix={arXiv},
    primaryClass={cs.CV,cs.LG}
}
```
