# AnyDoor run on NVIDIA Blackwell GB10

> Ref. https://github.com/ali-vilab/AnyDoor

This guide provides specialized installation instructions for running AnyDoor on modern enterprise hardware, specifically tailored for **NVIDIA Blackwell (GB10) GPUs (Compute Capability `sm_121`) with CUDA 13** on **ARM64 (aarch64)** architecture.

> ---
> **IMPORTANT**: 
>
> AnyDoor is best suited for object & animal replacement (not recommended for human subjects).
>
> - Pros: Allows precise spatial control over the replacement location.
> - Cons: Highly sensitive to mask boundaries; an imprecise mask can result in object distortion.
>
> ---

## Prerequisites
- **Python 3.10** (Strictly required to avoid `distutils` errors).
- Linux environment (Ubuntu/Debian on DGX/ARM64 recommended).
- Miniconda / Miniforge installed.

```bash
git clone https://github.com/ali-vilab/AnyDoor
cd AnyDoor
```

### Step 1: Prepare the Conda Environment

Create a clean Python 3.10 Conda environment and activate it. We use Conda/Mamba to ensure smooth dependency resolution on ARM64.

```bash
conda create -n anydoor python=3.10 -y
conda activate anydoor
pip install --upgrade pip

```

### Step 2: Install PyTorch for Blackwell (CUDA 13)

Standard PyTorch releases do not include kernels for Blackwell (`sm_121`). You must install the **Nightly Build** compiled with CUDA 13 to avoid `no kernel image is available` errors.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

```

*(Verify your installation by running `python -c "import torch; print(torch.cuda.get_device_name(0))"`. It should display your GB10 GPU).*

### Step 3: Fix & Install Dependencies

Original dependencies conflict with modern backend servers and Blackwell architectures.

**1. Create a cleaned `requirements.txt`:**
Replace the contents of `requirements.txt` with the following (we removed `torch`, `xformers`, and `gradio` as we will install specific versions manually):

```text
albumentations==1.3.0
einops==0.3.0
fvcore==0.1.5.post20221221
numpy==1.23.1
omegaconf==2.1.1
open_clip_torch==2.17.1
opencv_contrib_python==4.7.0.72
opencv_python==4.7.0.72
opencv_python_headless==4.7.0.72
Pillow==9.4.0
pytorch_lightning==1.5.0
safetensors==0.2.7
scipy==1.9.1
setuptools==66.0.0
submitit==1.5.1
timm==0.6.12
torchmetrics==0.6.0
tqdm==4.65.0
transformers==4.19.2
```

**2. Install requirements and stable UI components:**
Run the following commands to install the base requirements and lock the Gradio UI backend to versions that won't cause `unhashable type: 'dict'` or `KeyError: 'dataset'` errors:

```bash
pip install -r requirements.txt
pip install "gradio==3.50.2" "fastapi==0.104.1" "starlette==0.27.0" "pydantic<2.0.0" "jinja2==3.1.2" "markupsafe==2.1.3"

```

### Step 4: Apply the Blackwell SDPA Patch (xFormers Bypass)

The original `xformers` library does not support `sm_121` and will crash. We need to hijack the `xformers` memory-efficient attention call and redirect it to PyTorch 2.5+'s native `scaled_dot_product_attention` (SDPA).

Run this script in your terminal to automatically patch `run_gradio_demo.py`:

```bash
cat << 'EOF' > fix_blackwell_sdpa.py
import re

filepath = "run_gradio_demo.py"
with open(filepath, "r") as f:
    content = f.read()

# Remove any existing patch
content = re.sub(r'import torch\nimport torch\.nn\.functional as F\nimport xformers\.ops\n.*?xformers\.ops\.memory_efficient_attention = _patched_mea\n+', '', content, flags=re.DOTALL)

# Inject smart SDPA patch
patch = """import torch
import torch.nn.functional as F
import xformers.ops

def _patched_mea(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
    q, k, v = query, key, value
    
    # Transpose if 4D tensor (DINOv2 Encoder)
    if query.ndim == 4:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
    out = F.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), 
        dropout_p=p if p is not None else 0.0, 
        scale=scale
    )
    
    # Transpose back if 4D tensor
    if query.ndim == 4:
        out = out.transpose(1, 2)
        
    return out

xformers.ops.memory_efficient_attention = _patched_mea
"""

with open(filepath, "w") as f:
    f.write(patch + "\n" + content)
print("✅ Blackwell Smart SDPA Patch Applied!")
EOF

python fix_blackwell_sdpa.py

```

### Step 5: Download Models (Checkpoints)

Create a directory for the models and download both the AnyDoor checkpoint and the DINOv2 vision encoder:

```bash
mkdir -p model_ckpt

# 1. Download AnyDoor Checkpoint (Using Hugging Face CLI)
pip install -U huggingface_hub
hf download xichenhku/AnyDoor "epoch=1-step=8687.ckpt" --repo-type space --local-dir ./model_ckpt

# 2. Download DINOv2 Checkpoint (Using wget)
wget -c https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth -P ./model_ckpt

```

### Step 6: Configure YAML Files

Point the configuration files to your downloaded checkpoints.

**1. Update `configs/anydoor.yaml**`
Find `FrozenDinoV2Encoder` (around line 83) and update the `weight` path:

```yaml
      params:
        weight: ./model_ckpt/dinov2_vitg14_pretrain.pth

```

**2. Update `configs/demo.yaml**`
Change the `pretrained_model` path at the top of the file:

```yaml
pretrained_model: ./model_ckpt/epoch=1-step=8687.ckpt
config_file: configs/anydoor.yaml
save_memory: False
use_interactive_seg: True

```

*(Note: If you encounter Out of Memory (OOM) errors on your GPU during inference, change `save_memory: True`).*

### Step 7: Run the App

To prevent Gradio from falsely assuming localhost is inaccessible due to enterprise proxy rules on DGX nodes, bypass the proxy for local interfaces, then launch the demo:

```bash
python run_gradio_demo.py

```

Access the UI via the local URL provided in your terminal (`http://0.0.0.0:7860`). If you are working on a remote server, use SSH port forwarding to access it on your local browser.
