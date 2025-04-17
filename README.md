# LLM-LongTailRec
Exploring distillation and alignment methods to enhance long-tail item recommendations using Large Language Models. The project's code is adapted from [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep)

### 1. Environment setup
If using Conda, follow these steps
```
conda create -n idl_proj_env python=3.9
conda activate idl_proj_env
conda install pip
pip install -r requirements.txt
```

### 2. Training baseline models
1. Log in to Weights & Biases
```
wandb login <YOUR_API_KEY>
```

2. Run:
```
cd src
python -m baselines.DeepFM
```
Supported models:
- DeepFM
- DeepFFM
- XDeepFM