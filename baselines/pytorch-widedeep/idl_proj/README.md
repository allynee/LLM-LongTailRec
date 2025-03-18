# Code Adapted from [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep)
## 1. Environment setup
If using Conda, follow these steps in the correct directory (LLM-LongTailRec/baselines/pytorch-widedeep)
```
conda create --name deepfm_env python=3.9
pip install -r requirements.txt
pip install -e .
```

## 2. Model training
1. Log in to Weights & Biases (WanDB)
```
wandb login <YOUR_API_KEY>
```

2. Run:
```
cd idl_proj
python DeepFM.py
```
Supported models:
- DeepFM
- DeepFFM
- XDeepFM