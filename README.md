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

3. Running custom metrics example
```
cd src
python3 -m baselines.custom_metrics_example
```

4. Training Results

- K = 5, top 5 highest ranked items
- Best diversity will be K / number of categories (5/102) which is 0.049.
  - Occurs when all items are recommended are unique categories
- MRR is calculated as 1/R if there is an item in the top 5 that is labelled 1.
  - Eg if MRR = 0.1605, 1/0.1605 = ~6, the labelled item appears on average at the 6th position
  - This is possible with K = 5 because there are situations where the labelled item does not appear in the top 5
  - The score will be 0 for those.

DeepFM 

============================ ****====================================================
wandb: Run summary:
wandb:      catalogue_coverage 1
wandb:               diversity 0.03683
wandb:                     mrr 0.1605
wandb: rank_bin_avg(diversity) 4.37704
wandb:          train_accuracy 0.75657
wandb:              train_loss 0.56272
wandb:            val_accuracy 0.29986
wandb:                val_loss 7.93579
================================================================================

DeepFFM

================================================================================
wandb: Run summary:
wandb:      catalogue_coverage 0.97059
wandb:               diversity 0.03629
wandb:                     mrr 0.05159
wandb: rank_bin_avg(diversity) 3.42104
wandb:          train_accuracy 0.78222
wandb:              train_loss 0.48125
wandb:            val_accuracy 0.01377
wandb:                val_loss 4.07832
================================================================================

XDeepFM

================================================================================
wandb:      catalogue_coverage 1
wandb:               diversity 0.03693
wandb:                     mrr 0.03388
wandb: rank_bin_avg(diversity) 4.2057
wandb:          train_accuracy 0.78371
wandb:              train_loss 0.47885
wandb:            val_accuracy 0.0392
wandb:                val_loss 5.39595
================================================================================