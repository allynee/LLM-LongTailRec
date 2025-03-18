# Code adapted from DIEN
https://arxiv.org/abs/1809.03672
## 1. Data preparation
Unzip the dataset files and move them into the `script/` directory:
```
tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
Ensure the following files are in `script/`:
```
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info
```
## 2. Environment setup
If using Conda, create a compatible environment:
```
conda create -n baseline python=3.5 -y
conda activate baseline
pip install -r requirements.txt
```
## 3. Model training
To train a model, run:
```
python train.py train [MODEL_NAME]
```
Supported models:
- DNN
- PNN 
- Wide (Wide&Deep)
- DIN
- DIEN
