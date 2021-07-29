# HAFnet

This is a PyTorch implementation of paper "AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning".

**This code will be open-sourced after the paper is published!**

# Dataset

The dataset can be downloaded from Baidu Yun or Google Drive:

```


```


# Training

```
python train.py --data_dir [DATADIR] --learning_rate 1e-2 --num_epoch 1000, --device 'cuda:0'
```

# Testing

```
python test.py --data_dir [DATADIR] --model_file [TRAINED MODEL PATH] --device 'cuda:0'
```
