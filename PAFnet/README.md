# HAFnet

This is a PyTorch implementation of paper "AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning".

**This code will be open-sourced after the paper is published!**

# Dataset

The dataset can be downloaded from [BaiduYunPan](https://pan.baidu.com/s/1BW8ZsP2TXqNU1MJFQrzZBQ) (accessed on 13 August 2021), the extraction code is ``d7fk``.


# Training

```
python train.py --data_dir [DATADIR] --learning_rate 1e-2 --num_epoch 1000, --device 'cuda:0'
```

# Testing

```
python test.py --data_dir [DATADIR] --model_file [TRAINED MODEL PATH] --device 'cuda:0'
```


# Citation

If you find the dataset or this code is useful, please kindly cite our paper and star our pakcage [AutofocusSAR](https://github.com/aisari/AutofocusSAR) on GitHub::

```bib
@article{Liu2021Fast,
  title={Fast SAR Autofocus Based on Ensemble Convolutional Extreme Learning Machine},
  author={Liu, Zhi and Yang, Shuyuan and Feng, Zhixi and Gao, Quanwei and Wang, Min},
  journal={Remote Sensing},
  volume={13},
  number={14},
  pages={2683},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}



```


