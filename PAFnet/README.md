# PAFnet

This is a PyTorch implementation of paper "AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning".

Autofocus plays a key role in synthetic aperture radar (SAR) imaging, especially for high-resolution imaging. In the literature, the minimum-entropy-based algorithms (MEA) have been proved to be robust and have been widely applied in SAR. However, this kind of method needs hundreds of iterations and is computationally expensive. In this paper, we proposed a non-iterative autofocus scheme based on deep learning and minimum-entropy criterion. It’s an unsupervised framework, which utilizes entropy as the loss function. In this scheme, deep neural networks are utilized for feature extraction and parameter estimation. Based on this scheme, two autofocus models (autofocus network and progressive autofocus network) are proposed. After training, the network learned the rules of autofocus from a large number of examples. Experimental results on real SAR data show that the proposed methods have focusing quality close to the state-of-the-art but with real-time focusing speed.

**This code will be open-sourced fully after the paper is published!**

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


