# AutofocusSAR

Codes and dataset for machine learning-based Synthetic Aperture Radar (SAR) autofocus algorithms.

# Folder Description

- ``Dataset``: contains all information about dataset. 
- ``ECELMs`` : contains code of  *Ensemble Convolutional Extreme Learning Machine based Autofocus Algorithms*, such as Bagging-ECELMs.
- ``PAFnet`` : contains code of  *AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning*.


# Algorithms

1. Bagging-ECELMs: Fast SAR Autofocus based on Ensemble Convolutional Extreme Learning Machine, 2021,  [pdf](https://www.mdpi.com/2072-4292/13/14/2683/pdf), [doi](https://www.mdpi.com/2072-4292/13/14/2683)


# Usage

## Dependencies

You need first to install our SAR library ( ``torchsar`` ) by excuting the following command:

Please see [torchsar](https://aisari.iridescent.ink/torchsar/) for details.

```bash
pip install torchsar
```

# Citation


If you find the dataset or code is useful, please kindly cite our papers:

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



