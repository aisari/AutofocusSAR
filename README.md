# AutofocusSAR

Codes and dataset for machine learning-based Synthetic Aperture Radar (SAR) autofocus algorithms. Please see  [AutofocusSAR Github](https://github.com/aisari/AutofocusSAR/) or [AutofocusSAR Webpage](https://aisari.iridescent.ink/AutofocusSAR/).  **Any commercial use is prohibited!**

# Folder Description

- [Dataset](./Dataset/Readme.md): contains all information about dataset. 
- [ECELMs](./ECELMs/README.md) : contains code of  *Ensemble Convolutional Extreme Learning Machine based Autofocus Algorithms*, such as ``Bagging-ECELMs``.
- [CNNAF](./CNNAF/README.md) : contains code of  *SAR Autofocus based on Convolutional Neural Networks*.
- [PAFnet](./PAFnet/README.md) : contains code of  *AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning*.


# Algorithms

1. ``Bagging-ECELMs``: Fast SAR Autofocus based on Ensemble Convolutional Extreme Learning Machine, 2021,  [pdf](https://www.mdpi.com/2072-4292/13/14/2683/pdf), [doi](https://www.mdpi.com/2072-4292/13/14/2683)
2. ``CNN-AF``: SAR Autofocus based on Convolutional Neural Networks, 2021, submitted to Journal of Radars
3. ``AFnet`` and ``PAFnet``: Fast and Accurate SAR Autofocus Based on Deep Learning, 2021, submitted to TIP

# Usage

## Dependencies

You need first to install our SAR library ( ``torchsar`` ) by excuting the following command:

Please see [torchsar](https://aisari.iridescent.ink/torchsar/) for details. The package can be installed by

```bash
pip install torchsar
```

**Now, all platforms are supported and part of the source code is open!**

# Citation

If you find the datasets or codes are useful, please kindly cite our papers and star our pakcage [AutofocusSAR](https://github.com/aisari/AutofocusSAR) on GitHub:

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



