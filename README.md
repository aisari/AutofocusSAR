# AutofocusSAR

Codes and dataset for machine learning-based Synthetic Aperture Radar (SAR) autofocus algorithms. Please see  [AutofocusSAR Github](https://github.com/aisari/AutofocusSAR/) or [AutofocusSAR Webpage](https://aisari.iridescent.ink/AutofocusSAR/).  **Any commercial use is prohibited!**

# Folder Description

- [Dataset](./Dataset/Readme.md): contains all information about dataset. 
- [ECELMs](./ECELMs/README.md) : contains code of  *Ensemble Convolutional Extreme Learning Machine based Autofocus Algorithms*, models: ``Bagging-ECELMs``.
- [PAFnet](./PAFnet/README.md) : contains code of  *AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning*, models: ``AFnet``, ``PAFnet``.
- [VDNNAF](./VDNNAF/README.md) : contains code of  *A Finely Focusing Method of SAR Using Very Deep Neural Network*, (implenmented by me, not the official).
- [CNNAF](./CNNAF/README.md) : contains code of  *SAR Autofocus based on Convolutional Neural Networks*.

# Algorithms

1. ``Dataset`` or ``Bagging-ECELMs``: Fast SAR Autofocus based on Ensemble Convolutional Extreme Learning Machine, 2021, [pdf](https://www.mdpi.com/2072-4292/13/14/2683/pdf), [doi](https://www.mdpi.com/2072-4292/13/14/2683)
2. ``CNN-AF``: SAR Autofocus based on Convolutional Neural Networks
3. ``AFnet`` and ``PAFnet``: Fast and Accurate SAR Autofocus Based on Deep Learning, 2022, [pdf](https://ieeexplore.ieee.org/document/9931653), [doi](https://doi.org/10.1109/TGRS.2022.3217063)

# Usage

## Dependencies

You need first to install our SAR library ( ``torchsar`` ) by excuting the following command:

Please see [torchsar](https://aisari.iridescent.ink/torchsar/) for details (e.g. install on Windows). The package can be installed by

```bash
pip install torchsar
```

**Now, all platforms are supported and part of the source code is open!**

# Citation

If you find the datasets or codes are useful, please kindly cite our papers and star our pakcage [AutofocusSAR](https://github.com/aisari/AutofocusSAR), [torchsar](https://github.com/aisari/torchsar), [torchbox](https://github.com/antsfamily/torchbox) on GitHub:

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

@article{Liu2022PAFnet,
  title={AFnet and PAFnet: Fast and Accurate SAR Autofocus Based on Deep Learning},
  author={Liu, Zhi and Yang, Shuyuan and Gao, Quanwei and Feng, Zhixi and Wang, Min and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3217063}
}

```



