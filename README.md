# Feature Pyramid Fusion  for Detection and Segmentation of Morphologically Complex Eukaryotic Cells

Fusion of feature pyramids for nucleus segmentation and cell segmentation. The model uses nucleus and cytoplasma signal to achieve better cell segmentation results.

The code is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation of https://github.com/matterport/Mask_RCNN and extends it for feature pyramid fusion.

![Example segmentation result](assets/fpf_example_error2.png)


## Installation

Please see https://github.com/matterport/Mask_RCNN for requirements and installation

## Run model

* Jupyter notebook demo for running models and vizualization: [segmentation_demo.ipynb](samples/cells_and_cores/segmentation_demo.ipynb)

* Training scripts: [cells_and_cores](samples/cells_and_cores)

## Dataset

Full dataset available at: https://box.uni-marburg.de/index.php/s/N934NJi7IsvOphf

## Models

* [Pretrained Grayscale ResNet](https://box.uni-marburg.de/index.php/s/ILxQpw1aSgzUv6R)

* [FPF &oplus; weighted loss](https://box.uni-marburg.de/index.php/s/cBv2jsXqWJ1GaK6)

* [FPF &odot; weighted loss](https://box.uni-marburg.de/index.php/s/IdAfDmIm6mR64Ek)

* [FPF &oplus;](https://box.uni-marburg.de/index.php/s/hiF6CoIbGDyLIxw)

* [FPF &odot;](https://box.uni-marburg.de/index.php/s/238ujIRoD0cfzSi)

* [with nucleus channel](https://box.uni-marburg.de/index.php/s/OZVCTSHAWvLcQuO)

* [without nucleus channel](https://box.uni-marburg.de/index.php/s/e9mm45kDupUXumj)

* [Cores](https://box.uni-marburg.de/index.php/s/SBjrDZSIZXegcGw)
