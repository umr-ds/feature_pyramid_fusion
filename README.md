# Feature Pyramid Fusion  for Detection and Segmentation of Morphologically Complex Eukaryotic Cells

Fusion of feature pyramids for nucleus segmentation and cell segmentation. The model uses nucleus and cytoplasma signal to achieve better cell segmentation results.

![Example segmentation result](assets/overall_architecture.png)

The code is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation of https://github.com/matterport/Mask_RCNN and extends it for feature pyramid fusion.

![Example segmentation result](assets/fpf_example_error2.png)


## Installation

Please see https://github.com/matterport/Mask_RCNN for requirements and installation

## Run model

* Jupyter notebook demo for running models and vizualization: [segmentation_demo.ipynb](samples/cells_and_cores/segmentation_demo.ipynb)

* Training scripts: [cells_and_cores](samples/cells_and_cores)

## Dataset

Full dataset available at: https://data.uni-marburg.de/bitstream/handle/dataumr/231/synmikro_macrophages.tar.gz

## Models

* [Pretrained Grayscale ResNet](https://data.uni-marburg.de/bitstream/handle/dataumr/231/pretrained_grayscale_resnet.zip)

* [FPF &oplus; weighted loss](https://data.uni-marburg.de/bitstream/handle/dataumr/231/fpf_add_weighted.zip)

* [FPF &odot; weighted loss](https://data.uni-marburg.de/bitstream/handle/dataumr/231/fpf_concat_weighted.zip)

* [FPF &oplus;](https://data.uni-marburg.de/bitstream/handle/dataumr/231/fpf_add.zip)

* [FPF &odot;](https://data.uni-marburg.de/bitstream/handle/dataumr/231/fpf_concat.zip)

* [with nucleus channel](https://data.uni-marburg.de/bitstream/handle/dataumr/231/with_nucleus.zip)

* [without nucleus channel](https://data.uni-marburg.de/bitstream/handle/dataumr/231/without_nucleus.zip)

* [Cores](https://data.uni-marburg.de/bitstream/handle/dataumr/231/cores.zip)

## Reference

```
@article{korfhage2020detection,
  title={Detection and segmentation of morphologically complex eukaryotic cells in fluorescence microscopy images via feature pyramid fusion},
  author={Korfhage, Nikolaus and M{\"u}hling, Markus and Ringshandl, Stephan and Becker, Anke and Schmeck, Bernd and Freisleben, Bernd},
  journal={PLOS Computational Biology},
  volume={16},
  number={9},
  pages={e1008179},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

