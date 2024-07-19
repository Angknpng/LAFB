# LAFB
Code repository for our paper entilted "Learning Adaptive Fusion Bank for Multi-modal Salient Object Detection" accepted at TCSVT 2024.

arXiv version: https://arxiv.org/abs/2406.01127.
## Citing our work

If you think our work is helpful, please cite

```
@article{wang2024learning,
  title={Learning Adaptive Fusion Bank for Multi-modal Salient Object Detection},
  author={Wang, Kunpeng and Tu, Zhengzheng and Li, Chenglong and Zhang, Cheng and Luo, Bin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
## Overview
### Framework
[![avatar](https://github.com/Angknpng/LAFB/raw/main/figures/Framework.png)](https://github.com/Angknpng/LAFB/blob/main/figures/Framework.png)
### RGB-D SOD Performance
[![avatar](https://github.com/Angknpng/LAFB/raw/main/figures/RGBDCompare.png)](https://github.com/Angknpng/LAFB/blob/main/figures/RGBDCompare.png)
### RGB-T SOD Performance
[![avatar](https://github.com/Angknpng/LAFB/raw/main/figures/RGBTCompare.png)](https://github.com/Angknpng/LAFB/blob/main/figures/RGBTCompare.png)
## Data Preparation
RGB-D and RGB-T  SOD datasets can be found here. [[baidu pan](https://pan.baidu.com/s/1bJcV2QTH-tWp358p5oGgeg?pwd=chjo) fetch code: chjo] 
## Predictions

Saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/19GNoKIW-sDCgOPjdv_XqSQ?pwd=uodf) fetch code: uodf]

## Pretrained Models
Pretrained parameters can be found here.[[baidu pan](https://pan.baidu.com/s/1T17meMMASEDZNIjdohQHvQ?pwd=3ed6) fetch code: 3ed6]

## Usage

### Prepare

1. Create directories for the experiment and parameter files.
2. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
3. Install other packages: `pip install -r requirements.txt`.
4. Set your path of all datasets in `./Code/utils/options.py`.

### Train

```
python train.py
```

### Test

```
python test_produce_maps.py
```

## Contact

If you have any questions, please contact us (kp.wang@foxmail.com).
