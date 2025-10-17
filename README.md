# LAFB
Code repository for our paper entilted "Learning Adaptive Fusion Bank for Multi-modal Salient Object Detection" accepted at TCSVT 2024.

arXiv version: https://arxiv.org/abs/2406.01127.

24.7.19. The prediction results and weights based on VGG and ResNet backbones have been updated in the Baidu network disk link below.
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
RGB-D and RGB-T  SOD datasets can be found here. [[baidu pan](https://pan.baidu.com/s/1RE48go1wzGWymMblawG2wQ?pwd=vvgq) fetch code: vvgq] 
## Predictions

Saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1VRJG35qH2-aMqqeu9bIA8g?pwd=ytjh) fetch code: ytjh] or [[google drive](https://drive.google.com/drive/folders/1RSrdmZxdizrb58ULvuoAAF1UgxwpVHVp?usp=drive_link)]

## Pretrained Models
Pretrained parameters can be found here.[[baidu pan](https://pan.baidu.com/s/1Fct6_SB5HQf1yOd1GupL4w?pwd=etcn) fetch code: etcn] or [[google drive](https://drive.google.com/drive/folders/1RSrdmZxdizrb58ULvuoAAF1UgxwpVHVp?usp=drive_link)]

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
