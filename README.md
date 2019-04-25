# Improve AttnGAN with Object Relationship Detection Model

## Introduction

Implementation code for the COMS 4995 final project.

## Dependencies

Python 2.7
Pytorch 1.0

Python Packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`
- `pyyaml`

## File Structure

### Data

Coco 2014 Training Dataset: [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`. Please download the 2014 version datasets.

The metadata for COCO also needs to be downloaded.
[coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9). Save to `data/`

### Training

The `*.yml` files under `code/cfg/` are configuration files for training. 
- Pre-train DAMSM models:
  - `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 0`
 
- Train AttnGAN models:
  - `python main.py --cfg cfg/coco_attn2.yml --gpu 0`

## Sample Images

- Under `./code` folder, run `python main.py --cfg cfg/eval_coco.yml --gpu 0` to generate examples.

### Image Demonstration
 Network            | a black table on top of white floor | a man riding a bicycle |
:-------------------------:|:-----------------------------------:|:----------------------:|
G0 | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_0_g0.png) | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_1_g0.png)
G1 | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_0_g1.png) | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_1_g1.png)
G2(Final) | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_0_g2.png) | ![](https://github.com/YHCU/coms_4995_project/tree/master/demo/0_s_1_g2.png)

**Reference**

- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks] [[code]] (https://github.com/taoxugit/AttnGAN)
