# SASAN: Spectrum-Axial Spatial Approach Networks for Medical Image Segmentation

### [Project page](https://github.com/IMOP-lab/SASAN-Pytorch) | [Paper](https://ieeexplore.ieee.org/abstract/document/10486971) | [Our laboratory home page](https://github.com/IMOP-lab) 

Our paper has been accepted by IEEE Transactions on Medical Imaging!

by Xingru Huang, Jian Huang, Kai Zhao, Tianyun Zhang, Zhi Li, Changpeng Yue, Wenhao Chen, Ruihao Wang, Xuanbin Chen, Qianni Zhang, Ying Fu, Yangyundou Wang, and Yihao Guo

Hangzhou Dianzi University IMOP-lab

<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/figures/network.png"width=90% height=90%>
</div>
<p align=left>
  Figure 1: Detailed network structure of the SASAN.
</p>

The proposed SASAN is an innovative 3D medical image segmentation network that integrates spectrum information. It achieves state-of-the-art performance over 13 previous methods on the CMED and OIMHS datasets.

We will first introduce our method and principles, then introduce the experimental environment and provide Github links to previous methods we have compared. Finally, we will present the experimental results.

## Installation
We run SASAN and previous methods on a system running Ubuntu 22.04, with Python 3.9, PyTorch 2.0.0, and CUDA 11.8. For a full list of software packages and version numbers, see the experimental environment file 'environment.yaml'. 

## Experiment
### Baselines

We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[3D U-Net](https://github.com/wolny/pytorch-3dunet); [RAUNet](https://github.com/nizhenliang/RAUNet); [UNETR](https://github.com/tamasino52/UNETR); [SwinUNETR](https://github.com/LeonidAlekseev/Swin-UNETR); [ResUNet](https://github.com/rishikksh20/ResUnet);
[MultiResUNet](https://github.com/nibtehaz/MultiResUNet); [V-Net](https://github.com/mattmacy/vnet.pytorch); [3D UX-Net](https://github.com/MASILab/3DUX-Net); [SegResNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet.py); [HighRes3DNet](https://github.com/fepegar/highresnet);
[TransBTS](https://github.com/Rubics-Xuan/TransBTS); [nnFormer](https://github.com/282857341/nnFormer); [SETR](https://github.com/fudan-zvg/SETR)

### Compare with others on the CMED dataset

<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/tables/Compare.png"width=80% height=80%>
</div>
<p align=left>
  Figure 2: Comparison experiments between our method and 13 previous segmentation methods on the CMED dataset.
</p>

<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/figures/Rendering.png"width=80% height=80%>
</div>
<p align=left>
  Figure 3: The visual results of our method compared to the existing 13 segmentation methods on the CMED dataset.
</p>
    
Our method demonstrates the best performance across all categories and metrics. SASAN outperforms previous methods in detail segmentation of choroidal and macular edema categories, highlighting the role of low-frequency data in enhancing details and reducing noise. In addition, SASAN performs well in boundary segmentation of retinal categories, thanks to its BoundaryRea loss and self-updating mechanism, which improves sensitivity to boundary distance.

### Ablation study

#### Key components of SASAN
<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/tables/Ablation1.png"width=80% height=80%>
</div>
<p align=left>
  Figure 4: Ablation experiments on key components of SASAN on the CMED dataset.
</p>

FINE introduces a wide range of low-frequency features, which has a good effect in reducing high-frequency noise and enhancing detail extraction. ASEM has a strong ability to enhance the network for the analysis of features that are difficult to distinguish objects.

#### Loss function strategy
<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/tables/Ablation2.png"width=80% height=80%>
</div>
<p align=left>
  Figure 5: Ablation experiments on Loss function strategy on the CMED dataset.
</p>

The self-updating mechanism and BoundaryRea Loss enhance the network's boundary segmentation ability. To a certain extent, it also improves the overall segmentation ability of the network.

### Model Complexity

<div align=left>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/figures/Parameters.png"width=60% height=60%>
</div>
<p align=left>
  Figure 6: Comparative assessment of parameters, FLOPs, and inference time for our proposed method versus classical models under uniform evaluation settings and computer configuration.
</p>

# Citation
If SASAN is useful for your research,  please consider citing:

    @article{huang2024sasan,
      title={SASAN: Spectrum-Axial Spatial Approach Networks for Medical Image Segmentation},
      author={Huang, Xingru and Huang, Jian and Zhao, Kai and Zhang, Tianyun and Li, Zhi and Yue, Changpeng and Chen, Wenhao and Wang, Ruihao and Chen, Xuanbin and Zhang, Qianni and others},
      journal={IEEE Transactions on Medical Imaging},
      year={2024},
      publisher={IEEE}
    }

# Question
if you have any questions, please contact 'j.huang@hdu.edu.cn'
