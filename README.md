# Multi-view Masked Contrastive Representation Learning for Endoscopic Video Analysis
This repository provides the official PyTorch implementation of the paper [Multi-view Masked Contrastive Representation Learning for Endoscopic Video Analysis]()
![image](https://github.com/MLMIP/MMCRL/blob/main/img/MMCRL.png)

## Installation
We can install packages using provided `environment.yaml`.

```shell
cd MMCRL
conda env create -f environment.yaml
conda activate MMCRL
```

## Data Preparation
We use the datasets provided by [Endo-FM](https://github.com/med-air/Endo-FM) and are grateful for their valuable work.

## weights
pretrain weight:

[pretrain]()

downstream weight:

[Classification]()

[Segmentation]()

[Detection]()

## Pre-training
```shell
cd MMCRL
wget -P checkpoints/ https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth
bash scripts/pretrain.sh
```

## Fine-tuning
```shell
# PolypDiag (Classification)
cd MMCRL
bash scripts/eval_finetune_polypdiag.sh

# CVC (Segmentation)
cd MMCRL/TransUNet
python train.py

# KUMC (Detection)
cd MMCRL/STMT
bash script/train_stft.sh
```

## Acknowledgement
Our code is based on [DINO](https://github.com/facebookresearch/dino), [TimeSformer](https://github.com/facebookresearch/TimeSformer), [SVT](https://github.com/kahnchana/svt), [Endo-FM](https://github.com/med-air/Endo-FM), [TransUNet](https://github.com/Beckschen/TransUNet), and [STFT](https://github.com/lingyunwu14/STFT). Thanks them for releasing their codes.


## Citation
```
@article{hu2024one,
  title={Multi-view Masked Contrastive Representation Learning for Endoscopic Video Analysis},
  author={Hu, Kai and Xiao, Ye and Zhang, Yuan and Gao, Xieping}
  journal={Advances in Neural Information Processing Systems},
  volume={},
  year={2024}
}
```
