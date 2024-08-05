# Enhanced Multimodal Depth Estimation via Dynamic Weighted Audio-Visual Fusion and Adaptive Bins

This repository contains snippets of test code related to [DA-UNet] that are used to demonstrate and validate the methods mentioned in the paper. To protect the integrity of the project and sensitive information, we have not released the full research code.We provide the test code of the model to verify the accuracy of the model

# Dataset
## Simulated Datasets
Replica-VisualEchoes can be obatined from  [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment.
MatterportEchoes is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset.

## Real-World Dataset
Batvision1 and Batvision2 can be obatined from [here](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M). More information about these two datasets can be obtained [here](https://amandinebtto.github.io/Batvision-Dataset/).

You need to modify the dataset path in each `config.yml` folder.
# Pre-trained Models

You need to download pre trained weights and place them in the folder corresponding to the dataset name.

| Dataset | Checkpoint|
|----------|----------|
| Replica| [Download](https://github.com/anyone-see/DA-Net/releases/tag/v1.Replica)|
| Matterport3D| [Download](https://github.com/anyone-see/DA-Net/releases/tag/v1.Matterport3D) |
|Batavision v1|[Download](https://github.com/anyone-see/DA-Net/releases/tag/v1.BV1)|
|Batavision v2|[Download](https://github.com/anyone-see/DA-Net/releases/tag/v1.BV2)|


# Build Environment
The version and name of the library we are using are in the `env.yaml` file
```
conda env create -f env.yaml
```

# Evaluation

```
python test.py bv1
python test.py bv2
python test.py replica
python test.py mp3d
```
