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
| Replica| [Download](https://drive.google.com/drive/folders/1HnX8Lq2Bu87eObwdoSiR2zApH22Fa8d9?usp=drive_link)|
| Matterport3D| [Download](https://drive.google.com/drive/folders/1r2f2lSBoKKkAHAcXASdFTWUsaMKof0vc?usp=drive_link) |
|Batavision v1|[Download](https://drive.google.com/drive/folders/13vDIFtQLYMKdylaXyLZS1iaswQz8If1U?usp=drive_link)|
|Batavision v2|[Download](https://drive.google.com/drive/folders/1F9fAdYjKNhx5PnRS7cOP8iNSIh2WeRBN?usp=drive_link)|


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