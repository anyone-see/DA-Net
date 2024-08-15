# Enhanced Multimodal Depth Estimation via Dynamic Weighted Audio-Visual Fusion and Adaptive Bins

This repository contains snippets of test code for `DA-Net` used to demonstrate and validate the methods described in the paper. While the full research code is not released to protect project integrity and sensitive information, we provide this test code to verify the model's accuracy. We guarantee to publish the complete training code after the paper is accepted.

# Dataset
## Simulated Datasets
Replica-VisualEchoes can be obatined from  [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment.
MatterportEchoes is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset.

## Real-World Dataset
Batvision1 and Batvision2 can be obatined from [here](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M). More information about these two datasets can be obtained [here](https://amandinebtto.github.io/Batvision-Dataset/).

You need to modify the dataset path in each `config.yml` folder.
# Pre-trained Models

You need to download pre trained weights and place them in the folder corresponding to the dataset name. [All Download](https://drive.google.com/drive/folders/1CVjyjuTAktMcgaYjFf-rCQFLNu7D-dUa?usp=drive_link)

| Dataset | Checkpoint|
|----------|----------|
| Replica| [Download](https://drive.google.com/drive/folders/1Y8OXga9ZHaUrU8lg9iYzUgLHze4s1xV3?usp=drive_link)|
| Matterport3D| [Download](https://drive.google.com/drive/folders/1PFVSYVh-x0ZuFMRbqqoapmK-LAlGyDCz?usp=drive_link)|
|Batavision v1|[Download](https://drive.google.com/drive/folders/1Tj2jy3OZQP3rtxgzArieG9zElyKMDZss?usp=drive_link)|
|Batavision v2|[Download](https://drive.google.com/drive/folders/12yCzdMvEvvU1mG3f3kzzk_N9FfRe1VF_?usp=drive_link)|


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
