# Enhanced Multimodal Depth Estimation via Dynamic Weighted Audio-Visual Fusion and Adaptive Bins

This repository contains snippets of test code related to [AD-UNet] that are used to demonstrate and validate the methods mentioned in the paper. To protect the integrity of the project and sensitive information, we have not released the full research code.We provide the test code of the model to verify the accuracy of the model

# Dataset
## Simulated Datasets
Replica-VisualEchoes can be obatined from  [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment.
MatterportEchoes is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset.

## Real-World Dataset
Batvision1 and Batvision2 can be obatined from [here](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M). More information about these two datasets can be obtained [here](https://amandinebtto.github.io/Batvision-Dataset/).



# Evaluation
Configure the relevant yml files before testing.We will give the pre-trained model parameters [here](https://drive.google.com/file/d/1BiNgFQNvO8n4_RZGusPzk4qksGiGQgX6/view?usp=drive_link)
```
pip install requirements.txt -r
python test.py
```