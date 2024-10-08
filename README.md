

---

# Deep Learning Project - Meta AI DETR Model with ResNet-50 Backbone

[Google Colab Link](https://colab.research.google.com/drive/1J_Y-F5mEhT3hHAyTP7fNyb-mcnTVQwyV?usp=sharing)
This project explores the DETR (DEtection TRansformer) model developed by Meta AI, utilizing a ResNet-50 backbone. The DETR model is designed for end-to-end object detection, based on the paper titled “End-to-End Object Detection with Transformers” by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. The model is trained on the Common Objects in Context (COCO) 2017 object detection dataset, containing 118k annotated images.

- Model Architecture - The DETR model combines a convolutional backbone network (ResNet-50) with a Transformer-based encoder-decoder architecture to detect objects in an image without using traditional methods like non-maximum suppression  
- Model Reference - [Meta AI DETR Model with ResNet-50 on Hugging Face](https://huggingface.co/facebook/detr-resnet-50)  
- Research Paper - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
![image](https://github.com/user-attachments/assets/f71d1cb7-d9d7-4be7-a817-a27c1a865ed4)

## Exploration Objectives

The main goal of this project is to explore the capabilities of the DETR model by testing it on three unseen datasets. The exploration involves the following:

1. Evaluating an untrained model - Test the DETR model’s ability to make predictions on new datasets without any prior training  
2. Training the model manually - Train the DETR model on three new datasets and evaluate its performance  
3. Exploring image augmentation techniques - Apply various image augmentation techniques to understand their effect on model performance  
4. Hyperparameter tuning - Adjust hyperparameters like learning rate, batch size, and optimizer settings to analyze their impact on the model's results  

## Datasets

The following datasets are used to test the model's ability to generalize to new contexts

1. Animals Dataset - A collection of images featuring various animal species  
   [Dataset Link](https://huggingface.co/datasets/Francesco/animals-ij5d2)  
2. Traffic Dataset - Images of road traffic scenes, including different vehicles and traffic signs  
   [Dataset Link](https://huggingface.co/datasets/Francesco/road-traffic)  
3. Football Dataset - Images of football players in different game scenarios  
   [Dataset Link](https://huggingface.co/datasets/manot/football-players)  

## Implementation Plan

- Load the DETR model - Use the pre-trained DETR model with the ResNet-50 backbone from the Hugging Face Model Hub  
- Dataset preparation - Preprocess the datasets (animals, traffic, football) and apply relevant data augmentation techniques  
- Training - Fine-tune the DETR model on the new datasets, using various hyperparameter settings to optimize performance  
- Evaluation - Assess the model’s performance using standard metrics (e.g., mAP, precision, recall) and compare results across datasets and training conditions  
- Analysis - Document the impact of different hyperparameters and augmentation techniques on the model's effectiveness  

---
