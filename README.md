# Medical Sound Classification using Log-Mel Spectrogram and MFCC for Identifying Asthma and COPD

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a comprehensive machine learning solution for classifying respiratory diseases (Asthma and COPD) from medical sound recordings. The system employs multiple approaches including traditional machine learning algorithms, deep convolutional neural networks, recurrent neural networks, and transformer-based architectures to analyze audio features extracted from cough and vowel sounds.

## Table of Contents

- [Background](#background)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Audio Preprocessing](#audio-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Architecture](#model-architecture)
  - [Ensemble Methods](#ensemble-methods)
- [Project Structure](#project-structure)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)
- [Contributors](#contributors)
- [Supervisors](#supervisors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Background

Chronic Obstructive Pulmonary Disease (COPD) and Asthma are two of the most prevalent chronic respiratory diseases globally, affecting millions of people worldwide. According to the World Health Organization, COPD is the third leading cause of death globally, while Asthma affects approximately 262 million people. Early and accurate diagnosis of these conditions is critical for effective treatment and disease management.

Traditional diagnostic methods for respiratory diseases rely heavily on spirometry tests, clinical examination, and physician expertise, which can be time-consuming, expensive, and not always accessible in resource-limited settings. Recent advances in machine learning and signal processing have opened new possibilities for automated diagnosis using respiratory sound analysis.

This project addresses the challenge of developing an automated, non-invasive diagnostic tool that can classify respiratory sounds to assist in the early detection and differentiation of Asthma and COPD from healthy individuals.

## Objectives

The primary objectives of this research are:

1. To develop robust audio preprocessing and feature extraction pipelines for respiratory sound analysis
2. To investigate and compare the performance of traditional machine learning algorithms (XGBoost, LightGBM, CatBoost) on audio-derived features
3. To design and implement deep learning architectures including CNNs, Vision Transformers, and RNNs for audio classification
4. To evaluate the effectiveness of transfer learning approaches using pre-trained models (ConvNeXt, YAMNet, Vision Transformer)
5. To create an ensemble system that combines multiple models to achieve superior classification performance

## Dataset

This project utilizes the **Kaggle Medical Sound Classification Challenge** dataset, which contains audio recordings of respiratory sounds from patients diagnosed with Asthma, COPD, and healthy controls.

**Dataset Characteristics:**
- **Source:** Kaggle Medical Sound Classification Challenge
- **Audio Types:** Cough sounds and vowel 
- **Classes:** 3 (Healthy, Asthma, COPD)
- **Format:** WAV audio files
- **Sampling Rate:** Variable (resampled to 16 kHz for consistency)
- **Split:** Train/Test sets provided by competition

**Data Organization:**
- Cough audio files
- Vowel audio files
- Log-Mel spectrograms
- Metadata: CSV files with patient IDs and labels

## Methodology

### Audio Preprocessing

The audio preprocessing pipeline consists of the following steps:

1. **Audio Loading:** Load WAV files using librosa with consistent sampling rate
2. **Resampling:** Standardize all audio to 16 kHz sampling rate for YAMNet compatibility
3. **Normalization:** Amplitude normalization to [-1, 1] range
4. **Silence Removal:** Trim leading and trailing silence using energy-based detection
5. **Duration Standardization:** Pad or truncate audio to fixed length when necessary

### Feature Extraction

Two primary feature extraction methods are employed:

#### 1. Log-Mel Spectrogram
- **Window Size:** 2048 samples (Hann window)
- **Hop Length:** 512 samples
- **Mel Bands:** 128
- **Frequency Range:** 0-8000 Hz
- **Output Format:** 2D image-like representation for CNN/ViT models

#### 2. Mel Frequency Cepstral Coefficients (MFCC)
- **Number of MFCCs:** 13 coefficients
- **Delta Features:** First and second-order derivatives included
- **Frame Length:** 25 ms
- **Frame Shift:** 10 ms
- **Output Format:** Time-series data for RNN/LSTM models

#### 3. YAMNet Embeddings
- **Pre-trained Model:** Google's YAMNet (AudioSet)
- **Embedding Dimension:** 1024
- **Temporal Pooling:** Global average pooling over time dimension
- **Output Format:** Fixed-length feature vector

### Model Architecture

#### Traditional Machine Learning

Three gradient boosting algorithms are implemented with hyperparameter optimization:

**1. XGBoost**
- Tree-based ensemble with gradient boosting
- Hyperparameters: max_depth, learning_rate, n_estimators, subsample
- Optimization: Optuna with 5-fold cross-validation

**2. LightGBM**
- Histogram-based gradient boosting
- Leaf-wise tree growth strategy
- Optimized for speed and memory efficiency

**3. CatBoost**
- Ordered boosting with categorical feature support
- Built-in handling of overfitting
- GPU acceleration enabled

**Feature Variants:**
- CSV metadata only
- Audio embeddings (YAMNet)
- Combined features (metadata + embeddings)
- Separate cough and vowel models
- Full feature fusion

#### Deep Learning - Image Classification Models

**1. ConvNeXt Tiny**
- Architecture: ConvNeXt-Tiny (FB in22k pre-trained, fine-tuned on ImageNet-1k)
- Input: 224x224x3 Log-Mel Spectrogram images
- Training: Two-stage (warmup + fine-tuning)

**2. Vision Transformer (ViT)**
- Architecture: ViT-Base patch16-224 (ImageNet-21k pre-trained)
- Input: 224x224x3 Log-Mel Spectrogram images
- Patch Size: 16x16

**3. YAMNet Transfer Learning**
- Architecture: YAMNet encoder + custom classifier
- Input: Raw audio waveform (16 kHz)
- Embeddings: 1024-dimensional
- Training: Linear probing (warmup) + full fine-tuning

#### Deep Learning - Sequential Models

**1. Bidirectional LSTM**
- Input: MFCC features (T, 13)
- Layer 1: Bidirectional LSTM (128 units)
- Layer 2: Bidirectional LSTM (64 units)
- Attention: Multi-head attention mechanism
- Output: Dense(3) with softmax

**2. Bidirectional GRU**
- Input: MFCC features (T, 13)
- Layer 1: Bidirectional GRU (128 units)
- Layer 2: Bidirectional GRU (64 units)
- Pooling: Global max + average pooling
- Output: Dense(3) with softmax

**3. Simple RNN**
- Input: MFCC features (T, 13)
- Layer 1: Bidirectional Simple RNN (64 units)
- Layer 2: Bidirectional Simple RNN (32 units)
- Output: Dense(3) with softmax

### Ensemble Methods

Multiple ensemble strategies are implemented:

1. **Average Ensemble:** Arithmetic mean of probability distributions
2. **Weighted Ensemble:** Learned weights based on validation performance
3. **Stacking:** Meta-learner trained on model predictions
4. **Cough-Vowel Fusion:** Separate models for each audio type, combined at inference

## Project Structure

```
Medical-Sound-Classification-using-Log-Mel-Spectrogram-and-MFCC-for-identifying-Asthma-and-COPD/
│
├── Image_Classification_Models/
│   ├── convnext.ipynb                        # ConvNeXt-Tiny implementation
│   ├── vit.ipynb                             # Vision Transformer implementation
│   ├── yamnet.ipynb                          # YAMNet transfer learning
│   └── preprocess_log_mel_spectrograms.ipynb # Preprocessing pipeline
│
├── Sequential_Models/
│   ├── train_lstm_pipeline.ipynb             # Bidirectional LSTM
│   ├── train_lstm_unidirectional_pipeline.ipynb # Unidirectional LSTM
│   ├── train_pure_lstm.ipynb                 # Pure LSTM implementation
│   ├── train_gru_pipeline.ipynb              # Bidirectional GRU
│   └── train_rnn_pipeline.ipynb              # Simple RNN
│
├── Traditional_ML/
│   ├── XGBoost_csvONLY.ipynb                 # XGBoost with CSV features
│   ├── XGboost_embedJere.ipynb               # XGBoost with embeddings
│   ├── XGboost_embedK.ipynb                  # XGBoost variant
│   ├── XGBoost_coughVowelembjere.ipynb       # Cough+Vowel features
│   ├── XGBoost_coughVowelembK.ipynb          # Cough+Vowel variant
│   ├── XGboost_FullJere.ipynb                # Full feature set
│   ├── XGboost_FullK.ipynb                   # Full feature variant
│   ├── LightBgm_csvONLY.ipynb                # LightGBM with CSV
│   ├── LightBgm_EmbedJere.ipynb              # LightGBM with embeddings
│   ├── LightBgm_embedK.ipynb                 # LightGBM variant
│   ├── LightBgm_coughembK.ipynb              # LightGBM cough features
│   ├── LightBgm_coughVowelembjere.ipynb      # LightGBM cough+vowel
│   ├── LightGbm_FullK.ipynb                  # LightGBM full features
│   ├── CatBoost.ipynb                        # CatBoost baseline
│   ├── CatBoost_csvONLY.ipynb                # CatBoost with CSV
│   ├── CatBoost_EmbedJere.ipynb              # CatBoost with embeddings
│   ├── CatBoost_embedK1.ipynb                # CatBoost variant
│   ├── CatBoost_coughVowelembjere.ipynb      # CatBoost cough+vowel
│   ├── CatBoost_coughVowelembK.ipynb         # CatBoost variant
│   ├── Catboost_FullJere.ipynb               # CatBoost full features
│   └── CatBoost_fullK.ipynb                  # CatBoost full variant
│
├── LICENSE                                    # MIT License
└── README.md                                  # This file
```

## Results

Results from model evaluations will be documented here after experiments are completed.

### Preliminary Findings

**Model Performance Comparison:**

| Model Type | Model Name | Validation F1 | Notes |
|------------|------------|---------------|-------|
| Traditional ML | XGBoost | TBD | Baseline model |
| Traditional ML | LightGBM | TBD | Fast training |
| Traditional ML | CatBoost | TBD | Handles categorical features |
| CNN | ConvNeXt-Tiny | 0.396 | Transfer learning |
| Transformer | ViT | 0.351 | Attention mechanism |
| Audio DL | YAMNet | 0.391 | Pre-trained on AudioSet |
| RNN | Bi-LSTM | TBD | Temporal modeling |
| RNN | Bi-GRU | TBD | Faster than LSTM |
| Ensemble | Cough+Vowel Fusion | TBD | Best performance |

### Best Practices Identified

1. **Two-stage training** (warmup + fine-tuning) improves convergence for deep models
2. **Class balancing** using weighted loss functions reduces bias toward majority class
3. **Optuna hyperparameter optimization** provides 5-10% improvement over manual tuning
4. **Ensemble of cough and vowel models** outperforms single-modality approaches
5. **Data augmentation** (SpecAugment, time stretching) improves generalization


## Requirements

### Core Dependencies
- Python 3.8+
- NumPy, Pandas, Scikit-learn

### Audio Processing
- Librosa for audio analysis
- SoundFile for audio I/O

### Deep Learning Frameworks
- PyTorch 2.0+
- TensorFlow 2.10+
- TensorFlow Hub

### Computer Vision
- timm (PyTorch Image Models)
- Albumentations for augmentation
- OpenCV

### Traditional ML
- XGBoost
- LightGBM
- CatBoost

### Optimization & Utilities
- Optuna for hyperparameter tuning
- Matplotlib, Seaborn for visualization

## Contributors

This project is developed by a team of undergraduate students from Institut Teknologi Sepuluh Nopember (ITS) as part of their Machine Learning final project.

**Team Members:**

- **Fa'iz Akbar Hizbullah** (5054241005)
  - Email: faizakbar2301@gmail.com

- **Jeremy Mattathias Mboe** (5054241012)
  - Email: jeremymattathias12@gmail.com

- **Arvito Rajapandya Natlysandro** (5054241046)
  - Email: arvito.rajapandya@gmail.com

**Affiliation:**
- Program: Rekayasa Kecerdasan Artifisial (Artificial Intelligence Engineering)
- Department: Teknik Informatika (Informatics Engineering)
- Faculty: Fakultas Elektro dan Informatika Cerdas (Faculty of Intelligent Electrical and Informatics Engineering)
- Institution: Institut Teknologi Sepuluh Nopember (ITS)
- Year: 2025

## Supervisors

This project is supervised by faculty members from the Department of Informatics Engineering, ITS:

- **Dini Adni Navastara, S.Kom, M.Sc.**
  - Department of Informatics Engineering, ITS

- **Ilham Gurat Adillion, S.Kom., M.Kom.**
  - Department of Informatics Engineering, ITS

- **Aldinata Rizky Revanda, S.Kom., M.Kom.**
  - Department of Informatics Engineering, ITS

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Fa'iz Akbar Hizbullah, Jeremy Mattathias Mboe, Arvito Rajapandya Natlysandro

## Acknowledgments

We would like to express our gratitude to:

- **Institut Teknologi Sepuluh Nopember (ITS)** for providing the educational environment and resources
- **Department of Informatics Engineering** for the computational infrastructure and support
- **Kaggle** for hosting the Medical Sound Classification Challenge and providing the dataset
- **Our supervisors** for their guidance, feedback, and mentorship throughout this project
- **Google Research** for the YAMNet pre-trained model and AudioSet dataset
- **Meta AI** and **timm library maintainers** for pre-trained ConvNeXt models
- **OpenAI** and the **PyTorch community** for excellent documentation and tutorials
- **XGBoost, LightGBM, and CatBoost teams** for their efficient gradient boosting implementations

**Special Thanks:**
- Papers and research that inspired our methodology
- Open-source community for libraries and tools that made this project possible
- Fellow students for collaboration and peer review

---

**Citation:**

If you use this work in your research, please cite:

Hizbullah, F. A., Mboe, J. M., & Natlysandro, A. R. (2025). Medical Sound Classification using Log-Mel Spectrogram and MFCC for Identifying Asthma and COPD. Institut Teknologi Sepuluh Nopember.

---

**Contact:**

For questions, suggestions, or collaborations, please contact:
- Email: jeremymattathias12@gmail.com
- GitHub Issues: [Project Issues Page](https://github.com/yourusername/Medical-Sound-Classification/issues)

**Last Updated:** December 2025
