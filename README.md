# DDoS Attack Detection using Machine Learning

A comprehensive machine learning solution for detecting Distributed Denial of Service (DDoS) attacks using Random Forest and Decision Tree classifiers with optimized feature selection.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Requirements](#-requirements)
- [Contributing](#-contributing)

## 🎯 Overview

This project implements a machine learning-based DDoS attack detection system that can accurately classify network traffic as either benign or malicious. The system uses advanced preprocessing techniques, feature engineering, and ensemble learning methods to achieve high detection accuracy with low false positive rates.

### Key Highlights
- **High Accuracy**: Achieves >99% accuracy on balanced test sets
- **Low False Positive Rate**: Optimized to minimize false alarms
- **Real-time Ready**: Fast inference time suitable for real-time detection
- **Multiple Attack Types**: Detects 11 different DDoS attack variants
- **Optimized Features**: Uses only top 12 most important features for efficiency

## ✨ Features

- **Comprehensive Data Preprocessing**
  - Handles missing values, infinite values, and outliers
  - Feature normalization and encoding
  - Removal of constant and identifier columns

- **Advanced Feature Engineering**
  - XGBoost-based feature importance analysis
  - Selection of top 12 most discriminative features
  - Reduced dimensionality for faster inference

- **Multiple Model Support**
  - Random Forest Classifier (primary model)
  - Decision Tree Classifier
  - XGBoost for feature selection

- **Robust Evaluation**
  - Multiple test scenarios (60:40 and 80:20 benign-attack ratios)
  - Comprehensive metrics (accuracy, precision, recall, F1-score)
  - Confusion matrices and ROC curves
  - False positive rate analysis

## 📊 Dataset

### Dataset Source
This project uses the **CICDDoS2019** dataset, which contains realistic network traffic data including both benign traffic and various types of DDoS attacks. The dataset was collected by the Canadian Institute for Cybersecurity and provides comprehensive network flow features extracted using CICFlowMeter.

### Dataset Organization
- **CSV-01-12/01-12/**: Original raw training dataset files containing 11 different attack types
- **CSV-03-11/03-11/**: Additional raw test dataset files
- **10-1 attack-benign/**: **PRIMARY FOLDER** - Contains the final optimized model with 10:1 attack-to-benign ratio balancing
- **50-50/**: Experimental folder with 50-50 balanced datasets (initial experiments)

> **Note**: The final production-ready model is located in the `10-1 attack-benign/` folder. The 50-50 folder contains early experimental work with perfectly balanced datasets, which showed good results but the 10:1 ratio proved more robust for real-world scenarios.

### Attack Types Covered
The system can detect the following DDoS attack types:
1. **DrDoS_DNS** - DNS Reflection/Amplification
2. **DrDoS_LDAP** - LDAP Reflection/Amplification
3. **DrDoS_MSSQL** - MSSQL Reflection/Amplification
4. **DrDoS_NetBIOS** - NetBIOS Reflection/Amplification
5. **DrDoS_NTP** - NTP Reflection/Amplification
6. **DrDoS_SNMP** - SNMP Reflection/Amplification
7. **DrDoS_SSDP** - SSDP Reflection/Amplification
8. **DrDoS_UDP** - UDP Flood
9. **Syn** - SYN Flood
10. **TFTP** - TFTP Reflection/Amplification
11. **UDPLag** - UDP Lag Attack

### Data Distribution (Final Model - 10:1 Ratio)
- **Training Set**: 10:1 attack-to-benign ratio (realistic imbalanced scenario mimicking real-world traffic)
- **Test Set 1**: 60% benign, 40% attack
- **Test Set 2**: 80% benign, 20% attack

Each attack type CSV contains network flow features such as packet lengths, flag counts, flow duration, bytes per second, and various statistical measures of the traffic patterns.

## 🤖 Models

### Primary Model: Random Forest (Top 12 Features)
- **Algorithm**: Random Forest Classifier
- **Features**: 12 optimized features
- **Hyperparameters**:
  - n_estimators: 150
  - max_depth: 8
  - min_samples_split: 10
  - class_weight: balanced

### Top 12 Selected Features
1. MinPacketLength
2. URGFlagCount
3. Inbound
4. Init_Win_bytes_forward
5. min_seg_size_forward
6. ACKFlagCount
7. AveragePacketSize
8. FwdPacketLengthMean
9. FwdPacketLengthMin
10. TotalBackwardPackets
11. TotalLengthofFwdPackets
12. FlowBytes/s

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/AyusK4/DDoS-Detection.git
cd DDoS-Detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

1. **Prepare the dataset**
   - Place your raw CSV files in the appropriate directories
   - Run the first cell in `New-Model.ipynb` to create balanced datasets

2. **Preprocess the data**
   - Execute cells 3-7 in the notebook to preprocess training and test sets
   - Preprocessed files will be saved in the `Preprocessed/` folder

3. **Train the model**
   - Run cells 8-22 to train and evaluate models
   - The trained model will be saved as `models/rf_top12_tuned.joblib`

### Making Predictions

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("10-1 attack-benign/models/rf_top12_tuned.joblib")

# Prepare your sample data (must include all 12 features)
sample_data = {
    'MinPacketLength': 100.0,
    'URGFlagCount': 0,
    'Inbound': 1,
    'Init_Win_bytes_forward': 8192,
    'min_seg_size_forward': 20,
    'ACKFlagCount': 2,
    'AveragePacketSize': 300.5,
    'FwdPacketLengthMean': 150.2,
    'FwdPacketLengthMin': 50.0,
    'TotalBackwardPackets': 5,
    'TotalLengthofFwdPackets': 600.0,
    'FlowBytes/s': 1200.0
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_data])

# Make prediction
prediction = model.predict(sample_df)
print("Prediction:", "ATTACK" if prediction[0] == 1 else "BENIGN")
```

## 📈 Results

### Performance Metrics

#### Test Set 1 (60% Benign, 40% Attack)
- **Accuracy**: >99%
- **Precision**: >99%
- **Recall**: >99%
- **F1-Score**: >99%
- **False Positive Rate**: <1%

#### Test Set 2 (80% Benign, 20% Attack)
- **Accuracy**: >99%
- **Precision**: >99%
- **Recall**: >99%
- **F1-Score**: >99%
- **False Positive Rate**: <1%

### Inference Time
- **Single Sample Prediction**: <5ms
- Suitable for real-time network traffic analysis

### Visual Results
All evaluation plots including confusion matrices, ROC curves, and precision-recall curves are available in the `plots_rf_top12/` directory.

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
jupyter>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Note**: Due to size constraints, trained models and large CSV files are not included in the repository. Please follow the training instructions to generate your own models.
