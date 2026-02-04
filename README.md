

# EEG Stress Classification using Machine Learning

## 📋 Project Overview
This project implements a comprehensive machine learning pipeline for classifying stress levels from EEG (Electroencephalography) data. The system compares multiple ML algorithms to identify the most effective approach for stress detection.

## 🎯 Objectives
- Classify EEG signals into "Stressed" (1) and "Non-Stressed" (0) states
- Compare performance of 10+ machine learning algorithms
- Identify optimal model for EEG-based stress detection
- Provide reproducible research code for academic/industrial applications

## 📊 Dataset
- **Source**: EEG recordings from 30 subjects
- **Samples**: 54,914 instances
- **Features**: 27 EEG-derived features (mean, std, entropy, etc.)
- **Classes**: Binary classification (0: Non-Stressed, 1: Stressed)

## 🤖 Models Implemented

### Traditional Machine Learning
1. **Random Forest** - 84.40% accuracy
2. **Decision Tree** - 79.99% accuracy
3. **Logistic Regression** - 58.54% accuracy
4. **Support Vector Machine (SVM)** - 59.36% accuracy
5. **K-Nearest Neighbors (KNN)** - 63.31% accuracy

### Ensemble Methods
6. **Gradient Boosting** - 75.62% accuracy
7. **AdaBoost** - 77.51% accuracy
8. **XGBoost** - Competitive performance
9. **LightGBM** - 83.50% accuracy
10. **Extra Trees** - 85.40% accuracy (Best)

### Other Models
11. **Bernoulli Naive Bayes** - 59.36% accuracy
12. **Neural Network** - Competitive performance (~85% accuracy)

## 🚀 Installation & Setup

### Prerequisites
```bash
python>=3.8




