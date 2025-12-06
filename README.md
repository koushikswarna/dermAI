# DermAI 🔬

> AI-powered dermatological diagnostic system for automated skin lesion segmentation and disease classification

An end-to-end deep learning pipeline that analyzes dermoscopic images to identify and classify skin lesions across 7 disease categories, including life-threatening melanoma. Combines semantic segmentation with multi-class classification for comprehensive dermatological analysis.

## 🎯 Overview

DermAI uses a dual-model architecture to provide pixel-perfect lesion boundaries and accurate disease diagnosis:

- **U-Net Segmentation Network** - Identifies precise lesion boundaries with pixel-level accuracy
- **Custom CNN Classifier** - Classifies lesions into 7 diagnostic categories with confidence scores
- **Complete Pipeline** - Seamless integration from raw image input to clinical prediction output

## 🏥 Clinical Significance

Melanoma detection at early stages achieves 99% survival rates, but drops dramatically when diagnosed late. This system demonstrates the potential of AI-assisted screening to improve diagnostic accuracy and accessibility, particularly in regions with limited dermatological expertise.

## 📊 Technical Details

**Architecture:**
- U-Net encoder-decoder with skip connections for segmentation
- 5-layer CNN with batch normalization and dropout for classification
- Binary cross-entropy loss (segmentation) + cross-entropy loss (classification)

**Dataset:**
- HAM10000: 10,015 dermoscopic images
- ISIC 2018: Additional segmentation masks for training
- Real-world clinical data with diverse skin types and lesion presentations

**Disease Categories:**
1. Melanoma (MEL) - Malignant skin cancer
2. Melanocytic Nevus (NV) - Common moles
3. Basal Cell Carcinoma (BCC) - Most common skin cancer
4. Actinic Keratoses (AKIEC) - Pre-cancerous lesions
5. Benign Keratosis (BKL) - Non-cancerous growth
6. Dermatofibroma (DF) - Benign fibrous nodule
7. Vascular Lesions (VASC) - Blood vessel abnormalities

## 🛠️ Technology Stack

- **PyTorch** - Deep learning framework
- **OpenCV & PIL** - Image preprocessing and augmentation
- **NumPy & Pandas** - Data manipulation and analysis
- **Medical Imaging** - HAM10000, ISIC datasets
