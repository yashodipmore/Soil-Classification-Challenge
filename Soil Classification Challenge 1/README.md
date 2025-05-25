<<<<<<< HEAD
# Soil-Classification_anam
=======
# 🌱 Soil Image Classification Challenge - IIT Ropar (Annam.ai)

## 📌 Project Overview
This project solves the **Soil Image Classification Challenge** by accurately classifying soil images into four categories:
- Alluvial Soil
- Black Soil
- Clay Soil
- Red Soil

Our final model achieves a **perfect F1-score of 1.0 across all classes**, meeting the competition’s objective to maximize the *minimum* per-class F1.

---

## 👨‍💻 Team Members
- **Yashodip More** – Electrical Engineering, RCPIT, Maharashtra – yashodipmore2004@gmail.com  
- **S.M. Sakthivel** – AI & Data Science, Achariya College, Puducherry – s.m.sakthivelofficial@gmail.com  
- **Komal Kumavat** – Electrical Engineering, RCPIT, Maharashtra – komalkumavat025@gmail.com

---

## 🧠 Project Logic & Architecture

### 1. **Preprocessing**
- Standardized all input images to **224×224 pixels**.
- Applied **transformations**: RandomHorizontalFlip, RandomRotation, Normalization (ImageNet mean/std).
- Split dataset into **80% Train, 20% Validation** using `train_test_split`.

### 2. **Custom Dataset Class**
- Created a `SoilDataset` PyTorch class to load and preprocess images dynamically from folders.
- Ground truth labels were extracted from a CSV file using `pandas`.

### 3. **Model Architecture**
- Utilized a **pretrained ResNet18** from `torchvision.models`.
- Replaced the final layer to output **4 classes**.
- Trained using `CrossEntropyLoss` and `Adam` optimizer.

### 4. **Training Pipeline**
- Trained the model for multiple epochs using GPU (if available).
- Used `tqdm` for monitoring progress.
- Monitored and printed **per-class F1-scores** during each validation epoch.
- Saved the best model based on validation F1-score.

### 5. **Prediction & Submission**
- Loaded test images, applied same transformations.
- Generated predictions using the trained model.
- Created a **submission CSV** with image IDs and predicted labels.

---

## 🛠️ Setup Instructions

### 1. Environment Requirements
Create a virtual environment and install dependencies:

```bash
conda create -n soil-classification python=3.10 -y
conda activate soil-classification
pip install -r requirements.txt
>>>>>>> d9c2ad1 (Initial commit)
