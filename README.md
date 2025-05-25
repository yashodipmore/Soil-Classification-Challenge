# Soil-Classification-Challenge

##  Soil Image Classification Challenge (IIT Ropar, Annam.ai)

This repository contains our complete solutions to two soil image classification challenges hosted by [Annam.ai](https://annam.ai) at **IIT Ropar**. The primary goal was to develop high-performing deep learning models to accurately classify soil images, maximizing **F1-score** for balanced predictions.

##  Competition Results

- **Challenge 1 (Multiclass):**  
  - **Kaggle Rank:** 18th  
  - **Best F1-score:** 1.0000

- **Challenge 2 (Binary):**  
  - **Kaggle Rank:** 24th  
  - **Best F1-score:** 0.9751

---

##  Project Structure

- `Soil Classification Challenge 1/`  
  Multiclass classification of soil images into four types: *Alluvial*, *Black*, *Clay*, and *Red*.

- `Soil Classification Challenge 2/`  
  Binary classification: Soil vs. Non-Soil images, with special focus on handling class imbalance.

Each folder contains:
- All code, Jupyter notebooks, and `README.md` for that specific challenge.
- Example requirements and setup instructions.
- Datasets and submission templates (see inside respective folders).

---

##  Team Members
- **Team Name - Winners**
- **Yashodip More** – Electrical Engineering, RCPIT, Maharashtra – yashodipmore2004@gmail.com  
- **S.M. Sakthivel** – AI & Data Science, Achariya College, Puducherry – s.m.sakthivelofficial@gmail.com  
- **Komal Kumavat** – Electrical Engineering, RCPIT, Maharashtra – komalkumavat025@gmail.com

---

## 🏆 Highlights

- **Challenge 1:** Achieved a perfect F1-score of 1.0 across all soil classes using a fine-tuned ResNet18 model.
- **Challenge 2:** Addressed class imbalance by manually augmenting the dataset with non-soil images, significantly improving generalization and reducing false positives.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yashodipmore/Soil-Classification-Challenge.git
cd Soil-Classification-Challenge
```

### 2. Environment Setup

- **Option 1: Conda**

  ```bash
  conda create -n soil-classification python=3.10 -y
  conda activate soil-classification
  pip install -r requirements.txt
  ```

- **Option 2: Virtualenv**

  ```bash
  python -m venv soil-env
  source soil-env/bin/activate  # Linux/macOS
  soil-env\Scripts\activate     # Windows
  pip install -r requirements.txt
  ```

### 3. Directory Structure

```
Soil-Classification-Challenge/
│
├── Soil Classification Challenge 1/
│   ├── train/
│   ├── test/
│   ├── train_labels.csv
│   ├── test_ids.csv
│   ├── submission.csv
│   ├── soil-classification-1.ipynb
│   └── README.md
│
├── Soil Classification Challenge 2/
│   ├── train/
│   ├── test/
│   ├── train_labels.csv
│   ├── test_ids.csv
│   ├── submission.csv
│   ├── soil-classification-2.ipynb
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## 🧠 Project Logic & Approach

### Challenge 1: Four-Class Soil Classification

- **Preprocessing:**  
  - All images resized to 224x224.  
  - Applied augmentations: flips, rotation, normalization.
- **Dataset:**  
  - Custom PyTorch `SoilDataset` class, labels from CSV.
- **Model:**  
  - Pretrained ResNet18, last layer adapted for 4 classes.
- **Training:**  
  - CrossEntropyLoss, Adam optimizer, F1-score monitored per class.
  - Best model checkpointed based on validation F1.
- **Prediction:**  
  - Test set inference, submission CSV generated.

### Challenge 2: Soil vs. Non-Soil (Binary Classification)

- **Problem:**  
  - Binary task (`1`: Soil, `0`: Not Soil).  
  - Training set had only soil images; non-soil images added manually for generalization.
- **Model:**  
  - Pretrained ResNet18, last FC layer changed to binary output.
  - BCEWithLogitsLoss, Adam optimizer.
- **Augmentation:**  
  - Flips, color jitter, rotation.
- **Evaluation:**  
  - F1-score used for model selection and validation.
- **Submission:**  
  - `submission.csv` with image_id and predicted label.

---

## 📦 Requirements

See `requirements.txt`. Key dependencies:
- torch >= 2.0.0
- torchvision >= 0.15.0
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.7.0
- Pillow >= 9.4.0
- tqdm >= 4.65.0
- (optional) jupyterlab >= 3.6.0

---

## ⚡ GPU Training

- Training is optimized for GPU (NVIDIA Tesla T4/RTX A6000 recommended).
- In the notebooks, ensure CUDA is detected:
  ```python
  import torch
  print(torch.cuda.get_device_name(0))
  ```
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

---

## 📝 License

This repository is made available for academic and non-commercial use only.

---

## 🙏 Acknowledgements

Thanks to [Annam.ai](https://annam.ai/) and [IIT Ropar](https://www.iitrpr.ac.in/) for organizing this valuable challenge and providing the dataset.

---

For more details, see the README in each challenge folder!
