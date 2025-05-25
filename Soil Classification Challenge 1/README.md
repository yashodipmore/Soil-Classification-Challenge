# 🌱 Soil Image Classification Challenge 1 - IIT Ropar (Annam.ai)

##  Project Overview
This repository contains the solution for the **Soil Image Classification Challenge**, where the objective is to classify soil images into four categories:
- Alluvial Soil
- Black Soil
- Clay Soil
- Red Soil

Our final deep learning model achieves a **perfect F1-score of 1.0 across all classes**, effectively maximizing the *minimum* per-class F1 as required by the competition.

---

##  Team Members
- **Yashodip More** – Electrical Engineering, RCPIT, Maharashtra – yashodipmore2004@gmail.com  
- **S.M. Sakthivel** – AI & Data Science, Achariya College, Puducherry – s.m.sakthivelofficial@gmail.com  
- **Komal Kumavat** – Electrical Engineering, RCPIT, Maharashtra – komalkumavat025@gmail.com

---

##  Project Logic & Architecture

### 1. **Data Preparation & Preprocessing**
- **Image Standardization:** All input images are resized to **224×224 pixels** to ensure uniformity for the deep learning model.
- **Augmentations:** Applied the following transformations to enhance model generalization:
  - `RandomHorizontalFlip`
  - `RandomRotation`
  - `Normalization` (using ImageNet mean and standard deviation)
- **Train/Validation Split:** The labeled dataset is divided into **80% training** and **20% validation** using `train_test_split` from `sklearn.model_selection`.

### 2. **Custom PyTorch Dataset**
- **Dynamic Loading:** Implemented a `SoilDataset` class using PyTorch's `Dataset` class to load and preprocess images on-the-fly from corresponding directories.
- **Label Extraction:** Ground-truth labels are mapped from a CSV using `pandas`.

### 3. **Model Architecture**
- **Base Model:** Used a **pretrained ResNet18** from `torchvision.models` for robust feature extraction.
- **Customization:** The final fully-connected layer is replaced to output four classes, matching the soil types.
- **Loss & Optimizer:** Model is trained using `CrossEntropyLoss` and optimized with the `Adam` optimizer.

### 4. **Training Pipeline**
- **GPU Acceleration:** Training is performed on GPU if available, falling back to CPU otherwise.
- **Progress Monitoring:** Training and validation progress are tracked with `tqdm` progress bars.
- **F1 Metrics:** After each epoch, per-class F1-scores are computed and displayed to ensure balanced performance.
- **Model Checkpointing:** The best-performing model (based on validation F1) is saved automatically.

### 5. **Prediction & Submission**
- **Test Data:** Test images are loaded and subjected to the same preprocessing pipeline as training images.
- **Inference:** Predictions are generated using the trained model.
- **Submission Format:** Output is stored in `submission.csv` with two columns: image IDs and predicted soil labels, formatted as required by the challenge.

---

## 🛠️ Setup Instructions

### 1. Environment Requirements

A. **Clone the Repository**
```bash
git clone https://github.com/yashodipmore/Soil-Classification-Challenge.git
cd "Soil Classification Challenge 1"
```

B. **Create and Activate a Virtual Environment**
```bash
# Using conda (recommended)
conda create -n soil-classification python=3.10 -y
conda activate soil-classification
```

C. **Install Dependencies**
```bash
pip install -r requirements.txt
```
**Contents of `requirements.txt`:**
```
PIL
numpy
os
pandas
sklearn
torch
torchvision
tqdm
```

### 2. Data Setup
- Place the training, validation, and test images in the respective folders (see project structure).
- Ensure the labels CSV is present in the `data/` directory.

### 3. Running the Solution

A. **Training**
- Run the main training notebook/script in the `notebooks/` folder.
- Example (if using Jupyter):
  1. Open `notebooks/<training_notebook>.ipynb`.
  2. Execute all cells after configuring the paths as needed.

B. **Prediction**
- After training, run the inference cell/script to generate predictions for the test set.
- The predictions will be saved as `submission.csv`.

### 4. Project Structure

```
Soil Classification Challenge 1/
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── labels.csv
├── notebooks/
│   ├── training_notebook.ipynb
│   └── inference_notebook.ipynb
├── requirements.txt
├── submission.csv
└── README.md
```

---

##  Detailed Logic Flow

1. **Load Data:** Read image paths and labels from CSV.
2. **Transform:** Apply PyTorch transforms (resize, augment, normalize).
3. **Custom Dataset:** Dynamically fetch images and labels.
4. **Model Initialization:** Load pretrained ResNet18, modify output layer.
5. **Training Loop:** For each epoch:
   - Train on batches.
   - Validate and compute per-class F1.
   - Save model if validation F1 improves.
6. **Testing:** Load best model, preprocess test images, predict classes.
7. **Submission:** Format predictions as per competition and export to CSV.

---

##  Troubleshooting

- If CUDA is not available, the code falls back to CPU but will be slower.
- Ensure all required libraries are installed and match the specified versions.
- If you encounter missing files or directories, check the project structure above.

---

##  Acknowledgements

- Challenge by [IIT Ropar & Annam.ai](https://annam.ai/)
- Pretrained model weights from [torchvision](https://pytorch.org/vision/stable/index.html)

---

##  Contact

For queries, contact any team member (see above) or open an issue in this repo.

---

*Happy Coding and Good Luck with Soil Classification!*
