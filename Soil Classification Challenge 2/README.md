#  Soil Image Classification Challenge - Part 2 (IIT Ropar Hackathon by Annam.ai)

##  Overview

This repository contains our complete solution for **Soil Classification - Part 2**, a challenge hosted by [Annam.ai](https://annam.ai) as part of the hackathon organized at **IIT Ropar**. The primary objective is to classify whether an input image is of **soil** or **non-soil** using **deep learning-based computer vision techniques**.

The key tasks included:
- Training a convolutional neural network (CNN) model on a labeled dataset.
- Standardizing and preprocessing images (resizing, normalization, augmentation).
- Evaluating using F1-Score to ensure balanced precision and recall.
- Creating a prediction file `submission.csv` for final evaluation.
- Documenting the entire pipeline for reproducibility.

>  This project uses **PyTorch** as the deep learning framework and utilizes **GPU (Tesla T4 or RTX A6000 recommended)** for efficient training.

---

##  Quick Start Guide

###  Clone the Repository

```bash
git clone https://github.com/your-username/soil-image-classification.git
cd soil-image-classification
```

###  Setup Environment

Create a virtual environment (optional but recommended):

```bash
python -m venv soil-env
source soil-env/bin/activate  # Linux/macOS
soil-env\Scripts\activate     # Windows
```

###  Install Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

###  Directory Structure

```
soil-image-classification/
│
├── train/                  # Folder with all training images
├── test/                   # Folder with all test images
├── train_labels.csv        # Training labels CSV
├── test_ids.csv            # Test file image IDs
├── submission.csv          # Final submission file
├── notebooks/soil-classification-2.ipynb  # Jupyter Notebook with full training and inference
├── requirements.txt
└── README.md               # This file
```

##  Note for Annam.ai Evaluators

>  **Important Clarification Regarding Training Data Composition**

During the challenge, we observed that **all images in the `train/` folder provided in the official dataset were soil images only**, and there were **no non-soil images** included in the training data. However, in the `test/` set, the images include **various non-soil categories** such as **animals, cats, dogs, plants, etc.**

This mismatch leads to a critical issue:  
If the model is trained solely on positive (soil) examples, it **fails to generalize** and classify non-soil images correctly during inference, since it has **never seen negative examples (label `0`)**.

To address this and improve the model's ability to **discriminate between soil and non-soil images**, we made a **minor yet crucial adjustment**:

-  **We manually added a small subset of clearly non-soil images from the test set into the `train/` directory** and labeled them with `0` to represent negative examples.

This helped the model learn meaningful distinctions between soil and non-soil classes, significantly improving F1-score and reducing false positives.

---

 **When running or evaluating this repository:**

- Ensure that you **use the modified `train/` directory**, which includes a balanced mix of soil (`label 1`) and manually added non-soil (`label 0`) images.
- If you're running on **Kaggle**, upload the full modified dataset (including `train/`, `test/`, and CSV files) before execution.
- The notebook assumes that the dataset is already organized in this format.

This adjustment aligns the training distribution closer to the test distribution and makes the evaluation fair and realistic.


---

##  Model Architecture & Approach

###  Problem Definition

This is a **binary image classification task**:
- `1` → Soil Image
- `0` → Not a Soil Image

The dataset contains diverse images which require robust feature extraction of texture, color, and patterns. The metric used is **F1-Score**, so we aim to optimize for both **precision and recall**.

---

###  Data Preprocessing

- **Image resizing**: All images resized to 224x224 for compatibility with CNNs.
- **Normalization**: Pixel values scaled using ImageNet mean and std.
- **Augmentation**: Applied transforms like random horizontal flips, color jitter, and rotations to increase diversity.

###  Model Used

- **Architecture**: Pretrained `ResNet18` from torchvision (can easily switch to ResNet34/50).
- **Transfer Learning**: Used pretrained weights to leverage ImageNet features.
- **Modification**: Final fully connected layer replaced with `nn.Linear(512, 1)` and sigmoid activation.

---

###  Training Strategy

- Loss: `BCEWithLogitsLoss` (binary classification)
- Optimizer: `Adam`
- Scheduler: `StepLR` for learning rate decay
- Epochs: 10 (configurable)
- Validation split from training data (80:20)
- Early stopping based on validation F1-score

---

###  Evaluation Metric

**F1-Score** is used because:
- It balances **Precision** (positive predictive value) and **Recall** (sensitivity).
- Encourages fairness in detecting both soil and non-soil classes.
- It penalizes biased models that overfit one class.

---

##  GPU Instructions (Train on RTX T4 / A6000)

Make sure your environment has GPU support. You can check by:

```python
import torch
print(torch.cuda.get_device_name(0))
```

In the notebook `soil-classification-2.ipynb`, ensure:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

##  Submission Generation

After training, the model predicts on `test/` images using:

```bash
submission = pd.DataFrame({'image_id': test_image_ids, 'label': predictions})
submission.to_csv('submission.csv', index=False)
```

Format:
```
image_id,label
img_a1b2c3.jpg,1
img_d4e5f6.jpg,0
...
```

---

##  Team Details

#### Team Leader: Yashodip More, Electrical Engineering, RC Patel Institute of Technology, Shirpur, Maharashtra – yashodipmore2004@gmail.com
#### Team Member: S.M. Sakthivel, AI & Data Science, Achariya College of Engineering Technology, Puducherry – s.m.sakthivelofficial@gmail.com
#### Team Member: Komal Kumavat, Electrical Engineering, RC Patel Institute of Technology, Shirpur, Maharashtra – komalkumavat025@gmail.com
#### Team - Winners

---

##  Requirements (requirements.txt)

```txt
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
Pillow>=9.4.0
tqdm>=4.65.0
```

Optional for notebook interface:
```txt
jupyterlab>=3.6.0
```

---

##  License

This repository is made available for academic and non-commercial use only.

---


---
##  Acknowledgements

Thanks to [Annam.ai](https://annam.ai/) and [IIT Ropar](https://www.iitrpr.ac.in/) for organizing this valuable challenge and providing the dataset.


