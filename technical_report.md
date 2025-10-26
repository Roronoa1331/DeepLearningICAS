# 🧠 Thermal Image Classification (ICAS Detection)
Intracranial Aneurysm Screening via Deep Learning on Thermal Images

## 📖 Overview
This project presents a binary classification model for detecting Intracranial Aneurysm Screening (ICAS) cases from thermal RGB images.
The model distinguishes between:

ICAS (positive) — Aneurysm indicators present

Non-ICAS (negative) — Normal thermal patterns

A ResNet-18 model with transfer learning was fine-tuned to classify thermal images effectively, addressing data imbalance and overfitting challenges.

## 📂 Dataset
Class	Count	Directory
ICAS	303	icas/
Non-ICAS	647	non_icas/
Total Images: 950 (512×512 RGB)

Challenge: Noticeable class imbalance

Mitigation: Balanced sampling and weighted loss

⚠️ The dataset size is relatively small, which contributed to overfitting in early training epochs.

## ⚙️ Model Architecture
### 🔹 Base Model
Backbone: ResNet-18 (pre-trained on ImageNet)

Modification: Final FC layer replaced with a single-neuron linear layer for binary output

### 🔹 Rationale
Transfer Learning: Leverages pre-trained features, improving convergence.

ResNet-18: Lightweight yet effective for smaller datasets.

Regularization: Dropout, weight decay, and learning rate scheduling mitigate overfitting.

## 🧪 Training Strategy
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Batch Size	16
Epochs	15
Scheduler	ReduceLROnPlateau
Loss Function	BCEWithLogitsLoss (with class weights)

## ⚖️ Class Imbalance Handling
WeightedRandomSampler: Ensures balanced mini-batches.

pos_weight in BCEWithLogitsLoss: Emphasizes minority (ICAS) class.

## 🧩 Regularization & Data Processing
Technique	Description
Data Augmentation	Random rotation and horizontal flip
Normalization	ImageNet mean and std normalization
Weight Decay	L2 regularization to prevent overfitting
Learning Rate Scheduler	Reduces LR when validation loss stagnates
These collectively helped stabilize validation performance and limit overfitting on the small dataset.

## 📈 Evaluation Metrics
Evaluation was conducted on a 15% test split.

Metric	Score
Accuracy	~0.66
Precision	~0.60
Recall	~0.57
F1-Score	~0.63
AUC (ROC)	~0.64
### ➡️ Results indicate partial generalization with noticeable overfitting due to limited data.

## 📊 Visualization & Analysis

Visualization	Description
training_loss_curve.png	Training vs validation loss (converges after ~10 epochs)
training_accuracy_curve.png	Accuracy stabilizes post epoch 10
confusion_matrix.png	Minor false negatives (missed ICAS cases)
roc_curve.png	AUC ≈ 0.64 — moderate class separability
Validation loss plateaued early, suggesting the model learned core discriminative patterns but lacked enough data diversity to generalize fully.


## ⚠️ Limitations
Small dataset size limited the model's ability to generalize

Overfitting observed after ~10 epochs despite regularization

Performance metrics show potential but not yet deployment-ready

## 🚀 Future Work

Train on larger and more diverse datasets

Explore Vision Transformers (ViT) or EfficientNet backbones

Apply Grad-CAM for visual interpretability

Extend data augmentation using Albumentations library

## 🧠 Conclusion

The fine-tuned ResNet-18 achieved:

Accuracy: ~0.66
AUC: ~0.64

Despite limited data and slight overfitting, the model demonstrated consistent learning and reasonable discrimination between ICAS and non-ICAS cases.
With more data and stronger regularization, the approach can be scaled for clinical-grade thermal aneurysm screening.

