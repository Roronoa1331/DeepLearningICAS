🧠 Thermal Image Classification (ICAS Detection)
Intracranial Aneurysm Screening via Deep Learning on Thermal Images
📖 Overview

This project presents a binary classification model for detecting Intracranial Aneurysm Screening (ICAS) cases from thermal RGB images.
The model distinguishes between:

ICAS (positive) — Aneurysm indicators present

Non-ICAS (negative) — Normal thermal patterns

A ResNet-18 model with transfer learning was fine-tuned to classify thermal images effectively, addressing data imbalance and overfitting challenges.

📂 Dataset

Total Images: 950 (512×512 RGB)

Class	Count	Directory
ICAS	303	icas/
Non-ICAS	647	non_icas/

Class imbalance was mitigated during training through sampling and weighted loss adjustments.

⚙️ Model Architecture
🔹 Base Model

Backbone: ResNet-18 (pre-trained on ImageNet)

Modification: Final FC layer replaced with a single-neuron linear layer for binary output

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)

🔹 Rationale

Transfer Learning: Leverages pre-trained features, improving convergence.

ResNet-18: Lightweight yet effective for smaller datasets.

Regularization: Dropout, weight decay, and scheduler-based learning rate reduction.

🧪 Training Strategy
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Batch Size	16
Epochs	15
Scheduler	ReduceLROnPlateau
Loss Function	BCEWithLogitsLoss (with class weights)
⚖️ Class Imbalance Handling

WeightedRandomSampler: Ensures balanced mini-batches.

pos_weight in BCEWithLogitsLoss: Emphasizes minority (ICAS) class.

🧩 Regularization & Data Processing

Data Augmentation: Random rotation, horizontal flip

Normalization: ImageNet mean and std

Weight Decay: Prevents overfitting

Learning Rate Scheduler: Reduces LR on plateau

These collectively stabilized validation performance and prevented overfitting.

📈 Evaluation Metrics

Evaluation was conducted on a 15% test split.

Metric	Score
Accuracy	~0.88
Precision	~0.86
Recall	~0.83
F1-Score	~0.84
AUC (ROC)	~0.91

➡️ Metrics indicate strong generalization and good discrimination between classes.

📊 Visualization & Analysis
Visualization	Description
training_loss_curve.png	Training vs validation loss (converges after ~10 epochs)
training_accuracy_curve.png	Accuracy stabilization post epoch 10
confusion_matrix.png	Minor false negatives observed (missed ICAS cases)
roc_curve.png	AUC ≈ 0.91 — strong class separability
💬 Discussion
✅ Strengths

Robust handling of class imbalance

Effective transfer learning using ResNet-18

Visual insights via ROC and confusion matrix

⚠️ Limitations

Small dataset may limit generalization

Early epochs showed minor overfitting

🚀 Future Work

Explore Vision Transformers (ViT) or EfficientNet

Apply Grad-CAM for explainability

Expand data via Albumentations augmentations

🧠 Conclusion

The fine-tuned ResNet-18 achieved ~0.88 accuracy and 0.91 AUC on ICAS detection.
With optimized transfer learning, data balancing, and regularization, it demonstrates reliable performance and interpretability — highlighting the feasibility of thermal imaging for aneurysm screening.

🗂️ Repository Structure
📁 project_root/
├── icas/                      # ICAS images
├── non_icas/                  # Non-ICAS images
├── ThermalDeepLearning.py     # Training & evaluation script
├── confusion_matrix.png
├── roc_curve.png
├── training_loss_curve.png
├── training_accuracy_curve.png
└── README.md

⚙️ Environment & Dependencies
Library	Version
PyTorch	2.4
Torchvision	0.19
NumPy	≥1.26
Matplotlib	≥3.8
scikit-learn	≥1.5

Install dependencies:

pip install torch torchvision numpy matplotlib scikit-learn

🧪 Usage
# Train model
python ThermalDeepLearning.py --train

# Evaluate on test data
python ThermalDeepLearning.py --eval

# Generate analysis plots
python ThermalDeepLearning.py --visualize