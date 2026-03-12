# AOI Defect Classification with CNN

A Convolutional Neural Network (CNN) built with PyTorch to classify surface defects detected by Automated Optical Inspection (AOI) systems in PCB manufacturing.

## 📌 Project Overview

AOI (Automated Optical Inspection) is widely used in PCB and semiconductor manufacturing to detect product defects. This project trains a custom CNN to classify grayscale images into **6 defect categories**, automating the inspection process that would otherwise require manual review.

**Dataset source:** [AIdea — Defect Classifications of AOI](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)  
*(Provided by Industrial Technology Research Institute, ITRI)*

| Label | Class |
|-------|-------|
| 0 | Normal |
| 1 | Void |
| 2 | Horizontal Defect |
| 3 | Vertical Defect |
| 4 | Edge Defect |
| 5 | Particle |

- **Training images:** 2,528 PNG images
- **Test images:** 10,142 PNG images
- **Image format:** Grayscale PNG

## 🏗️ Model Architecture

A custom CNN (`Net`) built from scratch with PyTorch:

```
Input (1-channel grayscale image, 512×512)
  → Conv2d(1→6, kernel=5) → ReLU → MaxPool2d(2,2)
  → Conv2d(6→16, kernel=5) → ReLU → MaxPool2d(2,2)
  → Flatten
  → Linear(16×125×125, 120) → ReLU
  → Linear(120, 84) → ReLU
  → Linear(84, 6)
Output: 6-class prediction
```

| Hyperparameter | Value |
|----------------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |
| Learning Rate | 0.001 |
| Momentum | 0.9 |
| Epochs | 100 |
| Batch Size | 16 |

## 📁 Project Structure

```
AOI-Defect-Classification/
├── AOI.py               # Main training and inference script
├── train.csv            # Training annotations (ID, Label)
├── test.csv             # Test annotations (ID, Label)
├── train_images/        # Training images (download from AIdea)
├── test_images/         # Test images (download from AIdea)
├── cifar_net.pth        # Saved model weights (generated after training)
├── cnn.csv              # Prediction output (generated after inference)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ned0624/AOI-Defect-Classification.git
cd AOI-Defect-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Register and download the dataset from [AIdea](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4), then place the files as follows:

```
train.csv
test.csv
train_images/   ← unzip train_images.zip here
test_images/    ← unzip test_images.zip here
```

### 4. Train & run inference

```bash
python AOI.py
```

This will:
1. Train the CNN for 100 epochs
2. Save model weights to `cifar_net.pth`
3. Run inference on the test set
4. Output predictions to `cnn.csv`

## 🛠️ Tech Stack

- **Python** 3.8+
- **PyTorch** — model definition, training loop, GPU support
- **Torchvision** — image transforms
- **Pandas** — data loading and CSV I/O
- **Matplotlib / NumPy** — visualization

## 💡 Key Concepts Demonstrated

- Custom `Dataset` class using `torch.utils.data.Dataset`
- CNN architecture design from scratch
- GPU/CPU device handling with `torch.device`
- Model serialization with `torch.save` / `torch.load`
- Batch inference with prediction aggregation

---

*Dataset provided by ITRI via the AIdea platform.*
