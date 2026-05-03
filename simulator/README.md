# Saliency Map Interactive Simulator

## Overview

This interactive simulator shows the **input-space interpretability methods** that are used in deep learning for computer vision
This tool allows users to visualize how a convolutional neural network discovers important areas of an image when making predictions.

The simulator compares three saliency-based attribution methods:

- **Vanilla Saliency Maps**
- **Guided Backpropagation**
- **SmoothGrad**

Users can upload their own images, adjust parameters in real time, and compare attribution methods side-by-side.

---

## Features

### Interactive Image Upload
Upload any RGB image for saliency analysis.

### Multiple Attribution Methods
Select between:

- Vanilla Saliency
- Guided Backpropagation
- SmoothGrad

### Adjustable Parameters

#### Overlay Transparency
Controls visibility of saliency heatmap overlay.
**Range:** 0.0 – 1.0

---

#### SmoothGrad Noise Level (σ)
Controls Gaussian noise added to input.

**Typical values:** 0.05 – 0.20

Higher values:
- More smoothing
- Less sharp localisation

Lower values:
- Sharper detail
- More noise

---

#### Number of Samples
Number of noisy forward/backward passes.

**Range:** 5 – 50

Higher values:
- Better smoothing
- Slower execution

---

### Side-by-Side Comparison
Compare:

- Selected attribution method
- Default SmoothGrad baseline

---

### CPU-Only Execution
No GPU required.

Designed to run on standard CPU hardware.

---
## Installation

### 1. Clone Repository

The GitHub Repository is cloned by using the following command
```bash
git clone https://github.com/LucaDeGabriele2004/ARI5118DeepLearningForComputerVisionAssignmentSaliencyMapsandInputSpaceInterpretability
```
This command was then used to be able to access the directory to be able to install and load the simulator:
```bash
cd simulator
```

### 2. Install Requirements for Simulator

To install the neccessary Python packages that are required for the simulator to run, a requirements.txt text file was created. 
This text file contains the following Python packages:

streamlit
torch
torchvision
numpy
opencv-python
Pillow

The following bash command was then used to install these Python packages:
```bash
pip install -r requirements.txt
```

### 3. Execute the Simulator

To start the Streamlit simulator, the following bash command was used:
```bash
python -m streamlit run simulatorApp.py
```
The simulator will open automatically on your browser. If it does not open automatically, go to the URL: http://localhost:8501
Note: Port Number may be different

### 4. How to Use the Simulator

**Step 1:** Upload an image
Recommended examples:

- Animals
- Vehicles
- Faces
- Objects with clear foreground/background separation

**Step 2:** Select the Attribution Method (Vanilla Saliency, Guided Backpropagation and SmoothGrad)

**Step 3:** Adjust the parameters

**Step 4:** Look at the differences in the highlighted areas

Compare:

- Localisation sharpness
- Noise level
- Attention focus

### 5. Limitations

- Uses pretrained ResNet18
- ImageNet-based feature understanding
- Not suitable for production interpretability validation
- Saliency does not guarantee causal explanation
