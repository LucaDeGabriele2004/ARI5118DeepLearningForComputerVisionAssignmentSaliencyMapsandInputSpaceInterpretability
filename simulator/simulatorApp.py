import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

# ===============================
# MODEL
# ===============================
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# ===============================
# PREPROCESSING
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess(image):
    return transform(image).unsqueeze(0)

# ===============================
# VANILLA SALIENCY
# ===============================
def compute_saliency(input_tensor):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    model.zero_grad()
    output = model(input_tensor)
    score = output.max()
    score.backward()

    saliency = input_tensor.grad.abs().squeeze().numpy()
    saliency = np.max(saliency, axis=0)
    return saliency

# ===============================
# GUIDED BACKPROP
# ===============================
def guided_backprop(input_tensor):
    def relu_hook(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_backward_hook(relu_hook))

    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    model.zero_grad()
    output = model(input_tensor)
    score = output.max()
    score.backward()

    saliency = input_tensor.grad.abs().squeeze().numpy()
    saliency = np.max(saliency, axis=0)

    # remove hooks
    for h in hooks:
        h.remove()

    return saliency

# ===============================
# SMOOTHGRAD
# ===============================
def smoothgrad(input_tensor, n_samples=20, noise_level=0.1):
    total_grad = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = (input_tensor + noise).detach().requires_grad_(True)

        model.zero_grad()
        output = model(noisy_input)
        score = output.max()
        score.backward()

        total_grad += noisy_input.grad.abs()

    saliency = total_grad / n_samples
    saliency = saliency.squeeze().numpy()
    saliency = np.max(saliency, axis=0)
    return saliency

# ===============================
# NORMALIZATION
# ===============================
def normalize_map(saliency):
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)
    return saliency

# ===============================
# HEATMAP OVERLAY
# ===============================
def overlay_heatmap(image, saliency, alpha=0.5):
    saliency = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))

    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ===============================
# UI
# ===============================
st.title("Saliency Map Interactive Simulator")

uploaded_file = st.file_uploader("Upload an image")

method = st.selectbox(
    "Select Method",
    ["Vanilla", "Guided Backprop", "SmoothGrad"]
)

alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)

if method == "SmoothGrad":
    noise = st.slider("Noise Level (σ)", 0.0, 0.5, 0.1)
    samples = st.slider("Number of Samples", 5, 50, 20)

# ===============================
# MAIN DISPLAY
# ===============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(image)

    st.image(image, caption="Original Image", use_column_width=True)

    col1, col2 = st.columns(2)

    # LEFT SIDE
    with col1:
        st.subheader("Method A")

        if method == "Vanilla":
            sal1 = compute_saliency(input_tensor)
        elif method == "Guided Backprop":
            sal1 = guided_backprop(input_tensor)
        else:
            sal1 = smoothgrad(input_tensor, samples, noise)

        sal1 = normalize_map(sal1)
        overlay1 = overlay_heatmap(image, sal1, alpha)
        st.image(overlay1)

    # RIGHT SIDE (comparison)
    with col2:
        st.subheader("Method B (SmoothGrad)")

        sal2 = smoothgrad(input_tensor, 20, 0.1)
        sal2 = normalize_map(sal2)
        overlay2 = overlay_heatmap(image, sal2, alpha)
        st.image(overlay2)