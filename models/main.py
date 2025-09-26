import torch
from torchvision import transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import math

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
MODEL_PATH = "ensemble_traced.pt"  # TorchScript model
CLASSES_PATH = "classes.json"      # Species mapping
TEST_DIR = "test_images"           # Folder containing test images

# --------------------
# LOAD MODEL
# --------------------
ensemble = torch.jit.load(MODEL_PATH, map_location=DEVICE)
ensemble.eval()
print("✅ TorchScript model loaded successfully!")

# --------------------
# LOAD CLASSES
# --------------------
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# --------------------
# IMAGE TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------
# PREDICTION FUNCTION
# --------------------
def predict_image(img_path, model, transform, device=DEVICE):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()
    return classes[pred_idx], img

# --------------------
# LOAD ALL IMAGES AND PREDICTIONS
# --------------------
images = []
labels = []
filenames = []

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(TEST_DIR, fname)
    pred_label, img = predict_image(img_path, ensemble, transform)
    images.append(img)
    labels.append(pred_label)
    filenames.append(fname)
    print(f"{fname} → {pred_label}")

print("✅ All predictions completed!")

# --------------------
# DISPLAY ALL IMAGES IN A GRID
# --------------------
num_images = len(images)
cols = 4  # images per row
rows = math.ceil(num_images / cols)

plt.figure(figsize=(cols*4, rows*4))

for i, (img, label, fname) in enumerate(zip(images, labels, filenames)):
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label}", fontsize=10)

plt.tight_layout()
plt.show()
