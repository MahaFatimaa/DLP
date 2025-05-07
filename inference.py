import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F  # Importing softmax for probability calculation

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
model_dir = "model_dir"  # Path to the folder where your model is saved
pth_file = r"model_output\model.pth"  # Use raw string to avoid escape sequence issues
image_path = r"test\2\00002_00000_00012.png"  # Path to the image for inference

# Load the image processor and model
image_processor = ViTImageProcessor.from_pretrained(model_dir)
model = AutoModelForImageClassification.from_pretrained(
    model_dir,
    num_labels=43,  # Ensure this matches the number of classes your model was trained on
    ignore_mismatched_sizes=True
).to(device)

# Load the trained weights from the .pth file
model.load_state_dict(torch.load(pth_file))
model.eval()  # Set model to evaluation mode

# Define class label mapping
id2label = {i: str(i) for i in range(43)}
model.config.id2label = id2label

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt").to(device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get probabilities
logits = outputs.logits
probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

# Get top prediction
predicted_class_idx = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
confidence = probabilities[predicted_class_idx] * 100

# Output result
print(f"\nPredicted Class Index: {predicted_class_idx}")
print(f"Predicted Class Label: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")
