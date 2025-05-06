import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_dir = 'data'  # Path to the original data folder containing all 43 class folders
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# Create directories for train, test, and validation if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class directory in the 'data' folder
classes = os.listdir(data_dir)
for class_name in classes:
    class_folder = os.path.join(data_dir, class_name)
    
    # Skip if not a directory (in case of hidden files or non-class folders)
    if not os.path.isdir(class_folder):
        continue

    # Get all file paths in the class folder
    file_paths = [os.path.join(class_folder, file) for file in os.listdir(class_folder) if file.endswith('.jpg') or file.endswith('.png')]

    # Split the data into train, valid, and test
    train_files, temp_files = train_test_split(file_paths, test_size=0.3, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=2/3, random_state=42)  # 0.33 x 0.3 = 0.10 for validation

    # Create subdirectories for each class inside train, valid, and test folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move files to corresponding directories
    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, class_name, os.path.basename(file)))
    for file in valid_files:
        shutil.copy(file, os.path.join(valid_dir, class_name, os.path.basename(file)))
    for file in test_files:
        shutil.copy(file, os.path.join(test_dir, class_name, os.path.basename(file)))

# Load the image processor for preprocessing images
model_dir = "model_dir"  # Update this to the folder where you saved the downloaded files
image_processor = ViTImageProcessor.from_pretrained(model_dir)

# Load the pre-trained ViT model with 43 output classes (change num_labels to 43)
model = AutoModelForImageClassification.from_pretrained(
    model_dir,
    num_labels=43,
    ignore_mismatched_sizes=True
).to(device)

# Define the image transformations
def get_transforms(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = image_processor.size.get("shortest_edge", (image_processor.size["height"], image_processor.size["width"]))
    return Compose([
        RandomResizedCrop(size),
        ToTensor(),
        normalize,
    ])

# Apply transformations
transforms = get_transforms(image_processor)

# Load datasets using ImageFolder and apply transformations
train_dataset = ImageFolder(train_dir, transform=transforms)
val_dataset = ImageFolder(valid_dir, transform=transforms)
test_dataset = ImageFolder(test_dir, transform=transforms)

# Verify dataset sizes
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Prepare label dictionaries
labels = train_dataset.classes
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

model.config.label2id = label2id
model.config.id2label = id2label

# Define training arguments with automatic checkpoint saving
output_dir = "./model_output"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,  # Keep only the last 5 checkpoints
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    push_to_hub=False,
    report_to="none",
)

# Define the metric computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')  # Weighted F1 score for multi-class
    return {"accuracy": accuracy, "f1": f1}

# Define a custom data collator to handle batches
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return {"pixel_values": torch.stack(images), "labels": torch.tensor(labels)}

# Function to find the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = [d for d in checkpoints if os.path.isdir(d)]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

# Check if a checkpoint exists
latest_checkpoint = get_latest_checkpoint(output_dir)

# Initialize the Trainer with built-in checkpoint saving
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_collate_fn,
)

# Train the model, resuming from the latest checkpoint if available
if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found, starting training from scratch.")
    trainer.train()

# Evaluate the model on the test set after training is complete
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("\nFinal Test Metrics:")
print(f"Test Loss: {test_results['eval_loss']:.4f}")
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {test_results['eval_f1']:.4f}")  # Print the F1 score

# Save the model as a .pth file
pth_save_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), pth_save_path)
print(f"Model saved as .pth file at {pth_save_path}")