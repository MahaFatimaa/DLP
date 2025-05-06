import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def data_preprocessing(data_dir):
    train_dir = os.path.join(data_dir, 'Train')

    print("----------- Starting Data Preprocessing -----------")
    
    # Load dataset
    data = []
    labels = []
    classes = 43
    
    # Load training images
    for label in range(classes):
        label_dir = os.path.join(train_dir, str(label))
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
            data.append(img)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Split 70% for training and 30% for the combined validation + test set
    X_train, X_remaining, y_train, y_remaining = train_test_split(data, labels, test_size=0.3, random_state=42)
    print(f"Train shape: {X_train.shape}, Remaining shape: {X_remaining.shape}")
    
    # Split the remaining 30% into 20% test and 10% validation
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=2/3, random_state=42)
    print(f"Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # One-hot encoding for labels
    y_train = to_categorical(y_train, classes)
    y_val = to_categorical(y_val, classes)
    y_test = to_categorical(y_test, classes)

    # Save preprocessed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    print("----------- Data Preprocessing Complete -----------")
    return X_train, X_val, X_test, y_train, y_val, y_test


def exploratory_data_analysis(data_dir):
    print("----------- Starting Exploratory Data Analysis (EDA) -----------")
    
    # Load the training data
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    # Count the occurrences of each class
    class_counts = np.sum(y_train, axis=0)
    class_labels = np.arange(len(class_counts))

    # Class distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_labels, y=class_counts)
    plt.title("Class Distribution of Traffic Signs")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(class_labels)
    plt.show()

    # Load sample images for visualization
    sample_images = []
    sample_labels = []
    classes = 43
    
    print("Loading sample images...")
    for label in range(classes):
        label_dir = os.path.join(data_dir, 'Train', str(label))
        sample_img = os.listdir(label_dir)[0]  # Get the first image
        img_path = os.path.join(label_dir, sample_img)
        img = plt.imread(img_path)
        sample_images.append(img)
        sample_labels.append(label)
    
    # Plot sample images
    plt.figure(figsize=(15, 10))
    for i in range(1, 11):
        plt.subplot(2, 5, i)
        plt.imshow(sample_images[i - 1])
        plt.title(f"Class: {sample_labels[i - 1]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("----------- EDA Complete -----------")


def build_and_train_model(X_train, y_train, X_val, y_val, model_dir, epochs=20, batch_size=64):
    print("----------- Starting Model Training -----------")
    print("Building model...")
    
    model = Sequential()

    # First Layer
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Second Layer 
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Dense Layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    # Save the model and history
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'best_model.h5'))
    np.save(os.path.join(model_dir, 'history.npy'), history.history)

    print("----------- Model Training Complete -----------")


def evaluate_model(X_test, y_test, model_dir):
    print("----------- Starting Model Evaluation -----------")
    print("Loading model...")
    
    model_path = os.path.join(model_dir, 'best_model.h5')
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate F1 score, Precision, and Recall
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')

    print(f"Test Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}%")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    
    # Load training history
    history_path = os.path.join(model_dir, 'history.npy')
    if os.path.exists(history_path):
        print("Loading training history...")
        history = np.load(history_path, allow_pickle=True).item()

        # Plot accuracy and loss
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    print("----------- Model Evaluation Complete -----------")


def run(data_dir, model_dir, epochs=40, batch_size=64):
    # Data Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_dir)
    
    # Exploratory Data Analysis (EDA)
    exploratory_data_analysis(data_dir)
    
    # Build and Train Model
    build_and_train_model(X_train, y_train, X_val, y_val, model_dir, epochs, batch_size)
    
    # Model Evaluation
    evaluate_model(X_test, y_test, model_dir)


if __name__ == "__main__":
    data_dir = 'data'
    model_dir = 'models'
    run(data_dir, model_dir)
