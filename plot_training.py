import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

def load_data(data_dir):
    X = []
    y = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    for emotion_idx, emotion in enumerate(emotions):
        train_path = os.path.join(data_dir, 'train', emotion)
        test_path = os.path.join(data_dir, 'test', emotion)
        
        # Load training images
        if os.path.exists(train_path):
            for img_file in os.listdir(train_path):
                img = cv2.imread(os.path.join(train_path, img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X.append(img)
                    y.append(emotion_idx)
                    
        # Load test images
        if os.path.exists(test_path):
            for img_file in os.listdir(test_path):
                img = cv2.imread(os.path.join(test_path, img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X.append(img)
                    y.append(emotion_idx)
    
    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y

def plot_training_history(history, save_dir='plots'):
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir='plots'):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(emotions)) + 0.5, emotions, rotation=45)
    plt.yticks(np.arange(len(emotions)) + 0.5, emotions, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    print("Loading model...")
    model = tf.keras.models.load_model('emotion_model.h5')
    
    print("Loading test data...")
    X, y = load_data("FER-2013")
    
    # Reshape data for the model
    X = X.reshape(-1, 48, 48, 1)
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(y, y_pred_classes)
    
    print("Plots have been saved in the 'plots' directory.")

if __name__ == '__main__':
    main() 