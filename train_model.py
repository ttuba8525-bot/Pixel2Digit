import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

def load_and_preprocess_data():
    """
    Loads the MNIST dataset and normalizes pixel values to the range [0.0, 1.0].
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test) as preprocessed numpy arrays.
    """
    try:
        print("Loading MNIST dataset...")
        # Load MNIST dataset (60,000 training, 10,000 testing)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0.0, 1.0] by dividing by 255.0
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        print(f"Data loaded successfully. Train shape: {x_train.shape}, Test shape: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Error during data loading/preprocessing: {str(e)}")
        raise

def build_model():
    """
    Builds the Artificial Neural Network (ANN) model according to the specifications:
    - Input: 28x28 grayscale image
    - Flatten()
    - Dense(256, activation='relu')
    - Dense(128, activation='relu')
    - Dropout(0.3)
    - Dense(64, activation='relu')
    - Dense(10, activation='softmax')
    
    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    print("Building the ANN model...")
    model = models.Sequential([
        # Flatten input 28x28 images into a 784-element vector
        layers.Flatten(input_shape=(28, 28)),
        # Hidden layer 1
        layers.Dense(256, activation='relu'),
        # Hidden layer 2
        layers.Dense(128, activation='relu'),
        # Dropout layer (30% rate) to prevent overfitting
        layers.Dropout(0.3),
        # Hidden layer 3
        layers.Dense(64, activation='relu'),
        # Output layer with 10 units for digits 0-9 (softmax probabilities)
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def train_and_save_model(model, x_train, y_train):
    """
    Compiles and trains the ANN model. Saves the best checkpoint.
    
    Args:
        model (tf.keras.Model): The built Keras model.
        x_train (np.ndarray): Training inputs.
        y_train (np.ndarray): Training labels.
        
    Returns:
        tf.keras.callbacks.History: Training history log.
    """
    # Ensure model output directory exists
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'model.keras')
    
    # Model compilation
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model checkpoint to save the best model weights
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Early stopping to halt training if validation loss stops improving
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print(f"Training completed. Best model saved to: {model_path}")
    return history

def evaluate_and_report(model_path, x_train, y_train, x_test, y_test, history):
    """
    Loads the best saved model, performs evaluation, prints metrics, 
    saves the classification report and confusion matrix image.
    """
    try:
        print(f"Loading best saved model from {model_path} for final evaluation...")
        best_model = tf.keras.models.load_model(model_path)
        
        # Determine the best epoch from training logs (lowest validation loss)
        val_losses = history.history['val_loss']
        best_epoch_idx = np.argmin(val_losses)
        
        train_acc = history.history['accuracy'][best_epoch_idx]
        val_acc = history.history['val_accuracy'][best_epoch_idx]
        
        # Predict on test set
        print("Evaluating on test dataset...")
        y_pred_probs = best_model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate test metrics
        test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
        
        # Calculate Precision, Recall, and F1 Score (weighted and macro)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        
        # Detailed Classification Report
        report_str = classification_report(y_test, y_pred, digits=4)
        
        # Save Classification Report to text file
        report_file_path = 'classification_report.txt'
        with open(report_file_path, 'w') as f:
            f.write("=== DigitVision AI - Classification Report ===\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report_str)
        print(f"Classification report saved to: {report_file_path}")
        
        # Generate and save Confusion Matrix image
        print("Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
        plt.title('DigitVision AI - Confusion Matrix (Test Set)', fontsize=14, pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        cm_image_path = 'confusion_matrix.png'
        plt.savefig(cm_image_path, dpi=300)
        plt.close()
        print(f"Confusion matrix image saved to: {cm_image_path}")
        
        # Print final required metrics to stdout
        print("\n" + "="*40)
        print("           MODEL PERFORMANCE METRICS")
        print("="*40)
        print(f"Training Accuracy:   {train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Test Accuracy:       {test_acc * 100:.2f}%")
        print(f"Precision (Macro):   {precision_macro * 100:.2f}%")
        print(f"Recall (Macro):      {recall_macro * 100:.2f}%")
        print(f"F1 Score (Macro):    {f1_macro * 100:.2f}%")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise

def main():
    # Load and preprocess
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build ANN
    model = build_model()
    model.summary()
    
    # Train model
    history = train_and_save_model(model, x_train, y_train)
    
    # Evaluate model
    model_path = os.path.join('model', 'model.keras')
    evaluate_and_report(model_path, x_train, y_train, x_test, y_test, history)

if __name__ == '__main__':
    main()
