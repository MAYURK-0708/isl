"""
Train LSTM Model for 6-Class ISL Recognition
This script trains the model using your existing training data
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Configuration
ACTIONS = np.array(['Hello', 'How are you', 'thank you', 'alright', 'good morning', 'good afternoon'])
DATA_PATH = 'training-data'
SEQUENCE_LENGTH = 45
NUM_FEATURES = 258  # 33*4 (pose) + 21*3 (left hand) + 21*3 (right hand)
EPOCHS = 200
BATCH_SIZE = 32

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    # Pose keypoints (33 landmarks * 4 features: x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Left hand keypoints (21 landmarks * 3 features: x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand keypoints (21 landmarks * 3 features: x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def process_video(video_path):
    """Process a single video and extract keypoint sequences"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Empty video: {video_path}")
        return None
    
    # Select frames evenly or pad
    if total_frames >= SEQUENCE_LENGTH:
        frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    else:
        frame_indices = list(range(total_frames))
    
    keypoints_sequence = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = holistic.process(image)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)
        
        cap.release()
    
    # Pad or truncate to SEQUENCE_LENGTH
    if len(keypoints_sequence) < SEQUENCE_LENGTH:
        # Pad with the last frame
        last_frame = keypoints_sequence[-1] if keypoints_sequence else np.zeros(NUM_FEATURES)
        while len(keypoints_sequence) < SEQUENCE_LENGTH:
            keypoints_sequence.append(last_frame)
    elif len(keypoints_sequence) > SEQUENCE_LENGTH:
        keypoints_sequence = keypoints_sequence[:SEQUENCE_LENGTH]
    
    return np.array(keypoints_sequence)

def load_dataset():
    """Load all videos and create dataset"""
    X = []  # Videos (keypoint sequences)
    y = []  # Labels
    
    print("\n" + "="*70)
    print("LOADING AND PROCESSING VIDEOS")
    print("="*70)
    
    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        
        if not os.path.exists(action_path):
            print(f"⚠ Warning: Folder not found: {action_path}")
            continue
        
        # Get all video files (case-insensitive)
        video_files = [f for f in os.listdir(action_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
        
        print(f"\n{action}: Processing {len(video_files)} videos...")
        
        processed = 0
        failed = 0
        
        for video_file in video_files:
            video_path = os.path.join(action_path, video_file)
            
            try:
                keypoints = process_video(video_path)
                
                if keypoints is not None and keypoints.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    X.append(keypoints)
                    y.append(action_idx)
                    processed += 1
                    
                    if processed % 10 == 0:
                        print(f"  Processed: {processed}/{len(video_files)}", end='\r')
                else:
                    failed += 1
            except Exception as e:
                print(f"  Error processing {video_file}: {str(e)}")
                failed += 1
        
        print(f"  Processed: {processed}/{len(video_files)} - ✓ Success: {processed}, ✗ Failed: {failed}")
    
    print("\n" + "="*70)
    print(f"Total videos processed: {len(X)}")
    print("="*70)
    
    return np.array(X), np.array(y)

def create_model():
    """Create LSTM model"""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(ACTIONS), activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✓ Training history saved to: {save_path}")

def main():
    print("\n" + "="*70)
    print("  ISL 6-CLASS LSTM MODEL TRAINING")
    print("="*70)
    print(f"Actions: {', '.join(ACTIONS)}")
    print(f"Sequence Length: {SEQUENCE_LENGTH} frames")
    print(f"Features per frame: {NUM_FEATURES}")
    print(f"Epochs: {EPOCHS}")
    print("="*70)
    
    # Load dataset
    print("\nStep 1: Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ No videos loaded! Check your training-data folder structure.")
        return
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=len(ACTIONS))
    
    # Split dataset
    print("\nStep 2: Splitting dataset (80% train, 20% validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nStep 3: Creating model...")
    model = create_model()
    model.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        'lstm-model-6classes-best.hdf5',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("\nStep 4: Training model...")
    print(f"This will take approximately {EPOCHS * len(X_train) / (BATCH_SIZE * 3600):.1f}-{EPOCHS * len(X_train) / (BATCH_SIZE * 1800):.1f} hours")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Save final model
    model.save('lstm-model-6classes.hdf5')
    print("\n✓ Final model saved as: lstm-model-6classes.hdf5")
    print("✓ Best model saved as: lstm-model-6classes-best.hdf5")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    print("\nStep 5: Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Update app.py line 47 to use the new model:")
    print("   model.load_weights(r'lstm-model-6classes-best.hdf5')")
    print("2. Restart the server: python app.py")
    print("3. Test all 6 gestures at http://localhost:8000")
    print("="*70)

if __name__ == "__main__":
    main()
