# Training ISL Model with More Data

This guide explains how to train the ISL recognition model with additional datasets.

## Prerequisites

1. Python environment with all dependencies installed
2. Jupyter Notebook or VS Code with Jupyter extension
3. Training data in video format (MP4, AVI, MOV)

## Steps to Add More Training Data

### 1. Prepare Your Video Data

Create a folder structure like this:
```
training-data/
├── Hello/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ... (100+ videos)
├── How are you/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ... (100+ videos)
└── thank you/
    ├── video1.mp4
    ├── video2.mp4
    └── ... (100+ videos)
```

**Important Requirements:**
- Each gesture should have at least 100 videos
- Videos should be 2-4 seconds long
- Good lighting and clear hand visibility
- Varied performers (different people)
- Different backgrounds and angles
- Consistent gesture execution

### 2. Record Your Own Videos

Use the website's live interaction feature to record training data:

1. Go to http://localhost:8000
2. Click "Live Interaction"
3. Start Camera
4. Perform the gesture
5. Record multiple times (aim for 50-100 recordings per gesture)
6. Save the recordings to appropriate folders

### 3. Data Augmentation (Optional but Recommended)

To increase your dataset, apply augmentations:
- Horizontal flip
- Slight rotations
- Brightness adjustments
- Speed variations (slow-motion, fast-forward)

The original project used video augmentation to expand from 25 to 340 videos per class.

### 4. Open the Training Notebook

Open `train-model.ipynb` in Jupyter or VS Code:

```bash
# In VS Code
code train-model.ipynb

# Or in Jupyter
jupyter notebook train-model.ipynb
```

### 5. Update Data Paths

In the notebook, update the data directory paths:

```python
# Find and update these paths to your data location
data_path = "path/to/your/training-data"
```

### 6. Process Videos to Numpy Arrays

The notebook will:
1. Read each video
2. Extract 45 frames evenly
3. Apply MediaPipe to detect keypoints
4. Save as numpy arrays (shape: 45, 258)

This takes several hours for 1000+ videos.

### 7. Train the Model

Execute the training cells:

```python
# The model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adam', 
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Train
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=[checkpoint])
```

Training may take several hours depending on:
- Dataset size
- Your GPU/CPU
- Number of epochs

### 8. Save the Trained Model

The best model will be saved automatically during training:

```python
model.save_weights('lstm-model/new-model.hdf5')
```

### 9. Test Your New Model

Before deploying, test accuracy:

```python
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 10. Update the Application

Replace the model weights in `app.py`:

```python
# Change this line:
model.load_weights(r"lstm-model\170-0.83.hdf5")

# To:
model.load_weights(r"lstm-model\new-model.hdf5")
```

### 11. Restart the Server

Kill the current server and restart:

```bash
# Stop server
taskkill /F /FI "WINDOWTITLE eq ISL FastAPI Server*"

# Start with new model
python app.py
```

## Tips for Better Training

### 1. **Data Quality Over Quantity**
- Clear, well-lit videos are better than many poor-quality ones
- Consistent gesture execution improves accuracy

### 2. **Balance Your Dataset**
- Equal number of videos per class (e.g., 100 each)
- Avoid class imbalance

### 3. **Use Validation Split**
- Keep 20% of data for validation
- Never train on your test set

### 4. **Monitor Training**
- Watch for overfitting (validation accuracy drops)
- Use early stopping if validation loss increases

### 5. **Experiment with Hyperparameters**
- Learning rate
- Batch size
- Number of LSTM layers
- Number of neurons

## Adding New Gestures

To add gestures beyond "Hello", "How are you", "thank you":

1. Create folders for new gestures
2. Collect 100+ videos per gesture
3. Update the actions array:

```python
# In app.py
actions = np.array(["Hello", "How are you", "thank you", "Please", "Sorry"])
```

4. Update model output layer:

```python
# Change from 3 to number of gestures
model.add(Dense(5, activation='softmax'))  # For 5 gestures
```

5. Retrain from scratch

## Troubleshooting

### Low Accuracy
- Collect more data
- Improve video quality
- Check gesture consistency
- Try data augmentation

### Overfitting
- Add dropout layers
- Reduce model complexity
- Get more training data
- Use data augmentation

### Out of Memory
- Reduce batch size
- Process videos in smaller batches
- Use a smaller model

## Resources

- Training Notebook: `train-model.ipynb`
- Helper Functions: `helper_functions.py`
- INCLUDE Dataset: [zenodo.org/record/4010759](https://zenodo.org/record/4010759)
- MediaPipe Docs: [google.github.io/mediapipe](https://google.github.io/mediapipe)

## Quick Start Commands

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Open training notebook
code train-model.ipynb

# 3. After training, test the model
python -c "from helper_functions import *; # test code here"

# 4. Restart server with new model
python app.py
```

## Expected Timeline

- **Data Collection**: 2-5 hours (for 300 videos)
- **Data Processing**: 3-6 hours (converting to numpy)
- **Model Training**: 2-4 hours (200 epochs)
- **Testing & Validation**: 1 hour
- **Total**: ~1-2 days for complete pipeline

## Need Help?

- Check the original notebook: `train-model.ipynb`
- Review README.md for model details
- Test with small dataset first (10 videos per class)
- Monitor training progress closely

---

**Remember**: Training a good model requires patience and quality data. Start with a small dataset to test the pipeline, then scale up!
