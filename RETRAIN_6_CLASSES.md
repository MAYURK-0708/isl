# Retraining Model for 6 Classes

## Overview
The model has been updated to support **6 sign language gestures**:
1. Hello
2. How are you
3. thank you
4. **alright** (NEW)
5. **good morning** (NEW)
6. **good afternoon** (NEW)

## Current Status
⚠️ **IMPORTANT**: The app.py has been updated to expect 6 classes, but you still need to retrain the model with the new data.

The current model weights (`lstm-model\170-0.83.hdf5`) are trained for only 3 classes. Until you retrain with 6 classes, predictions will be incorrect!

## Steps to Retrain

### Step 1: Prepare Your Data
You need video files organized like this:
```
training-data/
├── Hello/              (existing videos)
├── How are you/        (existing videos)
├── thank you/          (existing videos)
├── alright/            (NEW - add videos here)
├── good morning/       (NEW - add videos here)
└── good afternoon/     (NEW - add videos here)
```

**Recommended**: 100+ videos per gesture for good accuracy.

### Step 2: Download INCLUDE Dataset (Optional but Recommended)
The INCLUDE dataset contains high-quality ISL videos including the 3 new gestures.

```bash
# Download the dataset
wget https://zenodo.org/record/4010759/files/Greetings_1of2.zip
wget https://zenodo.org/record/4010759/files/Greetings_2of2.zip

# Unzip
unzip Greetings_1of2.zip
unzip Greetings_2of2.zip
```

The dataset includes:
- `50. Alright/` folder
- `51. Good Morning/` folder
- `52. Good afternoon/` folder

### Step 3: Organize Your Data
Copy videos from INCLUDE dataset to your training-data folder:

```python
import shutil
import os

# Create folders if they don't exist
os.makedirs('training-data/alright', exist_ok=True)
os.makedirs('training-data/good morning', exist_ok=True)
os.makedirs('training-data/good afternoon', exist_ok=True)

# Copy from INCLUDE dataset (adjust paths as needed)
source_base = 'Greetings/'
dest_base = 'training-data/'

# Copy alright videos
for file in os.listdir(os.path.join(source_base, '50. Alright')):
    shutil.copy(
        os.path.join(source_base, '50. Alright', file),
        os.path.join(dest_base, 'alright', file)
    )

# Copy good morning videos
for file in os.listdir(os.path.join(source_base, '51. Good Morning')):
    shutil.copy(
        os.path.join(source_base, '51. Good Morning', file),
        os.path.join(dest_base, 'good morning', file)
    )

# Copy good afternoon videos
for file in os.listdir(os.path.join(source_base, '52. Good afternoon')):
    shutil.copy(
        os.path.join(source_base, '52. Good afternoon', file),
        os.path.join(dest_base, 'good afternoon', file)
    )
```

### Step 4: Update train-model.ipynb

Open `train-model.ipynb` and update the actions array:

**Find this line (around cell 15-20):**
```python
actions = np.array(['Hello', 'How are you', 'Thank You'])
```

**Change it to:**
```python
actions = np.array(['Hello', 'How are you', 'thank you', 'alright', 'good morning', 'good afternoon'])
```

**Also update the data path** to point to your training-data folder:
```python
DATA_PATH = 'training-data'  # or your actual path
```

### Step 5: Run Training
1. Open `train-model.ipynb` in Jupyter Notebook or VS Code
2. Run all cells in sequence
3. The training process will:
   - Convert videos to keypoint arrays (this takes time!)
   - Split data into train/test sets
   - Train the LSTM model
   - Save model weights

**Note**: Training may take 1-4 hours depending on:
- Number of videos
- Your computer's CPU/GPU
- Number of epochs (default: 500-1000)

### Step 6: Save and Use New Model
After training completes, the notebook will save a new model file like:
```
new-model-6classes.hdf5
```

Update `app.py` to use the new model:

**Find this line (around line 47):**
```python
model.load_weights(r"lstm-model\170-0.83.hdf5")
```

**Change it to:**
```python
model.load_weights(r"lstm-model\new-model-6classes.hdf5")
```

### Step 7: Test Your Model
1. Restart the server
2. Visit http://localhost:8000
3. Test each gesture with live camera or upload videos
4. Check confidence scores to verify model accuracy

## Quick Training with Existing Data

If you already have the INCLUDE dataset videos in your training-data folder:

1. Update `train-model.ipynb` actions array to 6 classes
2. Make sure all 6 folders exist in training-data/
3. Run the preprocessing cells (converts videos to numpy arrays)
4. Run the training cells
5. Wait for training to complete
6. Update app.py with new model path
7. Restart server

## Model Architecture
The model automatically adjusts to 6 classes. The output layer will change from:
```python
Dense(3, activation='softmax')  # OLD: 3 classes
```
to:
```python
Dense(6, activation='softmax')  # NEW: 6 classes
```

This happens automatically in `initialize_model()` since it uses `actions.shape[0]`.

## Troubleshooting

### Error: "Input array shape mismatch"
- Make sure all videos are processed correctly
- Check that numpy arrays are saved with shape (45, 258)

### Error: "Not enough data for class X"
- You need at least 30-50 videos per class
- Collect more videos or download INCLUDE dataset

### Low Accuracy
- Collect more diverse videos (different people, angles, lighting)
- Increase training epochs
- Use data augmentation
- Balance dataset (similar number of videos per class)

### Predictions Still Wrong
- Make sure you updated BOTH:
  1. The actions array in app.py (DONE ✓)
  2. The model weights file (YOU NEED TO DO THIS)
- Old model weights won't work with 6 classes!

## Next Steps
1. ✅ Actions array updated in app.py (already done)
2. ⏳ Collect/organize video data for 3 new gestures
3. ⏳ Update train-model.ipynb with 6 classes
4. ⏳ Run training (1-4 hours)
5. ⏳ Update app.py to load new model weights
6. ⏳ Test all 6 gestures

## Need Help?
- See `TRAINING_GUIDE.md` for detailed training instructions
- Use `collect_training_data.py` to record your own videos
- Check the training notebook for preprocessing examples
